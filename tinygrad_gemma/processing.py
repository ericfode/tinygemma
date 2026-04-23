from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import GemmaConditionalConfig
from .tokenizer import GemmaTokenizer


def _load_multimodal_backends():
  try:
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
  except ImportError as exc:  # pragma: no cover - exercised in integration environments
    raise ImportError(
      "Gemma 4 multimodal preprocessing requires transformers and pillow; install tinygrad-gemma[multimodal]"
    ) from exc
  return Gemma4AudioFeatureExtractor, Gemma4ImageProcessorPil


def _replace_placeholders(prompt: str, placeholder: str, replacements: list[str]) -> str:
  if not replacements:
    return prompt
  count = len(re.findall(re.escape(placeholder), prompt))
  if count != len(replacements):
    raise ValueError(f"prompt contains {count} occurrences of {placeholder!r}, but {len(replacements)} inputs were provided")
  replacements_iter = iter(replacements)
  return re.sub(re.escape(placeholder), lambda _: next(replacements_iter), prompt)


def _decode_pcm_samples(raw: bytes, sample_width: int) -> np.ndarray:
  if sample_width == 1:
    audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    return (audio - 128.0) / 128.0
  if sample_width == 2:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
  if sample_width == 3:
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    value = packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8) | (packed[:, 2].astype(np.int32) << 16)
    sign = value & 0x800000
    value = value - (sign << 1)
    return value.astype(np.float32) / 8388608.0
  if sample_width == 4:
    return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
  raise ValueError(f"unsupported WAV sample width {sample_width}")


def _resample_audio(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
  if source_rate == target_rate or waveform.size == 0:
    return waveform.astype(np.float32, copy=False)
  target_length = max(1, int(round(waveform.shape[0] * target_rate / source_rate)))
  x_old = np.arange(waveform.shape[0], dtype=np.float32)
  x_new = np.linspace(0.0, max(waveform.shape[0] - 1, 0), target_length, dtype=np.float32)
  return np.interp(x_new, x_old, waveform.astype(np.float32)).astype(np.float32)


def load_wav(path: str | Path, *, target_rate: int = 16_000) -> np.ndarray:
  with wave.open(str(path), "rb") as wav_file:
    channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    sample_rate = wav_file.getframerate()
    frame_count = wav_file.getnframes()
    raw = wav_file.readframes(frame_count)
  waveform = _decode_pcm_samples(raw, sample_width)
  if channels > 1:
    waveform = waveform.reshape(-1, channels).mean(axis=1)
  waveform = _resample_audio(waveform, sample_rate, target_rate)
  return waveform.astype(np.float32, copy=False)


@dataclass(slots=True)
class GemmaPreparedInputs:
  input_ids: list[int]
  pixel_values: np.ndarray | None = None
  image_position_ids: np.ndarray | None = None
  input_features: np.ndarray | None = None
  input_features_mask: np.ndarray | None = None


class GemmaMultimodalProcessor:
  def __init__(self, config: GemmaConditionalConfig, tokenizer: GemmaTokenizer):
    self.config = config
    self.tokenizer = tokenizer
    self.image_processor = None
    self.audio_feature_extractor = None
    Gemma4AudioFeatureExtractor, Gemma4ImageProcessorPil = _load_multimodal_backends()
    if config.vision_config is not None:
      self.image_processor = Gemma4ImageProcessorPil(
        patch_size=config.vision_config.patch_size,
        pooling_kernel_size=config.vision_config.pooling_kernel_size,
        max_soft_tokens=config.vision_config.default_output_length,
      )
    if config.audio_config is not None:
      self.audio_feature_extractor = Gemma4AudioFeatureExtractor()

  @classmethod
  def from_pretrained(cls, model_dir: str | Path, config: GemmaConditionalConfig | None = None) -> "GemmaMultimodalProcessor":
    model_dir = Path(model_dir)
    conditional_config = config or GemmaConditionalConfig.from_path(model_dir / "config.json")
    return cls(conditional_config, GemmaTokenizer.from_pretrained(model_dir))

  def _default_prompt(self, num_images: int, num_audio: int) -> str:
    pieces = []
    if num_images:
      if self.tokenizer.image_token is None:
        raise ValueError("tokenizer does not define an image placeholder token")
      pieces.extend([self.tokenizer.image_token] * num_images)
    if num_audio:
      if self.tokenizer.audio_token is None:
        raise ValueError("tokenizer does not define an audio placeholder token")
      pieces.extend([self.tokenizer.audio_token] * num_audio)
    if not pieces:
      raise ValueError("multimodal processor requires text, images, or audio")
    return " ".join(pieces)

  def _prepare_images(self, image_paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if self.image_processor is None:
      raise ValueError("image inputs were provided, but this checkpoint has no vision tower")
    try:
      from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised in integration environments
      raise ImportError("image inputs require pillow; install tinygrad-gemma[multimodal]") from exc
    images = []
    for path in image_paths:
      with Image.open(path) as image:
        images.append(image.convert("RGB"))
    batch = self.image_processor(images, return_tensors="np", do_convert_rgb=True)
    return batch["pixel_values"], batch["image_position_ids"].astype(np.int32), [int(x) for x in batch["num_soft_tokens_per_image"]]

  def _prepare_audio(self, audio_paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if self.audio_feature_extractor is None:
      raise ValueError("audio inputs were provided, but this checkpoint has no audio tower")
    audio = [load_wav(path, target_rate=self.audio_feature_extractor.sampling_rate) for path in audio_paths]
    batch = self.audio_feature_extractor(audio, return_tensors="np")
    masks = batch["input_features_mask"].astype(bool)
    token_counts = [int(mask[::2][::2].sum()) for mask in masks]
    return batch["input_features"], masks, token_counts

  def prepare_inputs(
    self,
    text: str | None,
    *,
    images: list[str | Path] | None = None,
    audio: list[str | Path] | None = None,
    add_bos: bool = True,
  ) -> GemmaPreparedInputs:
    images = [] if images is None else [Path(path) for path in images]
    audio = [] if audio is None else [Path(path) for path in audio]
    prompt = text if text not in (None, "") else self._default_prompt(len(images), len(audio))

    pixel_values = None
    image_position_ids = None
    if images:
      pixel_values, image_position_ids, num_soft_tokens = self._prepare_images(images)
      if not self.tokenizer.image_token or not self.tokenizer.boi_token or not self.tokenizer.eoi_token:
        raise ValueError("tokenizer does not define the Gemma 4 image special tokens")
      prompt = _replace_placeholders(
        prompt,
        self.tokenizer.image_token,
        [f"{self.tokenizer.boi_token}{self.tokenizer.image_token * count}{self.tokenizer.eoi_token}" for count in num_soft_tokens],
      )

    input_features = None
    input_features_mask = None
    if audio:
      input_features, input_features_mask, num_audio_tokens = self._prepare_audio(audio)
      if not self.tokenizer.audio_token or not self.tokenizer.boa_token or not self.tokenizer.eoa_token:
        raise ValueError("tokenizer does not define the Gemma 4 audio special tokens")
      prompt = _replace_placeholders(
        prompt,
        self.tokenizer.audio_token,
        [f"{self.tokenizer.boa_token}{self.tokenizer.audio_token * count}{self.tokenizer.eoa_token}" for count in num_audio_tokens],
      )

    return GemmaPreparedInputs(
      input_ids=self.tokenizer.encode(prompt, add_bos=add_bos),
      pixel_values=pixel_values,
      image_position_ids=image_position_ids,
      input_features=input_features,
      input_features_mask=input_features_mask,
    )
