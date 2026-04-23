from __future__ import annotations

import json
from pathlib import Path

from tinygrad import nn
from tinygrad.helpers import Context

from .config import GemmaConditionalConfig, GemmaConfig, load_config_dict
from .model import GemmaForCausalLM
from .multimodal import GemmaForConditionalGeneration
from .runtime import prepare_device


GemmaAnyConfig = GemmaConfig | GemmaConditionalConfig
GemmaAnyModel = GemmaForCausalLM | GemmaForConditionalGeneration


def _has_multimodal_towers(config: GemmaAnyConfig) -> bool:
  return isinstance(config, GemmaConditionalConfig) and (config.vision_config is not None or config.audio_config is not None)


def load_config(model_dir: str | Path) -> GemmaAnyConfig:
  model_dir = Path(model_dir)
  raw_config = load_config_dict(model_dir / "config.json")
  if raw_config.get("vision_config") is not None or raw_config.get("audio_config") is not None:
    return GemmaConditionalConfig.from_dict(raw_config)
  return GemmaConfig.from_dict(raw_config)


def load_text_config(model_dir: str | Path) -> GemmaConfig:
  config = load_config(model_dir)
  return config.text_config if isinstance(config, GemmaConditionalConfig) else config


def resolve_weight_files(model_dir: str | Path) -> list[Path]:
  model_dir = Path(model_dir)
  index_path = model_dir / "model.safetensors.index.json"
  if index_path.exists():
    index_data = json.loads(index_path.read_text())
    files = list(dict.fromkeys(index_data["weight_map"].values()))
    return [model_dir / name for name in files]
  direct = model_dir / "model.safetensors"
  if direct.exists():
    return [direct]
  files = sorted(model_dir.glob("*.safetensors"))
  if files:
    return files
  raise FileNotFoundError(f"no .safetensors checkpoint found under {model_dir}")


def load_state_dict(model_dir: str | Path) -> dict:
  state_dict = {}
  for path in resolve_weight_files(model_dir):
    state_dict.update(nn.state.safe_load(path))
  return state_dict


def normalize_state_dict_keys(state_dict: dict, config: GemmaAnyConfig) -> dict:
  normalized = {}
  multimodal = _has_multimodal_towers(config)
  for key, value in state_dict.items():
    if multimodal and ".depthwise_conv1d.weight" in key:
      key = key.replace(".depthwise_conv1d.weight", ".depthwise_conv1d.conv.weight")
    if key.startswith("model.language_model."):
      mapped = ("model.language_model." if multimodal else "model.") + key[len("model.language_model."):]
      normalized[mapped] = value
    elif key.startswith("language_model."):
      mapped = ("model.language_model." if multimodal else "model.") + key[len("language_model."):]
      normalized[mapped] = value
    elif key.startswith("model.text_model."):
      mapped = ("model.language_model." if multimodal else "model.") + key[len("model.text_model."):]
      normalized[mapped] = value
    else:
      normalized[key] = value
  if config.tie_word_embeddings:
    embed_key = "model.language_model.embed_tokens.weight" if multimodal else "model.embed_tokens.weight"
    if "lm_head.weight" not in normalized and embed_key in normalized:
      normalized["lm_head.weight"] = normalized[embed_key]
  return normalized


def load_pretrained(
  model_dir: str | Path,
  *,
  strict: bool = True,
  verbose: bool = False,
  device: str | None = None,
) -> GemmaAnyModel:
  config = load_config(model_dir)
  state_dict = normalize_state_dict_keys(load_state_dict(model_dir), config)
  model_type = GemmaForConditionalGeneration if _has_multimodal_towers(config) else GemmaForCausalLM
  if device is None:
    model = model_type(config)  # type: ignore[arg-type]
    nn.state.load_state_dict(model, state_dict, strict=strict, verbose=verbose, consume=True)
    return model
  target_device = prepare_device(device)
  with Context(DEV=target_device):
    model = model_type(config)  # type: ignore[arg-type]
    nn.state.load_state_dict(model, state_dict, strict=strict, verbose=verbose, consume=True)
  return model
