from __future__ import annotations

import argparse
import os
from pathlib import Path

from tinygrad.helpers import Context

from .loader import load_pretrained
from .multimodal import GemmaForConditionalGeneration
from .processing import GemmaMultimodalProcessor
from .runtime import default_device
from .tokenizer import GemmaTokenizer

DEFAULT_MAX_BEAM = 16


def parse_prompt_tokens(raw: str) -> list[int]:
  return [int(part.strip()) for part in raw.split(",") if part.strip()]


def resolve_beam(raw: str) -> int:
  value = raw.strip().lower()
  if value == "max":
    beam = int(os.getenv("TINYGRAD_GEMMA_MAX_BEAM", str(DEFAULT_MAX_BEAM)))
    if beam < 1:
      raise ValueError("TINYGRAD_GEMMA_MAX_BEAM must be positive when --beam max is used")
    return beam
  beam = int(value)
  if beam < 0:
    raise ValueError("beam must be non-negative")
  return beam


def main():
  parser = argparse.ArgumentParser(description="Run Gemma 4 as a native tinygrad model")
  parser.add_argument("--model-dir", required=True, help="Local Hugging Face checkpoint directory")
  parser.add_argument("--prompt", help="Text prompt (supports tokenizer.json or sentencepiece tokenizers)")
  parser.add_argument("--prompt-tokens", help="Comma-separated token ids")
  parser.add_argument("--image", action="append", default=[], help="Path to an image file. Repeat for multiple images.")
  parser.add_argument("--audio", action="append", default=[], help="Path to a WAV file. Repeat for multiple audio segments.")
  parser.add_argument("--max-new-tokens", type=int, default=64)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--device", default="auto", help=f"Tinygrad device to use. 'auto' resolves to {default_device().lower()} on this machine")
  parser.add_argument("--beam", default="0", help="Tinygrad BEAM width. Use 0 to disable or 'max' for the widest supported search.")
  args = parser.parse_args()

  has_multimodal_inputs = bool(args.image or args.audio)
  if args.prompt and args.prompt_tokens:
    raise SystemExit("provide at most one of --prompt or --prompt-tokens")
  if not args.prompt and not args.prompt_tokens and not has_multimodal_inputs:
    raise SystemExit("provide --prompt, --prompt-tokens, or at least one multimodal input")
  if has_multimodal_inputs and args.prompt_tokens:
    raise SystemExit("--prompt-tokens cannot be combined with --image or --audio; use --prompt so placeholders can be expanded")

  model_dir = Path(args.model_dir)
  model = load_pretrained(model_dir, device=args.device)

  tokenizer = None
  generation_kwargs = {}
  if has_multimodal_inputs:
    if not isinstance(model, GemmaForConditionalGeneration):
      raise SystemExit("this checkpoint does not expose Gemma 4 vision/audio towers")
    processor = GemmaMultimodalProcessor.from_pretrained(model_dir, model.config)
    prepared = processor.prepare_inputs(args.prompt, images=args.image, audio=args.audio)
    tokenizer = processor.tokenizer
    prompt_ids = prepared.input_ids
    generation_kwargs = {
      "pixel_values": prepared.pixel_values,
      "image_position_ids": prepared.image_position_ids,
      "input_features": prepared.input_features,
      "input_features_mask": prepared.input_features_mask,
    }
  elif args.prompt_tokens:
    prompt_ids = parse_prompt_tokens(args.prompt_tokens)
  else:
    tokenizer = GemmaTokenizer.from_pretrained(model_dir)
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)

  stop_ids = {tokenizer.eos_id} if tokenizer is not None else None
  beam = resolve_beam(args.beam)
  if beam > 0 and "PARALLEL" not in os.environ and str(getattr(model, "device", "CPU")).upper() in ("CPU", "PYTHON"):
    os.environ["PARALLEL"] = str(os.cpu_count() or 1)
  with Context(BEAM=beam):
    generated = list(
      model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stop_token_ids=stop_ids,
        **generation_kwargs,
      )
    )

  if tokenizer is None:
    print(",".join(str(tok) for tok in generated))
  else:
    print(tokenizer.decode(generated))
