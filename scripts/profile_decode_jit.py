from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tinygrad import Tensor, TinyJit, Variable
from tinygrad.device import Buffer
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import Context, GlobalCounters
from tinygrad.nn import state as nn_state
from tinygrad.uop.ops import sym_infer

import tinygrad
from tinygrad_gemma import load_pretrained
from tinygrad_gemma.loader import load_quantization_manifest, resolve_weight_files
from tinygrad_gemma.model import (
  GemmaAttention,
  GemmaCache,
  GemmaCacheEntry,
  GemmaDecoderLayer,
  GemmaMLP,
  GemmaModel,
  RMSNorm,
  TextScaledEmbedding,
  gelu_pytorch_tanh,
)
from tinygrad_gemma.multimodal import GemmaForConditionalGeneration
from tinygrad_gemma.runtime import prepare_device


DEFAULT_MODEL_DIR = Path("/Users/ericfode/Downloads/tinygrad-gemma/checkpoints/gemma-4-E2B-int8")
DEFAULT_OUT = Path("benchmarks/gemma4-metal-postwindow-jit-profile-current.json")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _source_range(obj) -> tuple[int, int]:
  lines, start = inspect.getsourcelines(obj)
  return start, start + len(lines) - 1


CATEGORY_RANGES = {
  "tinygrad_gemma.model": [
    ("mlp", _source_range(GemmaMLP.__call__)),
    ("mlp", _source_range(gelu_pytorch_tanh)),
    ("attention", _source_range(GemmaAttention.__call__)),
    ("embedding_per_layer", _source_range(TextScaledEmbedding.__call__)),
    ("embedding_per_layer", _source_range(GemmaModel.project_per_layer_inputs)),
    ("decoder_residual", _source_range(GemmaDecoderLayer.__call__)),
    ("norm", _source_range(RMSNorm.__call__)),
  ],
  "tinygrad_gemma.multimodal": [
    ("logits_argmax", _source_range(GemmaForConditionalGeneration.logits)),
    ("logits_argmax", _source_range(GemmaForConditionalGeneration.sample_next)),
  ],
}


def _line_from_caller(caller: str) -> int | None:
  parts = caller.split(":")
  if len(parts) < 2:
    return None
  try:
    return int(parts[1])
  except ValueError:
    return None


def _metadata_payload(metadata) -> list[dict[str, str]]:
  return [{"name": str(getattr(m, "name", "")), "caller": str(getattr(m, "caller", ""))} for m in metadata]


def strip_ansi(value: str) -> str:
  return ANSI_RE.sub("", value)


def classify_kernel(metadata) -> str:
  callers = [str(getattr(m, "caller", "")) for m in metadata]
  categories = set()
  for caller in callers:
    caller_module = caller.split(":", 1)[0]
    line = _line_from_caller(caller)
    if line is None:
      continue
    for category, (start, end) in CATEGORY_RANGES.get(caller_module, []):
      if start <= line <= end:
        categories.add(category)
  if "logits_argmax" in categories:
    return "logits_argmax"
  if "mlp" in categories:
    return "mlp"
  if "attention" in categories:
    return "attention"
  if "embedding_per_layer" in categories:
    return "embedding_per_layer"
  if "decoder_residual" in categories:
    return "decoder_residual"
  if "norm" in categories:
    return "norm"
  return "other"


def checkpoint_quantization_summary(model_dir: Path) -> dict[str, Any]:
  manifest = load_quantization_manifest(model_dir)
  tensors = {} if manifest is None else manifest.get("tensors", {})
  dtypes: dict[str, int] = defaultdict(int)
  sample_shapes: dict[str, str] = {}
  for path in resolve_weight_files(model_dir):
    state = nn_state.safe_load(path)
    for name, tensor in state.items():
      dtype_name = str(getattr(tensor.dtype, "name", tensor.dtype))
      dtypes[dtype_name] += 1
      if name in (
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.per_layer_model_projection.weight",
      ):
        sample_shapes[name] = f"{tuple(tensor.shape)} {tensor.dtype}"
  return {
    "manifest_method": None if manifest is None else manifest.get("method"),
    "manifest_quantized_tensor_count": len(tensors),
    "raw_dtype_counts": dict(sorted(dtypes.items())),
    "sample_tensors": sample_shapes,
  }


def language_model(model):
  return getattr(model.model, "language_model", model.model)


def build_zero_cache(model, context_length: int, max_length: int) -> GemmaCache:
  lm = language_model(model)
  dtype = lm.embed_tokens.weight.dtype
  cache = GemmaCache.empty(len(lm.layers), max_length=max_length)
  for idx, layer in enumerate(lm.layers):
    attn = layer.self_attn
    key = Tensor.zeros(
      1,
      attn.num_key_value_heads,
      max_length,
      attn.head_dim,
      device=model.device,
      dtype=dtype,
    ).contiguous().realize()
    value = Tensor.zeros(
      1,
      attn.num_key_value_heads,
      max_length,
      attn.head_dim,
      device=model.device,
      dtype=dtype,
    ).contiguous().realize()
    cache.entries[idx] = GemmaCacheEntry(key=key, value=value, length=context_length)
  cache.past_seen_tokens = context_length
  cache.decode_sliding_window = True
  return cache


def run_captured_items(captured, token: Tensor, start_var) -> tuple[list[dict[str, Any]], dict[str, int]]:
  input_buffers, var_vals, names, expected_info = _prepare_jit_inputs((token, start_var), {})
  if names != captured.expected_names:
    raise RuntimeError(f"JIT input names changed: {names!r} != {captured.expected_names!r}")
  if expected_info != captured.expected_input_info:
    raise RuntimeError("JIT input metadata changed before profiling")

  for idx, offset, device, size, dtype in captured.extra_view_inputs:
    input_buffers.append(Buffer(device, size, dtype, base=input_buffers[idx], offset=offset).ensure_allocated())
  for (item_idx, buffer_idx), input_idx in captured._input_replace.items():
    captured._jit_cache[item_idx].bufs[buffer_idx] = input_buffers[input_idx]

  rows = []
  try:
    if captured._first_run:
      for item in captured.jit_cache:
        for buffer in item.bufs:
          if buffer is not None:
            buffer.ensure_allocated()
      captured._first_run = False

    for ordinal, item in enumerate(captured._jit_cache):
      item.lower()
      prg = item.prg
      metadata = _metadata_payload(item.metadata)
      category = classify_kernel(item.metadata)
      display_name = "" if prg is None else strip_ansi(prg.display_name)
      device = "" if prg is None else prg.device
      est_ops = sym_infer(prg.estimates.ops, var_vals) if isinstance(prg, CompiledRunner) else 0
      est_mem = sym_infer(prg.estimates.mem, var_vals) if isinstance(prg, CompiledRunner) else 0
      start = time.perf_counter()
      elapsed = item.run(var_vals, wait=True, jit=True, do_update_stats=False)
      wall_elapsed = time.perf_counter() - start
      rows.append({
        "ordinal": ordinal,
        "category": category,
        "display_name": display_name,
        "device": device,
        "elapsed_ms": (elapsed if elapsed is not None else wall_elapsed) * 1000.0,
        "est_ops": int(est_ops),
        "est_mem": int(est_mem),
        "metadata": metadata,
      })
  finally:
    captured._clear_inputs()
  return rows, var_vals


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
  by_category: dict[str, dict[str, Any]] = {}
  for row in rows:
    category = row["category"]
    entry = by_category.setdefault(category, {"kernel_count": 0, "elapsed_ms": 0.0, "est_ops": 0, "est_mem": 0})
    entry["kernel_count"] += 1
    entry["elapsed_ms"] += row["elapsed_ms"]
    entry["est_ops"] += row["est_ops"]
    entry["est_mem"] += row["est_mem"]
  total_elapsed = sum(row["elapsed_ms"] for row in rows)
  total_ops = sum(row["est_ops"] for row in rows)
  total_mem = sum(row["est_mem"] for row in rows)
  for entry in by_category.values():
    entry["elapsed_share"] = entry["elapsed_ms"] / total_elapsed if total_elapsed > 0 else 0.0
    entry["ops_share"] = entry["est_ops"] / total_ops if total_ops else 0.0
  return {
    "kernel_count": len(rows),
    "elapsed_ms": total_elapsed,
    "est_ops": total_ops,
    "est_mem": total_mem,
    "by_category": dict(sorted(by_category.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
    "top_elapsed_kernels": sorted(rows, key=lambda row: row["elapsed_ms"], reverse=True)[:20],
    "top_est_ops_kernels": sorted(rows, key=lambda row: row["est_ops"], reverse=True)[:20],
  }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["ordinal", "category", "display_name", "device", "elapsed_ms", "est_ops", "est_mem", "metadata"])
    writer.writeheader()
    for row in rows:
      writer.writerow({
        **{key: row[key] for key in ("ordinal", "category", "display_name", "device", "elapsed_ms", "est_ops", "est_mem")},
        "metadata": json.dumps(row["metadata"], sort_keys=True),
      })


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile a synthetic post-window Gemma 4 decode TinyJit on tinygrad METAL.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--device", default="METAL")
  parser.add_argument("--context-length", type=int, default=700)
  parser.add_argument("--profile-start", type=int, help="Decode position to profile. Defaults to context length plus 3.")
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--csv-out", type=Path)
  args = parser.parse_args()

  resolved_device = prepare_device(args.device)
  model = load_pretrained(args.model_dir, device=resolved_device)
  lm = language_model(model)
  sliding_start = model._sliding_decode_start()
  if sliding_start is None:
    raise SystemExit("model does not expose a Metal sliding decode start")
  if args.context_length < sliding_start:
    raise SystemExit(f"--context-length must be >= sliding start {sliding_start}")

  profile_start = args.profile_start if args.profile_start is not None else args.context_length + 3
  max_length = profile_start + 2
  cache = build_zero_cache(model, args.context_length, max_length)
  token = Tensor([[2]], dtype="int32", device=model.device).realize()

  with Context(JIT=2, BEAM=0, TRACEMETA=2):
    rollout_jit = TinyJit(lambda token, start_pos: model._rollout_next_token(token, start_pos, cache, 0.0, decode_sliding_window=True))
    for offset in range(3):
      out = rollout_jit(token, Variable("gemma_start_pos_window", sliding_start, max_length - 1).bind(args.context_length + offset))
      out.realize()
      cache.set_active_length(args.context_length + offset + 1)
    captured = rollout_jit.captured
    if captured is None:
      raise RuntimeError("decode TinyJit did not capture")
    GlobalCounters.reset()
    rows, var_vals = run_captured_items(captured, token, Variable("gemma_start_pos_window", sliding_start, max_length - 1).bind(profile_start))

  csv_out = args.csv_out or args.out.with_suffix(".csv")
  write_csv(csv_out, rows)
  payload = {
    "model_dir": str(args.model_dir),
    "device": resolved_device,
    "tinygrad_path": str(Path(tinygrad.__file__).resolve()),
    "context_length": args.context_length,
    "profile_start": profile_start,
    "sliding_start": sliding_start,
    "max_length": max_length,
    "jit_mode": "JIT=2 ungraphed per-kernel timing",
    "tensor_dtype": str(lm.embed_tokens.weight.dtype),
    "text_config": {
      "num_hidden_layers": lm.config.num_hidden_layers,
      "num_attention_heads": lm.config.num_attention_heads,
      "num_key_value_heads": lm.config.num_key_value_heads,
      "hidden_size": lm.config.hidden_size,
      "intermediate_size": lm.config.intermediate_size,
      "vocab_size": lm.config.vocab_size,
      "sliding_window": lm.config.sliding_window,
      "layer_types": lm.config.layer_types,
    },
    "quantization": checkpoint_quantization_summary(args.model_dir),
    "var_vals": var_vals,
    "summary": summarize_rows(rows),
    "csv": str(csv_out),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "out": str(args.out),
    "csv": str(csv_out),
    "kernel_count": payload["summary"]["kernel_count"],
    "elapsed_ms": round(payload["summary"]["elapsed_ms"], 3),
    "by_category": {
      key: {
        "kernel_count": value["kernel_count"],
        "elapsed_ms": round(value["elapsed_ms"], 3),
        "elapsed_share": round(value["elapsed_share"], 4),
      }
      for key, value in payload["summary"]["by_category"].items()
    },
  }, sort_keys=True))


if __name__ == "__main__":
  main()
