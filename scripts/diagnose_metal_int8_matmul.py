from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import tinygrad
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.codegen.opt import tc
from tinygrad.device import Buffer
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import Context
from tinygrad.uop.ops import sym_infer

from tinygrad_gemma.runtime import prepare_device


DEFAULT_OUT = Path("benchmarks/gemma4-metal-int8-matmul-lowering-diagnostic-current.json")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _build_inputs(device: str, token_count: int, in_features: int, out_features: int, seed: int) -> dict[str, Tensor]:
  rng = np.random.default_rng(seed + token_count)
  x_np = rng.standard_normal((1, token_count, in_features)).astype(np.float32)
  q_np = rng.integers(-127, 128, size=(out_features, in_features), dtype=np.int8)
  scale_np = rng.uniform(0.001, 0.02, size=(out_features,)).astype(np.float32)
  w_np = q_np.astype(np.float32) * scale_np.reshape(out_features, 1)
  return {
    "x": Tensor(x_np, device=device).cast(dtypes.bfloat16).contiguous().realize(),
    "qweight": Tensor(q_np, device=device, dtype=dtypes.int8).contiguous().realize(),
    "scale": Tensor(scale_np, device=device, dtype=dtypes.float32).contiguous().realize(),
    "dequant_weight": Tensor(w_np, device=device).cast(dtypes.bfloat16).contiguous().realize(),
  }


def _rowwise_int8_linear(x: Tensor, qweight: Tensor, scale: Tensor) -> Tensor:
  out = x.matmul(qweight.transpose(), dtype="float")
  out = out * scale.reshape(*([1] * (out.ndim - 1)), scale.shape[0])
  return out.cast(x.dtype).contiguous()


def _bf16_linear(x: Tensor, weight: Tensor) -> Tensor:
  return x.matmul(weight.transpose(), dtype="float").cast(x.dtype).contiguous()


def _source_flags(src: str) -> dict[str, bool]:
  lowered = src.lower()
  return {
    "has_simdgroup_multiply_accumulate": "simdgroup_multiply_accumulate" in src,
    "has_wmma_name": "wmma" in lowered,
    "mentions_int8": "int8" in lowered or "char" in lowered,
  }


def _strip_ansi(value: str) -> str:
  return ANSI_RE.sub("", value)


def _run_captured_items(captured, args: tuple[Tensor, ...]) -> list[dict[str, Any]]:
  input_buffers, var_vals, names, expected_info = _prepare_jit_inputs(args, {})
  if names != captured.expected_names:
    raise RuntimeError(f"JIT input names changed: {names!r} != {captured.expected_names!r}")
  if expected_info != captured.expected_input_info:
    raise RuntimeError("JIT input metadata changed before diagnostic replay")

  for idx, offset, device, size, dtype in captured.extra_view_inputs:
    input_buffers.append(Buffer(device, size, dtype, base=input_buffers[idx], offset=offset).ensure_allocated())
  for (item_idx, buffer_idx), input_idx in captured._input_replace.items():
    captured._jit_cache[item_idx].bufs[buffer_idx] = input_buffers[input_idx]

  rows: list[dict[str, Any]] = []
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
      src = getattr(getattr(prg, "p", None), "src", "") if isinstance(prg, CompiledRunner) else ""
      var_vals_with_fixed = var_vals | item.fixedvars
      elapsed = item.run(var_vals, wait=True, jit=True, do_update_stats=False)
      rows.append({
        "ordinal": ordinal,
        "display_name": "" if prg is None else _strip_ansi(prg.display_name),
        "device": "" if prg is None else prg.device,
        "elapsed_ms": 1000.0 * float(elapsed or 0.0),
        "est_ops": 0 if not isinstance(prg, CompiledRunner) else int(sym_infer(prg.estimates.ops, var_vals_with_fixed)),
        "est_mem": 0 if not isinstance(prg, CompiledRunner) else int(sym_infer(prg.estimates.mem, var_vals_with_fixed)),
        "source_flags": _source_flags(src),
      })
  finally:
    captured._clear_inputs()
  return rows


def _capture_and_measure(name: str, fn, args: tuple[Tensor, ...], repeats: int) -> dict[str, Any]:
  with Context(JIT=2, BEAM=0):
    jit = TinyJit(fn)
    for _ in range(3):
      jit(*args).realize()
    captured = jit.captured
    if captured is None:
      raise RuntimeError(f"{name} did not capture a TinyJit")
    samples = [_run_captured_items(captured, args) for _ in range(repeats)]

  total_ms = [sum(row["elapsed_ms"] for row in sample) for sample in samples]
  first_rows = samples[0]
  source_flags = {
    key: any(row["source_flags"][key] for row in first_rows)
    for key in ("has_simdgroup_multiply_accumulate", "has_wmma_name", "mentions_int8")
  }
  return {
    "name": name,
    "kernel_count": len(first_rows),
    "median_ms": statistics.median(total_ms),
    "min_ms": min(total_ms),
    "max_ms": max(total_ms),
    "source_flags": source_flags,
    "kernels": first_rows,
  }


def _metal_tensor_core_dtypes() -> list[dict[str, str]]:
  return [
    {"dtype_in": str(core.dtype_in), "dtype_out": str(core.dtype_out), "dims": str(core.dims)}
    for core in tc.metal
  ]


def _variant_for_token_count(shapes: list[dict[str, Any]], token_count: int, name: str) -> dict[str, Any] | None:
  for shape in shapes:
    if shape["token_count"] != token_count:
      continue
    for variant in shape["variants"]:
      if variant["name"] == name:
        return variant
  return None


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
  device = prepare_device(args.device)
  shapes = []
  for token_count in args.token_counts:
    tensors = _build_inputs(device, token_count, args.in_features, args.out_features, args.seed)
    rowwise = _capture_and_measure(
      "rowwise_int8_linear",
      _rowwise_int8_linear,
      (tensors["x"], tensors["qweight"], tensors["scale"]),
      args.repeats,
    )
    dequant = _capture_and_measure(
      "bf16_dequantized_weight_linear",
      _bf16_linear,
      (tensors["x"], tensors["dequant_weight"]),
      args.repeats,
    )
    shapes.append({
      "token_count": token_count,
      "in_features": args.in_features,
      "out_features": args.out_features,
      "variants": [rowwise, dequant],
    })

  int8_tensor_core_registered = any("int8" in item["dtype_in"] for item in _metal_tensor_core_dtypes())
  rowwise_uses_simdgroup_any_shape = any(
    variant["source_flags"]["has_simdgroup_multiply_accumulate"]
    for shape in shapes
    for variant in shape["variants"]
    if variant["name"] == "rowwise_int8_linear"
  )
  decode_rowwise = _variant_for_token_count(shapes, 1, "rowwise_int8_linear")
  decode_dequant = _variant_for_token_count(shapes, 1, "bf16_dequantized_weight_linear")
  decode_rowwise_ms = None if decode_rowwise is None else decode_rowwise["median_ms"]
  decode_dequant_ms = None if decode_dequant is None else decode_dequant["median_ms"]
  decode_ratio = None if not decode_rowwise_ms or not decode_dequant_ms else decode_rowwise_ms / decode_dequant_ms
  return {
    "device": device,
    "tinygrad_path": str(Path(tinygrad.__file__).resolve()),
    "diagnostic": "Gemma 4 decode-shape rowwise int8 linear lowering on tinygrad METAL",
    "seed": args.seed,
    "repeats": args.repeats,
    "metal_tensor_core_dtypes": _metal_tensor_core_dtypes(),
    "summary": {
      "metal_tensor_core_registers_int8": int8_tensor_core_registered,
      "rowwise_int8_linear_any_shape_uses_simdgroup_mma": rowwise_uses_simdgroup_any_shape,
      "decode_token1_rowwise_int8_uses_simdgroup_mma": None if decode_rowwise is None else decode_rowwise["source_flags"]["has_simdgroup_multiply_accumulate"],
      "decode_token1_rowwise_int8_median_ms": decode_rowwise_ms,
      "decode_token1_bf16_dequantized_weight_median_ms": decode_dequant_ms,
      "decode_token1_rowwise_int8_vs_bf16_dequantized_ratio": decode_ratio,
      "interpretation": (
        "Current one-token rowwise int8 decode linear is not using Metal simdgroup MMA and is not materially faster than bf16 dequantized-weight linear."
        if not int8_tensor_core_registered and decode_rowwise is not None and not decode_rowwise["source_flags"]["has_simdgroup_multiply_accumulate"]
        else "Inspect kernels before treating this as proof against int8 tensor-core use."
      ),
    },
    "shapes": shapes,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Diagnose tinygrad METAL lowering for Gemma 4 rowwise-int8 decode matmul.")
  parser.add_argument("--device", default="METAL")
  parser.add_argument("--token-counts", type=int, nargs="+", default=[1, 8])
  parser.add_argument("--in-features", type=int, default=1536)
  parser.add_argument("--out-features", type=int, default=12288)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--seed", type=int, default=20260424)
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = parser.parse_args()

  payload = run_diagnostic(args)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "out": str(args.out),
    "device": payload["device"],
    "metal_tensor_core_registers_int8": payload["summary"]["metal_tensor_core_registers_int8"],
    "decode_token1_rowwise_int8_uses_simdgroup_mma": payload["summary"]["decode_token1_rowwise_int8_uses_simdgroup_mma"],
    "decode_token1_rowwise_int8_vs_bf16_dequantized_ratio": (
      None if payload["summary"]["decode_token1_rowwise_int8_vs_bf16_dequantized_ratio"] is None
      else round(payload["summary"]["decode_token1_rowwise_int8_vs_bf16_dequantized_ratio"], 4)
    ),
    "rowwise_int8_linear_any_shape_uses_simdgroup_mma": payload["summary"]["rowwise_int8_linear_any_shape_uses_simdgroup_mma"],
    "shapes": [
      {
        "token_count": shape["token_count"],
        "variants": [
          {
            "name": variant["name"],
            "kernel_count": variant["kernel_count"],
            "median_ms": round(variant["median_ms"], 4),
            "uses_simdgroup_mma": variant["source_flags"]["has_simdgroup_multiply_accumulate"],
          }
          for variant in shape["variants"]
        ],
      }
      for shape in payload["shapes"]
    ],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
