from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import tinygrad
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer

from tinygrad_gemma.metal_int8 import ROWWISE_INT8_DECODE_LINEAR_SOURCE, metal_rowwise_int8_decode_linear
from tinygrad_gemma.runtime import prepare_device


DEFAULT_OUT = Path("benchmarks/gemma4-metal-rowwise-int8-linear-prototype-current.json")

GLOBAL_X_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void rowwise_int8_decode_linear_global_x(
  device float* out [[buffer(0)]],
  const device float* x [[buffer(1)]],
  const device char* qweight [[buffer(2)]],
  const device float* scale [[buffer(3)]],
  constant int& in_features [[buffer(4)]],
  constant int& out_features [[buffer(5)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= uint(out_features)) return;
  const uint base = gid * uint(in_features);
  float acc = 0.0f;
  for (int k = 0; k < in_features; k++) {
    acc += x[k] * float(qweight[base + uint(k)]);
  }
  out[gid] = acc * scale[gid];
}
"""


def _make_buffers(device: str, in_features: int, out_features: int, seed: int) -> tuple[dict[str, Buffer], np.ndarray]:
  rng = np.random.default_rng(seed)
  x_np = rng.standard_normal(in_features).astype(np.float32)
  q_np = rng.integers(-127, 128, size=(out_features, in_features), dtype=np.int8)
  scale_np = rng.uniform(0.001, 0.02, size=(out_features,)).astype(np.float32)
  ref = (q_np.astype(np.float32) @ x_np) * scale_np
  buffers = {
    "out": Buffer(device, out_features, dtypes.float, initial_value=np.zeros(out_features, dtype=np.float32).tobytes()),
    "x": Buffer(device, in_features, dtypes.float, initial_value=x_np.tobytes()),
    "qweight": Buffer(device, out_features * in_features, dtypes.int8, initial_value=q_np.tobytes()),
    "scale": Buffer(device, out_features, dtypes.float, initial_value=scale_np.tobytes()),
  }
  return buffers, ref


def _make_tensors(device: str, in_features: int, out_features: int, seed: int):
  rng = np.random.default_rng(seed)
  x_np = rng.standard_normal(in_features).astype(np.float32)
  q_np = rng.integers(-127, 128, size=(out_features, in_features), dtype=np.int8)
  scale_np = rng.uniform(0.001, 0.02, size=(out_features,)).astype(np.float32)
  ref = (q_np.astype(np.float32) @ x_np) * scale_np
  return (
    Tensor(x_np.reshape(1, 1, in_features), device=device).contiguous().realize(),
    Tensor(q_np, device=device, dtype=dtypes.int8).contiguous().realize(),
    Tensor(scale_np, device=device, dtype=dtypes.float32).contiguous().realize(),
    ref,
  )


def _copy_output(buffer: Buffer) -> np.ndarray:
  return np.frombuffer(buffer.as_buffer(), dtype=np.float32).copy()


def _run_variant(name: str, program, buffers: dict[str, Buffer], reference: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
  threadgroups = (math.ceil(args.out_features / args.local_size), 1, 1)
  local_size = (args.local_size, 1, 1)
  call_args = (
    buffers["out"]._buf,
    buffers["x"]._buf,
    buffers["qweight"]._buf,
    buffers["scale"]._buf,
  )

  warmup_ms = []
  for _ in range(args.warmups):
    elapsed = program(*call_args, vals=(args.in_features, args.out_features), global_size=threadgroups, local_size=local_size, wait=True)
    warmup_ms.append(1000.0 * float(elapsed or 0.0))

  samples_ms = []
  for _ in range(args.repeats):
    elapsed = program(*call_args, vals=(args.in_features, args.out_features), global_size=threadgroups, local_size=local_size, wait=True)
    samples_ms.append(1000.0 * float(elapsed or 0.0))

  output = _copy_output(buffers["out"])
  max_abs_error = float(np.max(np.abs(output - reference)))
  mean_abs_error = float(np.mean(np.abs(output - reference)))
  median_ms = statistics.median(samples_ms)
  return {
    "name": name,
    "correctness": {
      "max_abs_error": max_abs_error,
      "mean_abs_error": mean_abs_error,
      "passed": max_abs_error <= args.atol,
      "atol": args.atol,
    },
    "timing_ms": {
      "warmup": warmup_ms,
      "samples": samples_ms,
      "median": median_ms,
      "min": min(samples_ms),
      "max": max(samples_ms),
    },
  }


def run_prototype(args: argparse.Namespace) -> dict[str, Any]:
  if args.in_features > 4096:
    raise ValueError("--in-features must be <= 4096 for the threadgroup-x prototype")

  device = prepare_device(args.device)
  metal_device = Device[device]
  buffers, reference = _make_buffers(device, args.in_features, args.out_features, args.seed)
  variants = [
    _run_variant(
      "rowwise_int8_decode_linear_global_x",
      metal_device.runtime("rowwise_int8_decode_linear_global_x", GLOBAL_X_SOURCE.encode()),
      buffers,
      reference,
      args,
    ),
    _run_variant(
      "rowwise_int8_decode_linear_threadgroup_x",
      metal_device.runtime("rowwise_int8_decode_linear_threadgroup_x", ROWWISE_INT8_DECODE_LINEAR_SOURCE.encode()),
      buffers,
      reference,
      args,
    ),
  ]

  x_tensor, qweight_tensor, scale_tensor, tensor_reference = _make_tensors(device, args.in_features, args.out_features, args.seed)
  tensor_output = metal_rowwise_int8_decode_linear(x_tensor, qweight_tensor, scale_tensor, local_size=args.local_size)
  tensor_values = tensor_output.numpy().reshape(args.out_features)
  tensor_max_abs_error = float(np.max(np.abs(tensor_values - tensor_reference)))

  best = min(variants, key=lambda item: item["timing_ms"]["median"])
  all_passed = all(item["correctness"]["passed"] for item in variants)
  return {
    "device": device,
    "tinygrad_path": str(Path(tinygrad.__file__).resolve()),
    "prototype": "raw tinygrad MetalProgram rowwise-int8 decode linear",
    "shape": {
      "in_features": args.in_features,
      "out_features": args.out_features,
      "local_size": args.local_size,
      "threadgroups": (math.ceil(args.out_features / args.local_size), 1, 1),
    },
    "seed": args.seed,
    "warmups": args.warmups,
    "repeats": args.repeats,
    "variants": variants,
    "module_check": {
      "output_shape": tuple(tensor_output.shape),
      "max_abs_error": tensor_max_abs_error,
      "passed": tensor_max_abs_error <= args.atol,
    },
    "best_variant": {
      "name": best["name"],
      "median_ms": best["timing_ms"]["median"],
      "min_ms": best["timing_ms"]["min"],
    },
    "interpretation": (
      "The repo can compile and launch raw Metal rowwise-int8 decode-linear kernels through tinygrad buffers."
      if all_passed and tensor_max_abs_error <= args.atol
      else "At least one raw Metal rowwise-int8 prototype variant failed the NumPy reference check."
    ),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Prototype a raw Metal rowwise-int8 decode linear kernel through tinygrad buffers.")
  parser.add_argument("--device", default="METAL")
  parser.add_argument("--in-features", type=int, default=1536)
  parser.add_argument("--out-features", type=int, default=12288)
  parser.add_argument("--local-size", type=int, default=128)
  parser.add_argument("--warmups", type=int, default=5)
  parser.add_argument("--repeats", type=int, default=20)
  parser.add_argument("--seed", type=int, default=20260424)
  parser.add_argument("--atol", type=float, default=1e-3)
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = parser.parse_args()

  payload = run_prototype(args)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "out": str(args.out),
    "device": payload["device"],
    "shape": payload["shape"],
    "best_variant": payload["best_variant"],
    "module_check": payload["module_check"],
    "variants": [
      {
        "name": item["name"],
        "median_ms": round(item["timing_ms"]["median"], 6),
        "min_ms": round(item["timing_ms"]["min"], 6),
        "max_abs_error": item["correctness"]["max_abs_error"],
        "passed": item["correctness"]["passed"],
      }
      for item in payload["variants"]
    ],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
