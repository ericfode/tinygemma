from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tinygrad
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import Context

from tinygrad_gemma.metal_int8 import metal_rowwise_int8_decode_linear
from tinygrad_gemma.runtime import prepare_device


DEFAULT_OUT = Path("benchmarks/gemma4-metal-runtime-bridge-tinyjit-diagnostic-current.json")


def _dtype_for_name(name: str):
  if name == "float32":
    return dtypes.float32
  if name == "bfloat16":
    return dtypes.bfloat16
  raise ValueError(f"unsupported dtype {name!r}")


def _make_x(device: str, dtype_name: str, in_features: int, seed: int) -> tuple[Tensor, np.ndarray]:
  rng = np.random.default_rng(seed)
  x_np = rng.standard_normal((1, 1, in_features)).astype(np.float32)
  x = Tensor(x_np, device=device).cast(_dtype_for_name(dtype_name)).contiguous().realize()
  actual = x.float().numpy().reshape(in_features).astype(np.float32)
  return x, actual


def _max_abs_error(got: np.ndarray, ref: np.ndarray) -> float:
  return float(np.max(np.abs(got.astype(np.float32) - ref.astype(np.float32))))


def _run_dtype_probe(
  *,
  device: str,
  dtype_name: str,
  q_np: np.ndarray,
  scale_np: np.ndarray,
  qweight: Tensor,
  scale: Tensor,
  args: argparse.Namespace,
  jit_mode: int,
) -> dict[str, Any]:
  def fn(x: Tensor) -> Tensor:
    return metal_rowwise_int8_decode_linear(x, qweight, scale, local_size=args.local_size)

  jit = TinyJit(fn)
  rows: list[dict[str, Any]] = []
  capture_output: np.ndarray | None = None
  captured_kernel_count: int | None = None
  captured_input_replace_count: int | None = None

  with Context(JIT=jit_mode, BEAM=0):
    for call_idx in range(args.calls):
      x, x_actual = _make_x(device, dtype_name, args.in_features, args.seed + 1000 * call_idx + 7)
      ref = (q_np.astype(np.float32) @ x_actual) * scale_np
      try:
        out = jit(x)
        got = out.numpy().reshape(args.out_features)
      except Exception as exc:
        rows.append({
          "call_index": call_idx,
          "jit_count_after_call": jit.cnt,
          "ok": False,
          "error": f"{type(exc).__name__}: {exc}",
        })
        break

      err = _max_abs_error(got, ref)
      if call_idx == 1:
        capture_output = got.copy()
      if jit.captured is not None and captured_kernel_count is None:
        captured_kernel_count = len(jit.captured.jit_cache)
        captured_input_replace_count = len(jit.captured.input_replace)
      rows.append({
        "call_index": call_idx,
        "jit_count_after_call": jit.cnt,
        "ok": err <= args.atol,
        "max_abs_error": err,
        "matches_capture_output": None if capture_output is None else _max_abs_error(got, capture_output) <= args.atol,
      })

  replay_rows = [row for row in rows if row["call_index"] >= 2 and row.get("error") is None]
  error_rows = [row for row in rows if row.get("error")]
  replay_all_correct = bool(replay_rows) and all(bool(row["ok"]) for row in replay_rows)
  replay_any_stale = any(row.get("matches_capture_output") is True and not row.get("ok", False) for row in replay_rows)

  if error_rows and jit.captured is None:
    status = "blocked_no_capture"
    interpretation = "The raw Metal runtime call did not produce a TinyJit capture for this dtype."
  elif replay_all_correct:
    status = "safe_for_tinyjit_replay"
    interpretation = "Replay outputs changed with the input and matched the NumPy reference."
  elif replay_any_stale:
    status = "unsafe_stale_replay"
    interpretation = "TinyJit replay returned the capture-time raw Metal output when the input changed."
  else:
    status = "unsafe_or_unclassified"
    interpretation = "TinyJit replay did not prove correctness for changed inputs."

  return {
    "dtype": dtype_name,
    "jit_mode": jit_mode,
    "status": status,
    "interpretation": interpretation,
    "jit_count_final": jit.cnt,
    "captured": jit.captured is not None,
    "captured_kernel_count": captured_kernel_count,
    "captured_input_replace_count": captured_input_replace_count,
    "rows": rows,
  }


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
  if args.in_features > 4096:
    raise ValueError("--in-features must be <= 4096 for the raw Metal threadgroup staging path")
  if args.calls < 3:
    raise ValueError("--calls must be at least 3 to exercise TinyJit replay")

  device = prepare_device(args.device)
  rng = np.random.default_rng(args.seed)
  q_np = rng.integers(-127, 128, size=(args.out_features, args.in_features), dtype=np.int8)
  scale_np = rng.uniform(0.001, 0.02, size=(args.out_features,)).astype(np.float32)
  qweight = Tensor(q_np, device=device, dtype=dtypes.int8).contiguous().realize()
  scale = Tensor(scale_np, device=device, dtype=dtypes.float32).contiguous().realize()

  probes = []
  for jit_mode in args.jit_modes:
    for dtype_name in args.x_dtypes:
      probes.append(_run_dtype_probe(
        device=device,
        dtype_name=dtype_name,
        q_np=q_np,
        scale_np=scale_np,
        qweight=qweight,
        scale=scale,
        args=args,
        jit_mode=jit_mode,
      ))
  all_safe = all(probe["status"] == "safe_for_tinyjit_replay" for probe in probes)
  any_stale = any(probe["status"] == "unsafe_stale_replay" for probe in probes)
  any_blocked = any(probe["status"] == "blocked_no_capture" for probe in probes)
  if all_safe:
    status = "safe_for_decode_tinyjit_replay"
    interpretation = "The raw Metal tensor wrapper is safe to try inside decode TinyJit replay."
  elif any_stale:
    status = "unsafe_for_decode_tinyjit_replay"
    interpretation = "The raw MetalProgram launch is not captured as a TinyJit ExecItem; direct decode wiring would risk stale outputs."
  elif any_blocked:
    status = "blocked_for_decode_tinyjit_replay"
    interpretation = "The raw MetalProgram launch does not create a usable TinyJit replay path."
  else:
    status = "unclassified_for_decode_tinyjit_replay"
    interpretation = "The diagnostic did not establish a safe replay boundary."

  return {
    "device": device,
    "tinygrad_path": str(Path(tinygrad.__file__).resolve()),
    "diagnostic": "raw Metal rowwise-int8 tensor wrapper under TinyJit replay",
    "shape": {
      "in_features": args.in_features,
      "out_features": args.out_features,
      "local_size": args.local_size,
    },
    "seed": args.seed,
    "calls": args.calls,
    "jit_modes": args.jit_modes,
    "atol": args.atol,
    "summary": {
      "status": status,
      "safe_for_decode_tinyjit_replay": all_safe,
      "interpretation": interpretation,
    },
    "probes": probes,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Diagnose whether the raw Metal int8 runtime bridge is safe under TinyJit replay.")
  parser.add_argument("--device", default="METAL")
  parser.add_argument("--in-features", type=int, default=1536)
  parser.add_argument("--out-features", type=int, default=12288)
  parser.add_argument("--local-size", type=int, default=128)
  parser.add_argument("--calls", type=int, default=4)
  parser.add_argument("--jit-modes", type=int, nargs="+", default=[1, 2], choices=[1, 2])
  parser.add_argument("--seed", type=int, default=20260424)
  parser.add_argument("--atol", type=float, default=1e-3)
  parser.add_argument("--x-dtypes", nargs="+", default=["float32", "bfloat16"], choices=["float32", "bfloat16"])
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = parser.parse_args()

  payload = run_diagnostic(args)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "out": str(args.out),
    "device": payload["device"],
    "shape": payload["shape"],
    "summary": payload["summary"],
    "probes": [
      {
        "dtype": probe["dtype"],
        "jit_mode": probe["jit_mode"],
        "status": probe["status"],
        "captured": probe["captured"],
        "captured_kernel_count": probe["captured_kernel_count"],
        "jit_count_final": probe["jit_count_final"],
      }
      for probe in payload["probes"]
    ],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
