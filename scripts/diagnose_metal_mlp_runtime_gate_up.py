from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tinygrad
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import Context

from tinygrad_gemma.config import GemmaConfig
from tinygrad_gemma.model import GemmaMLP
from tinygrad_gemma.quantization import RowwiseInt8Linear
from tinygrad_gemma.runtime import prepare_device


DEFAULT_OUT = Path("benchmarks/gemma4-metal-mlp-runtime-gate-up-diagnostic-current.json")


def _dtype_for_name(name: str):
  if name == "float32":
    return dtypes.float32
  if name == "bfloat16":
    return dtypes.bfloat16
  raise ValueError(f"unsupported dtype {name!r}")


def _random_linear(rng: np.random.Generator, rows: int, cols: int, device: str) -> RowwiseInt8Linear:
  q = rng.integers(-8, 9, size=(rows, cols), dtype=np.int8)
  scale = rng.uniform(1e-4, 8e-4, size=(rows,)).astype(np.float32)
  return RowwiseInt8Linear(
    Tensor(q, device=device, dtype=dtypes.int8).contiguous().realize(),
    Tensor(scale, device=device, dtype=dtypes.float32).contiguous().realize(),
    original_dtype="bfloat16",
  )


def _make_mlps(args: argparse.Namespace, device: str) -> tuple[GemmaMLP, GemmaMLP]:
  config = GemmaConfig(
    hidden_size=args.hidden_size,
    intermediate_size=args.intermediate_size,
    num_hidden_layers=1,
    num_attention_heads=1,
    num_key_value_heads=1,
    head_dim=args.hidden_size,
    hidden_act="silu",
    hidden_activation="silu",
    layer_types=["full_attention"],
    sliding_window=None,
    hidden_size_per_layer_input=0,
    vocab_size_per_layer_input=0,
    global_head_dim=args.hidden_size,
  )
  rng = np.random.default_rng(args.seed)
  gate = _random_linear(rng, args.intermediate_size, args.hidden_size, device)
  up = _random_linear(rng, args.intermediate_size, args.hidden_size, device)
  down = _random_linear(rng, args.hidden_size, args.intermediate_size, device)

  fast = GemmaMLP(config, 0)
  fallback = GemmaMLP(config, 0)
  fast.gate_proj = fallback.gate_proj = gate
  fast.up_proj = fallback.up_proj = up
  fast.down_proj = fallback.down_proj = down
  fast._force_metal_fused_int8_gate_up = True
  fallback._can_use_metal_fused_int8_gate_up = lambda x: False
  return fast, fallback


def _make_x(device: str, dtype_name: str, hidden_size: int, seed: int) -> Tensor:
  rng = np.random.default_rng(seed)
  x_np = rng.standard_normal((1, 1, hidden_size)).astype(np.float32)
  return Tensor(x_np, device=device).cast(_dtype_for_name(dtype_name)).contiguous().realize()


def _max_abs_error(got: Tensor, ref: Tensor) -> float:
  got_np = got.float().numpy()
  ref_np = ref.float().numpy()
  return float(np.max(np.abs(got_np - ref_np)))


def _run_case(args: argparse.Namespace, *, device: str, dtype_name: str, jit_mode: int) -> dict[str, Any]:
  fast, fallback = _make_mlps(args, device)
  jit = TinyJit(lambda x: fast(x))
  rows: list[dict[str, Any]] = []
  capture_output: np.ndarray | None = None

  with Context(JIT=jit_mode, BEAM=0):
    for call_idx in range(args.calls):
      x = _make_x(device, dtype_name, args.hidden_size, args.seed + 1000 * call_idx + 31)
      out = jit(x).realize()
      ref = fallback(x).realize()
      err = _max_abs_error(out, ref)
      got_np = out.float().numpy()
      if call_idx == 1:
        capture_output = got_np.copy()
      rows.append({
        "call_index": call_idx,
        "jit_count_after_call": jit.cnt,
        "max_abs_error": err,
        "ok": err <= args.atol,
        "matches_capture_output": None if capture_output is None else bool(np.max(np.abs(got_np - capture_output)) <= args.atol),
      })

  replay_rows = [row for row in rows if row["call_index"] >= 2]
  replay_all_correct = bool(replay_rows) and all(bool(row["ok"]) for row in replay_rows)
  replay_any_stale = any(row["matches_capture_output"] is True and not row["ok"] for row in replay_rows)
  if replay_all_correct:
    status = "safe_for_tinyjit_replay"
    interpretation = "Metal MLP gate/up replay matched the existing tinygrad fused-int8 path for changed inputs."
  elif replay_any_stale:
    status = "unsafe_stale_replay"
    interpretation = "Metal MLP gate/up replay returned capture-time output for changed inputs."
  else:
    status = "unsafe_or_unclassified"
    interpretation = "Metal MLP gate/up replay did not prove correctness."

  return {
    "dtype": dtype_name,
    "jit_mode": jit_mode,
    "status": status,
    "interpretation": interpretation,
    "jit_count_final": jit.cnt,
    "captured": jit.captured is not None,
    "captured_kernel_count": None if jit.captured is None else len(jit.captured.jit_cache),
    "rows": rows,
  }


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
  if args.hidden_size > 4096:
    raise ValueError("--hidden-size must be <= 4096 for the raw Metal gate/up path")
  if args.calls < 3:
    raise ValueError("--calls must be at least 3 to exercise TinyJit replay")

  device = prepare_device(args.device)
  cases = [
    _run_case(args, device=device, dtype_name=dtype_name, jit_mode=jit_mode)
    for jit_mode in args.jit_modes
    for dtype_name in args.x_dtypes
  ]
  all_safe = all(case["status"] == "safe_for_tinyjit_replay" for case in cases)
  return {
    "device": device,
    "tinygrad_path": str(Path(tinygrad.__file__).resolve()),
    "diagnostic": "GemmaMLP raw Metal runtime gate/up against existing fused rowwise-int8 path",
    "shape": {
      "hidden_size": args.hidden_size,
      "intermediate_size": args.intermediate_size,
      "gate_up_out_features": 2 * args.intermediate_size,
    },
    "seed": args.seed,
    "calls": args.calls,
    "jit_modes": args.jit_modes,
    "atol": args.atol,
    "summary": {
      "status": "safe_for_decode_tinyjit_replay" if all_safe else "unsafe_for_decode_tinyjit_replay",
      "safe_for_decode_tinyjit_replay": all_safe,
    },
    "cases": cases,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Diagnose GemmaMLP raw Metal rowwise-int8 gate/up replay correctness.")
  parser.add_argument("--device", default="METAL")
  parser.add_argument("--hidden-size", type=int, default=1536)
  parser.add_argument("--intermediate-size", type=int, default=6144)
  parser.add_argument("--calls", type=int, default=4)
  parser.add_argument("--jit-modes", type=int, nargs="+", default=[1, 2], choices=[1, 2])
  parser.add_argument("--x-dtypes", nargs="+", default=["bfloat16"], choices=["float32", "bfloat16"])
  parser.add_argument("--seed", type=int, default=20260424)
  parser.add_argument("--atol", type=float, default=2e-2)
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
    "cases": [
      {
        "dtype": case["dtype"],
        "jit_mode": case["jit_mode"],
        "status": case["status"],
        "captured": case["captured"],
        "captured_kernel_count": case["captured_kernel_count"],
        "jit_count_final": case["jit_count_final"],
        "max_abs_error": max(row["max_abs_error"] for row in case["rows"]),
      }
      for case in payload["cases"]
    ],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
