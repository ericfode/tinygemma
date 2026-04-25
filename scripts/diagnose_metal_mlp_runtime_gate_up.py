from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import tinygrad


DEFAULT_OUT = Path("benchmarks/gemma4-metal-mlp-runtime-gate-up-diagnostic-current.json")
ABANDONED_RAW_GATE_UP_STATUS = "abandoned_graph_breaking_runner"
ABANDONED_RAW_GATE_UP_INTERPRETATION = (
  "The raw Metal gate/up Runner was replay-safe but fragmented MetalGraph batching and later failed raw-mode profiler validation with an invalid Metal library file. "
  "It is no longer a live Gemma decode optimization path; use archived artifacts for historical evidence."
)


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
  return {
    "device": args.device,
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
      "status": ABANDONED_RAW_GATE_UP_STATUS,
      "safe_for_decode_tinyjit_replay": False,
      "interpretation": ABANDONED_RAW_GATE_UP_INTERPRETATION,
    },
    "cases": [],
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
