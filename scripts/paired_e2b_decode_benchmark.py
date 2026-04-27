#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK_SCRIPT = Path("benchmarks/evo_e2b_int8_metal_decode.py")


def utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


def resolve_benchmark_script(benchmark_script: Path, baseline_target: Path, *, cwd: Path) -> Path:
  candidate = benchmark_script if benchmark_script.is_absolute() else cwd / benchmark_script
  if candidate.exists():
    return benchmark_script
  if benchmark_script != DEFAULT_BENCHMARK_SCRIPT:
    return benchmark_script

  baseline = baseline_target if baseline_target.is_absolute() else cwd / baseline_target
  for parent in baseline.resolve().parents:
    worktree_candidate = parent / DEFAULT_BENCHMARK_SCRIPT
    if worktree_candidate.exists():
      try:
        return worktree_candidate.relative_to(cwd)
      except ValueError:
        return worktree_candidate
  return benchmark_script


def parse_result(stdout: str, *, label: str) -> dict[str, Any]:
  text = stdout.strip()
  try:
    payload = json.loads(text)
  except json.JSONDecodeError as exc:
    raise SystemExit(f"{label} benchmark stdout was not valid JSON: {exc}\nstdout:\n{text}") from exc
  score = payload.get("score")
  if not isinstance(score, int | float):
    raise SystemExit(f"{label} benchmark JSON must contain numeric 'score'; got {score!r}")
  return payload


def run_one(
  *,
  label: str,
  benchmark_script: Path,
  target: Path,
  benchmark_args: list[str],
  cwd: Path,
) -> dict[str, Any]:
  command = [sys.executable, str(benchmark_script), "--target", str(target), *benchmark_args]
  started_at = utc_now()
  proc = subprocess.run(
    command,
    cwd=cwd,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  ended_at = utc_now()
  if proc.returncode != 0:
    raise SystemExit(
      f"{label} benchmark failed with exit code {proc.returncode}\n"
      f"command: {' '.join(command)}\n"
      f"stdout:\n{proc.stdout}\n"
      f"stderr:\n{proc.stderr}"
    )
  result = parse_result(proc.stdout, label=label)
  return {
    "target": str(target),
    "command": command,
    "started_at": started_at,
    "ended_at": ended_at,
    "returncode": proc.returncode,
    "stdout": proc.stdout,
    "stderr": proc.stderr,
    "score": float(result["score"]),
    "result": result,
  }


def build_payload(
  *,
  label: str,
  benchmark_script: Path,
  baseline_target: Path,
  candidate_target: Path,
  benchmark_args: list[str],
  cwd: Path,
  min_delta: float | None = None,
) -> dict[str, Any]:
  started_at = utc_now()
  baseline = run_one(
    label="baseline",
    benchmark_script=benchmark_script,
    target=baseline_target,
    benchmark_args=benchmark_args,
    cwd=cwd,
  )
  candidate = run_one(
    label="candidate",
    benchmark_script=benchmark_script,
    target=candidate_target,
    benchmark_args=benchmark_args,
    cwd=cwd,
  )
  ended_at = utc_now()
  delta = candidate["score"] - baseline["score"]
  relative_delta = delta / baseline["score"] if baseline["score"] else None
  passed_min_delta = min_delta is None or delta >= min_delta
  return {
    "label": label,
    "started_at": started_at,
    "ended_at": ended_at,
    "benchmark_script": str(benchmark_script),
    "benchmark_args": benchmark_args,
    "baseline": baseline,
    "candidate": candidate,
    "delta": delta,
    "relative_delta": relative_delta,
    "candidate_improved": delta > 0,
    "min_delta": min_delta,
    "passed_min_delta": passed_min_delta,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description="Run the same decode benchmark against baseline and candidate targets in one session and report score deltas."
  )
  parser.add_argument("--benchmark-script", type=Path, default=DEFAULT_BENCHMARK_SCRIPT)
  parser.add_argument("--baseline-target", type=Path, required=True)
  parser.add_argument("--candidate-target", type=Path, required=True)
  parser.add_argument("--out", type=Path)
  parser.add_argument("--label", default="paired-decode-benchmark")
  parser.add_argument(
    "--min-delta",
    type=float,
    help="Exit nonzero after writing output if candidate_score - baseline_score is below this threshold.",
  )
  args, benchmark_args = parser.parse_known_args(argv)
  if benchmark_args and benchmark_args[0] == "--":
    benchmark_args = benchmark_args[1:]

  cwd = Path.cwd()
  benchmark_script = resolve_benchmark_script(args.benchmark_script, args.baseline_target, cwd=cwd)
  payload = build_payload(
    label=args.label,
    benchmark_script=benchmark_script,
    baseline_target=args.baseline_target,
    candidate_target=args.candidate_target,
    benchmark_args=benchmark_args,
    cwd=cwd,
    min_delta=args.min_delta,
  )
  text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
  print(text, end="")
  if not payload["passed_min_delta"]:
    print(
      f"candidate delta {payload['delta']:.6g} below --min-delta {args.min_delta:.6g}",
      file=sys.stderr,
    )
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
