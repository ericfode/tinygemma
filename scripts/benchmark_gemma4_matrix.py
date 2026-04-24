from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from tinygrad.helpers import Context

from tinygrad_gemma import load_pretrained
from tinygrad_gemma.runtime import prepare_device
from tinygrad_gemma.tokenizer import GemmaTokenizer


SIZES = ("E2B", "E4B", "26B-A4B", "31B")
FORMATS = ("bf16", "int8")


def checkpoint_dir(root: Path, size: str, fmt: str) -> Path:
  suffix = "" if fmt == "bf16" else f"-{fmt}"
  return root / f"gemma-4-{size}{suffix}"


def output_digest(tokens: list[int]) -> str:
  return hashlib.sha256(" ".join(str(token) for token in tokens).encode()).hexdigest()


def validate_decode_warmup_tokens(max_new_tokens: int, decode_warmup_tokens: int) -> None:
  if decode_warmup_tokens < 0 or decode_warmup_tokens >= max_new_tokens:
    raise ValueError("--decode-warmup-tokens must satisfy 0 <= value < --max-new-tokens")


class DecodeSuffixTimer:
  def __init__(self, *, decode_warmup_tokens: int, started: float, clock: Callable[[], float] = time.perf_counter):
    self.decode_warmup_tokens = decode_warmup_tokens
    self._clock = clock
    self._measured_start = started if decode_warmup_tokens == 0 else None

  def note_token_count(self, generated_count: int) -> None:
    if generated_count == self.decode_warmup_tokens:
      self._measured_start = self._clock()

  def finish(self, generated_count: int, ended: float | None = None) -> dict[str, float | int]:
    measured_tokens = max(0, generated_count - self.decode_warmup_tokens)
    if measured_tokens == 0:
      measured_seconds = 0.0
    else:
      ended = self._clock() if ended is None else ended
      measured_start = self._measured_start if self._measured_start is not None else ended
      measured_seconds = ended - measured_start
    return {
      "decode_warmup_tokens": self.decode_warmup_tokens,
      "measured_decode_tokens": measured_tokens,
      "measured_decode_seconds": measured_seconds,
      "measured_decode_tokens_per_second": measured_tokens / measured_seconds if measured_seconds > 0 else 0.0,
    }


def append_generated_token(generated: list[int], token_id: int, timer: DecodeSuffixTimer) -> None:
  generated.append(token_id)
  timer.note_token_count(len(generated))


def append_progress(progress_path: Path | None, payload: dict) -> None:
  if progress_path is None:
    return
  progress_path.parent.mkdir(parents=True, exist_ok=True)
  with progress_path.open("a") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")


def error_rows(*, beams: list[int], status: str, error: str, decode_warmup_tokens: int) -> list[dict]:
  return [
    {
      "beam": beam,
      "status": status,
      "generated_tokens": 0,
      "seconds": "",
      "tokens_per_second": "",
      "decode_warmup_tokens": decode_warmup_tokens,
      "measured_decode_tokens": 0,
      "measured_decode_seconds": "",
      "measured_decode_tokens_per_second": "",
      "load_seconds": "",
      "rollout_jit_count": "",
      "output_prefix": "",
      "output_sha256": "",
      "error": error,
    }
    for beam in beams
  ]


def benchmark_checkpoint(
  model_dir: Path,
  *,
  prompt: str,
  beams: list[int],
  max_new_tokens: int,
  decode_warmup_tokens: int,
  device: str,
  size: str,
  fmt: str,
  progress_every: int,
  progress_path: Path | None,
) -> list[dict]:
  started = time.perf_counter()
  try:
    resolved_device = prepare_device(device)
  except Exception as exc:
    return error_rows(beams=beams, status="device_error", error=repr(exc), decode_warmup_tokens=decode_warmup_tokens)
  try:
    model = load_pretrained(model_dir, device=resolved_device)
    tokenizer = GemmaTokenizer.from_pretrained(model_dir)
  except Exception as exc:
    return error_rows(beams=beams, status="load_error", error=repr(exc), decode_warmup_tokens=decode_warmup_tokens)
  prompt_ids = tokenizer.encode(prompt, add_bos=True)
  load_seconds = time.perf_counter() - started
  rows = []
  if str(getattr(model, "device", "CPU")).upper() in ("CPU", "PYTHON") and "PARALLEL" not in os.environ:
    os.environ["PARALLEL"] = str(os.cpu_count() or 1)
  for beam in beams:
    run_start = time.perf_counter()
    decode_timer = DecodeSuffixTimer(decode_warmup_tokens=decode_warmup_tokens, started=run_start)
    status = "ok"
    generated: list[int] = []
    error = ""
    try:
      with Context(BEAM=beam):
        for token_id in model.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=0.0, stop_token_ids=None):
          append_generated_token(generated, token_id, decode_timer)
          if progress_every > 0 and len(generated) % progress_every == 0:
            elapsed = time.perf_counter() - run_start
            decode_metrics = decode_timer.finish(len(generated), ended=run_start + elapsed)
            progress = {
              "size": size,
              "format": fmt,
              "device": device,
              "beam": beam,
              "generated_tokens": len(generated),
              "target_new_tokens": max_new_tokens,
              "seconds": round(elapsed, 6),
              "tokens_per_second": round(len(generated) / elapsed, 6) if elapsed > 0 else 0.0,
              "decode_warmup_tokens": decode_metrics["decode_warmup_tokens"],
              "measured_decode_tokens": decode_metrics["measured_decode_tokens"],
              "measured_decode_seconds": round(decode_metrics["measured_decode_seconds"], 6),
              "measured_decode_tokens_per_second": round(decode_metrics["measured_decode_tokens_per_second"], 6),
            }
            print({"progress": progress}, file=sys.stderr, flush=True)
            append_progress(progress_path, progress)
    except KeyboardInterrupt:
      status = "interrupted"
      error = "KeyboardInterrupt"
    except Exception as exc:  # keep the matrix moving
      status = "error"
      error = repr(exc)
    run_end = time.perf_counter()
    seconds = run_end - run_start
    decode_metrics = decode_timer.finish(len(generated), ended=run_end)
    rollout_jit = getattr(model, "_last_rollout_jit", None)
    rows.append({
      "beam": beam,
      "status": status,
      "prompt_tokens": len(prompt_ids),
      "generated_tokens": len(generated),
      "seconds": f"{seconds:.6f}",
      "tokens_per_second": f"{(len(generated) / seconds) if seconds > 0 else 0.0:.6f}",
      "decode_warmup_tokens": decode_metrics["decode_warmup_tokens"],
      "measured_decode_tokens": decode_metrics["measured_decode_tokens"],
      "measured_decode_seconds": f"{decode_metrics['measured_decode_seconds']:.6f}",
      "measured_decode_tokens_per_second": f"{decode_metrics['measured_decode_tokens_per_second']:.6f}",
      "load_seconds": f"{load_seconds:.6f}",
      "rollout_jit_count": getattr(rollout_jit, "cnt", "") if rollout_jit is not None else "",
      "output_prefix": " ".join(str(token) for token in generated[:32]),
      "output_sha256": output_digest(generated) if generated else "",
      "error": error,
    })
  del model
  gc.collect()
  return rows


def main() -> None:
  parser = argparse.ArgumentParser(description="Benchmark Gemma 4 native tinygrad checkpoints across beam widths.")
  parser.add_argument("--root", type=Path, default=Path("checkpoints"))
  parser.add_argument("--sizes", nargs="*", default=list(SIZES), choices=list(SIZES))
  parser.add_argument("--formats", nargs="*", default=list(FORMATS), choices=list(FORMATS))
  parser.add_argument("--beams", nargs="*", type=int, default=[1, 2, 3, 4])
  parser.add_argument("--devices", nargs="*", default=["METAL"])
  parser.add_argument("--prompt", default="hello")
  parser.add_argument("--max-new-tokens", type=int, default=1000)
  parser.add_argument("--decode-warmup-tokens", type=int, default=0, help="Initial generated tokens excluded from measured_decode_* suffix metrics.")
  parser.add_argument("--out", type=Path, default=Path("benchmarks/gemma4-matrix.csv"))
  parser.add_argument("--resume", action="store_true", help="Append to an existing CSV and skip completed size/format/device/beam rows.")
  parser.add_argument("--progress-every", type=int, default=50, help="Emit progress after this many generated tokens. Use 0 to disable.")
  parser.add_argument("--progress-out", type=Path, help="Optional JSONL sidecar for progress events.")
  args = parser.parse_args()
  try:
    validate_decode_warmup_tokens(args.max_new_tokens, args.decode_warmup_tokens)
  except ValueError as exc:
    parser.error(str(exc))

  args.out.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = [
    "size",
    "format",
    "device",
    "checkpoint",
    "beam",
    "status",
    "prompt_tokens",
    "target_new_tokens",
    "generated_tokens",
    "seconds",
    "tokens_per_second",
    "decode_warmup_tokens",
    "measured_decode_tokens",
    "measured_decode_seconds",
    "measured_decode_tokens_per_second",
    "load_seconds",
    "rollout_jit_count",
    "output_prefix",
    "output_sha256",
    "error",
  ]
  completed = set()
  if args.resume and args.out.exists():
    with args.out.open(newline="") as handle:
      for row in csv.DictReader(handle):
        completed.add((row["size"], row["format"], row["device"], row["beam"], row.get("decode_warmup_tokens", "0")))

  write_header = not args.resume or not args.out.exists()
  with args.out.open("a" if args.resume else "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if write_header:
      writer.writeheader()
      handle.flush()
    for size in args.sizes:
      for fmt in args.formats:
        model_dir = checkpoint_dir(args.root, size, fmt)
        for device in args.devices:
          base = {
            "size": size,
            "format": fmt,
            "device": device,
            "checkpoint": str(model_dir),
            "prompt_tokens": "",
            "target_new_tokens": args.max_new_tokens,
            "decode_warmup_tokens": args.decode_warmup_tokens,
          }
          if not model_dir.exists():
            writer.writerow({**base, "beam": "", "status": "missing", "generated_tokens": 0, "seconds": "", "tokens_per_second": "", "measured_decode_tokens": 0, "measured_decode_seconds": "", "measured_decode_tokens_per_second": "", "load_seconds": "", "rollout_jit_count": "", "output_prefix": "", "output_sha256": "", "error": "checkpoint directory missing"})
            handle.flush()
            continue
          pending_beams = [beam for beam in args.beams if (size, fmt, device, str(beam), str(args.decode_warmup_tokens)) not in completed]
          if not pending_beams:
            print(f"skip completed {size} {fmt} {device}", file=sys.stderr, flush=True)
            continue
          progress_path = args.progress_out or args.out.with_suffix(args.out.suffix + ".progress.jsonl")
          for row in benchmark_checkpoint(
            model_dir,
            prompt=args.prompt,
            beams=pending_beams,
            max_new_tokens=args.max_new_tokens,
            decode_warmup_tokens=args.decode_warmup_tokens,
            device=device,
            size=size,
            fmt=fmt,
            progress_every=args.progress_every,
            progress_path=progress_path,
          ):
            writer.writerow({**base, **row})
            handle.flush()
            print(base | row, flush=True)


if __name__ == "__main__":
  main()
