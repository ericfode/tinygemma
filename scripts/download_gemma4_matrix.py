from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


OFFICIAL_MODELS = {
  "E2B": "google/gemma-4-E2B",
  "E4B": "google/gemma-4-E4B",
  "26B-A4B": "google/gemma-4-26B-A4B",
  "31B": "google/gemma-4-31B",
}


def checkpoint_dir(root: Path, size: str) -> Path:
  return root / f"gemma-4-{size}"


def main() -> None:
  parser = argparse.ArgumentParser(description="Download the official Gemma 4 checkpoint matrix.")
  parser.add_argument("--root", type=Path, default=Path("checkpoints"))
  parser.add_argument("--sizes", nargs="*", default=list(OFFICIAL_MODELS), choices=list(OFFICIAL_MODELS))
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  args.root.mkdir(parents=True, exist_ok=True)
  for size in args.sizes:
    model_id = OFFICIAL_MODELS[size]
    local_dir = checkpoint_dir(args.root, size)
    cmd = [
      "hf",
      "download",
      model_id,
      "--local-dir",
      str(local_dir),
      "--include",
      "*.json",
      "--include",
      "*.safetensors",
      "--include",
      "README.md",
    ]
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
      subprocess.run(cmd, check=True)


if __name__ == "__main__":
  main()
