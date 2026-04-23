from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tinygrad import nn

from tinygrad_gemma.quantization import QUANTIZATION_MANIFEST, ROWWISE_INT8, quantize_state_dict


SIZES = ("E2B", "E4B", "26B-A4B", "31B")


def source_dir(root: Path, size: str) -> Path:
  return root / f"gemma-4-{size}"


def quantized_dir(root: Path, size: str) -> Path:
  return root / f"gemma-4-{size}-int8"


def copy_metadata(src: Path, dst: Path) -> None:
  dst.mkdir(parents=True, exist_ok=True)
  for path in src.iterdir():
    if path.suffix == ".safetensors":
      continue
    if path.is_file():
      shutil.copy2(path, dst / path.name)


def quantize_file(src_file: Path, dst_file: Path, manifest: dict) -> None:
  state_dict = nn.state.safe_load(src_file)
  quantized, shard_manifest = quantize_state_dict(state_dict, quantize=ROWWISE_INT8)
  manifest["tensors"].update(shard_manifest["tensors"])
  nn.state.safe_save(quantized, str(dst_file))


def main() -> None:
  parser = argparse.ArgumentParser(description="Create native tinygrad-gemma row-wise int8 checkpoints.")
  parser.add_argument("--root", type=Path, default=Path("checkpoints"))
  parser.add_argument("--sizes", nargs="*", default=list(SIZES), choices=list(SIZES))
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args()

  for size in args.sizes:
    src = source_dir(args.root, size)
    dst = quantized_dir(args.root, size)
    if not src.exists():
      raise FileNotFoundError(f"missing source checkpoint: {src}")
    if dst.exists() and not args.force and (dst / QUANTIZATION_MANIFEST).exists():
      print(f"skip existing {dst}", flush=True)
      continue
    if dst.exists() and args.force:
      shutil.rmtree(dst)
    copy_metadata(src, dst)
    manifest = {"format": "tinygrad_gemma.quantized.v1", "method": ROWWISE_INT8, "source": str(src), "tensors": {}}
    for src_file in sorted(src.glob("*.safetensors")):
      print(f"quantize {src_file} -> {dst / src_file.name}", flush=True)
      quantize_file(src_file, dst / src_file.name, manifest)
    (dst / QUANTIZATION_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
  main()
