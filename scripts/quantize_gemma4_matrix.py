from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np

from tinygrad_gemma.quantization import QUANTIZATION_MANIFEST, ROWWISE_INT8, quantize_numpy_rowwise_int8


SIZES = ("E2B", "E4B", "26B-A4B", "31B")
_SCALE_PREFIX = "_quant_scale."
_SOURCE_FLOAT_DTYPES = {"BF16": "bfloat16", "F16": "half", "F32": "float", "F64": "double"}
_NUMPY_DTYPES = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}


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


def _read_safetensors_header(path: Path) -> tuple[dict, int]:
  with path.open("rb") as handle:
    header_len = struct.unpack("<Q", handle.read(8))[0]
    header = json.loads(handle.read(header_len))
  return header, 8 + header_len


def _bf16_bytes_to_float32(raw: bytes, shape: list[int]) -> np.ndarray:
  uint16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32, copy=False)
  return (uint16 << 16).view(np.float32).reshape(shape)


def _tensor_bytes_to_float32(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
  if dtype == "BF16":
    return _bf16_bytes_to_float32(raw, shape)
  return np.frombuffer(raw, dtype=_NUMPY_DTYPES[dtype]).astype(np.float32, copy=False).reshape(shape)


def _tensor_nbytes(dtype: str, shape: list[int]) -> int:
  itemsize = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "BF16": 2, "F16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}[dtype]
  return itemsize * int(np.prod(shape, dtype=np.int64))


def _write_safetensors_header(handle, specs: dict[str, tuple[str, list[int], int]], metadata: dict | None = None) -> None:
  headers = {}
  if metadata:
    headers["__metadata__"] = metadata
  offset = 0
  for name, (dtype, shape, nbytes) in specs.items():
    headers[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
    offset += nbytes
  header = json.dumps(headers, separators=(",", ":")).encode("utf-8")
  header += b" " * ((8 - len(header) % 8) % 8)
  handle.write(struct.pack("<Q", len(header)))
  handle.write(header)


def quantize_file(src_file: Path, dst_file: Path, manifest: dict) -> None:
  header, data_start = _read_safetensors_header(src_file)
  metadata = header.get("__metadata__")
  specs: dict[str, tuple[str, list[int], int]] = {}
  entries: list[tuple[str, dict, bool]] = []
  for name, entry in header.items():
    if name == "__metadata__":
      continue
    dtype = entry["dtype"]
    shape = entry["shape"]
    if dtype not in _SOURCE_FLOAT_DTYPES or len(shape) < 2:
      specs[name] = (dtype, shape, _tensor_nbytes(dtype, shape))
      entries.append((name, entry, False))
      continue
    scale_key = f"{_SCALE_PREFIX}{name}"
    specs[name] = ("I8", shape, _tensor_nbytes("I8", shape))
    specs[scale_key] = ("F32", [shape[0]], 4 * shape[0])
    entries.append((name, entry, True))
    manifest["tensors"][name] = {"scheme": "symmetric_rowwise_int8", "scale_key": scale_key, "dtype": _SOURCE_FLOAT_DTYPES[dtype]}

  dst_file.unlink(missing_ok=True)
  with src_file.open("rb") as src, dst_file.open("wb") as dst:
    _write_safetensors_header(dst, specs, metadata=metadata)
    for name, entry, should_quantize in entries:
      start, end = entry["data_offsets"]
      src.seek(data_start + start)
      raw = src.read(end - start)
      if not should_quantize:
        dst.write(raw)
        continue
      qweight, scale = quantize_numpy_rowwise_int8(_tensor_bytes_to_float32(raw, entry["dtype"], entry["shape"]))
      dst.write(qweight.tobytes(order="C"))
      dst.write(scale.astype("<f4", copy=False).tobytes(order="C"))


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
