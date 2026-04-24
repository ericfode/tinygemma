from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes

QUANTIZATION_MANIFEST = "quantization.json"
ROWWISE_INT8 = "int8"
SUPPORTED_QUANTIZATIONS = (ROWWISE_INT8,)
_SCALE_PREFIX = "_quant_scale."


def supported_quantizations() -> tuple[str, ...]:
  return SUPPORTED_QUANTIZATIONS


def _is_quantizable_tensor(tensor: Tensor) -> bool:
  return dtypes.is_float(tensor.dtype) and tensor.ndim >= 2


def _scale_key(name: str) -> str:
  return f"{_SCALE_PREFIX}{name}"


def _dtype_manifest_name(dtype) -> str:
  return "bfloat16" if dtype == dtypes.bfloat16 else dtype.name


def _dtype_from_manifest(dtype: str):
  return "bfloat16" if dtype == "__bf16" else dtype


class RowwiseInt8Linear:
  is_rowwise_int8 = True

  def __init__(self, qweight: Tensor, scale: Tensor, *, original_dtype: str, bias: Tensor | None = None):
    self.weight = qweight
    self.scale = scale
    self.original_dtype = original_dtype
    self.bias = bias

  def __call__(self, x: Tensor) -> Tensor:
    out = x.matmul(self.weight.transpose(), dtype="float")
    out = out * self.scale.reshape(*([1] * (out.ndim - 1)), self.scale.shape[0])
    if self.bias is not None:
      out = out + self.bias.reshape(*([1] * (out.ndim - 1)), self.bias.shape[0]).float()
    return out.cast(x.dtype)


def _quantize_rowwise_int8(tensor: Tensor) -> tuple[Tensor, Tensor, str]:
  quantized, scales = quantize_numpy_rowwise_int8(tensor.numpy())
  return Tensor(quantized, dtype="int8"), Tensor(scales, dtype="float32"), _dtype_manifest_name(tensor.dtype)


def quantize_numpy_rowwise_int8(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  array = array.astype(np.float32, copy=False)
  rows = array.reshape(array.shape[0], -1)
  max_abs = np.max(np.abs(rows), axis=1, keepdims=True)
  scales = np.maximum(max_abs / 127.0, 1e-12).astype(np.float32, copy=False)
  quantized = np.clip(np.rint(rows / scales), -127.0, 127.0).astype(np.int8, copy=False).reshape(array.shape)
  return quantized, scales.reshape(array.shape[0])


def quantize_state_dict(state_dict: dict[str, Tensor], *, quantize: str) -> tuple[dict[str, Tensor], dict[str, Any]]:
  quantize_name = quantize.strip().lower()
  if quantize_name not in SUPPORTED_QUANTIZATIONS:
    raise ValueError(f"unsupported quantization {quantize!r}")

  quantized_state: dict[str, Tensor] = {}
  manifest: dict[str, Any] = {"format": "tinygrad_gemma.quantized.v1", "method": quantize_name, "tensors": {}}
  for name, tensor in state_dict.items():
    if not _is_quantizable_tensor(tensor):
      quantized_state[name] = tensor
      continue
    qweight, scale, original_dtype = _quantize_rowwise_int8(tensor)
    scale_key = _scale_key(name)
    quantized_state[name] = qweight
    quantized_state[scale_key] = scale
    manifest["tensors"][name] = {"scheme": "symmetric_rowwise_int8", "scale_key": scale_key, "dtype": original_dtype}
  return quantized_state, manifest


def load_quantization_manifest(model_dir: str | Path) -> dict[str, Any] | None:
  manifest_path = Path(model_dir) / QUANTIZATION_MANIFEST
  return json.loads(manifest_path.read_text()) if manifest_path.exists() else None


def dequantize_rowwise_int8_tensor(tensor: Tensor, scale: Tensor, dtype: str) -> Tensor:
  qweight = tensor.numpy().astype(np.float32, copy=False)
  scales = scale.numpy().astype(np.float32, copy=False).reshape(qweight.shape[0], 1)
  restored = (qweight.reshape(qweight.shape[0], -1) * scales).reshape(qweight.shape)
  return Tensor(restored, dtype=_dtype_from_manifest(dtype))


def dequantize_state_dict(state_dict: dict[str, Tensor], manifest: dict[str, Any]) -> dict[str, Tensor]:
  tensors = manifest.get("tensors", {})
  dequantized: dict[str, Tensor] = {}
  consumed = {entry["scale_key"] for entry in tensors.values()}

  for name, tensor in state_dict.items():
    if name in consumed:
      continue
    if name not in tensors:
      dequantized[name] = tensor
      continue

    entry = tensors[name]
    scale_key = entry["scale_key"]
    if scale_key not in state_dict:
      raise KeyError(f"missing quantization scale tensor {scale_key!r} for {name!r}")

    dequantized[name] = dequantize_rowwise_int8_tensor(tensor, state_dict[scale_key], entry.get("dtype", "float"))
  return dequantized
