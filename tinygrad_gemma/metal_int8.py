from __future__ import annotations

import math
from functools import lru_cache

from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.engine.realize import ExecItem, Runner, capturing
from tinygrad.helpers import CAPTURING
from tinygrad.uop.ops import Ops, UOp

from .runtime import prepare_device


ROWWISE_INT8_DECODE_LINEAR_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void rowwise_int8_decode_linear_threadgroup_x(
  device float* out [[buffer(0)]],
  const device float* x [[buffer(1)]],
  const device char* qweight [[buffer(2)]],
  const device float* scale [[buffer(3)]],
  constant int& in_features [[buffer(4)]],
  constant int& out_features [[buffer(5)]],
  uint gid [[thread_position_in_grid]],
  uint lid [[thread_index_in_threadgroup]],
  uint tpg [[threads_per_threadgroup]]
) {
  threadgroup float x_local[4096];
  for (int k = int(lid); k < in_features; k += int(tpg)) {
    x_local[k] = x[k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (gid >= uint(out_features)) return;
  const uint base = gid * uint(in_features);
  float acc = 0.0f;
  for (int k = 0; k < in_features; k++) {
    acc += x_local[k] * float(qweight[base + uint(k)]);
  }
  out[gid] = acc * scale[gid];
}
"""


@lru_cache(maxsize=None)
def _rowwise_int8_decode_linear_program(device: str):
  resolved_device = prepare_device(device)
  if resolved_device != "METAL":
    raise RuntimeError("raw rowwise-int8 decode linear is METAL-only")
  return Device[resolved_device].runtime(
    "rowwise_int8_decode_linear_threadgroup_x",
    ROWWISE_INT8_DECODE_LINEAR_SOURCE.encode(),
  )


def _realized_buffer(tensor: Tensor) -> Buffer:
  realized = tensor.contiguous().realize()
  buffer = realized.uop.buffer
  if not isinstance(buffer, Buffer):
    raise TypeError("raw rowwise-int8 decode linear expects a single-device tensor")
  return buffer.ensure_allocated()


class RowwiseInt8DecodeLinearRunner(Runner):
  def __init__(self, device: str, in_features: int, out_features: int, local_size: int):
    self.in_features = in_features
    self.out_features = out_features
    self.local_size = local_size
    super().__init__("rowwise_int8_decode_linear_threadgroup_x", device)

  def __reduce__(self):
    return (self.__class__, (self.device, self.in_features, self.out_features, self.local_size))

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[str, int], wait=False):
    output_buffer, x_buffer, qweight_buffer, scale_buffer = rawbufs
    program = _rowwise_int8_decode_linear_program(self.device)
    return program(
      output_buffer._buf,
      x_buffer._buf,
      qweight_buffer._buf,
      scale_buffer._buf,
      vals=(self.in_features, self.out_features),
      global_size=(math.ceil(self.out_features / self.local_size), 1, 1),
      local_size=(self.local_size, 1, 1),
      wait=wait,
    )


def _run_or_capture_rowwise_int8_decode_linear(
  output_buffer: Buffer,
  x_buffer: Buffer,
  qweight_buffer: Buffer,
  scale_buffer: Buffer,
  *,
  in_features: int,
  out_features: int,
  local_size: int,
  device: str,
) -> None:
  runner = RowwiseInt8DecodeLinearRunner(device, in_features, out_features, local_size)
  item = ExecItem(
    UOp(Ops.NOOP),
    [output_buffer, x_buffer, qweight_buffer, scale_buffer],
    prg=runner,
  )
  if len(capturing) and CAPTURING:
    capturing[0].add(item)
  item.run()


def metal_rowwise_int8_decode_linear(
  x: Tensor,
  qweight: Tensor,
  scale: Tensor,
  *,
  local_size: int = 128,
) -> Tensor:
  if not isinstance(x.device, str) or x.device != "METAL":
    raise RuntimeError("raw rowwise-int8 decode linear expects x on METAL")
  if qweight.device != x.device or scale.device != x.device:
    raise RuntimeError("raw rowwise-int8 decode linear expects all tensors on the same METAL device")
  if qweight.dtype != dtypes.int8:
    raise TypeError(f"qweight must be int8, got {qweight.dtype}")
  if scale.dtype != dtypes.float32:
    raise TypeError(f"scale must be float32, got {scale.dtype}")
  if qweight.ndim != 2:
    raise ValueError(f"qweight must be rank 2, got shape {qweight.shape}")
  if x.shape[-1] != qweight.shape[1]:
    raise ValueError(f"cannot multiply input shape {x.shape} by qweight shape {qweight.shape}")
  if scale.shape != (qweight.shape[0],):
    raise ValueError(f"scale shape {scale.shape} does not match qweight rows {qweight.shape[0]}")
  if math.prod(x.shape[:-1]) != 1:
    raise ValueError("raw rowwise-int8 decode linear currently supports exactly one decode row")
  if x.shape[-1] > 4096:
    raise ValueError("raw rowwise-int8 decode linear threadgroup staging supports at most 4096 input features")

  in_features = int(qweight.shape[1])
  out_features = int(qweight.shape[0])
  output = Tensor.empty(out_features, device=x.device, dtype=dtypes.float32).realize()
  output_buffer = output.uop.buffer
  if not isinstance(output_buffer, Buffer):
    raise TypeError("raw rowwise-int8 decode linear created a non-buffer output")

  x_buffer = _realized_buffer(x.float().reshape(in_features))
  qweight_buffer = _realized_buffer(qweight.reshape(out_features * in_features))
  scale_buffer = _realized_buffer(scale)
  _run_or_capture_rowwise_int8_decode_linear(
    output_buffer.ensure_allocated(),
    x_buffer,
    qweight_buffer,
    scale_buffer,
    in_features=in_features,
    out_features=out_features,
    local_size=local_size,
    device=x.device,
  )
  return output.reshape(*x.shape[:-1], out_features)
