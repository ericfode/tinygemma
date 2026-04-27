#!/usr/bin/env python3
"""Prototype a research-tinygrad capture bridge for the raw Metal int8 runner.

This is intentionally a spike, not production integration. Run it with the
research tinygrad checkout first on PYTHONPATH, for example:

  PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. \
    .venv/bin/python scripts/prototype_research_raw_metal_capture_bridge.py

The script proves that preserving UOps until replay is enough for research
TinyJit to feed current input buffers to a custom raw Metal runner. It also
keeps the production helper fail-closed for add-linear-only capture.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np

from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import CAPTURING, Context
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat
import tinygrad.engine.jit as jit_mod
import tinygrad.engine.realize as realize
from tinygrad.engine.realize import capturing

from tinygrad_gemma.metal_int8 import (
  RowwiseInt8DecodeLinearRunner,
  metal_rowwise_int8_decode_linear,
)

RAW_CUSTOM_FUNCTION = "raw_rowwise_int8_decode_linear"


def _require_research_tinygrad() -> None:
  if not hasattr(TinyJit, "add_linear") or hasattr(TinyJit, "add"):
    raise RuntimeError(
      "this prototype is meant for the research tinygrad add_linear capture API; "
      "run with PYTHONPATH=/Users/ericfode/src/.tinygrad_research:."
    )


def _custom_dims(ast: UOp) -> tuple[int, int, int]:
  dims = tuple(int(src.arg) for src in ast.src if src.op is Ops.CONST)
  if len(dims) != 3:
    raise RuntimeError(f"{RAW_CUSTOM_FUNCTION} expected three integer constants, got {dims}")
  return dims


def _exec_raw_rowwise_int8(ctx, call: UOp, ast: UOp):
  resolved = realize.resolve_params(ctx, call)
  bufs = [u.buffer.ensure_allocated() for u in resolved]
  if len(bufs) != 4:
    raise RuntimeError(f"{RAW_CUSTOM_FUNCTION} expected 4 buffers, got {len(bufs)}")
  output, x, qweight, scale = bufs
  in_features, out_features, local_size = _custom_dims(ast)
  if output.size != out_features or x.size != in_features:
    raise RuntimeError(
      f"{RAW_CUSTOM_FUNCTION} buffer/dimension mismatch: "
      f"output={output.size}/{out_features} x={x.size}/{in_features}"
    )
  runner = RowwiseInt8DecodeLinearRunner(output.device, in_features, out_features, local_size)
  with realize.track_stats(
    ctx,
    call,
    runner.device,
    runner.display_name,
    runner.estimates,
    bufs,
    ctx.var_vals,
    outputs=(0,),
    inputs=(1, 2, 3),
    first_run=runner.first_run,
  ) as timing:
    timing[0] = runner(bufs, ctx.var_vals, wait=False)
    runner.first_run = False


@contextmanager
def installed_raw_rowwise_research_bridge() -> Iterator[None]:
  old_pm_exec = realize.pm_exec
  old_call_outs_ins = jit_mod._call_outs_ins
  raw_pm = PatternMatcher([
    (
      UPat(
        Ops.CALL,
        src=(UPat(Ops.CUSTOM_FUNCTION, arg=RAW_CUSTOM_FUNCTION, name="ast"),),
        name="call",
        allow_any_len=True,
      ),
      _exec_raw_rowwise_int8,
    )
  ])

  def patched_call_outs_ins(call: UOp) -> tuple[set[int], set[int]]:
    ast = call.src[0]
    if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == RAW_CUSTOM_FUNCTION:
      return {0}, {1, 2, 3}
    return old_call_outs_ins(call)

  realize.pm_exec = raw_pm + old_pm_exec
  jit_mod._call_outs_ins = patched_call_outs_ins
  try:
    yield
  finally:
    realize.pm_exec = old_pm_exec
    jit_mod._call_outs_ins = old_call_outs_ins


def raw_rowwise_int8_decode_linear_bridge(
  x: Tensor,
  qweight: Tensor,
  scale: Tensor,
  *,
  local_size: int = 4,
) -> Tensor:
  if qweight.ndim != 2:
    raise ValueError(f"qweight must be rank 2, got {qweight.shape}")
  if x.shape[-1] != qweight.shape[1]:
    raise ValueError(f"cannot multiply input shape {x.shape} by qweight shape {qweight.shape}")
  if scale.shape != (qweight.shape[0],):
    raise ValueError(f"scale shape {scale.shape} does not match qweight rows {qweight.shape[0]}")

  in_features = int(qweight.shape[1])
  out_features = int(qweight.shape[0])
  x_work = x.float().reshape(in_features).contiguous().realize()
  qweight_work = qweight.reshape(out_features * in_features).contiguous().realize()
  scale_work = scale.contiguous().realize()
  output_uop = UOp.new_buffer(x_work.device, out_features, dtypes.float32)
  custom = UOp(
    Ops.CUSTOM_FUNCTION,
    dtypes.void,
    src=(
      UOp.const(dtypes.int, in_features),
      UOp.const(dtypes.int, out_features),
      UOp.const(dtypes.int, local_size),
    ),
    arg=RAW_CUSTOM_FUNCTION,
  )
  call = custom.call(
    output_uop,
    x_work.uop.buf_uop,
    qweight_work.uop.buf_uop,
    scale_work.uop.buf_uop,
    metadata=(),
  )
  linear = UOp(Ops.LINEAR, src=(call,))
  if len(capturing) and CAPTURING:
    capture = capturing[0]
    add_linear = getattr(capture, "add_linear", None)
    if add_linear is None:
      raise RuntimeError("research raw bridge requires TinyJit.add_linear capture")
    add_linear(linear, {})
  else:
    realize.run_linear(linear)
  return Tensor(output_uop).reshape(*x.shape[:-1], out_features)


def _reference(x_np: np.ndarray, qweight_np: np.ndarray, scale_np: np.ndarray) -> np.ndarray:
  return (x_np.astype(np.float32).reshape(-1) @ qweight_np.T.astype(np.float32)) * scale_np


def _make_weights() -> tuple[np.ndarray, np.ndarray, Tensor, Tensor]:
  qweight_np = np.array([
    [1, -2, 3, 0, -1, 2, 1, -3],
    [0, 1, -1, 2, 3, -2, 1, 1],
    [2, 0, 1, -1, 1, 0, -2, 3],
    [-1, 2, 0, 1, -2, 3, 0, 1],
  ], dtype=np.int8)
  scale_np = np.array([0.5, 1.25, -0.75, 2.0], dtype=np.float32)
  qweight = Tensor(qweight_np, device="METAL", dtype=dtypes.int8).realize()
  scale = Tensor(scale_np, device="METAL", dtype=dtypes.float32).realize()
  return qweight_np, scale_np, qweight, scale


def _prove_existing_helper_rejects(qweight: Tensor, scale: Tensor) -> None:
  def fn(x: Tensor) -> Tensor:
    return metal_rowwise_int8_decode_linear(x, qweight, scale, local_size=4).realize()

  runner = TinyJit(fn)
  with Context(JIT=1):
    runner(Tensor(np.ones(8, dtype=np.float32), device="METAL", dtype=dtypes.float32).realize())
    try:
      runner(Tensor(np.full(8, 2.0, dtype=np.float32), device="METAL", dtype=dtypes.float32).realize())
    except RuntimeError as exc:
      message = str(exc)
      if "add(ExecItem)" not in message:
        raise
      print("existing helper rejects research capture:", message)
      return
  raise AssertionError("existing raw helper unexpectedly captured under research TinyJit")


def _prove_bridge_replay(qweight_np: np.ndarray, scale_np: np.ndarray, qweight: Tensor, scale: Tensor) -> None:
  inputs = [
    np.array([1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 1.5, 0.0], dtype=np.float32),
    np.array([-2.0, 0.25, 1.0, 3.0, -1.5, 2.5, 0.5, -0.75], dtype=np.float32),
    np.array([0.5, -1.0, 2.0, -2.0, 1.0, 0.0, 3.0, -3.0], dtype=np.float32),
  ]

  def fn(x: Tensor) -> Tensor:
    return raw_rowwise_int8_decode_linear_bridge(x, qweight, scale, local_size=4).realize()

  runner = TinyJit(fn)
  outputs = []
  with installed_raw_rowwise_research_bridge(), Context(JIT=1):
    for row in inputs:
      out = runner(Tensor(row, device="METAL", dtype=dtypes.float32).realize()).numpy()
      np.testing.assert_allclose(out, _reference(row, qweight_np, scale_np), rtol=1e-5, atol=1e-5)
      outputs.append(out.tolist())
  assert runner.captured is not None
  custom_calls = [call for call in runner.captured.linear.src if call.src[0].op is Ops.CUSTOM_FUNCTION]
  assert len(custom_calls) == 1, f"expected one custom raw call, got {len(custom_calls)}"
  assert custom_calls[0].src[0].arg == RAW_CUSTOM_FUNCTION
  print("bridge replay outputs:", outputs)


def _prove_bridge_output_is_consumable(qweight_np: np.ndarray, scale_np: np.ndarray, qweight: Tensor, scale: Tensor) -> None:
  row = np.array([0.25, 0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0], dtype=np.float32)

  def fn(x: Tensor) -> Tensor:
    return (raw_rowwise_int8_decode_linear_bridge(x, qweight, scale, local_size=4) + 1.25).realize()

  runner = TinyJit(fn)
  with installed_raw_rowwise_research_bridge(), Context(JIT=1):
    for multiplier in (1.0, 2.0, -1.0):
      current = row * multiplier
      out = runner(Tensor(current, device="METAL", dtype=dtypes.float32).realize()).numpy()
      np.testing.assert_allclose(out, _reference(current, qweight_np, scale_np) + 1.25, rtol=1e-5, atol=1e-5)
  print("bridge output consumed by later tinygrad op")


def main() -> None:
  _require_research_tinygrad()
  qweight_np, scale_np, qweight, scale = _make_weights()
  _prove_existing_helper_rejects(qweight, scale)
  _prove_bridge_replay(qweight_np, scale_np, qweight, scale)
  _prove_bridge_output_is_consumable(qweight_np, scale_np, qweight, scale)
  print("ok research raw metal capture bridge prototype")


if __name__ == "__main__":
  main()
