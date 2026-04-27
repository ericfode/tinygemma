# Metal Int8 Raw Runner Capture Safety Plan

Date: 2026-04-27
Increment: `metal-int8-raw-runner-capture-safety-050`

## Context

Increment 049 made `_rowwise_int8_decode_linear_program` compatible with the research tinygrad Metal runtime by compiling the Metal source to MTLB bytes before calling `Device.runtime`. That removed the immediate `Invalid library file` blocker, but did not prove that the raw runner is safe inside TinyJit capture/replay.

The capture APIs differ materially:

- stock tinygrad TinyJit capture stores `ExecItem`s through `capturing[0].add(item)`;
- the research tinygrad runtime stores linear UOps through `capturing[0].add_linear(linear, var_vals)` and has no generic `add(ExecItem)` hook.

`metal_int8.py` currently assumes the stock-style hook whenever `capturing` and `CAPTURING` are active. Under the research runtime, integrating this raw runner into a jitted decode path would either fail with an unhelpful attribute error or, if later papered over incorrectly, risk an even worse failure: immediate execution during capture with no replay item recorded.

## Target

`tinygrad_gemma/metal_int8.py`, specifically `_run_or_capture_rowwise_int8_decode_linear`.

## Objective

Make capture behavior explicit and testable before any model hot-path integration:

1. If the active capture object supports stock-style `add(ExecItem)`, record the raw runner `ExecItem` and execute it for the capture pass.
2. If capture is active but the object does not support `add(ExecItem)`, raise a clear `RuntimeError` before running the kernel.
3. Preserve non-capture eager behavior.
4. Preserve the post-049 MTLB program construction behavior.

## Non-goals

- Do not integrate `metal_rowwise_int8_decode_linear` into `RowwiseInt8Linear`, `GemmaMLP`, or decode.
- Do not claim throughput improvement.
- Do not attempt to encode the raw runner into research tinygrad linear UOps in this increment.
- Do not tune `local_size`.

## Test strategy

Add focused tests that avoid requiring real Metal hardware:

- fake a stock-style capture object with `add(item)` and monkeypatch `RowwiseInt8DecodeLinearRunner.__call__` so the test proves both capture registration and immediate execution happen;
- fake a research-style capture object with only `add_linear(...)` and prove `_run_or_capture_rowwise_int8_decode_linear` raises a clear `RuntimeError` before executing;
- keep the existing MTLB program construction contract from 049.

Then run real canaries:

- stock tinygrad focused test suite;
- research tinygrad import/capture canary that expects the explicit capture error rather than silent bypass;
- full pytest and the cheap repo gates.

## Acceptance criteria

- Focused tests pass.
- Research tinygrad capture canary demonstrates explicit rejection under `add_linear`-only capture.
- Full pytest passes.
- CLI help and Metal smoke pass.
- The repo-loop state and evolution log record that 050 is a safety increment, not a performance increment.
