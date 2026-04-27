# Metal Int8 MTLB Compatibility Plan

Date: 2026-04-27
Increment: `metal-int8-research-mtlb-compatibility-049`

## Context

The active evo frontier remains `exp_0005` at default E2B int8 METAL decode score `28.6153`. The recent model.py experiment family has negative evidence across scaling guards, K/V rank/view reshapes, attention-output view elision, RMSNorm scale caching, and active shared-KV view reuse. No new model.py micro-surface is currently evidence-backed.

The separate raw rowwise-int8 Metal runner in `tinygrad_gemma/metal_int8.py` is still dormant. Increment 047 showed that its stock-tinygrad prototype can run, but the actual research tinygrad path used by the E2B harness fails before launch with `RuntimeError: Invalid library file`.

## Root cause

`_rowwise_int8_decode_linear_program()` passes raw Metal source bytes directly into `Device["METAL"].runtime(...)`.

That works on stock tinygrad versions whose `MetalProgram` accepts raw source and compiles it internally. The local research tinygrad runtime under `/Users/ericfode/src/.tinygrad_research` expects compiled Metal library bytes (`MTLB...ENDT`) and calls `newLibraryWithData_error` directly. Source bytes are therefore interpreted as an invalid compiled library.

## Goal

Make the raw rowwise-int8 Metal program construction compatible with both stock and research tinygrad by compiling `ROWWISE_INT8_DECODE_LINEAR_SOURCE` through `Device["METAL"].compiler.compile_cached(...)` before passing bytes to `Device["METAL"].runtime(...)`.

## Non-goals

- Do not wire `metal_rowwise_int8_decode_linear` into `RowwiseInt8Linear`.
- Do not re-enable raw Metal fused gate/up in `GemmaMLP`.
- Do not tune `local_size` as a runtime optimization.
- Do not claim an E2B decode throughput improvement from this compatibility patch.

## Implementation plan

1. Add a focused test proving `_rowwise_int8_decode_linear_program()` sends compiled bytes from `compiler.compile_cached(...)` to `runtime(...)`, not raw source bytes.
2. Run the focused test and confirm RED under the current implementation.
3. Patch `tinygrad_gemma/metal_int8.py` minimally:
   - resolve and validate device as today;
   - fetch `metal_device = Device[resolved_device]`;
   - compile `ROWWISE_INT8_DECODE_LINEAR_SOURCE` with `metal_device.compiler.compile_cached(...)`;
   - pass the compiled library bytes to `metal_device.runtime(...)`.
4. Run the focused test and existing RowwiseInt8/metal_int8 tests.
5. Run a small research-tinygrad METAL canary that asserts compiled bytes begin with `MTLB` and end with `ENDT`, then executes a tiny raw rowwise-int8 decode-linear call.
6. Run cheap repo gates: focused tests, CLI help, and Metal smoke.
7. Record the outcome in `state/evolution-log.md` and `configs/repo-loop-state.json`.

## Acceptance

- Focused compile-before-runtime test passes.
- Existing RowwiseInt8/metal_int8 tests pass.
- Research tinygrad METAL canary passes without `RuntimeError: Invalid library file`.
- `tinygrad-gemma --help` and `scripts/smoke_metal.py` still pass.
- No model.py integration or throughput claim is made.
