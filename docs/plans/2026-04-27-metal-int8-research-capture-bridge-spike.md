# Metal Int8 Research Capture Bridge Spike Plan

Date: 2026-04-27
Increment: `metal-int8-research-capture-bridge-spike-051`

## Context

Increments 049 and 050 established two facts:

1. the dormant raw rowwise-int8 METAL decode-linear runner can now construct its Metal program under both stock tinygrad and the research tinygrad runtime by passing compiled MTLB bytes to `Device.runtime`;
2. stock tinygrad TinyJit can capture/replay the raw runner through `add(ExecItem)`, while research tinygrad only exposes `add_linear(linear, var_vals)`, so the production helper now fails closed when asked to capture through an add-linear-only runtime.

The active E2B path uses `/Users/ericfode/src/.tinygrad_research`, so any future raw-runner model integration must be represented as a replayable research tinygrad linear/UOp. A buffer-only shortcut is unsound: it loses `PARAM` identity and reads capture-time buffers during replay.

## Findings from 051 feasibility review

Research tinygrad can execute custom calls in `run_linear`, but only if `tinygrad.engine.realize.pm_exec` has a matching handler. Unknown `Ops.CUSTOM_FUNCTION` calls are otherwise ignored by the pattern matcher, which is an unpleasantly quiet failure mode.

The smallest honest bridge shape is:

```text
Ops.LINEAR
  Ops.CALL
    Ops.CUSTOM_FUNCTION(arg="raw_rowwise_int8_decode_linear")
      const(in_features)
      const(out_features)
      const(local_size)
    output_buffer_uop
    x_buffer_uop
    qweight_buffer_uop
    scale_buffer_uop
```

The executor must resolve the call operands through `resolve_params(ctx, call)` at replay time, then invoke `RowwiseInt8DecodeLinearRunner` on the current buffers. This is the key property: preserve UOps until replay, do not convert to concrete buffers during capture.

## Target for this increment

Add an inspectable prototype script only:

- `scripts/prototype_research_raw_metal_capture_bridge.py`

The script may install a process-local runtime patch while it runs, then restore globals before exit. It must not change the production model path.

## Non-goals

- Do not modify `GemmaMLP`, `RowwiseInt8Linear`, or the default raw runner helper.
- Do not enable the bridge by default.
- Do not patch `/Users/ericfode/src/.tinygrad_research` source files.
- Do not claim benchmark throughput improvement.
- Do not add broad global monkeypatching to import-time package code.

## Acceptance criteria

The prototype script must, under the research tinygrad import path:

1. reproduce the current explicit add-linear-only rejection for the existing production helper inside TinyJit capture;
2. install a temporary `pm_exec` handler for `raw_rowwise_int8_decode_linear`;
3. build the raw op as a UOp-preserving custom function linear, not from concrete buffers;
4. run a TinyJit capture/replay sequence across at least three different METAL inputs;
5. verify every replay output against a NumPy reference;
6. prove a later tinygrad op can consume the raw output;
7. restore patched globals before exit.

## Decision rule

If the script passes, record 051 as a prototype-only feasibility result. The next increment may inspect whether the bridge can be made first-class and graphable enough to avoid the known graph-fragmentation regression.

If the script fails, record the raw-runner research capture bridge as infeasible for now and return to graphable `model.py` / profiler-backed surfaces.
