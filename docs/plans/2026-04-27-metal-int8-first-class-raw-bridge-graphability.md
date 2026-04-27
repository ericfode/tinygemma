# Metal Int8 First-Class Raw Bridge Graphability Decision

## Objective

Resolve `metal-int8-first-class-raw-bridge-graphability-052`: after the
prototype-only research capture bridge proved that a raw rowwise-int8 Metal
runner can be replayed under a local `pm_exec` monkeypatch, decide whether there
is an honest first-class integration path that is graphable enough for the E2B
METAL decode benchmark.

## Decision

Reject production integration of the raw Metal custom Runner. Keep
`tinygrad_gemma/metal_int8.py` only as a compatibility/research canary surface,
and prune live-facing hooks that make the raw gate/up path look selectable from
the model or profiler.

The next optimization surface should return to graphable Tensor/tinygrad program
paths, with the profiler as the arbiter. A faster standalone Metal kernel is not
useful if it fragments the decode replay graph; the machinery is exacting, and
rather unsentimental about beautiful orphan kernels.

## Evidence

### Research bridge status

The accepted 051 prototype demonstrated feasibility only under process-local
patching:

- `tinygrad.engine.realize.pm_exec` needed a custom handler for
  `Ops.CUSTOM_FUNCTION(arg="raw_rowwise_int8_decode_linear")`.
- `tinygrad.engine.jit._call_outs_ins` needed a matching input/output mapping.
- The custom call had to preserve `x`, `qweight`, and `scale` UOps until replay
  time; a buffer-only bridge was unsound because it lost `PARAM` identity.

That is not a production API surface. It is a controlled research instrument.

### MetalGraph admits compiled tinygrad programs, not raw host-call runners

The research tinygrad checkout used by profiling/benchmarks reports:

```text
GraphRunner.supports_exec_item:
  return new_call.src[0].op in (Ops.SINK, Ops.PROGRAM) and len(GraphRunner._all_devs(batch_devs, new_call)) == 1

MetalGraph.supports_exec_item:
  ...
  return GraphRunner.supports_exec_item(batch_devs, new_call)
```

`MetalGraph.__init__` then assumes every scheduled item is backed by a
`CompiledRunner` with a Metal pipeline and launch dimensions:

```text
prg: CompiledRunner = cast(CompiledRunner, ji.prg)
all_pipelines.append(prg._prg.pipeline_state)
icb_command.setComputePipelineState(prg._prg.pipeline_state)
global_size, local_size = prg.p.launch_dims({v: 0 for v in self.vars})
```

A project-local `RowwiseInt8DecodeLinearRunner` is a host-call runner around raw
Metal source/library bytes. It does not naturally provide the compiled tinygrad
program metadata required by `MetalGraph` ICB construction. Making it fit would
require a tinygrad-runtime extension, not a narrow repo optimization.

### Existing benchmark evidence already showed fragmentation risk

The raw gate/up profile artifact (`benchmarks/gemma4-metal-decode-graph-raw-gate-up-current.json`)
recorded `raw_gate_up_runner_count=35` and post-graph execution with many more
items than the accepted graphable surface. The accepted frontier remains the
calibrated evo baseline `exp_0005` at `28.6153` tok/s. No raw-runner row has
superseded it.

## Code cleanup in this increment

- Removed the dead `GemmaMLP._can_use_metal_fused_int8_gate_up(...)` method and
  its test. It always returned `False` and was not used by `GemmaMLP.__call__`.
- Removed the `scripts/profile_decode_jit.py --metal-int8-gate-up` user-facing
  switch. New profile payloads keep the historical `metal_int8_gate_up` field
  fixed at `"default"` for artifact-schema continuity, but the profiler no
  longer advertises a raw selectable path.
- Marked `tinygrad_gemma/metal_int8.py` as prototype-only in its module docstring.

## Invalidation criteria for revisiting

Reopen this branch only if tinygrad gains one of the following first-class
surfaces:

1. A graphable custom-call abstraction that `GraphRunner.supports_exec_item` and
   `MetalGraph` can batch without per-run monkeypatching.
2. A way to express the rowwise-int8 decode-linear kernel as a normal tinygrad
   `CompiledRunner`/`ProgramSpec` while preserving replay parameter identity and
   MetalGraph batching.
3. A benchmarked upstream/runtime patch showing raw custom runners can remain
   inside the same graph-batched decode replay path without fragmenting into
   host-call items.

Until then, graphable Tensor program transformations remain the honest path.

## Next target

`evo-frontier-profiler-backed-graphable-surface-053`: refresh the current
profile/frontier evidence and pick the next graphable `model.py` or profiler
surface. Do not spend real E2B throughput gates on raw custom Runner integration
without first satisfying the invalidation criteria above.
