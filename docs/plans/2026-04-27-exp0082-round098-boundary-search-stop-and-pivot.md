# exp0082 Round 098 Boundary-Search Stop and Pivot Plan

Date: 2026-04-27 08:44 PDT

## Summary

The `exp_0082` `model.py` layer-boundary search has now produced two consecutive no-improvement rounds after reaching the current frontier.

Current frontier remains:

- Experiment: `exp_0082`
- Score: `66.3476` tok/s on the registered E2B int8 METAL `128/20` benchmark
- Effective E2B layer-output cutpoints: `[0,1,2,3,4,6,8,9,11,12]`
- Final hidden boundary: retained post-norm `.contiguous().realize()` before logits
- Evo state after round 098: `experiments=91`, `committed=25`, `discarded=66`, `failed=0`, `active=0`

## Round 098 worker results

All children passed correctness gates and regressed by performance.

| Experiment | Parent | Probe | Score | Decision |
| --- | --- | --- | ---: | --- |
| `exp_0087` | `exp_0079` | side branch: add only layer-1 earliest boundary to layer-13-positive branch | `56.7466` | discarded; regressed vs parent `63.9473` |
| `exp_0088` | `exp_0082` | layer-8 post-attention/pre-MLP decode boundary | `51.9819` | discarded; severe regression |
| `exp_0089` | `exp_0082` | layer-11 post-attention/pre-MLP decode boundary | `64.9409` | discarded; regression |
| `exp_0090` | `exp_0082` | layer-13 post-attention/pre-MLP decode boundary | `63.7468` | discarded; regression |

Common validation from workers:

- Focused `tests/test_profile_decode_jit.py`: `32 passed` for each worktree.
- Evo gates passed: `_init_gate`, `metal_smoke`, `cli_help`, `e2b_int8_metal_hash16`.
- Short benchmark invariants preserved: `generated_tokens=128`, `measured_decode_tokens=108`, `rollout_jit_count=127`, `decode_fallback=false`.
- Output hash preserved: `1c39dd289bb7363f0f600ae1087ad39926e9733447df72bf03955c92d06066b0`.

## Interpretation

The boundary family is exhausted for the current epoch:

1. Additive layer-output cuts on `exp_0082` at 5, 7, 10, and 13 regressed in round 097.
2. Intra-layer post-attention/pre-MLP boundaries at layers 8, 11, and 13 regressed in round 098.
3. The partial `exp_0079 + layer1` side branch also regressed, so the earlier layer-13-positive branch does not recover with a partial early-frontier composition.

This is now more than a local stumble. The frontier is probably sitting on a real graph-capture/layout ridge, and additional `.contiguous().realize()` placement is behaving as graph fragmentation rather than graph simplification.

## Stop rule

Stop the current `model.py` boundary-placement search unless a new profiler artifact identifies a different targetable source class. Specifically, do not run more children that merely:

- add/remove/shift layer-output cutpoints,
- add intra-layer `.contiguous().realize()` seams,
- compose layer 13 with more early cutpoints,
- touch final-hidden narrowing/movement,
- revisit K/V projection fusion/caching/cache geometry/raw Metal runners without new evidence.

## Pivot options

### Preferred next move: measurement-first profiler instrumentation

Construct a profiling-only increment that separates the dominant layer-13 `kv_projection` bucket into finer subphases without changing runtime behavior:

- fused int8 matmul,
- scale broadcast/cast,
- chunk/split,
- head reshape,
- RoPE/RMSNorm if still conflated by source attribution.

This should be a measurement branch or repo instrumentation increment, not a performance child. Success criterion: a profile artifact that reveals whether the `644`-source layer-13 `kv_projection` bucket is one indivisible lowering wall or a smaller reshape/cast/source-attribution leak.

### Secondary move: separate profile/source-count evo run

If continuing with evo automation, initialize a separate run whose metric is profile-derived and lower-is-better, for example:

- `source_count` from `scripts/profile_decode_jit.py`,
- `graph_batch_count`,
- profiled `post_graph_execution` elapsed time,
- or a weighted combination such as `source_count + 100 * graph_batch_count`.

This would be a new benchmark contract, not a continuation of the current throughput-only child stream. It needs a Goodhart gate preserving the E2B short hash/JIT/fallback invariants.

### Deferred move: target outside `model.py`

Only after measurement clarifies the subphase should a new target/run consider non-`model.py` surfaces such as `tinygrad_gemma/metal_int8.py`. Existing raw-runner compatibility failures and K/V/cache-geometry regressions remain negative evidence.

## Immediate recommendation

Do not spawn another performance-worker round from `exp_0082` under the current target. Record the stall, keep `exp_0082` as the frontier, and pivot to measurement-first instrumentation before further code-search.
