# Graphable Rowwise-Int8 QKV Fusion Probe

## Objective

Resolve `evo-frontier-profiler-backed-graphable-surface-053` by testing the most credible post-raw-runner runtime surface: a tinygrad-native, graphable rowwise-int8 Q/K/V projection fusion for one-token decode.

## Rationale

The previous raw Metal bridge was rejected because it could not enter `MetalGraph` as a first-class compiled program. The next plausible path was therefore not another host-call runner, but a plain Tensor-level fusion modeled after the accepted fused rowwise-int8 MLP gate/up and K/V projection paths.

Three independent read-only reviews converged on the same candidate:

1. Layer-13 source attribution says the remaining shared-source sliding bucket is overwhelmingly K/V projection rather than cache store.
2. `GemmaAttention._project_kv` already has a graphable rowwise-int8 fused K/V path.
3. A Q/K/V fusion is distinct from the rejected K/V dequant cache, transposed cache, rank flattening, output view, RMSNorm cache, active shared-KV view reuse, and raw runner surfaces.

## Experiment

Evo child:

- Parent: `exp_0005`
- Child: `exp_0011`
- Hypothesis: `probe: graphable rowwise-int8 qkv decode fusion`
- Commit in child worktree: `94ed982 Probe graphable rowwise-int8 QKV fusion`

Implementation shape inside the evo worktree:

- Added `_FUSED_INT8_QKV` cache in `tinygrad_gemma/model.py`.
- Added `GemmaAttention._can_use_fused_int8_qkv(query_len)` guarded to:
  - one-token decode only
  - inference only
  - non-shared-KV layers only
  - Q/K/V all `RowwiseInt8Linear`
  - no biases
- Added `_project_qkv(...)` using one Tensor matmul over concatenated rowwise-int8 Q/K/V weights and scales.
- Added focused parity test `test_runtime_int8_attention_fused_qkv_matches_separate_path`.

Pre-evo gates:

- `tests/test_tinygrad_gemma.py::test_runtime_int8_attention_fused_qkv_matches_separate_path`: passed.
- `tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py`: `74 passed, 1 skipped, 2 warnings`.
- `scripts/smoke_metal.py` under research tinygrad: passed with `rollout_jit_count=3`, `decode_fallback=False`.
- Real E2B int8 METAL hash16 gate from main repo root: passed at `29.464043 tok/s`, expected hash `12c16f74...`.

## Result

`evo run exp_0011` rejected the child:

- Default score: `27.7816 tok/s`
- Parent `exp_0005` score: `28.6153 tok/s`
- Failure: `e2b_int8_metal_hash1000_current_floor`
- Evo action: discarded `exp_0011`

Discard reason recorded in evo:

> graphable rowwise-int8 QKV decode fusion regressed default score to 27.7816 versus exp_0005 28.6153 and failed e2b_int8_metal_hash1000_current_floor; do not fuse q with K/V projection under this benchmark.

## Decision

Rejected. Do not retry Q/K/V rowwise-int8 projection fusion under the current benchmark unless a later profiler artifact shows a different scheduling/capture behavior.

The likely practical lesson is that folding Q into the K/V projection enlarges the producer graph and worsens scheduling or cache-write dependency structure, even though the operation is graphable and numerically correct.

## Next credible increment

Move to a profiler-first increment:

- Generalize `scripts/profile_decode_jit.py` phase attribution target selection beyond hardcoded layer-13 shared-source sliding attention.
- Use it to classify local producer layers and other shared-source layers before spending more E2B runtime gates.

This is a diagnostic step rather than another blind runtime patch, and it directly addresses the remaining uncertainty: where the dominant local cache-write source mass actually lives.
