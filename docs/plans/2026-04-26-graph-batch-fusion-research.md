# Graph-Batch Fusion Research / Source-Coherence Gate

Date: 2026-04-26

## Objective

Continue the `100 tok/s` optimization campaign with tokens per second as the hard metric. The narrow question for this pass was whether the latest `METAL` graph profile exposes graph-fusion boundaries that can be attacked directly, or whether the apparent boundaries are tinygrad graph batching artifacts.

## Metric frame

- 100 tok/s means `10.000000 ms/token`.
- Current artifact-level long floor remains `19.424036 tok/s`, or `51.482606 ms/token`, from `benchmarks/gemma4-metal-e2b-int8-1000-attribution-only-restored-current.csv`.
- Clean repeat artifact: `19.613212 tok/s`, or `50.986039 ms/token`.
- Short SDPA-off control row: `30.808261 tok/s`, or `32.458827 ms/token`, but this is a short gate and must not be treated as the accepted long floor.

## Profile boundary finding

Primary artifact: `benchmarks/gemma4-metal-decode-graph-attribution-only-restored-profile-512.json`.

The profile contains `7` post-graph `MetalGraph` executions over `2994` original source items with complete attribution and `34.319667 ms/token` elapsed time. The graph rows are:

| ordinal | display | source range | source count | elapsed ms | dominant rollup |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | `<batched 32>` | `0..31` | 32 | 0.304750 | local packed cache writes |
| 1 | `<batched 64>` | `32..95` | 64 | 0.522500 | local packed cache writes |
| 2 | `<batched 128>` | `96..223` | 128 | 1.086625 | local packed cache writes |
| 3 | `<batched 256>` | `224..479` | 256 | 2.921375 | local packed cache writes |
| 4 | `<batched 512>` | `480..991` | 512 | 4.727583 | local packed cache writes |
| 5 | `<batched 1024>` | `992..2015` | 1024 | 10.283125 | local + shared-source packed cache writes |
| 6 | `<batched 978>` | `2016..2993` | 978 | 14.473708 | shared-source packed cache writes + MLP |

Interpretation: these are not semantic model fusion boundaries. They match tinygrad's `JIT_BATCH_SIZE` growth policy: start at 32, then double the graph batch size until the remainder. The graph cuts split layer/category scopes mid-stream, so treating the seven rows as seven layer-level barriers would be a category error with unusually good indentation.

A synthetic probe was recorded at `benchmarks/tinygrad-metal-graph-jitbatch-policy-probe.json`:

| `JIT_BATCH_SIZE` | synthetic graph source counts |
| ---: | --- |
| 32 | `[32, 64, 4]` |
| 64 | `[64, 36]` |
| 128 | `[100]` |
| 512 | `[100]` |
| 0 | `[100]` |

This confirms that larger or zero `JIT_BATCH_SIZE` can collapse graph batches for graphable kernels on the local tinygrad research runtime. It does **not** prove a Gemma tok/s win; it only earns a measured Gemma probe after source coherence is restored.

## Bottleneck finding

Source-attributed profile rollup:

| category | source items | elapsed ms | share |
| --- | ---: | ---: | ---: |
| `attention_packed_cache_write_local` | 1709 | 16.763029 | 48.84% |
| `attention_packed_cache_write_shared_source` | 1178 | 15.973113 | 46.54% |
| `mlp` | 107 | 1.583524 | 4.61% |

Top detailed categories:

| category | source items | elapsed ms | share |
| --- | ---: | ---: | ---: |
| `attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention` | 885 | 11.798664 | 34.38% |
| `attention_packed_cache_write__role_shared_source__layer_14__type_full_attention` | 293 | 4.174449 | 12.16% |
| `attention_packed_cache_write__role_local__layer_12__type_sliding_attention` | 253 | 2.540655 | 7.40% |
| `attention_packed_cache_write__role_local__layer_11__type_sliding_attention` | 233 | 2.339813 | 6.82% |
| `attention_packed_cache_write__role_local__layer_10__type_sliding_attention` | 213 | 2.138970 | 6.23% |
| `mlp` | 107 | 1.583524 | 4.61% |

The still-credible semantic target is not the MetalGraph batch boundary. It is the dependency-closed packed cache-write scope, especially layer 13 shared-source sliding attention.

## Source-coherence blocker

The checked-out `tinygrad_gemma/model.py` currently does **not** match the packed-cache runtime state described by the accepted artifacts and tests:

- `GemmaCacheEntry` has only `key`, `value`, and `length`; no `packed` backing tensor.
- `realize_cache_update` performs two split writes: `key_cache[:, :, start:end, :].assign(key).realize()` and `value_cache[:, :, start:end, :].assign(value).realize()`.
- `model.py` has no `make_packed_cache_entry`, no `packed_cache` update argument, and no rolled-cache helper APIs referenced by modified tests and evolution-log entries.
- `scripts/profile_decode_jit.py` now includes a compatibility fallback for this split-cache state, so it can import, but a fresh profile on this source would not be apples-to-apples with the accepted packed-cache artifacts.

Therefore no new runtime speed claim should be made from this dirty source state. Restore or identify the source snapshot that produced the accepted packed-cache artifacts before running further Gemma throughput gates.

## Council synthesis

Three independent reviewers agreed on the main points:

1. The seven `MetalGraph` rows are tinygrad `JIT_BATCH_SIZE` grouping artifacts, not semantic fusion boundaries.
2. The top profile bottleneck remains packed cache-write work, especially layer 13 shared-source sliding attention.
3. The current live source is split-cache and incoherent with the packed-cache artifacts/tests.
4. Do not claim the current tree implements the accepted runtime checkpoint; first restore source coherence, then measure.

## Prioritized next probes

### 0. Restore source/artifact coherence before more Gemma gates

Required before a credible `tok/s` claim:

- Restore or locate the final-lane packed-cache runtime implementation recorded in `packed-kv-lastdim-cache-assignment-034` and later artifacts.
- Ensure `tinygrad_gemma/model.py`, `scripts/profile_decode_jit.py`, tests, and evolution logs agree on cache geometry.
- Run focused packed-cache tests before any `METAL` benchmark.

### 1. Graph-batch knob probe after coherence

Once source coherence is restored, run profile control and `JIT_BATCH_SIZE=0` in the same coherent state:

```bash
PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. \
JIT_BATCH_SIZE=0 \
.venv/bin/python scripts/profile_decode_jit.py \
  --model-dir checkpoints/gemma-4-E2B-int8 \
  --device METAL \
  --context-length 512 \
  --jit-mode 1 \
  --out benchmarks/gemma4-metal-decode-graph-jitbatch0-profile-512.json \
  --csv-out benchmarks/gemma4-metal-decode-graph-jitbatch0-profile-512.csv
```

Acceptance for moving to a long tok/s gate:

- `source_attribution.original_exec_count` remains `2994` or lower.
- `post_graph_execution.graph_batch_count` drops from `7` toward `1`.
- `elapsed_ms` improves meaningfully versus `34.319667 ms/token` without attribution loss.

If earned, run the hard long row and a same-session default control:

```bash
PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. \
JIT_BATCH_SIZE=0 \
.venv/bin/python scripts/benchmark_gemma4_matrix.py \
  --root checkpoints \
  --sizes E2B \
  --formats int8 \
  --devices METAL \
  --beams 0 \
  --max-new-tokens 1000 \
  --decode-warmup-tokens 20 \
  --progress-every 100 \
  --out benchmarks/gemma4-metal-e2b-int8-1000-jitbatch0-current.csv
```

A runtime claim requires stable hash, `rollout_jit_count=999`, `decode_fallback=false`, and repeated lower row above `19.424036 tok/s`.

### 2. Layer-13 producer/store sub-attribution

If `JIT_BATCH_SIZE=0` does not improve real tok/s, continue with profiler-only `layer13-shared-source-producer-store-attribution-038`: split the layer-13 shared-source packed write scope into K/V projection, RMSNorm, RoPE, RHS packing, and actual store components. Do not write another runtime patch until this attribution shows what portion is actually removable/fusible.

### 3. Runtime candidates after sub-attribution only

Only after sub-attribution:

- Restore/verify final-lane packed cache as the baseline runtime surface.
- Consider graphable fused int8 K/V or already-packed K/V production only if it targets the dominant layer-13 dependency mass without pre-realizing RHS tensors.
- Continue rejecting raw Metal custom runners, SDPA, prepacked RHS materialization, broad sliding-roll parity, flat-slot layout, active-cache guard, and deferred readback unless new measurements beat the long floor.

## Verification performed in this pass

- Parsed `benchmarks/gemma4-metal-decode-graph-attribution-only-restored-profile-512.json` and confirmed `2994` source items, `7` `MetalGraph` rows, complete attribution, and `34.319667 ms/token`.
- Created `benchmarks/tinygrad-metal-graph-jitbatch-policy-probe.json` with synthetic `JIT_BATCH_SIZE` grouping evidence.
- Updated `scripts/profile_decode_jit.py` so it imports under both packed-cache and current split-cache `model.py` states. This is profiler compatibility only, not a runtime optimization.
- Ran `.venv/bin/python -m pytest tests/test_profile_decode_jit.py -q` successfully after the compatibility patch.

## Non-claim

This pass does not supersede the accepted `19.424036 tok/s` long floor. It records a research conclusion and a source-coherence gate. The path to 100 tok/s is still roughly a `5.15x` reduction from the current artifact floor; pleasant arithmetic, if not yet pleasant performance.

## Follow-up execution result

Source coherence was restored and the ordered probe was executed. The result confirms the graph-batch diagnosis but rejects `JIT_BATCH_SIZE=0` as a runtime optimization:

| artifact | source items | graph batches | profile ms/token | hard `1000/20` tok/s | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `benchmarks/gemma4-metal-decode-graph-source-coherent-control-profile-512.json` / `benchmarks/gemma4-metal-e2b-int8-1000-source-coherent-control-current.csv` | 2994 | 7 | 37.935 | 18.954322 | same-source control |
| `benchmarks/gemma4-metal-decode-graph-jitbatch0-profile-512.json` / `benchmarks/gemma4-metal-e2b-int8-1000-jitbatch0-current.csv` | 2994 | 1 | 36.396 | 18.685010 | rejected |

Both hard rows used the same stable output hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`, `rollout_jit_count=999`, and `decode_fallback=false`. The graph knob reduced launch grouping and improved the same-source profile by `4.06%`, but the long throughput metric regressed by `1.42%` versus the same-session control and remained below the accepted `19.424036 tok/s` floor. Therefore graph-batch count is not the next optimization target.

The next target is `layer13-shared-source-producer-store-subattribution-042`: split the dominant layer-13 shared-source packed cache-write scope into K/V projection, RMSNorm/RoPE, RHS packing, and actual store components before proposing another runtime patch.
