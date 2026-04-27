# Shared-Source Layer13 Non-Rejected Runtime Hypothesis Review

## Objective

Resolve `shared-source-layer13-nonrejected-runtime-hypothesis-058`: decide whether the corrected shared-source-layer13 exact phase target justifies a new runtime evo child.

## Evidence

The corrected neighbor-target comparison identified `shared-source-layer13` as the dominant exact target:

- 888 phase source items
- ~14.503 ms phase elapsed
- ~14.422 ms in `kv_projection`
- 7 MetalGraph batches preserved
- 0 raw gate/up runners
- overlap reporting maps phase buckets back to `attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention`

The runtime path is `GemmaAttention._project_kv(...)` followed by normalization/RoPE, cache update, and sharing of `current_entry` for later shared-KV consumer layers.

## Rejection Ledger

The apparent runtime levers are already covered by rejected or exhausted probes:

- Raw Metal runner / capture bridge: rejected because it is not first-class MetalGraph compatible.
- Q/K/V rowwise-int8 projection fusion: rejected as `exp_0011`, regressed to 27.7816 tok/s and failed the long floor.
- K/V projection rank/view reshapes and split/no-split variants: rejected in prior evo/runtime attempts.
- Producer materialization / prepacked K/V repeat / deferred assign-realize variants: rejected.
- Shared-source windowed allocation: rejected; physical cache length did not reduce source count or elapsed profile time.
- Store/RHS packing tweaks: attribution shows store and pack are ~single-source tiny buckets; the hot mass is projection.

## Decision

Do not launch a new shared-source-layer13 runtime evo child without a genuinely new transformation. The current evidence points at the same K/V projection mass already attacked from several angles. Repeating it with a new label would be numerology with a commit hash.

## Next Direction

Move to an orthogonal model.py runtime surface that was previously shortlisted but not exhausted in this loop: decode-only attention-output view elision. It should be tested only as a narrow evo child with correctness gates and the current exp_0005 floor, not committed directly to main unless it beats the benchmark.

## Verification

No code changed in this increment. The decision is grounded in:

- `docs/plans/2026-04-27-compare-neighbor-exact-phase-targets.md`
- `state/evolution-log.md` rejection entries for 037, 042, 043, 047-053
- direct inspection of `tinygrad_gemma/model.py::GemmaAttention._project_kv` and cache update flow
