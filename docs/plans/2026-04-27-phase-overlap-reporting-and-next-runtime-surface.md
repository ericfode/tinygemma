# Phase Overlap Reporting and Next Runtime Surface

## Objective

Resolve `phase-overlap-reporting-and-next-runtime-surface-056`: make cache-write phase attribution report whether a phase bucket actually belongs to its claimed parent source category.

## Change

`scripts/profile_decode_jit.py` now emits parent-category overlap for cache-write phase attribution:

- Per graph/source row: `cache_write_phase_parent_category_counts`
- Original capture summary: `original_capture.cache_write_phase_parent_category_counts`
- Aggregate phase summary: `source_attributed_cache_write_phase_summary.by_phase_parent_category`

This makes the profiler answer the important question directly: "does this phase bucket overlap only the expected parent, or is it borrowing time from another graph/source category?"

## Test

Added `test_profile_phase_summary_reports_parent_category_overlap`, which constructs local-layer12 and shared-source-layer13 phase-tagged fake source items and verifies that overlap reporting preserves the exact parent category for each phase.

## Artifact

Generated:

- `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-overlap-700.json`
- `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-overlap-700.csv`

Key values:

- targets: [`local-layer12`]
- graph batches: 7
- raw gate/up runners: 0
- phase source count: 256
- conflict source count: 6
- unclassified source count: 0
- `kv_projection`: 250 sources, ~2.954 ms, ~5.96% elapsed
- `by_phase_parent_category` shows every local-layer12 phase bucket maps back to `attention_packed_cache_write__role_local__layer_12__type_sliding_attention`

## Decision

The corrected and overlap-checked local-layer12 surface is exclusive but small. It gives a reliable profiling target, not yet a runtime patch. The next runtime candidate should require a concrete transformation tied to the 250-source local-layer12 `kv_projection` bucket and should avoid already-rejected K/V projection reshapes/fusions.

## Verification

- RED: new overlap test failed before the field existed.
- Profiler test module: 35 passed, 2 warnings.
- Full suite: 84 passed, 2 warnings.
- METAL smoke: `rollout_jit_count=3`, `decode_fallback=False`.
