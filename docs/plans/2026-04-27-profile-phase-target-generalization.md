# Profile Phase Target Generalization

## Objective

Resolve `profile-phase-target-generalization-054`: make the profiler’s
cache-write phase cutpoint instrumentation configurable so future experiments
can isolate local/shared-source cache-write phases without editing `scripts/profile_decode_jit.py` each time.

## Changes

- Replaced the hardcoded layer-13-only phase target with `CacheWritePhaseTarget` selectors.
- Added scoped target selection through `cache_write_phase_target_scope(...)`.
- Added parser support for:
  - `layer13` (backward-compatible default)
  - `all-shared-source`
  - `all-local`
  - `all-packed`
  - exact selectors: `local-layer<N>`, `shared-source-layer<N>`, `shared-consumer-layer<N>`
- Added CLI flags:
  - `--phase-cutpoints`
  - repeated `--phase-target TARGET`
- Kept `--layer13-phase-cutpoints` as a backward-compatible alias for existing workflows.
- Added JSON/stdout metadata fields:
  - `cache_write_phase_targets`
  - `cache_write_phase_cutpoints`

## Verification

- RED tests failed before implementation:
  - `test_profile_cache_write_phase_target_accepts_scoped_presets`
  - `test_profile_cache_write_phase_target_parser_supports_multiple_presets`
  - `test_profile_cache_write_phase_target_parser_supports_exact_role_layer_selectors`
- GREEN tests passed after implementation:
  - `tests/test_profile_decode_jit.py`: 32 passed.
  - Full suite: 82 passed, 2 warnings.
  - CLI help includes `--phase-target`, `--phase-cutpoints`, and dynamic exact selector help.
  - METAL smoke passed with `rollout_jit_count=3` and `decode_fallback=False`.

## Real profiler artifacts

Generated but left untracked as benchmark artifacts:

- `benchmarks/gemma4-metal-decode-graph-all-local-phase-profile-700.json`
- `benchmarks/gemma4-metal-decode-graph-all-local-phase-profile-700.csv`
- `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-profile-700.json`
- `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-profile-700.csv`

Observed summaries:

- `--phase-target all-local --phase-cutpoints` preserved `MetalGraph` batching with 7 graph batches and `42.066 ms` profile elapsed time.
- `--phase-target local-layer12 --phase-cutpoints` preserved `MetalGraph` batching with 7 graph batches and `50.537 ms` profile elapsed time.
- Exact local-layer12 targeting produced phase attribution under `attention_packed_cache_write__role_local__layer_12__type_sliding_attention__phase_kv_projection`.

## Notes

Broad targets such as `all-local` are useful for coarse scanning but can produce phase metadata conflicts across graph batches. Exact selectors are the safer next diagnostic surface for runtime experiments.

## Next candidate

Use the new exact selector to profile one local packed-cache layer at a time, then test a runtime change only if the phase attribution identifies a non-conflicting, high-cost, graphable surface.
