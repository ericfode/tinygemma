# Exp0091 head-first KV scope review

## Objective

Execute the next non-boundary runtime/profile increment after the layer-13 KV subphase profile showed `kv_head_reshape` as the dominant shared-source surface.

## Starting evidence

- Parent frontier before this thread: `exp_0082`, score `66.3476` tok/s.
- Measurement-pivot artifact: `benchmarks/gemma4-metal-decode-graph-exp0082-layer13-kv-subphase-cutpoints-512.json`.
- The layer-13 shared-source cache-write bucket split into:
  - `kv_head_reshape`: `377` sources / `~6.555 ms`.
  - `store`: `228` sources / `~3.951 ms`.
  - `kv_scale_cast`: `29` sources / `~0.499 ms`.
  - `rmsnorm_rope`: `18` sources / `~0.305 ms`.
  - `kv_fused_int8_matmul`: `2` sources / `~0.022 ms`.
  - `rhs_pack`: `1` source / `~0.011 ms`.

## Accepted runtime child

`exp_0091` changed the one-token decode K/V projection layout for the sliding shared-source cache-write producer:

- `_project_kv(..., head_first_single_token=True)` returns `(batch, kv_heads, query_len, head_dim)` directly.
- The decode cache-write path skips the legacy `(batch, query_len, kv_heads, head_dim) -> transpose(1, 2)` shape route for `store_full_length_kv` sliding producers.
- This is a runtime change in the evo worktree only; it is not a main-branch runtime merge.

Result:

- `exp_0091`: `72.8749` tok/s, committed.
- Parent `exp_0082`: `66.3476` tok/s.
- Delta: `+6.5273` tok/s (`+9.84%`).
- Gates passed: `_init_gate`, `metal_smoke`, `cli_help`, `e2b_int8_metal_hash16`.

Follow-up profile generated from the accepted frontier:

- `benchmarks/gemma4-metal-decode-graph-exp0091-head-first-kv-layer13-kv-subphase-profile-512.json`.
- `benchmarks/gemma4-metal-decode-graph-exp0091-head-first-kv-layer13-kv-subphase-profile-512.csv`.
- Post-graph elapsed estimate dropped from `15.5166 ms` to `14.1998 ms` on the comparable profile.
- Layer-13 `kv_head_reshape` attribution remains structurally large (`377` sources / `~5.967 ms`), so source-count metadata still treats the reshape/view roots as dominant even after the throughput improvement.

## Rejected scope widenings

Two natural generalizations were tested and rejected as clean performance regressions:

1. `exp_0092`: extend head-first K/V projection to full-attention shared-source producers.
   - Change: removed the `self.sliding_window is not None` guard.
   - Score: `72.6278` vs parent `72.8749` (`-0.2471` tok/s, `-0.34%`).
   - Gates passed.
   - Decision: do not apply the head-first layout to full-attention shared producers.

2. `exp_0093`: use head-first K/V projection for all sliding decode producers.
   - Change: removed the `self.store_full_length_kv` guard while keeping `self.sliding_window is not None`.
   - RED: `tests/test_tinygrad_gemma.py::test_local_sliding_decode_uses_head_first_single_token_kv_projection` failed before implementation.
   - GREEN: focused tests plus `tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py` passed (`75 passed, 1 skipped, 2 warnings`).
   - Score: `72.7641` vs parent `72.8749` (`-0.1108` tok/s, `-0.15%`).
   - Gates passed.
   - Decision: do not generalize the layout beyond the accepted sliding shared-source producer under the current short-floor metric.

## Main-branch support work

Profiler/test support landed on `main` in commit `6d4bb0f`:

- `scripts/profile_decode_jit.py` now preserves `*args`/`**kwargs` when wrapping `_project_kv` and forwards `head_first_single_token` in the profiler-only fused-int8 sidecar mirror.
- `tests/test_profile_decode_jit.py` pins the head-first single-token projection sidecar behavior.

This is instrumentation/test support only; the `exp_0091` runtime change remains in the evo branch/worktree.

## Decision

Keep `exp_0091` as the evo frontier. The head-first K/V layout win is narrow: it belongs to the layer-13 sliding shared-source producer shape, not to all full-attention or all sliding producers.

Do not spend more children merely widening this flag. The next credible non-boundary direction should either:

1. refine profiler attribution inside the remaining `kv_head_reshape`/`store` surfaces so we can distinguish view-root attribution from actual producer/store cost, or
2. test a different, narrowly justified runtime surface that changes the layer-13 shared-source producer/cache-write graph without broadening the accepted layout scope.

## Stop conditions applied

- No realization-boundary child was spawned.
- Both scope widenings preserved correctness gates and failed only on score, so they are negative performance evidence rather than correctness failures.
- The frontier remains `exp_0091`; repeated head-first scope broadening is now on the local rejection ledger.
