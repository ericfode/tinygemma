# exp0091 Layer-13 KV Structural Attribution

Date: 2026-04-27

## Purpose

Refine the accepted `exp_0091` frontier profile without changing runtime behavior. The goal is to make the remaining layer-13 `kv_head_reshape` / `store` attribution inspectable enough to choose the next non-boundary experiment.

## Scope

Runtime frontier remains `exp_0091`:

- Parent frontier: `exp_0082` at `66.3476` tok/s.
- Accepted child: `exp_0091` at `72.8749` tok/s.
- Rejected widenings: `exp_0092` and `exp_0093` both passed gates but regressed versus `exp_0091`.

This increment only changes profiler/test infrastructure on main.

## Changes

`scripts/profile_decode_jit.py` now reports per cache-write phase structural summaries:

- `cache_write_phase_structural_summary` in `original_capture` and per source-attribution row.
- `source_attributed_cache_write_phase_summary.by_phase_structure` in JSON output.
- The summary includes source count, program/root counts, category basis counts, display-name counts, op-signature counts, effect-kind counts, and `store_effect_count` / `store_effect_share`.

`tests/test_profile_decode_jit.py` pins that the structural phase summary survives both the original capture summary and the elapsed source-attributed phase rollup.

## Verification

Commands run from `/Users/ericfode/Downloads/tinygrad-gemma`:

```bash
.venv/bin/python -m pytest tests/test_profile_decode_jit.py -q
.venv/bin/tinygrad-gemma --help >/dev/null
PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/smoke_metal.py
git diff --check
```

Result: `40 passed, 2 warnings`; CLI help passed; METAL smoke reported `decode_fallback=False`; whitespace check passed.

## New frontier artifact

Generated with the accepted `exp_0091` worktree first on `PYTHONPATH`:

```bash
WORKTREE=/Users/ericfode/Downloads/tinygrad-gemma/.evo/run_0000/worktrees/exp_0091
PYTHONPATH="$WORKTREE:/Users/ericfode/src/.tinygrad_research" .venv/bin/python scripts/profile_decode_jit.py \
  --model-dir /Users/ericfode/Downloads/tinygrad-gemma/checkpoints/gemma-4-E2B-int8 \
  --device METAL \
  --context-length 512 \
  --jit-mode 1 \
  --phase-cutpoints \
  --phase-target layer13 \
  --out benchmarks/gemma4-metal-decode-graph-exp0091-head-first-kv-layer13-kv-structural-profile-512.json \
  --csv-out benchmarks/gemma4-metal-decode-graph-exp0091-head-first-kv-layer13-kv-structural-profile-512.csv
```

Artifact:

- `benchmarks/gemma4-metal-decode-graph-exp0091-head-first-kv-layer13-kv-structural-profile-512.json`
- `benchmarks/gemma4-metal-decode-graph-exp0091-head-first-kv-layer13-kv-structural-profile-512.csv`

Observed summary:

- `source_attribution.status=complete`
- `original_exec_count=954`
- `kernel_count=5`
- `elapsed_ms=14.400458429008722`
- layer-13 phase attribution:
  - `kv_fused_int8_matmul`: `2` sources / `0.020110 ms`
  - `kv_scale_cast`: `29` sources / `0.460615 ms`
  - `kv_head_reshape`: `377` sources / `6.043306 ms`
  - `rmsnorm_rope`: `18` sources / `0.276917 ms`
  - `rhs_pack`: `1` source / `0.010055 ms`
  - `store`: `228` sources / `3.636602 ms`

Structural observation: every reported phase source currently has `store_effect_share=1.0`, because the lowered source roots are dependency-closed sink/store-effect kernels. That is useful negative evidence: a naive `STORE`/sink split will not isolate literal store cost. The next refinement should separate structural motifs inside `kv_head_reshape` and `store`, not merely ask whether a lowered root is store-effectful.

## Next increment

Use the new `by_phase_structure` fields to split the dominant `kv_head_reshape` and `store` source families by display-name/op-signature motifs, then choose one minimal non-boundary runtime hypothesis only if the split identifies a credible producer shape that was not already rejected by `exp_0092`/`exp_0093`.
