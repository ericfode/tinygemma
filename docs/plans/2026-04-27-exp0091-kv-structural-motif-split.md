# exp0091 KV Structural Motif Split

Date: 2026-04-27

## Purpose

Turn the regenerated exp0091 structural phase profile into a reusable motif split before launching another runtime child. This is a measurement/planning increment only: it changes no model runtime behavior.

## Inputs

- Frontier: `exp_0091`
- Frontier score: `72.8749 tok/s`
- Source profile: `benchmarks/gemma4-metal-decode-graph-exp0091-head-first-kv-layer13-kv-structural-profile-512.json`
- Motif artifact: `benchmarks/gemma4-metal-decode-graph-exp0091-head-first-kv-layer13-kv-structural-motifs-512.json`
- Helper: `scripts/analyze_profile_phase_structure.py`

## Selected hot phases

The helper selected `kv_head_reshape` and `store` from the structural phase summary.

Combined selected phase count:

- `phase_count`: 2
- `total_selected_sources`: 605
- `total_selected_store_effects`: 605
- `all_selected_sources_store_effect_roots`: true

This matters: these are dependency-closed store-effect roots, not isolated literal memcpy/store kernels. A broad cache-store rewrite would be a poor next move unless it also changes the RHS graph shape. Beautifully named buckets can still lie by implication.

## `kv_head_reshape`

- `source_count`: 377
- `elapsed_ms`: 6.043305669447003
- `store_effect_note`: `all_sources_are_store_effect_roots`
- dominant display motif: `r_16_96` at 176 / 377 sources, share 0.46684350132625996
- dominant op signature: `Ops.CONST:4,Ops.INDEX:2,Ops.MUL:2,Ops.PARAM:2,Ops.ADD:1,Ops.RANGE:1,Ops.RECIPROCAL:1,Ops.REDUCE:1` at the same 176 / 377 share
- category basis: `repo_sidecar_uop_creation_metadata`: 377

Top display motifs:

1. `r_16_96`: 176 (46.684%)
2. `E_16_2_16_4`: 28 (7.427%)
3. `r_2_32_4_2_4_64`: 28 (7.427%)
4. `r_8_16_16`: 28 (7.427%)
5. `r_8_16_32n1`: 28 (7.427%)
6. `E_1120_4_8_16_4`: 11 (2.918%)
7. `r_35_16_16`: 10 (2.653%)
8. `r_70_32_4_384_4`: 10 (2.653%)

Interpretation:

`kv_head_reshape` remains structurally large after exp0091, but the dominant family is reduce-like `r_16_96` metadata, not merely a post-projection transpose node. The previous accepted head-first specialization already removed the narrow reshape/transpose shape that was safe. The remaining family should be treated as a producer/source-count problem inside the dependency-closed cache-write root, not a reason to broaden head-first layout again.

## `store`

- `source_count`: 228
- `elapsed_ms`: 3.6366022332387673
- `store_effect_note`: `all_sources_are_store_effect_roots`
- dominant display motif: `E_16_32_3` at 70 / 228 sources, share 0.30701754385964913
- dominant op signature: `Ops.INDEX:5,Ops.PARAM:5,Ops.CONST:2,Ops.MUL:2,Ops.ADD:1,Ops.CAST:1,Ops.END:1,Ops.RANGE:1` at the same 70 / 228 share
- category basis: `repo_sidecar_uop_creation_metadata`: 227, `repo_sidecar_realize_scope_metadata`: 1

Top display motifs:

1. `E_16_32_3`: 70 (30.702%)
2. `r_4_32_4_2_4_32`: 28 (12.281%)
3. `r_8_16_32`: 28 (12.281%)
4. `E_16_32_3n2`: 25 (10.965%)
5. `r_256_32_3_384_4`: 20 (8.772%)
6. `r_128_32_3_384_4`: 15 (6.579%)
7. `E_16_32_3n1`: 10 (4.386%)
8. `r_32_32_4_384_4`: 7 (3.070%)

Interpretation:

The store phase is dominated by index/cast expansion motifs and smaller reduce families. Since all selected sources are store-effect roots, a bare `.assign()` rearrangement is likely to be dependency-closed over the same RHS unless it changes the producer shape feeding the assignment. Earlier split/realize and broader cache-layout variants have already supplied enough negative evidence to reject decorative store-only churn.

## Next runtime candidate

The non-rejected next child should be narrower than exp0092/exp0093 and should not be another realization-boundary probe. Candidate class:

- parent: `exp_0091`
- candidate: minimal layer-13 sliding shared-source producer source-count reduction targeting the dominant `r_16_96`/index-cast motif family
- required RED coverage: prove the exact producer/cache-write graph shape being changed, and prove the accepted exp0091 scope does not broaden to full-attention shared producers or all sliding producers
- stop condition: if the focused source-count/profile smoke does not reduce the selected motif family, do not spend a full throughput run; if it does, run `evo run exp_0094` with inherited hash/JIT/fallback gates

## Verification

Executed:

```bash
.venv/bin/python -m pytest -q
.venv/bin/tinygrad-gemma --help
PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/smoke_metal.py
.venv/bin/python -m json.tool configs/repo-loop-state.json
.venv/bin/python -m json.tool benchmarks/gemma4-metal-decode-graph-exp0091-head-first-kv-layer13-kv-structural-motifs-512.json
git diff --check
```

Result: `104 passed, 2 warnings`; METAL smoke reported `default_device=METAL`, `loaded_model_device=METAL`, `logits_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`.
