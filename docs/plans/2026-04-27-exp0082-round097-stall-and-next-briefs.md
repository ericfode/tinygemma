# exp0082 Round 097 Stall Review and Next Briefs

Date: 2026-04-27 08:32 PDT

## Current state

- Evo frontier: `exp_0082`
- Short score: `66.3476` tok/s on the registered E2B int8 METAL `128/20` row
- Effective E2B cutpoints: `[0,1,2,3,4,6,8,9,11,12]` plus the retained final post-norm hidden-state `.contiguous().realize()`
- Effective gates: `metal_smoke`, `cli_help`, `e2b_int8_metal_hash16`
- `evo status`: `experiments=87`, `committed=25`, `evaluated=0`, `discarded=62`, `failed=0`, `active=0`, `best=66.3476`

## Round 097 worker results

All four children preserved semantics and failed by performance, so the result is clean negative evidence rather than infrastructure noise.

| Experiment | Parent | Probe | Score | Decision |
| --- | --- | --- | ---: | --- |
| `exp_0083` | `exp_0082` | add layer-10 local cut (`full_source_idxs[1] + 1`) | `65.7860` | discarded; small regression |
| `exp_0084` | `exp_0082` | add layer-13 adjacent output boundary on top of the dense frontier | `54.5064` | discarded; severe negative interaction |
| `exp_0085` | `exp_0082` | add layer-5 local cut (`full_source_idxs[0] + 1`) | `65.6087` | discarded; small regression |
| `exp_0086` | `exp_0082` | add layer-7 local-gap cut on top of the dense frontier | `52.8999` | discarded; severe negative interaction |

Common trace invariants reported by the review scans:

- `generated_tokens=128`
- `measured_decode_tokens=108`
- `rollout_jit_count=127`
- `decode_fallback=false`
- output hash `1c39dd289bb7363f0f600ae1087ad39926e9733447df72bf03955c92d06066b0`
- inherited gates passed

## Long confirmation

A manual long confirmation for unchanged `exp_0082` completed successfully:

- Artifact: `benchmarks/gemma4-metal-e2b-int8-1000-exp0082-long-confirm.json`
- Score: `40.841211` tok/s (`score=40.8412`)
- Row: E2B int8 METAL `1000/20`
- Started: `2026-04-27T15:18:50+00:00`
- Ended: `2026-04-27T15:20:46+00:00`

Caveat: this artifact records score/task timestamps rather than the fuller row metadata. Treat it as a completed long-score artifact, not a replacement for a full row-level invariant trace if a future acceptance hinges on the long floor.

## Interpretation

The simple additive layer-output cutpoint lattice around `exp_0082` is saturated or harmful:

- Layer 5 and layer 10 are near-frontier but negative.
- Layer 7 and layer 13 compose badly with the dense earliest-frontier cut set.
- The previous fact that layer 7 or layer 13 could help from weaker parents does not license adding them to `exp_0082`; the interaction is now measured.

The current `exp_0082` profile remains useful but does not justify repeating K/V internals:

- Profile artifact: `benchmarks/gemma4-metal-decode-graph-exp0082-earliest-frontier-512.json`
- Complete attribution: `950/950` source items, `5` MetalGraph batches, `15.071791713126004 ms`
- Dominant residual: `attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention`
- Residual size: `651` sources / `11.022088929398997 ms`
- Phase attribution: `kv_projection=644` sources / `10.894385540742658 ms`; `store=7` sources / `0.12770338865633907 ms`

The store path is not the problem. Prior K/V projection/cache-geometry/raw-runner attempts remain rejected unless new instrumentation changes the question.

## Next round briefs

Run one narrow worker per brief. Discard immediately on gate failure or short-score regression. Do not broaden a successful brief without another review.

### Brief 1: layer-13 intra-layer post-attention boundary

Parent: `exp_0082`

Surface: `GemmaDecoderLayer.__call__` around the attention residual, before the MLP path.

Concrete hypothesis: add exactly one decode-only `.contiguous().realize()` after the attention residual is formed for layer 13 and before `pre_feedforward_layernorm` / `mlp`.

Guarding constraints:

- Only for one-token bounded-cache decode: `seq_len == 1`, `cache is not None`, `cache.max_length is not None`, `not Tensor.training`.
- Only for `self.self_attn.layer_idx == 13`.
- Preserve `decode_realize_cut_idxs` exactly.
- Do not touch `_project_kv`, cache layout/update, raw Metal runners, layer-output cutpoints, final-hidden boundary, or layer 14.

Why: tests a new boundary class near the dominant layer-13 bucket without repeating K/V internals or the rejected layer-13 output boundary.

### Brief 2: layer-11 intra-layer post-attention boundary

Parent: `exp_0082`

Surface: same `GemmaDecoderLayer.__call__` attention-residual seam.

Concrete hypothesis: add one decode-only post-attention/pre-MLP boundary for layer 11 only.

Guarding constraints: same as Brief 1, but `layer_idx == 11`.

Why: layer 11 is an already accepted output-cut layer and still contributes a heavy local tranche in the profile. This avoids adding a rejected adjacent output cut while testing whether MLP/local source mass benefits from an intra-layer seam.

### Brief 3: layer-8 intra-layer post-attention boundary

Parent: `exp_0082`

Surface: same `GemmaDecoderLayer.__call__` attention-residual seam.

Concrete hypothesis: add one decode-only post-attention/pre-MLP boundary for layer 8 only.

Guarding constraints: same as Brief 1, but `layer_idx == 8`.

Why: layer 8 is another accepted heavy local tranche. This is a sibling diagnostic to layer 11, not a general multi-layer policy.

### Brief 4: partial side-branch composition from exp0079

Parent: `exp_0079`, not `exp_0082`

Concrete hypothesis: add only the layer-1 earliest boundary (`full_source_idxs[0] - 3`) to `exp_0079`'s layer-13-positive branch.

Expected candidate set: start from `exp_0079` `[2,3,4,6,8,9,11,12,13]` and add `1`, yielding `[1,2,3,4,6,8,9,11,12,13]`.

Do not add layer 0 in the same child. Do not recreate the rejected full `exp_0082 + layer13` set `[0,1,2,3,4,6,8,9,11,12,13]`.

Why: `exp_0079` proved layer 13 can help on a weaker dense parent, while `exp_0084` proved it composes badly with full `exp_0082`. This probes whether a partial early composition exists without importing the known-bad layer-0 interaction.

## No-go list after this review

Do not repeat, without new evidence:

- Additive `exp_0082` layer-output cuts at 5, 7, 10, or 13.
- Layer-14 after-cut.
- K/V projection fusion/caching/cache-layout/raw Metal runner variants.
- Final-hidden narrowing or moving the final-hidden boundary.
- Cutpoint removals/replacements as a blind next step.
- Cursor/frozenset/localized lookup/member-condition rewrites.

If the intra-layer and side-branch probes also regress, stop the current `model.py` layer-boundary search and pivot to measurement-first instrumentation or a separate profile/source-count evo run.
