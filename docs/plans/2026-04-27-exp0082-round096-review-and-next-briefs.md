# exp_0082 Round 096 Review and Next Briefs

## Objective

Cross-cut the worker round launched from `exp_0078`, source-confirm the new `exp_0082` frontier, and prepare the next narrow optimization briefs.

## Current frontier

- Previous frontier: `exp_0078` at `62.4154`
- New frontier: `exp_0082` at `66.3476`
- Winning path addition:
  - `exp_0080`: add earliest local decode cut before the accepted leading guard, E2B layer `1`, score `64.5288`
  - `exp_0082`: add layer `0` decode cut before that earliest local boundary, score `66.3476`
- Effective E2B cutpoint set: `[0, 1, 2, 3, 4, 6, 8, 9, 11, 12]`
- Final-hidden post-norm `.contiguous().realize()` boundary remains unchanged.

## Round worker results

All inspected workers passed inherited gates and preserved the short decode hash gate.

| Experiment | Parent | Change | Score | Delta vs parent | Status |
|---|---:|---|---:|---:|---|
| `exp_0079` | `exp_0078` | add layer-13 adjacent shared-source sliding output boundary | `63.9473` | `+1.5319` | committed |
| `exp_0080` | `exp_0078` | add layer-1 earliest local decode cut | `64.5288` | `+2.1134` | committed |
| `exp_0081` | `exp_0078` | add layer-7 second-tranche local-gap cut | `63.7228` | `+1.3074` | committed |
| `exp_0082` | `exp_0080` | add layer-0 earliest local decode cut | `66.3476` | `+1.8188` | committed |

Read-only scan findings:

- Three sibling `exp_0078` decode cut/boundary probes committed and are positive combination candidates: `exp_0079`, `exp_0080`, `exp_0081`.
- The strongest observed improver is the `exp_0080` branch extended by the layer-0 decode cut: `exp_0082`.
- All inspected probes preserved protected decode invariants: passed task status, full 128-token generation, no fallback, and stable output hash.

Structural aggregation:

- `gate_failures=[]` for `exp_0079`, `exp_0080`, `exp_0081`, and `exp_0082`.
- No evaluated or failed nodes remain after the worker round.
- No pairwise failure intersections exist this round; all candidate nodes committed.
- Improver set is the entire worker batch: `{exp_0079, exp_0080, exp_0081, exp_0082}`.

## exp_0082 source profile

Command run from repo root:

```bash
WORKTREE=/Users/ericfode/Downloads/tinygrad-gemma/.evo/run_0000/worktrees/exp_0082
PYTHONPATH=/Users/ericfode/src/.tinygrad_research:$WORKTREE \
  /Users/ericfode/Downloads/tinygrad-gemma/.venv/bin/python \
  "$WORKTREE/scripts/profile_decode_jit.py" \
  --model-dir /Users/ericfode/Downloads/tinygrad-gemma/checkpoints/gemma-4-E2B-int8 \
  --device METAL \
  --context-length 512 \
  --jit-mode 1 \
  --out /Users/ericfode/Downloads/tinygrad-gemma/benchmarks/gemma4-metal-decode-graph-exp0082-earliest-frontier-512.json \
  --csv-out /Users/ericfode/Downloads/tinygrad-gemma/benchmarks/gemma4-metal-decode-graph-exp0082-earliest-frontier-512.csv
```

Profile result:

- `source_attribution.status=complete`
- `original_exec_count=950`
- `attributed_source_count=950`
- `source_count_mismatches=0`
- `unattributed_tail_count=0`
- `unparsed_graph_batches=0`
- post-graph execution: `5` MetalGraph batches
- profiled elapsed: `15.071791713126004 ms`

Comparison to `exp_0078` profile:

| Profile | Sources | Graph batches | Profile elapsed | layer-13 shared-source parent |
|---|---:|---:|---:|---:|
| `exp_0078` | `1012` | `6` | `15.7192915212363 ms` | `646` sources / `11.067749330777588 ms` |
| `exp_0082` | `950` | `5` | `15.071791713126004 ms` | `651` sources / `11.022088929398997 ms` |

The earliest-boundary branch improved the global graph/source shape, but it did not materially remove the layer-13 shared-source sliding K/V projection bucket. That bucket remains the dominant residual surface:

- `attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention`: `651` source items, `11.022088929398997 ms`
- Phase summary:
  - `kv_projection`: `644` source items, `10.894385540742658 ms`
  - `store`: `7` source items, `0.12770338865633907 ms`

Graph rows for `exp_0082`:

| Ordinal | Sources | Elapsed ms | Main rollup |
|---:|---:|---:|---|
| 0 | 32 | `0.3321249969303608` | local `7`, shared-source `20`, MLP `5` |
| 1 | 64 | `0.556792...` | local `14`, shared-source `40`, MLP `10` |
| 2 | 128 | `1.384083...` | local `41`, shared-source `71`, MLP `16` |
| 3 | 256 | `2.862...` | local `80`, shared-source `148`, MLP `28` |
| 4 | 470 | `9.936792...` | shared-source `402`, MLP `68` |

## Next optimization briefs

### Brief 1: compose layer-13 adjacent boundary with the new earliest frontier

Objective: `exp_0079` proved the layer-13 adjacent output boundary is positive on `exp_0078`, and the `exp_0082` profile still shows layer-13 shared-source sliding K/V projection as the dominant residual bucket, so test that same single boundary on the new earliest-boundary frontier.

Parent node: `exp_0082`

Boundaries / anti-patterns:

- Preserve the current cutpoint set `[0, 1, 2, 3, 4, 6, 8, 9, 11, 12]` and final-hidden post-norm boundary.
- Add only the layer-13 adjacent shared-source sliding output boundary first.
- Do not add the rejected layer-14 final after-cut from `exp_0075`.
- Do not touch `_project_kv`, K/V fusion/caching, raw Metal runners, cache geometry, final-hidden narrowing, or bookkeeping representations.
- If this regresses on `exp_0082`, annotate, discard, and stop; do not decorate the branch.

Pointer traces:

- `exp_0079`: same boundary was positive on `exp_0078`, scoring `63.9473`.
- `exp_0082`: current best with earliest cuts, scoring `66.3476`.
- `benchmarks/gemma4-metal-decode-graph-exp0082-earliest-frontier-512.json`: layer-13 shared-source parent remains `651` sources / `11.022088929398997 ms`.

### Brief 2: compose layer-7 local-gap boundary with the new earliest frontier

Objective: `exp_0081` proved the layer-7 second-tranche local-gap boundary is positive on `exp_0078`, so test whether it composes with the stronger earliest-boundary frontier.

Parent node: `exp_0082`

Boundaries / anti-patterns:

- Preserve the current cutpoint set and final-hidden boundary.
- Add only the known-positive layer-7 local-gap boundary first.
- Do not add the layer-13 boundary in this branch; that is Brief 1.
- Do not add layer-14, move existing full-source brackets, remove cutpoints, or alter attention/cache internals.
- If the composition regresses, annotate, discard, and stop.

Pointer traces:

- `exp_0081`: layer-7 local-gap boundary scored `63.7228` on `exp_0078`.
- `exp_0082`: current best earliest-boundary parent.
- `exp_0082` profile row ordinal `3`: still has `256` sources with local/shared-source mixture.

### Brief 3: probe remaining layer-5 first-post-full local gap

Objective: The accepted earliest frontier now realizes layers `0..4` and layer `6`, leaving layer `5` as the first remaining local gap after the first full-attention source; test one strictly additive boundary there.

Parent node: `exp_0082`

Boundaries / anti-patterns:

- Preserve all current cutpoints and final-hidden boundary.
- Add only the layer-5 local boundary first.
- Do not move the layer-4 full-source boundary or the layer-6 accepted boundary.
- Do not combine with layer-7 or layer-13 in the first experiment.
- Do not change attention projection, cache layout, raw Metal runner code, or guard representation.

Pointer traces:

- `exp_0082`: current cutpoints skip layer `5` while retaining `4` and `6`.
- `exp_0082` profile row ordinal `2`: `128` sources with local work still mixed into the shared-source bucket.
- `exp_0063`: replacing earlier cutpoints regressed; this brief is additive only.

### Brief 4: probe remaining layer-10 second-post-full local gap

Objective: The second tranche still skips layer `10` between full-attention layer `9` and accepted later local cuts, so test whether one strictly additive boundary there reduces residual middle-row graph pressure.

Parent node: `exp_0082`

Boundaries / anti-patterns:

- Preserve all current cutpoints and final-hidden boundary.
- Add only the layer-10 local boundary first.
- Do not combine with layer-7 or layer-13 in the first experiment.
- Do not add layer-14 or move the layer-9 full-source boundary.
- Avoid cursor/frozenset/localized lookup rewrites, final-hidden narrowing, K/V internals, cache geometry, and raw Metal runners.

Pointer traces:

- `exp_0082`: current cutpoints skip layer `10` while retaining layers `9`, `11`, and `12`.
- `exp_0081`: nearby layer-7 local-gap probe was positive from `exp_0078`.
- `exp_0082` profile row ordinal `3`: `256` sources with local/shared-source mixture remains.

## Verification

- Read `.evo/project.md`, `configs/repo-loop-state.json`, `state/evolution-log.md`, and the previous `exp_0078` plan.
- Ran `evo status`, `evo frontier`, `evo path`, and `evo gate list` for the new frontier.
- Spawned the mandatory read-only scan subagent for `exp_0079..exp_0082`.
- Parsed all round-096 `outcome.json` files structurally.
- Source-confirmed E2B cutpoint sets by direct model instantiation in each experiment worktree.
- Profiled `exp_0082` with `profile_decode_jit.py` and parsed the JSON artifact.
