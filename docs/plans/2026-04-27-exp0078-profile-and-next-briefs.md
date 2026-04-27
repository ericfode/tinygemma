# exp_0078 Dense Frontier Profile and Next Briefs

## Objective

Complete the post-round-094 source confirmation for the current evo frontier and write the next narrow optimization briefs.

## Current frontier

- Frontier: `exp_0078`
- Short score: `62.4154`
- Long `1000/20` confirmation: `38.957688 tok/s`
- Worktree: `.evo/run_0000/worktrees/exp_0078`
- Winning source diff versus parent `exp_0076`: remove the redundant `and self.decode_realize_cut_idxs` guard before set membership.

## Source confirmation

Command run from repo root:

```bash
WORKTREE=/Users/ericfode/Downloads/tinygrad-gemma/.evo/run_0000/worktrees/exp_0078
PYTHONPATH=/Users/ericfode/src/.tinygrad_research:$WORKTREE \
  /Users/ericfode/Downloads/tinygrad-gemma/.venv/bin/python \
  "$WORKTREE/scripts/profile_decode_jit.py" \
  --model-dir /Users/ericfode/Downloads/tinygrad-gemma/checkpoints/gemma-4-E2B-int8 \
  --device METAL \
  --context-length 512 \
  --jit-mode 1 \
  --out /Users/ericfode/Downloads/tinygrad-gemma/benchmarks/gemma4-metal-decode-graph-exp0078-dense-frontier-512.json \
  --csv-out /Users/ericfode/Downloads/tinygrad-gemma/benchmarks/gemma4-metal-decode-graph-exp0078-dense-frontier-512.csv
```

Profile result:

- `source_attribution.status=complete`
- `original_exec_count=1012`
- `attributed_source_count=1012`
- `source_count_mismatches=0`
- `unattributed_tail_count=0`
- `unparsed_graph_batches=0`
- post-graph execution: `6` MetalGraph batches
- profiled elapsed: `15.7192915212363 ms`

Rollup:

- `attention_packed_cache_write_shared_source`: `676` source items, apportioned `11.498657529500633 ms`
- `attention_packed_cache_write_local`: `213` source items, apportioned `2.1694620359085093 ms`
- `mlp`: `123` source items, apportioned `2.051171955827158 ms`

Dominant exact bucket:

- `attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention`: `646` source items, apportioned `11.067749330777588 ms`
- Phase summary within that bucket:
  - `kv_projection`: `639` source items, `10.974117984005716 ms`
  - `store`: `7` source items, `0.0936313467718719 ms`

Correction: direct instantiation of `GemmaModel` from the E2B config reports the actual current dense decode cutpoint set as `[2, 3, 4, 6, 8, 9, 11, 12]`, not the previously summarized `{3,4,6,8,9,11,12}`. The omitted layer-2 cut is the `full_source_idxs[0] - 2` leading guard from `exp_0076`.

## Cross-cutting scan findings

Read-only scan subagents and structural aggregation found:

- Additive cutpoint lattice changes remain the only strong positive pattern: `exp_0060`, `exp_0065`, `exp_0068`, `exp_0072`, `exp_0076`, `exp_0078`.
- Final-hidden narrowing remains a wall: `exp_0064`, `exp_0067`, `exp_0070`.
- Cutpoint bookkeeping rewrites are mixed-to-bad: `exp_0078` helped by deleting a redundant condition, but `exp_0061`, `exp_0071`, and `exp_0077` regressed.
- Late/final full-source after-cuts are suspect: `exp_0075` regressed under the dense frontier.
- No gate failures were present in parsed round-094 outcomes; regressions were performance regressions, not correctness failures.

## Next optimization briefs

### Brief 1: earliest-region residual graph pressure

Objective: The accepted leading guard was a repeat improver and the profile still shows early graph batches before the dense first full-source cluster, so test whether a strictly additive earliest-region boundary reduces residual short-floor graph pressure without moving existing boundaries.

Parent node: `exp_0078`

Boundaries / anti-patterns:

- Preserve the existing cutpoint set `[2, 3, 4, 6, 8, 9, 11, 12]` and the final post-norm hidden-state `.contiguous().realize()` boundary.
- Do not remove or replace the accepted layer-2 guard.
- Do not repeat the rejected earliest-cut replacement from `exp_0063`.
- Do not use cursor, frozenset, localized-lookup, or other bookkeeping rewrites.
- First experiment should be one additive early boundary only; if it regresses, discard and stop rather than decorating the branch.

Pointer traces:

- `exp_0076`: leading guard committed on the dense frontier.
- `exp_0078`: current best and guard simplification parent.
- Profile artifact `benchmarks/gemma4-metal-decode-graph-exp0078-dense-frontier-512.json`, source rows ordinal `0` and `1`: early batches still exist before the large layer-13 bucket.

### Brief 2: second-tranche local gap

Objective: The second full-source bracket was additive, and the profile still mixes local layer work with the shared-source bucket in the middle graph rows, so test one strictly additive boundary in the remaining second-tranche local gap.

Parent node: `exp_0078`

Boundaries / anti-patterns:

- Preserve the current dense cutpoint set and final-hidden boundary.
- Do not add the final pre-shared full-source after-cut rejected by `exp_0075`.
- Do not move existing full-source brackets or remove cutpoints.
- Do not modify attention projection, cache geometry, or raw Metal runner code.
- Run the smallest single-boundary probe first; abandon on short-score regression.

Pointer traces:

- `exp_0068`: second full-source bracketing committed.
- `exp_0075`: final after-cut regressed and defines the late-boundary wall.
- Profile artifact row ordinal `3`: `256` sources mixing local layers 6-11 with shared-source layer 13.

### Brief 3: layer-13 adjacent output boundary, not K/V internals

Objective: The source-confirmed frontier is still dominated by layer-13 shared-source sliding K/V projection, so test whether a single adjacent layer-output boundary helps isolate that producer without touching rejected K/V projection internals.

Parent node: `exp_0078`

Boundaries / anti-patterns:

- Do not change `_project_kv`, K/V projection fusion, K/V weight caching, scale caching, raw Metal runners, or cache storage geometry.
- Do not pre-realize K/V tensors inside the producer; `exp_0032` and earlier producer materialization evidence are negative.
- Do not add the layer-14 final after-cut from `exp_0075`.
- This is one layer-output boundary probe only; preserve all existing dense cutpoints and final-hidden boundary.

Pointer traces:

- Profile artifact `source_attributed_cache_write_phase_summary`: layer-13 shared-source sliding attention has `646` source items, `11.067749330777588 ms`; `kv_projection` accounts for `639` source items and `10.974117984005716 ms`.
- `exp_0036`: early negative evidence for adding a full-source cut after the sliding source under a much weaker parent.
- `exp_0075`: late adjacent after-cut regressed under a dense parent, so avoid broad late-boundary changes.

### Brief 4: semantics-identical guard cleanup only

Objective: `exp_0078` proved that deleting one redundant condition can improve the dense frontier, but broader bookkeeping rewrites are mostly harmful, so test at most one semantics-identical guard simplification around the accepted per-layer realization predicate.

Parent node: `exp_0078`

Boundaries / anti-patterns:

- Must prove the instantiated E2B cutpoint set remains exactly `[2, 3, 4, 6, 8, 9, 11, 12]`.
- Preserve the final-hidden boundary exactly as `seq_len == 1 and cache is not None and cache.max_length is not None and not Tensor.training` followed by `.contiguous().realize()`.
- Do not introduce cursor state, frozenset storage, boolean masks, localized set variables, or rollout flags; these have negative evidence in `exp_0061`, `exp_0071`, `exp_0047`, `exp_0077`, and `exp_0070`.
- If there is no obviously redundant predicate left, do not force an edit.

Pointer traces:

- `exp_0078`: redundant truthiness deletion committed.
- `exp_0071` and `exp_0077`: nearby bookkeeping rewrites regressed.
- `exp_0061`: sequential cursor rewrite collapsed throughput under the final-hidden boundary.

## Verification

- Read `.evo/project.md`, `configs/repo-loop-state.json`, and `state/evolution-log.md` before profiling.
- Ran `evo status`, `evo frontier`, `evo path exp_0078`, `evo gate list exp_0078`, and `evo scratchpad`.
- Ran two read-only scan subagents over `exp_0060..exp_0078`.
- Parsed round-094 `outcome.json` files structurally and emitted pattern intersections.
- Ran `profile_decode_jit.py` on the `exp_0078` worktree and parsed the resulting JSON artifact.
