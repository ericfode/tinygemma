# Lane-before-head-dim Packed Cache Evo Probe

## Objective

Resolve `short-floor-dominant-source-mass-candidate-091`: run one short-floor-first evo child from `exp_0005`, selecting a candidate that could plausibly reduce packed-cache source mass or MetalGraph pressure before spending any long-floor budget.

## Candidate

Probe packed K/V cache backing layout `(B,H,L,2,D)` instead of the accepted final-lane layout `(B,H,L,D,2)`.

Rationale:

- Prior packed-cache work produced real gains, but slot geometry still had one documented untested variant.
- The candidate is narrow and localized to cache backing layout plus packed assignment shape.
- It can be judged by both the short evo score and post-window graph/source evidence.

## Experiment

- Parent: `exp_0005`
- Child: `exp_0014`
- Hypothesis: `probe: lane-before-head-dim packed cache layout`
- Worktree: `.evo/run_0000/worktrees/exp_0014`

Implementation inside the experiment worktree:

- `make_packed_cache_entry(...)` allocated packed backing storage as `(batch, heads, length, 2, head_dim)`.
- Exposed key/value views remained `(batch, heads, length, head_dim)` via lane index `0` and `1`.
- `realize_cache_update(...)` packed K/V with `unsqueeze(-2).cat(..., dim=-2)` to match the new lane axis.
- `scripts/profile_decode_jit.py` received the same packed-assignment shape so graph attribution measured the candidate geometry rather than the old profiler geometry.
- Tests updated/added to pin the backing layout while preserving full-forward/cache behavior.

## Verification inside exp_0014

Focused RED/GREEN and local profile geometry tests:

```bash
/Users/ericfode/Downloads/tinygrad-gemma/.venv/bin/python -m pytest -q \
  tests/test_profile_decode_jit.py \
  tests/test_tinygrad_gemma.py::test_packed_cache_uses_lane_before_head_dim_backing_layout \
  tests/test_tinygrad_gemma.py::test_preallocated_cache_matches_full_forward_for_gemma4 \
  tests/test_tinygrad_gemma.py::test_roll_sliding_cache_entry_aligns_absolute_positions_by_modulo_slot \
  tests/test_tinygrad_gemma.py::test_rolled_sliding_cache_decode_matches_full_forward_for_gemma4
```

Result: `36 passed, 2 warnings`.

The experiment commit was:

```text
f4be8f3 Probe lane-before-head-dim packed cache layout
```

## Evo result

```bash
evo run exp_0014
```

Result:

```text
EVALUATED exp_0014 score=27.2326 score_regressed (parent=28.6153)
```

Gate failures: none.

The default short score regressed by `-1.3827 tok/s` versus the current frontier parent `exp_0005`.

## Graph profile

Command:

```bash
PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. \
  /Users/ericfode/Downloads/tinygrad-gemma/.venv/bin/python \
  scripts/profile_decode_jit.py \
  --model-dir /Users/ericfode/Downloads/tinygrad-gemma/checkpoints/gemma-4-E2B-int8 \
  --device METAL \
  --context-length 512 \
  --jit-mode 1 \
  --out /Users/ericfode/Downloads/tinygrad-gemma/benchmarks/gemma4-metal-decode-graph-lane-before-head-dim-exp0014-512.json \
  --csv-out /Users/ericfode/Downloads/tinygrad-gemma/benchmarks/gemma4-metal-decode-graph-lane-before-head-dim-exp0014-512.csv
```

Result:

- `source_attribution.status=complete`
- `original_capture.exec_count=2994`
- `original_capture.graph_batch_count=0`
- `post_graph_execution.graph_batch_count=7`
- `elapsed_ms=53.861`
- `kernel_count=7`

## Decision

Reject.

The candidate was worse on the active short-floor metric and did not reduce captured graph/source mass. Because source count and graph batch count remained `2994` / `7`, the regression cannot be justified as a graph-size tradeoff. It is simply slower in a different backing shape.

Executed discard:

```bash
evo discard exp_0014 --reason "lane-before-head-dim packed cache layout regressed short score 27.2326 vs exp_0005 28.6153 and preserved source_count=2994/7 graph batches with slower 53.861ms profile"
```

`evo frontier` remains `exp_0005` at `28.6153`.

## Future note

This closes the remaining documented packed-cache slot-layout sweep item. Do not retry final-lane vs lane-before-head-dim vs flat-slot layout variants under the current benchmark unless a new profiler epoch shows a different source-mass mechanism.
