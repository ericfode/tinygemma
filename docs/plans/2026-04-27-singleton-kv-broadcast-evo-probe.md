# Singleton KV Broadcast Attention Evo Probe

## Objective

Resolve `short-floor-graph-size-weighted-evo-child-090`: run one short-floor-first evo child from `exp_0005`, choosing a candidate with a graph/source-count reduction thesis rather than another generic micro-elision.

## User policy applied

- Optimize only the default short `128/20` floor until it appears maxed.
- Defer long `1000/20` floor evaluation.
- Weigh graph/source-count reductions more heavily when deciding whether a candidate is worth retaining or revisiting.

## Candidate

`exp_0013`: singleton-KV-head broadcast attention.

The probe changed `GemmaAttention.__call__` so `num_key_value_heads == 1` could use direct 4-D broadcast attention:

- avoid `q.reshape(batch, num_kv_heads, num_kv_groups, query_len, head_dim)`,
- avoid `k.unsqueeze(2)` and `v.unsqueeze(2)`,
- avoid `mask.unsqueeze(2)`,
- preserve the existing grouped path for multi-KV-head grouped attention.

A focused red test first proved the old path built the grouped GQA axis for singleton KV heads. The patch made that test pass.

## Verification inside experiment worktree

Worktree:
`/Users/ericfode/Downloads/tinygrad-gemma/.evo/run_0000/worktrees/exp_0013`

Focused test command:

```bash
/Users/ericfode/Downloads/tinygrad-gemma/.venv/bin/python -m pytest -q \
  tests/test_tinygrad_gemma.py::test_singleton_kv_attention_uses_broadcast_without_group_axis \
  tests/test_tinygrad_gemma.py::test_forward_matches_numpy_reference_for_gemma4 \
  tests/test_tinygrad_gemma.py::test_cache_matches_full_forward_for_gemma4 \
  tests/test_tinygrad_gemma.py::test_preallocated_cache_matches_full_forward_for_gemma4 \
  tests/test_tinygrad_gemma.py::test_rolled_shared_sliding_source_matches_full_forward_after_window
```

Result: `5 passed, 2 warnings`.

## Evo result

Command:

```bash
evo run exp_0013
```

Result:

- `exp_0013` score: `28.2658`
- Parent `exp_0005` score: `28.6153`
- Decision: reject; the short floor regressed.

## Graph-size evidence

Because graph-size reductions are now weighted more heavily, the probe was not discarded on score alone. A post-window profiler run measured whether it actually reduced captured source pressure:

```bash
PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. \
  /Users/ericfode/Downloads/tinygrad-gemma/.venv/bin/python scripts/profile_decode_jit.py \
  --model-dir /Users/ericfode/Downloads/tinygrad-gemma/checkpoints/gemma-4-E2B-int8 \
  --device METAL \
  --context-length 512 \
  --jit-mode 1 \
  --out /Users/ericfode/Downloads/tinygrad-gemma/benchmarks/gemma4-metal-decode-graph-singleton-kv-broadcast-exp0013-512.json \
  --csv-out /Users/ericfode/Downloads/tinygrad-gemma/benchmarks/gemma4-metal-decode-graph-singleton-kv-broadcast-exp0013-512.csv
```

Observed:

- `source_count`: `2994`
- `graph_batch_count`: `7`
- `elapsed_ms`: `43.54`

The probe did not reduce source count from the accepted packed-cache family. It merely changed local expression shape while tinygrad preserved the same captured source count. A pleasing branch, but no theorem closed.

## Disposition

`exp_0013` was discarded:

```bash
evo discard exp_0013 --reason "singleton-KV broadcast attention regressed short score 28.2658 vs exp_0005 28.6153 and preserved source_count=2994"
```

Frontier remains `exp_0005` at `28.6153`.

## Next target

Continue short-floor-first. Prefer candidates that can plausibly reduce the dominant captured source mass before running the default benchmark. Avoid retrying singleton-KV broadcast unless a later tinygrad lowering change makes broadcasting materially reduce source count.
