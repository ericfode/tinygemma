# Decode-Only Attention Output View Elision Evo Probe

## Objective

Resolve `decode-only-attention-output-view-elision-evo-probe-059`: test an orthogonal model.py runtime surface outside the exhausted K/V projection line.

## Hypothesis

For one-token decode, the query-length axis is singleton. Therefore the final attention-output reorder:

- ungrouped path: `(probs @ v).transpose(1, 2).reshape(batch, query_len, -1)`
- grouped path: `(probs @ grouped_v).permute(0, 3, 1, 2, 4).reshape(batch, query_len, -1)`

might be equivalent to a direct reshape when `query_len == 1`, saving a view/rewrite source edge without changing semantics.

## Evo Experiment

- Parent: `exp_0005`
- Child: `exp_0012`
- Hypothesis: `probe: decode-only attention output view elision`
- Commit in child worktree: `048b6fb Probe decode-only attention output view elision`

## Pre-Evo Verification

Inside the child worktree:

- `tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py`: `73 passed, 1 skipped, 2 warnings`
- research METAL smoke: `rollout_jit_count=3`, `decode_fallback=False`
- hash16 real-checkpoint gate from main root: score `30.415459 tok/s`, expected hash matched

## Evo Result

`evo run exp_0012`:

- default score: `28.4939 tok/s`
- parent score: `28.6153 tok/s`
- gate failure: `e2b_int8_metal_hash1000_current_floor`

Discarded with reason:

> decode-only attention output view elision regressed default score to 28.4939 versus exp_0005 28.6153 and failed e2b_int8_metal_hash1000_current_floor; do not skip transpose/permute before decode attention output reshape under this benchmark.

## Decision

Reject. The semantic equivalence is not a performance improvement in tinygrad/MetalGraph under this workload. Do not retry this view-elision shape without new profile evidence.
