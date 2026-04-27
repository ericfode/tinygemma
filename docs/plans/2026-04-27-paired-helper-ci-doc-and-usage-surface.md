# Paired Helper CI Documentation and Usage Surface

## Objective

Resolve `paired-helper-ci-doc-and-usage-surface-063`: make the paired decode benchmark helper discoverable from the main project documentation.

## Change

Updated `README.md` under the Gemma 4 Matrix Workflow with:

- the purpose of `scripts/paired_e2b_decode_benchmark.py`,
- baseline/candidate target usage,
- extra argument forwarding after `--`,
- default evo worktree benchmark resolution behavior,
- an example comparing `exp_0005` to a future candidate worktree,
- and the warning that very short paired runs are smoke tests, not throughput claims.

## Verification

- `.venv/bin/python -m pytest -q tests/test_paired_decode_benchmark.py`: `2 passed`
- `git diff --check`: passed

## Decision

Accepted as documentation-only. No runtime or benchmark semantics changed.
