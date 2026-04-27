# Paired Helper README Command Validation

## Objective

Resolve `paired-helper-baseline-candidate-readme-command-validation-069`: verify that the README paired-helper command shape is honest against the current evo worktrees.

## Inspection

- `.evo/run_0000/worktrees/exp_0005` exists and is the current best baseline.
- `.evo/run_0000/worktrees/exp_0013` does not exist in this workspace.

## Change

Updated the README text before the paired-helper example to say: replace `exp_0013` with the candidate experiment worktree you want to compare.

The command remains an example for a future candidate rather than claiming `exp_0013` exists today.

## Verification

- `.venv/bin/python -m pytest -q tests/test_paired_decode_benchmark.py`: `4 passed`
- `git diff --check`: passed

## Decision

Accepted as documentation honesty. No runtime or helper semantics changed.
