# Paired Helper Smoke on Evo Frontier

## Objective

Resolve `paired-helper-smoke-on-evo-frontier-062`: prove the paired helper works against the current evo frontier, not just a fake unit benchmark.

## Result

The first real invocation exposed a useful defect:

- default `--benchmark-script benchmarks/evo_e2b_int8_metal_decode.py` did not exist on main,
- because the benchmark harness lives in evo worktrees,
- so the helper failed unless `--benchmark-script .evo/run_0000/worktrees/exp_0005/benchmarks/evo_e2b_int8_metal_decode.py` was passed explicitly.

Fix landed in `scripts/paired_e2b_decode_benchmark.py`:

- if the default benchmark path is missing in the current working directory,
- walk upward from `--baseline-target`,
- and use `<baseline-worktree>/benchmarks/evo_e2b_int8_metal_decode.py` when present.

Added test coverage in `tests/test_paired_decode_benchmark.py` for this worktree-default resolution.

## Real smoke artifacts

- Explicit script self-pair: `benchmarks/paired-exp0005-self-hash16.json`
  - baseline: `30.4576`
  - candidate: `30.4196`
  - delta: `-0.0380`
  - same target; small same-session noise.
- Default-resolution self-pair: `benchmarks/paired-exp0005-self-default-hash4.json`
  - baseline: `0.2349`
  - candidate: `0.2359`
  - delta: `0.0010`
  - confirms default worktree benchmark resolution.

The hash4 micro-run score is not meaningful for throughput; it exists only as a fast helper smoke.

## Verification

- `.venv/bin/python -m pytest -q`: `86 passed, 2 warnings`
- `.venv/bin/tinygrad-gemma --help`: passed
- research METAL smoke: `rollout_jit_count=3`, `decode_fallback=False`
- `.venv/bin/python -m py_compile scripts/paired_e2b_decode_benchmark.py`: passed
- `git diff --check`: passed

## Decision

Accepted as benchmark infrastructure. The helper is now usable from main against evo worktree targets without manually passing the benchmark script path.
