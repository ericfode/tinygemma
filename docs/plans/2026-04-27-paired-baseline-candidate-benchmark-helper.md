# Paired Baseline/Candidate Decode Benchmark Helper

## Objective

Resolve `paired-baseline-candidate-benchmark-helper-061`: add an infrastructure helper that runs the same decode benchmark against a baseline target and a candidate target in one session, then records the delta and provenance.

## Change

Added `scripts/paired_e2b_decode_benchmark.py`.

It accepts:

- `--benchmark-script` (defaults to `benchmarks/evo_e2b_int8_metal_decode.py`)
- `--baseline-target`
- `--candidate-target`
- `--out`
- `--label`
- extra benchmark arguments after `--`

It runs baseline first, candidate second, parses each benchmark's JSON stdout, and writes a single JSON payload containing:

- commands used,
- stdout/stderr per row,
- numeric baseline and candidate scores,
- absolute delta,
- relative delta,
- and `candidate_improved`.

## Tests

Added `tests/test_paired_decode_benchmark.py`, which uses a fake benchmark script to verify:

- paired execution,
- extra argument forwarding,
- JSON output to stdout and file,
- score delta/relative delta computation,
- and captured command/stderr provenance.

## Verification

- `.venv/bin/python -m pytest -q`: `85 passed, 2 warnings`
- `.venv/bin/tinygrad-gemma --help`: passed
- `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/smoke_metal.py`: METAL smoke passed, `rollout_jit_count=3`, `decode_fallback=False`
- `.venv/bin/python -m py_compile scripts/paired_e2b_decode_benchmark.py`: passed
- `.venv/bin/python scripts/paired_e2b_decode_benchmark.py --help`: passed
- `git diff --check`: passed

## Decision

This is benchmark infrastructure only. It does not change the runtime or claim a new throughput score.
