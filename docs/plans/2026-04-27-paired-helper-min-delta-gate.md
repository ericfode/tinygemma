# Paired Helper Minimum Delta Gate

## Objective

Resolve `paired-helper-min-delta-gate-065`: make the paired decode helper usable as a CI/evo-side gate by failing when a candidate delta is below a required floor while still writing the comparison artifact.

## Change

Updated `scripts/paired_e2b_decode_benchmark.py`:

- Added `--min-delta <float>`.
- The helper still writes JSON to stdout and `--out` before gate evaluation.
- Payload now includes `min_delta` and `passed_min_delta`.
- If `delta < min_delta`, the helper prints a clear stderr diagnostic and exits `1`.

Updated `README.md` with the `--min-delta` behavior.

## Tests

Expanded `tests/test_paired_decode_benchmark.py`:

- factored a fake benchmark writer,
- added regression-gate coverage proving a worse candidate exits `1`,
- and verifies the JSON artifact still exists with `passed_min_delta: false`.

## Verification

- `.venv/bin/python -m pytest -q`: `87 passed, 2 warnings`
- `.venv/bin/tinygrad-gemma --help`: passed
- research METAL smoke: passed, `rollout_jit_count=3`, `decode_fallback=False`
- helper py_compile/help: passed
- `git diff --check`: passed

## Decision

Accepted as infrastructure. This does not create a new runtime score, but it makes future paired comparisons gateable without losing the artifact on failure.
