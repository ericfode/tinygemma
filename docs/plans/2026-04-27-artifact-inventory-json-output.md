# Artifact Inventory JSON Output

## Objective

Resolve `artifact-inventory-json-output-078`: add a machine-readable JSON output mode to `scripts/inventory_untracked_artifacts.py`.

## Change

- Added `--format markdown|json`, defaulting to markdown.
- Added strict JSON serialization with `allow_nan=False`.
- JSON payload includes:
  - `timestamp`,
  - `untracked_count`,
  - `total_size`,
  - `referenced_count`,
  - ranked `rows` with `path`, `category`, `size`, `refs`, `reference_count`, and `disposition`.
- Preserved stderr-only human summary for both markdown and JSON modes.
- Kept helper read-only.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py
.venv/bin/python scripts/inventory_untracked_artifacts.py \
  --format json \
  --timestamp 2026-04-27T03:50:00-0700 \
  >/tmp/tinygrad-gemma-inventory.json \
  2>/tmp/tinygrad-gemma-inventory-json.stderr
.venv/bin/python -m pytest -q \
  tests/test_artifact_inventory.py \
  tests/test_paired_decode_benchmark.py \
  tests/test_profile_decode_jit.py
.venv/bin/python -m py_compile scripts/inventory_untracked_artifacts.py
.venv/bin/python -m json.tool configs/repo-loop-state.json >/dev/null
git diff --check
```

Results:

- artifact inventory tests: 5 passed
- helper/profiler/paired-helper slice: 44 passed, 2 warnings
- real-repo JSON smoke: parsed 56 rows, first ranked row stable, stderr summary separate
- py_compile: passed
- repo-loop JSON: valid
- diff whitespace check: passed

## Decision

Accepted machine-readable helper output. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
