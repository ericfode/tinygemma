# Artifact Inventory Reference Exactness

## Objective

Resolve `artifact-inventory-reference-exactness-077`: reduce false-positive artifact references in `scripts/inventory_untracked_artifacts.py`.

## Change

- Changed untracked discovery to use `git status --porcelain --untracked-files=all -z` so nested untracked directories are expanded into file paths.
- Replaced raw substring reference checks with exact token matching for the artifact path and basename.
- Exact tokens reject adjacent path/name characters such as letters, digits, `_`, `.`, `/`, and `-`.
- Added regression coverage proving:
  - `artifact.csv` is not referenced by `my-artifact.csv` or `artifact.csv.backup`,
  - exact filename references still count,
  - exact relative-path references still count,
  - unreferenced artifacts remain at reference count 0.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py
.venv/bin/python scripts/inventory_untracked_artifacts.py \
  --timestamp 2026-04-27T03:40:00-0700 \
  >/tmp/tinygrad-gemma-inventory-exact.md \
  2>/tmp/tinygrad-gemma-inventory-exact.stderr
.venv/bin/python -m pytest -q \
  tests/test_artifact_inventory.py \
  tests/test_paired_decode_benchmark.py \
  tests/test_profile_decode_jit.py
.venv/bin/python -m py_compile scripts/inventory_untracked_artifacts.py
.venv/bin/python -m json.tool configs/repo-loop-state.json >/dev/null
git diff --check
```

Results:

- artifact inventory tests: 4 passed
- helper/profiler/paired-helper slice: 43 passed, 2 warnings
- real-repo helper smoke: inventoried 56 untracked path(s)
- top ranked evidence remained docs/state/config-referenced profile JSON artifacts
- py_compile: passed
- repo-loop JSON: valid
- diff whitespace check: passed

## Decision

Accepted helper precision improvement. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
