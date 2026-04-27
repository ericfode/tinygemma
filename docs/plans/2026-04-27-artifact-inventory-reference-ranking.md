# Artifact Inventory Reference Ranking

## Objective

Resolve `artifact-inventory-reference-ranking-075`: make `scripts/inventory_untracked_artifacts.py` rank likely evidence-to-keep artifacts ahead of uncited generated files.

## Change

- Sort inventory rows by:
  1. referenced artifacts before unreferenced artifacts,
  2. higher tracked-reference count first,
  3. path name as a deterministic tiebreaker.
- Rename the main table to `Reference-ranked candidates`.
- Add an explicit `Reference count` column.
- Keep all behavior read-only.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py
.venv/bin/python scripts/inventory_untracked_artifacts.py \
  --timestamp 2026-04-27T03:30:00-0700 \
  >/tmp/tinygrad-gemma-inventory-ranked.md \
  2>/tmp/tinygrad-gemma-inventory-ranked.stderr
.venv/bin/python -m pytest -q \
  tests/test_artifact_inventory.py \
  tests/test_paired_decode_benchmark.py \
  tests/test_profile_decode_jit.py
.venv/bin/python -m py_compile scripts/inventory_untracked_artifacts.py
.venv/bin/python -m json.tool configs/repo-loop-state.json >/dev/null
git diff --check
```

Results:

- artifact inventory tests: 3 passed
- focused helper/profiler/paired-helper test slice: 42 passed, 2 warnings
- real-repo helper smoke: inventoried 56 untracked path(s)
- top ranked rows were docs/state/config-referenced JSON/profile evidence, e.g. phase-profile and paired-helper artifacts with 4 tracked references
- py_compile: passed
- repo-loop JSON: valid
- diff whitespace check: passed

## Decision

Accepted helper evidence-ranking improvement. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
