# Artifact Inventory Invalid Category Validation

## Objective

Resolve `artifact-inventory-invalid-category-validation-086`: reject unknown `--only-category` values before inventory collection.

## Change

Updated `scripts/inventory_untracked_artifacts.py` to validate `--only-category` values against the known category set before running git inventory commands.

Behavior:

- typo categories exit nonzero,
- stderr names the unknown value,
- stderr points users to `--list-categories`,
- validation happens before git collection, avoiding irrelevant tracebacks outside a repo.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py \
  tests/test_paired_decode_benchmark.py \
  tests/test_profile_decode_jit.py
.venv/bin/python scripts/inventory_untracked_artifacts.py \
  --only-category definitely-not-real \
  >/tmp/tinygrad-gemma-bad-category.out \
  2>/tmp/tinygrad-gemma-bad-category.err
# exits nonzero; stderr contains unknown category and --list-categories guidance
.venv/bin/python -m py_compile scripts/inventory_untracked_artifacts.py
.venv/bin/python -m json.tool configs/repo-loop-state.json >/dev/null
git diff --check
```

Observed deterministic slice: `48 passed, 2 warnings`.

## Decision

Accepted typo protection. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
