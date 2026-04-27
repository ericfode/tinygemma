# Artifact Inventory Invalid Category Documentation

## Objective

Resolve `artifact-inventory-invalid-category-doc-087`: document that unknown `--only-category` values fail before inventory collection.

## Change

Updated README workflow guidance to state that:

- unknown category names fail before inventory collection,
- the error points back to `--list-categories`,
- typos do not silently produce empty reports.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py
.venv/bin/python scripts/inventory_untracked_artifacts.py \
  --only-category definitely-not-real \
  >/tmp/tinygrad-gemma-bad-category-doc.out \
  2>/tmp/tinygrad-gemma-bad-category-doc.err
# exits nonzero; stderr contains unknown category and --list-categories guidance
git diff --check
```

## Decision

Documentation-only. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
