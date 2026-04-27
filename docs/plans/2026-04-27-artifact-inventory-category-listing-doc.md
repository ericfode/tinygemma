# Artifact Inventory Category Listing Documentation

## Objective

Resolve `artifact-inventory-list-categories-doc-085`: document the artifact inventory helper's `--list-categories` flag in README.

## Change

Updated README workflow guidance to mention:

- `--list-categories` prints valid category names,
- use those names with repeatable `--only-category <category>`,
- example remains focused on `benchmark-progress-log` JSON review.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py
.venv/bin/python scripts/inventory_untracked_artifacts.py --list-categories \
  >/tmp/tinygrad-gemma-artifact-categories-doc.txt
git diff --check
```

## Decision

Documentation-only. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
