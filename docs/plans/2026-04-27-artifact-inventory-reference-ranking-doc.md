# Artifact Inventory Reference Ranking Documentation

## Objective

Resolve `artifact-inventory-reference-ranking-doc-076`: document the inventory helper's reference-ranked table in the main README workflow.

## Change

Updated README artifact-inventory guidance to state that:

- the main table is reference-ranked,
- artifacts cited by tracked docs/state/config appear before uncited generated files,
- higher reference counts sort first,
- the ranking is only a review aid, not permission to auto-commit generated evidence.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py
git diff --check
```

Results:

- artifact inventory tests: passed
- diff whitespace check: passed

## Decision

Accepted documentation-only follow-up. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
