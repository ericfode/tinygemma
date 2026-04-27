# Artifact Inventory README Documentation

## Objective

Resolve `artifact-inventory-readme-doc-073`: make the read-only untracked artifact inventory helper discoverable from the main workflow documentation.

## Change

Updated README Gemma 4 Matrix Workflow documentation to add:

```bash
python scripts/inventory_untracked_artifacts.py \
  --output docs/plans/$(date +%F)-untracked-artifact-inventory.md
```

The README now states that the helper reports untracked paths, sizes, categories, tracked-doc references, and suggested dispositions, but does not delete artifacts, edit ignore files, stage files, or decide which benchmark evidence should travel with the repo.

## Verification

- `tests/test_artifact_inventory.py`: confirms helper behavior and read-only semantics.
- Diff checks before commit.
