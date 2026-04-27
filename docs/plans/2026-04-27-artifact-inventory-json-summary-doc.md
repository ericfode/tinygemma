# Artifact Inventory JSON Summary Documentation

## Objective

Resolve `artifact-inventory-json-summary-doc-081`: document the artifact inventory helper's JSON summary fields in README.

## Change

Updated README workflow guidance to state that `scripts/inventory_untracked_artifacts.py --format json` includes:

- `untracked_count`,
- `total_size`,
- `referenced_count`,
- `counts_by_category`,
- `counts_by_suffix`,
- ranked `rows` with per-row `reference_count`.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py
.venv/bin/python scripts/inventory_untracked_artifacts.py \
  --format json \
  --timestamp 2026-04-27T04:20:00-0700 \
  >/tmp/tinygrad-gemma-inventory-summary-doc.json \
  2>/tmp/tinygrad-gemma-inventory-summary-doc.stderr
python3 -m json.tool /tmp/tinygrad-gemma-inventory-summary-doc.json >/dev/null
git diff --check
```

## Decision

Documentation-only. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
