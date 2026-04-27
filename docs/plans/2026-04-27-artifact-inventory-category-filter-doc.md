# Artifact Inventory Category Filter Documentation

## Objective

Resolve `artifact-inventory-only-filter-doc-083`: document the artifact inventory helper's `--only-category` filter in README.

## Change

Updated README workflow guidance to mention:

- repeatable `--only-category <category>`,
- example use with `--format json --only-category benchmark-progress-log`,
- filtering avoids downstream shell-side scraping.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py
.venv/bin/python scripts/inventory_untracked_artifacts.py \
  --format json \
  --only-category benchmark-progress-log \
  --timestamp 2026-04-27T04:40:00-0700 \
  >/tmp/tinygrad-gemma-inventory-filter-doc.json \
  2>/tmp/tinygrad-gemma-inventory-filter-doc.stderr
python3 -m json.tool /tmp/tinygrad-gemma-inventory-filter-doc.json >/dev/null
git diff --check
```

## Decision

Documentation-only. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
