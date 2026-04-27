# Artifact Inventory JSON Output Documentation

## Objective

Resolve `artifact-inventory-json-output-doc-079`: document the helper's machine-readable JSON mode in README.

## Change

Updated README artifact-inventory guidance to mention:

- `--format json`,
- stdout remains machine-readable JSON,
- the human `inventoried N untracked path(s)` summary stays on stderr.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py
.venv/bin/python scripts/inventory_untracked_artifacts.py \
  --format json \
  --timestamp 2026-04-27T04:00:00-0700 \
  >/tmp/tinygrad-gemma-inventory-doc-json.json \
  2>/tmp/tinygrad-gemma-inventory-doc-json.stderr
python3 -m json.tool /tmp/tinygrad-gemma-inventory-doc-json.json >/dev/null
git diff --check
```

## Decision

Documentation-only. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
