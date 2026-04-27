# Artifact Inventory Category Filter

## Objective

Resolve `artifact-inventory-only-filter-082`: add a focused category filter to the read-only untracked artifact inventory helper.

## Change

Added repeatable `--only-category <category>` to `scripts/inventory_untracked_artifacts.py`.

Behavior:

- filters collected rows before rendering,
- applies to both markdown and JSON output,
- summary counts describe the filtered set,
- human stderr summary reports the filtered row count,
- helper remains read-only.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py \
  tests/test_paired_decode_benchmark.py \
  tests/test_profile_decode_jit.py
.venv/bin/python scripts/inventory_untracked_artifacts.py \
  --format json \
  --only-category benchmark-progress-log \
  --timestamp 2026-04-27T04:30:00-0700 \
  >/tmp/tinygrad-gemma-inventory-filter.json \
  2>/tmp/tinygrad-gemma-inventory-filter.stderr
python3 - <<'PY'
import json
from pathlib import Path
payload=json.loads(Path('/tmp/tinygrad-gemma-inventory-filter.json').read_text())
assert payload['untracked_count'] == 14
assert payload['counts_by_category'] == {'benchmark-progress-log': 14}
assert all(row['category'] == 'benchmark-progress-log' for row in payload['rows'])
PY
.venv/bin/python -m py_compile scripts/inventory_untracked_artifacts.py
.venv/bin/python -m json.tool configs/repo-loop-state.json >/dev/null
git diff --check
```

Observed deterministic slice: `46 passed, 2 warnings`.

## Decision

Accepted focused filtering for review tooling. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
