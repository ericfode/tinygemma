# Artifact Inventory JSON Summary Counts

## Objective

Resolve `artifact-inventory-category-summary-json-080`: include category and suffix counts in the machine-readable artifact inventory JSON payload.

## Change

Updated `scripts/inventory_untracked_artifacts.py` so `--format json` now includes:

- `counts_by_category`,
- `counts_by_suffix`,
- the existing ranked `rows`, `untracked_count`, `total_size`, and `referenced_count` fields.

The helper remains read-only; markdown output is unchanged except for sharing the same underlying row sort.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py \
  tests/test_paired_decode_benchmark.py \
  tests/test_profile_decode_jit.py
.venv/bin/python scripts/inventory_untracked_artifacts.py \
  --format json \
  --timestamp 2026-04-27T04:10:00-0700 \
  >/tmp/tinygrad-gemma-inventory-summary.json \
  2>/tmp/tinygrad-gemma-inventory-summary.stderr
python3 - <<'PY'
import json
from pathlib import Path
payload=json.loads(Path('/tmp/tinygrad-gemma-inventory-summary.json').read_text())
assert payload['untracked_count'] == 56
assert payload['counts_by_category']['decode-profiler-artifact'] >= 1
assert payload['counts_by_suffix']['.json'] >= 1
assert payload['rows'][0]['reference_count'] >= payload['rows'][-1]['reference_count']
PY
.venv/bin/python -m py_compile scripts/inventory_untracked_artifacts.py
.venv/bin/python -m json.tool configs/repo-loop-state.json >/dev/null
git diff --check
```

Observed deterministic slice: `45 passed, 2 warnings`.

Real-repo summary snapshot:

- `untracked_count`: `56`
- category counts include `decode-profiler-artifact: 28`, `benchmark-progress-log: 14`, `benchmark-result-artifact: 11`
- suffix counts include `.csv: 23`, `.json: 18`, `.jsonl: 14`

## Decision

Accepted machine-readable summaries. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
