# Artifact Inventory Category Listing

## Objective

Resolve `artifact-inventory-list-categories-084`: make valid artifact categories discoverable from the CLI without reading source code.

## Change

Added `--list-categories` to `scripts/inventory_untracked_artifacts.py`.

Behavior:

- prints known categories one per line,
- exits without requiring a git repo,
- emits no human inventory summary on stderr,
- provides a stable source of names for `--only-category`.

Current categories:

- `benchmark-progress-log`
- `benchmark-result-artifact`
- `decode-profiler-artifact`
- `dependency-lockfile`
- `other-untracked`
- `paired-helper-smoke`

## Verification

```bash
.venv/bin/python scripts/inventory_untracked_artifacts.py --list-categories \
  >/tmp/tinygrad-gemma-artifact-categories.txt
python3 - <<'PY'
from pathlib import Path
cats=Path('/tmp/tinygrad-gemma-artifact-categories.txt').read_text().splitlines()
assert cats == sorted(cats)
assert 'benchmark-progress-log' in cats
assert 'decode-profiler-artifact' in cats
PY
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py \
  tests/test_paired_decode_benchmark.py \
  tests/test_profile_decode_jit.py
.venv/bin/python -m py_compile scripts/inventory_untracked_artifacts.py
.venv/bin/python -m json.tool configs/repo-loop-state.json >/dev/null
git diff --check
```

Observed deterministic slice: `47 passed, 2 warnings`.

## Decision

Accepted category discovery. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
