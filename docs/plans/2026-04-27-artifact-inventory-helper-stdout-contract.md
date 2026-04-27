# Artifact Inventory Helper stdout Contract

## Objective

Resolve `artifact-inventory-helper-stdout-contract-074`: make `scripts/inventory_untracked_artifacts.py` safe for shell pipelines by keeping markdown stdout clean.

## Problem

After moving the helper from a one-off script into a reusable CLI, the no-`--output` mode printed the generated markdown to stdout and then appended a human summary such as:

```text
inventoried 56 untracked path(s)
```

That corrupted the stdout stream for callers that redirect the generated markdown directly into a file.

## Change

- Added regression coverage in `tests/test_artifact_inventory.py` proving stdout mode emits pure markdown.
- Moved the human summary to stderr for both stdout and `--output` modes.
- Fixed the implementation import (`sys`) required for stderr printing.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_artifact_inventory.py
.venv/bin/python scripts/inventory_untracked_artifacts.py \
  --timestamp 2026-04-27T03:21:00-0700 \
  >/tmp/tinygrad-gemma-inventory-stdout.md \
  2>/tmp/tinygrad-gemma-inventory-stderr.txt
.venv/bin/python -m json.tool configs/repo-loop-state.json >/dev/null
git diff --check
```

Results:

- artifact inventory tests: 2 passed
- stdout smoke: markdown begins with `# Untracked Artifact Inventory`
- stdout smoke: no `inventoried` summary in stdout
- stderr smoke: summary printed to stderr, e.g. `inventoried 56 untracked path(s)`
- repo-loop JSON: valid
- diff whitespace check: passed

## Decision

Accepted as helper contract hardening. No generated benchmark/profile artifacts were deleted, staged, ignored, or committed.
