# Paired Helper Contract Documentation Refresh

## Objective

Resolve `paired-helper-contract-doc-refresh-067`: make the README mention the paired helper's finite-score JSON contract.

## Change

Updated `README.md` paired-helper paragraph to state that child benchmark stdout must be JSON containing a finite numeric `score`, and that `NaN`/infinities are rejected rather than recorded.

## Verification

- `.venv/bin/python -m pytest -q tests/test_paired_decode_benchmark.py`: `4 passed`
- `git diff --check`: passed

## Decision

Accepted as documentation-only. No runtime or helper semantics changed in this increment.
