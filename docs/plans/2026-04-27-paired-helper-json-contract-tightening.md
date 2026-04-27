# Paired Helper JSON Contract Tightening

## Objective

Resolve `paired-helper-json-contract-tightening-066`: prevent malformed benchmark score payloads from silently passing through paired comparison artifacts.

## Change

Updated `scripts/paired_e2b_decode_benchmark.py`:

- Import `math` and require benchmark `score` to be finite numeric.
- Reject `NaN`, `Infinity`, and `-Infinity` even though Python's JSON parser accepts those tokens by default.
- Serialize helper output with `allow_nan=False` as a second guard.

Expanded `tests/test_paired_decode_benchmark.py` with a fake benchmark that emits `{"score": NaN}` and verifies the helper exits nonzero with a finite-score diagnostic.

## Verification

- `.venv/bin/python -m pytest -q`: `88 passed, 2 warnings`
- `.venv/bin/tinygrad-gemma --help`: passed
- research METAL smoke: passed, `rollout_jit_count=3`, `decode_fallback=False`
- helper py_compile/help: passed
- `git diff --check`: passed

## Decision

Accepted as infrastructure hardening. No runtime or benchmark score semantics changed except rejecting invalid score payloads.
