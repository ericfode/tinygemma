# Codex Local Environment Bootstrap

Date: 2026-04-23

## Objective

Make this worktree self-describing for Codex and locally runnable for cheap
verification.

## Scope

- Add root repo instructions for `tinygrad-gemma`.
- Add repo-loop state and a terse evolution log.
- Add a Codex local environment action file.
- Preserve the existing `george-hotz-reviewer` skill and add reusable Codex
  workflow skills/scripts locally.
- Create `.venv` with Python 3.11 and install the repo with dev, tokenizer,
  and multimodal extras.
- Fix the tinygrad 0.12 device-selection compatibility issue exposed by the
  test gate.
- Add a repo-owned Metal smoke gate that opens tinygrad METAL and runs a tiny
  Gemma forward/cache pass on that device.

## Gates

```bash
.venv/bin/python -m pytest -q
.venv/bin/tinygrad-gemma --help
.venv/bin/python scripts/smoke_metal.py
```

## Non-Goals

- No model checkpoint download.
- No Metal benchmark run.
- No performance claim beyond cheap local gates.

## Acceptance

The setup is accepted when the harness files exist, the venv imports the
package, the Python gate passes, the CLI entrypoint resolves, and the Metal
smoke gate proves tinygrad METAL execution.

Accepted on 2026-04-23 with:

```bash
.venv/bin/python -m pytest -q
# 25 passed, 1 skipped, 2 warnings

.venv/bin/tinygrad-gemma --help
# exited 0

.venv/bin/python scripts/smoke_metal.py
# default_device=METAL
# loaded_model_device=METAL
# logits_device=METAL
# logits_shape=(1, 3, 48)
# cache0_key_device=METAL
```
