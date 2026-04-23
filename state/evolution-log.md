# Evolution Log

## 2026-04-23 - Codex Local Environment Bootstrap

- Added root `AGENTS.md` so repo behavior is explicit for Codex sessions.
- Added repo-loop state, a bootstrap plan, procedure-candidate tracking, and a
  Codex environment action file.
- Copied Codex workflow skills/scripts into `.codex/skills/` and
  `.codex/scripts/` while preserving `.codex/skills/george-hotz-reviewer`.
- Created `.venv` with Python 3.11 and installed
  `.[dev,tokenizer,multimodal]`.
- Replaced removed tinygrad `Context(DEV=...)` usage with
  `temporary_default_device(...)` in loader/tests.
- Added `scripts/smoke_metal.py` as the cheap Metal gate. It requires tinygrad
  METAL, saves a tiny checkpoint, reloads it with
  `load_pretrained(..., device="METAL")`, runs a tiny Gemma forward/cache pass,
  realizes logits, and verifies the loaded model, logits, and cache tensors are
  on `METAL`.
- Cheap gates for this environment are `.venv/bin/python -m pytest -q`,
  `.venv/bin/tinygrad-gemma --help`, and
  `.venv/bin/python scripts/smoke_metal.py`.
- Verification on 2026-04-23: `pytest` reported 25 passed, 1 skipped, 2
  warnings; CLI help exited 0; `scripts/smoke_metal.py` reported
  `default_device=METAL`, `loaded_model_device=METAL`,
  `logits_device=METAL`, `logits_shape=(1, 3, 48)`, and
  `cache0_key_device=METAL`.
- Real-checkpoint Metal coverage remains outside this bootstrap until local
  checkpoints are present and a command loads them on `METAL`.
