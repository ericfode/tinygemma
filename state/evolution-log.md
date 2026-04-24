# Evolution Log

## 2026-04-23 - Metal Decode JIT Legality Fallback

- Found that the current tinygrad scheduler rejects the Metal decode JIT cache
  graph with `RuntimeError('input to kernel must be AFTER or BUFFER, not
  Ops.INDEX')`.
- Added explicit decode fallback tracking for both text and conditional Gemma 4
  generation paths. If the symbolic decode JIT raises that scheduler error,
  generation switches to eager token-id decode for the rest of the run instead
  of failing after the first token.
- Kept non-Metal generation on eager decode; the reusable decode JIT is now
  treated as a Metal-only optimization path.
- Extended `scripts/smoke_metal.py` so the cheap Metal gate verifies generation,
  prints `generated_tokens`, `rollout_jit_count`, and `decode_fallback`, and no
  longer stops at a forward/cache pass.
- Added `decode_fallback` to `scripts/benchmark_gemma4_matrix.py` rows so
  throughput artifacts do not hide whether a row used JIT replay.
- Added focused tests for non-Metal generate fallback behavior, preallocated
  sliding-cache correctness after the sliding window, and eliding the mask for
  already-cropped single-token sliding attention.
- Verification on 2026-04-23: focused tests passed; `scripts/smoke_metal.py`
  reported `generated_tokens=4`, `rollout_jit_count=0`, and
  `decode_fallback=True`; real E2B int8 `METAL`, `beam=0`,
  `max_new_tokens=4` completed with `status=ok`, `decode_fallback=true`,
  `rollout_jit_count=0`, `59.696246` seconds, and `0.067006` tok/s.
- This is not a throughput win. It converts a hidden Metal decode failure into a
  complete fallback path and names the real next target: restore legal TinyJit
  replay for symbolic cache stores.

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
