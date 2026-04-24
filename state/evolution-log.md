# Evolution Log

## 2026-04-23 - Fused MLP Gate-Up Decode Profile

- Added `scripts/profile_decode_jit.py`, a repo-owned synthetic post-window
  TinyJit profiler. It builds a zeroed KV cache at context length 700, forces
  `JIT=2` and `TRACEMETA=2`, then times the captured decode kernels by source
  category.
- The profile made the next bottleneck concrete: after the fused-MLP patch,
  synthetic post-window decode measured `110.302` ms total with MLP at
  `38.938` ms, attention at `26.236` ms, norm at `13.447` ms, and
  logits/argmax at `1.793` ms. Logits are not the current primary limiter.
- The same artifact records that the local `gemma-4-E2B-int8` checkpoint has an
  `int8` manifest with `0` quantized tensors and `2011` raw bfloat16 tensors.
  Treat the directory name as storage labeling, not runtime int8 proof.
- Added an inference-only fused gate/up path in `GemmaMLP`. When the gate and
  up weights are not trainable, decode caches a module-external concatenated
  gate/up weight and uses one wide linear before splitting gate and up. The
  fused cache is not part of `nn.state.get_state_dict`, and training still uses
  the original separate projections.
- Verification on 2026-04-23: full tests passed with `34 passed, 1 skipped,
  2 warnings in 37.79s`; CLI help exited 0; `scripts/smoke_metal.py`
  reported `generated_tokens=4`, `rollout_jit_count=3`, and
  `decode_fallback=False`; real E2B int8 `METAL`, `beam=0`,
  `max_new_tokens=700`, `decode_warmup_tokens=520` measured `11.891932`
  post-window tok/s with `rollout_jit_count=699` and
  `decode_fallback=false`; real E2B int8 `METAL`, `beam=0`,
  `max_new_tokens=1000`, `decode_warmup_tokens=20` measured `9.303522`
  tok/s with `rollout_jit_count=999` and `decode_fallback=false`.
- This improves the prior current 1000-token row from `8.300076` to
  `9.303522` measured tok/s, still far below the `50` tok/s target.
- Next target: reduce the remaining post-window floor around MLP/norm/attention
  kernel volume or introduce real runtime int8 matmul. Another logits-only
  optimization is unlikely to move the target materially.

## 2026-04-23 - Conditional Sliding Decode JIT Switch

- Diagnosed the post-window slowdown to a path split: `GemmaForCausalLM`
  already had a second Metal TinyJit for symbolic sliding-window decode, but
  the `GemmaForConditionalGeneration` path used by the official E2B int8
  checkpoint kept one symbolic decode JIT and never enabled
  `cache.decode_sliding_window`.
- Ported the Metal-only two-JIT sliding decode switch into
  `tinygrad_gemma/multimodal.py`. Conditional generation now uses the normal
  symbolic decode JIT before `sliding_window - 1`, then a bounded
  `gemma_start_pos_window` JIT with `decode_sliding_window=True` after the
  window.
- Added a unit test for the conditional Metal-only sliding decode start helper.
- Verification on 2026-04-23: focused tests passed; `scripts/smoke_metal.py`
  reported `generated_tokens=4`, `rollout_jit_count=3`, and
  `decode_fallback=False`; real E2B int8 `METAL`, `beam=0`,
  `max_new_tokens=700`, `decode_warmup_tokens=520` measured `10.295446`
  post-window tok/s with `rollout_jit_count=699` and `decode_fallback=false`;
  real E2B int8 `METAL`, `beam=0`, `max_new_tokens=1000`,
  `decode_warmup_tokens=20` measured `8.300076` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Full repo gates passed after the patch: `34 passed, 1 skipped, 2 warnings in
  38.65s`, CLI help exit 0, and the strengthened Metal smoke gate exit 0.
- This improves the prior current 1000-token row from `6.245549` measured tok/s
  to `8.300076` measured tok/s with the same output hash. The remaining gap is
  the second JIT capture around the window plus the full-attention layers that
  continue growing with context.
- Additional diagnostics on the same post-window slice: forcing the 7
  full-attention layers to use the local window measured `12.496987` tok/s, and
  a sync-free tensor loop measured `11.052106` tok/s. The full-attention layers
  and per-token CPU synchronization are real but insufficient to explain the
  remaining gap to `50` tok/s by themselves.
- Next target: produce a post-window kernel-level profile that separates
  attention/full-layer work, MLP work, and the tied logits/argmax path before
  making a more invasive optimization.

## 2026-04-23 - Long Metal Decode Floor Refresh

- Ran the queued real long-floor benchmark on the current grouped-query decode
  path and appended it to `benchmarks/gemma4-metal-1000-current.csv` instead of
  overwriting the existing interrupted beam=4 artifact.
- Verified E2B int8 `METAL`, `beam=0`, `max_new_tokens=1000`,
  `decode_warmup_tokens=20`: `status=ok`, `generated_tokens=1000`,
  `190.223269` seconds total, `5.256980` end-to-end tok/s,
  `156.911756` measured decode seconds, `6.245549` measured tok/s,
  `rollout_jit_count=999`, and `decode_fallback=false`.
- Progress sidecar evidence shows the long-context floor decays with generated
  length: `12.181298` measured tok/s at 100 generated tokens, `8.240744` at
  600, and `6.245961` at 1000.
- The existing `beam=4` row in the same artifact remains an interrupted run:
  `986` generated tokens, `438.910876` seconds, `status=interrupted`,
  `decode_fallback=false`. It is not the beam=0 acceptance row.
- Full repo gates passed after recording the artifact: `33 passed, 1 skipped,
  2 warnings in 37.08s`, CLI help exit 0, and the strengthened Metal smoke gate
  exit 0.
- Conclusion: the older 22 tok/s 1000-token note is stale for current code. The
  next code increment should target the long-context floor after separating
  sliding-layer cache-write/graph cost from the 7 full-attention layers.

## 2026-04-23 - Grouped-Query Decode Without KV Repeat

- Replaced the text attention decode math in `tinygrad_gemma/model.py` so
  grouped-query attention reshapes `q` into `(batch, kv_heads, groups, tokens,
  head_dim)` and broadcasts K/V with an extra group axis instead of physically
  expanding K/V through `repeat_kv()`.
- Left the shared `repeat_kv()` helper in place because multimodal vision/audio
  attention still imports and uses it outside the measured E2B text decode path.
- Verification on 2026-04-23: focused Gemma 4 forward/cache tests passed;
  `scripts/smoke_metal.py` reported `generated_tokens=4`,
  `rollout_jit_count=3`, and `decode_fallback=False`; real E2B int8 `METAL`,
  `beam=0`, `max_new_tokens=200`, `decode_warmup_tokens=20` measured
  `10.659780` warmup-excluded tok/s with `rollout_jit_count=199` and
  `decode_fallback=false`; full repo gates passed with `33 passed, 1 skipped,
  2 warnings in 43.23s`, CLI help exit 0, and the strengthened Metal smoke gate
  exit 0.
- Comparison row on the same target before this change measured `8.918411`
  warmup-excluded tok/s. The 50-token row was effectively neutral, so this is a
  modest longer-decode improvement, not the final throughput fix.
- Next target: refresh the 1000-token Metal row and then attack the long-context
  floor, likely around sliding-window cache behavior after the 512-token window.

## 2026-04-23 - Metal Decode JIT Replay Restored

- Read the tinygrad scheduler/JIT/cache-write path and compared this repo's
  symbolic cache mutation with tinygrad's own LLM examples in
  `.venv/lib/python3.11/site-packages/tinygrad/apps/llm.py` and
  `/Users/ericfode/src/.tinygrad_research/extra/models/llama.py`.
- Conclusion: the observed `RuntimeError('input to kernel must be AFTER or
  BUFFER, not Ops.INDEX')` was not proven to be a tinygrad bug. The repo was
  constructing symbolic cache writes with raw `UOp.after(...store...)`, while
  tinygrad's supported model pattern uses slice `.assign(...).realize()` inside
  the JIT path.
- Replaced the symbolic preallocated KV-cache update in
  `tinygrad_gemma/model.py` with the public slice-assign idiom. The integer
  prefill path already used this form.
- Strengthened `scripts/smoke_metal.py` so the Metal smoke gate now fails if
  decode generation falls back to eager mode or never runs a rollout TinyJit.
- Verification on 2026-04-23: focused cache/generate tests passed;
  `scripts/smoke_metal.py` reported `generated_tokens=4`,
  `rollout_jit_count=3`, and `decode_fallback=False`; real E2B int8 `METAL`,
  `beam=0`, `max_new_tokens=4` completed with `decode_fallback=false`,
  `rollout_jit_count=3`, `31.753536` seconds, and `0.125970` tok/s; real E2B
  int8 `METAL`, `beam=0`, `max_new_tokens=50`, `decode_warmup_tokens=4`
  measured `14.284291` warmup-excluded tok/s with `rollout_jit_count=49` and
  `decode_fallback=false`; full repo gates passed with `33 passed, 1 skipped,
  2 warnings in 39.12s`, CLI help exit 0, and the strengthened Metal smoke gate
  exit 0.
- Next target: improve the warmup-excluded Metal decode floor now that JIT
  replay is legal again. The most likely repo-local targets are sliding-window
  cache work and avoiding physical grouped-query KV repetition.

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
