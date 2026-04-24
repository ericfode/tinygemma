# Gemma 4 Speed Research

Date: 2026-04-23

Repo: `/Users/ericfode/Downloads/tinygrad-gemma`

Target checked here: E2B int8 on tinygrad `METAL`.

## Bottom Line

The main bottleneck was not Metal math capacity. It was the autoregressive decode
path failing to enter a reusable tinygrad JIT replay soon enough. `DEBUG=1`
showed repeated cache misses and growing kernel schedules around the per-layer KV
cache writes in `tinygrad_gemma/model.py`.

Update later on 2026-04-23: the current local tinygrad scheduler can reject the
symbolic Metal cache-store graph outright with `RuntimeError('input to kernel
must be AFTER or BUFFER, not Ops.INDEX')`. The repo now records
`decode_fallback` in benchmark rows and falls back to eager token-id decode
instead of failing after the first generated token. A real E2B int8 `METAL`,
`beam=0`, `max_new_tokens=4` row completed only through fallback:
`59.696246` seconds, `0.067006` tok/s, `rollout_jit_count=0`,
`decode_fallback=true`. Treat this as a regression/legality finding, not a speed
result.

Second update on 2026-04-23: this was not proven to be a tinygrad scheduler bug.
tinygrad's own LLM cache examples use public slice assignment,
`cache[..., start_pos:start_pos+T, ...].assign(...).realize()`, inside the JIT
path and then read from the realized cache. The failing repo graph hand-built
`AFTER(STORE(...))` with raw UOps for symbolic cache writes; after rangeification
that left an `Ops.INDEX` where `tinygrad.engine.schedule.create_schedule`
requires a `BUFFER`, `BIND`, `MSELECT`/`MSTACK`, or ordered `AFTER` input. Moving
the symbolic cache update to the public assign idiom restored Metal TinyJit
replay in this repo.

Current `HEAD` has a TinyJit decode experiment in `generate()` and rollout-JIT
metrics in the benchmark script. It is a real improvement: E2B int8 now
completes the 1000-token Metal gate without decode fallback. The previous
22 tok/s 1000-token note is stale for the current code path; the latest fused
MLP row measures 9.303522 warmup-excluded tok/s and still shows a large gap to
the 50 tok/s target.

| Run | Result |
| --- | ---: |
| Older E2B int8, beam=0, 10 tokens | 86.676913 s, 0.115371 tok/s |
| Older E2B int8, beam=1, 10 tokens | 89.059974 s, 0.112284 tok/s |
| Current E2B int8, beam=0, 10 tokens | 21.295810 s, 0.469576 tok/s, `rollout_jit_count=9` |
| Current E2B int8, beam=0, 100 tokens | 23.905944 s, 4.183060 tok/s, `rollout_jit_count=99` |
| Stale E2B int8, beam=0, 1000 tokens | 45.169393 s, 22.138885 tok/s, `rollout_jit_count=999` |
| Current E2B int8, beam=1, 1000 tokens | 90.487042-104.922802 s, 9.530817-11.051306 tok/s, `rollout_jit_count=999` |
| Fallback legality row, beam=0, 4 tokens | 59.696246 s, 0.067006 tok/s, `rollout_jit_count=0`, `decode_fallback=true` |
| Public-assign cache row, beam=0, 4 tokens | 31.753536 s, 0.125970 tok/s, `rollout_jit_count=3`, `decode_fallback=false` |
| Public-assign cache row, beam=0, 50 tokens, 4-token warmup | 34.948690 s total, 14.284291 measured tok/s, `rollout_jit_count=49`, `decode_fallback=false` |
| Repeat-KV baseline, beam=0, 200 tokens, 20-token warmup | 56.388236 s total, 8.918411 measured tok/s, `rollout_jit_count=199`, `decode_fallback=false` |
| Grouped-query no-repeat, beam=0, 200 tokens, 20-token warmup | 55.035400 s total, 10.659780 measured tok/s, `rollout_jit_count=199`, `decode_fallback=false` |
| Grouped-query no-repeat, beam=0, 1000 tokens, 20-token warmup | 190.223269 s total, 6.245549 measured tok/s, `rollout_jit_count=999`, `decode_fallback=false` |
| Conditional sliding-JIT switch, beam=0, 700 tokens, 520-token warmup | 118.606687 s total, 10.295446 measured post-window tok/s, `rollout_jit_count=699`, `decode_fallback=false` |
| Conditional sliding-JIT switch, beam=0, 1000 tokens, 20-token warmup | 149.649359 s total, 8.300076 measured tok/s, `rollout_jit_count=999`, `decode_fallback=false` |
| Fused MLP gate/up, beam=0, 700 tokens, 520-token warmup | 109.750742 s total, 11.891932 measured post-window tok/s, `rollout_jit_count=699`, `decode_fallback=false` |
| Fused MLP gate/up, beam=0, 1000 tokens, 20-token warmup | 136.105067 s total, 9.303522 measured tok/s, `rollout_jit_count=999`, `decode_fallback=false` |
| Diagnostic: window all 7 full-attention layers, 700 tokens, 520-token warmup | 114.692150 s total, 12.496987 measured post-window tok/s, `rollout_jit_count=699`, `decode_fallback=false` |
| Diagnostic: sync-free tensor loop, 700 tokens, 520-token warmup | 114.792884 s total, 11.052106 measured post-window tok/s, `rollout_jit_count=699`, `decode_fallback=false` |
| Interrupted beam=4 row, 1000-token target | 986 tokens, 438.910876 s, `status=interrupted`, `decode_fallback=false` |

`beam=1` is currently worse end-to-end because the BEAM compile/search cliff is
large. It does improve post-warmup token cadence in parts of the run, but the
interactive default should stay `beam=0` until compile cost is amortized or moved
out of the measured path. The repo-local beam=1 artifact is
`benchmarks/gemma4-metal-1000.csv`; an independent `/tmp` rerun measured
104.922802 s.

## DEBUG Evidence

`DEBUG=1` is the right instrument for the "jitter" question. In local tinygrad,
`TinyJit` prints `JIT captured ...` when the replay graph is captured. The
current benchmark process imports tinygrad from the repo venv at
`.venv/lib/python3.11/site-packages/tinygrad`.

Before the current JIT path is useful, the trace shows repeated cache misses at:

- `tinygrad_gemma/model.py:406`: `entry.key[:, :, past_seen_tokens:end_pos, :].assign(k).realize()`
- `tinygrad_gemma/model.py:407`: `entry.value[:, :, past_seen_tokens:end_pos, :].assign(v).realize()`

The scheduled kernel counts grow through the layer stack: 7, 28, 48, 67, 86,
105, and higher. That is scheduler/compile overhead, not a saturated GPU kernel.
In the current experiment, the trace eventually prints `JIT captured 1 linears
with 1 inputs`, and later tokens replay quickly.

The 10-token row makes the shape obvious:

- token 1: 15.502652 s
- token 2: 19.792983 s
- token 3: 21.172271 s
- token 10: 21.286981 s

After capture, tokens 4-10 arrive at roughly 16 ms/token. The 1000-token row
slows with context length but still completes, unlike the earlier long attempts.

## What To Do Next

1. Keep legal Metal TinyJit replay pinned before optimizing throughput claims.
   - `scripts/smoke_metal.py` now fails if generation falls back to eager decode
     or never runs the rollout JIT.
   - Keep `decode_fallback=true` rows out of speed comparisons except as failure
     evidence.
   - The next acceptable E2B speed row needs `decode_fallback=false`, nonzero
     `rollout_jit_count`, and a warmup-excluded measured decode suffix.

2. Keep the TinyJit decode direction, but make it first-class.
   - Move the closure-based rollout into an explicit decode runner/state object.
   - Own the preallocated cache buffers there.
   - Use a stable symbolic `cache_position`/`Variable`.
   - Add a warmup phase that captures the decode graph before measured generation.
   - Gate it with `DEBUG=1`: capture should happen by the third generated token,
     and no growing `CACHE MISS` chain should recur at the KV write lines.

3. Use `beam=0` for interactive inference and benchmark gates for now.
   - `beam=1` finished 1000 tokens, but only after a roughly 70-85 s first
     100-token cliff in local runs. It is an offline tuning setting until warmup
     is separated.

4. Implement real sliding-window KV behavior next.
   - E2B has 35 layers: 28 sliding layers with `sliding_window=512` and 7 full
     attention layers.
   - The refreshed 1000-token row decays from 12.181298 measured tok/s at
     100 generated tokens to 6.245961 measured tok/s at 1000 generated tokens.
   - Diagnosis: the text `GemmaForCausalLM` path already switched to a
     sliding-window decode JIT after the window, but the conditional/multimodal
     path used by E2B int8 did not. Porting that switch improved the refreshed
     1000-token row from 6.245549 measured tok/s to 8.300076 measured tok/s.
   - The post-window-only row, `--max-new-tokens 700
     --decode-warmup-tokens 520`, measured 10.295446 tok/s. The remaining
     full-row gap is partly the second JIT capture at the window boundary and
     partly the 7 full-attention layers that still grow with context.
   - A semantic-breaking diagnostic that forced the 7 full-attention layers to
     use the same local window measured 12.496987 post-window tok/s. That is a
     useful ceiling but not enough to explain the gap to 50 tok/s by itself.
   - A sync-free tensor-loop diagnostic measured 11.052106 post-window tok/s,
     so per-token `item()` synchronization is not the main limiter.

5. Keep grouped-query attention on the no-repeat decode path.
   - E2B has 8 attention heads and 1 KV head.
   - Replacing physical `repeat_kv()` in the text attention path with grouped
     attention was neutral at 50 generated tokens but improved the 200-token
     warmup-excluded row from 8.918411 tok/s to 10.659780 tok/s.
   - The multimodal vision/audio attention helpers still use `repeat_kv()`;
     leave those alone until they are on a measured hot path.

6. Treat repo `int8` as storage-only until runtime int8 exists.
   - `tinygrad_gemma/quantization.py` dequantizes checkpoint tensors back into
     ordinary tinygrad tensors at load.
   - Real speed from quantization needs compressed weights through matmul or a
     fused dequantize-matmul path.

7. Runtime int8 matmul is real now, but not enough.
   - The local `gemma-4-E2B-int8` checkpoint was regenerated after fixing bf16
     quantization. It now has 582 rowwise-int8 tensors and 582 scale tensors;
     `load_pretrained(..., device="METAL")` installs 525 runtime int8 linears.
   - The new rows are valid no-fallback TinyJit replay rows:
     20.063766 tok/s for 50/4, 15.751897 tok/s for 200/20, and 10.721864
     tok/s for 1000/20 with `rollout_jit_count=999`.
   - This proves compressed weights are flowing through matmul, but the
     long-context floor still decays far below 50 tok/s. The next target should
     profile post-int8 full-attention/norm cost rather than returning to
     logits-only changes.
   - Post-int8 profile at context length 700 measured 96.651 ms over 4807
     kernels: other 31.712 ms, MLP 22.190 ms, norm 16.797 ms, attention
     15.262 ms, logits/argmax 3.727 ms, and per-layer embedding 3.016 ms.
   - A conditional pre-embed experiment reduced the synthetic profile to
     89.465 ms but regressed the real 200/20 row to 11.272605 tok/s, so do not
     move token embedding/per-layer input prep out of the conditional JIT.
     Prefer a replay-local MLP/RowwiseInt8Linear kernel-volume reduction next.

8. Use the post-window JIT profile before guessing.
   - `scripts/profile_decode_jit.py` profiles a synthetic post-window decode
     token with zeroed KV cache and `JIT=2` so the captured TinyJit can be
     timed per kernel category.
   - The latest fused-MLP profile at context length 700 measured 110.302 ms
     total: MLP 38.938 ms, attention 26.236 ms, norm 13.447 ms,
     logits/argmax 1.793 ms. Logits are not the current primary limiter.
   - The same profile records that the local `gemma-4-E2B-int8` checkpoint has
     an `int8` manifest with 0 quantized tensors and 2011 raw bfloat16 tensors.
     Treat the directory name as storage labeling, not runtime int8 proof.

9. Prefill only the logits needed for generation.
   - Generation needs the last prompt logits, but the current prefill path still
     computes logits for every prompt position.
   - This matters more for long text prompts and multimodal inputs.

10. Cache or simplify decode masks and RoPE work.
   - For query length 1, full causal layers often need no explicit mask.
   - Sliding layers should not rebuild a full-position mask when the window is
     already cropped.

## Gates To Keep

- `env DEBUG=1 .venv/bin/python scripts/benchmark_gemma4_matrix.py --sizes E2B --formats int8 --devices METAL --beams 0 --max-new-tokens 2 ...`
  - Must show `JIT captured`.
  - Must not show an unbounded recurring KV-cache `CACHE MISS` chain.

- `env DEBUG=0 .venv/bin/python scripts/benchmark_gemma4_matrix.py --sizes E2B --formats int8 --devices METAL --beams 0 --max-new-tokens 1000 ...`
  - Older local result: 45.169393 s, 22.138885 tok/s.
  - Current shorter proof after the public-assign fix:
    `--max-new-tokens 50 --decode-warmup-tokens 4` measured 14.284291 tok/s
    with `decode_fallback=false`.
  - Current grouped-query proof:
    `--max-new-tokens 200 --decode-warmup-tokens 20` measured 10.659780 tok/s
    with `decode_fallback=false`.
  - Current 1000-token proof:
    `--max-new-tokens 1000 --decode-warmup-tokens 20` measured 6.245549 tok/s
    with `decode_fallback=false` and `rollout_jit_count=999`.
  - Current conditional sliding-JIT proof:
    `--max-new-tokens 1000 --decode-warmup-tokens 20` measured 8.300076 tok/s
    with `decode_fallback=false` and `rollout_jit_count=999`.
  - Current fused-MLP proof:
    `--max-new-tokens 1000 --decode-warmup-tokens 20` measured 9.303522 tok/s
    with `decode_fallback=false` and `rollout_jit_count=999`.
  - Current runtime-int8-matmul proofs:
    `--max-new-tokens 50 --decode-warmup-tokens 4` measured 20.063766 tok/s
    with `decode_fallback=false` and `rollout_jit_count=49`;
    `--max-new-tokens 200 --decode-warmup-tokens 20` measured 15.751897 tok/s
    with `decode_fallback=false` and `rollout_jit_count=199`;
    `--max-new-tokens 1000 --decode-warmup-tokens 20` measured 10.721864
    tok/s with `decode_fallback=false` and `rollout_jit_count=999`.
  - Current post-int8 profile proof:
    `scripts/profile_decode_jit.py --context-length 700` measured 96.651 ms
    and is recorded in
    `benchmarks/gemma4-metal-postwindow-jit-profile-int8matmul-current.json`.

- Full unit tests:
  - Current result: `35 passed, 1 skipped, 2 warnings in 39.60s`.

## Sources

- tinygrad speed guide: https://docs.tinygrad.org/developer/speed/
- tinygrad runtime docs: https://docs.tinygrad.org/developer/runtime/
- tinygrad overview: https://docs.tinygrad.org/
- Hugging Face Gemma4 docs: https://huggingface.co/docs/transformers/model_doc/gemma4
- Hugging Face cache explanation: https://huggingface.co/docs/transformers/en/cache_explanation
- Local performance envelope: `benchmarks/gemma4-metal-speed-targets.md`
