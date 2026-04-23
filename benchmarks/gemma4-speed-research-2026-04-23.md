# Gemma 4 Speed Research

Date: 2026-04-23

Repo: `/Users/ericfode/Downloads/tinygrad-gemma`

Target checked here: E2B int8 on tinygrad `METAL`.

## Bottom Line

The main bottleneck was not Metal math capacity. It was the autoregressive decode
path failing to enter a reusable tinygrad JIT replay soon enough. `DEBUG=1`
showed repeated cache misses and growing kernel schedules around the per-layer KV
cache writes in `tinygrad_gemma/model.py`.

Current `HEAD` has a TinyJit decode experiment in `generate()` and rollout-JIT
metrics in the benchmark script. It is a real improvement: E2B int8 now
completes the 1000-token Metal gate.

| Run | Result |
| --- | ---: |
| Older E2B int8, beam=0, 10 tokens | 86.676913 s, 0.115371 tok/s |
| Older E2B int8, beam=1, 10 tokens | 89.059974 s, 0.112284 tok/s |
| Current E2B int8, beam=0, 10 tokens | 21.295810 s, 0.469576 tok/s, `rollout_jit_count=9` |
| Current E2B int8, beam=0, 100 tokens | 23.905944 s, 4.183060 tok/s, `rollout_jit_count=99` |
| Current E2B int8, beam=0, 1000 tokens | 45.169393 s, 22.138885 tok/s, `rollout_jit_count=999` |
| Current E2B int8, beam=1, 1000 tokens | 90.487042-104.922802 s, 9.530817-11.051306 tok/s, `rollout_jit_count=999` |

`beam=1` is currently worse end-to-end because the BEAM compile/search cliff is
large. It does improve post-warmup token cadence in parts of the run, but the
interactive default should stay `beam=0` until compile cost is amortized or moved
out of the measured path. The repo-local beam=1 artifact is
`benchmarks/gemma4-metal-1000.csv`; an independent `/tmp` rerun measured
104.922802 s.

## DEBUG Evidence

`DEBUG=1` is the right instrument for the "jitter" question. In local tinygrad,
`TinyJit` prints `JIT captured ...` when the replay graph is captured; this is in
`/Users/ericfode/src/.tinygrad_research/tinygrad/engine/jit.py`.

Before the current JIT path is useful, the trace shows repeated cache misses at:

- `tinygrad_gemma/model.py:354`: `entry.key[:, :, past_seen_tokens:end_pos, :].assign(k).realize()`
- `tinygrad_gemma/model.py:355`: `entry.value[:, :, past_seen_tokens:end_pos, :].assign(v).realize()`

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

1. Keep the TinyJit decode direction, but make it first-class.
   - Move the closure-based rollout into an explicit decode runner/state object.
   - Own the preallocated cache buffers there.
   - Use a stable symbolic `cache_position`/`Variable`.
   - Add a warmup phase that captures the decode graph before measured generation.
   - Gate it with `DEBUG=1`: capture should happen by the third generated token,
     and no growing `CACHE MISS` chain should recur at the KV write lines.

2. Use `beam=0` for interactive inference and benchmark gates for now.
   - `beam=1` finished 1000 tokens, but only after a roughly 70-85 s first
     100-token cliff in local runs. It is an offline tuning setting until warmup
     is separated.

3. Implement real sliding-window KV behavior.
   - E2B has 35 layers: 28 sliding layers with `sliding_window=512` and 7 full
     attention layers.
   - Current attention still scores over the active cache length, then masks
     sliding layers.
   - Cropping or ring-buffering sliding layers should reduce both attention work
     and graph size as generation gets longer.

4. Avoid physical grouped-query KV repetition.
   - E2B has 8 attention heads and 1 KV head. `repeat_kv()` materializes an 8x
     expansion before attention.
   - Reshape/group the attention math so grouped-query attention does not copy K/V.

5. Treat repo `int8` as storage-only until runtime int8 exists.
   - `tinygrad_gemma/quantization.py` dequantizes checkpoint tensors back into
     ordinary tinygrad tensors at load.
   - Real speed from quantization needs compressed weights through matmul or a
     fused dequantize-matmul path.

6. Prefill only the logits needed for generation.
   - Generation needs the last prompt logits, but the current prefill path still
     computes logits for every prompt position.
   - This matters more for long text prompts and multimodal inputs.

7. Cache or simplify decode masks and RoPE work.
   - For query length 1, full causal layers often need no explicit mask.
   - Sliding layers should not rebuild a full-position mask when the window is
     already cropped.

## Gates To Keep

- `env DEBUG=1 .venv/bin/python scripts/benchmark_gemma4_matrix.py --sizes E2B --formats int8 --devices METAL --beams 0 --max-new-tokens 2 ...`
  - Must show `JIT captured`.
  - Must not show an unbounded recurring KV-cache `CACHE MISS` chain.

- `env DEBUG=0 .venv/bin/python scripts/benchmark_gemma4_matrix.py --sizes E2B --formats int8 --devices METAL --beams 0 --max-new-tokens 1000 ...`
  - Current result: 45.169393 s, 22.138885 tok/s.
  - This is now the minimum regression gate.

- Full unit tests:
  - Current result: `26 passed, 2 warnings in 51.17s`.

## Sources

- tinygrad speed guide: https://docs.tinygrad.org/developer/speed/
- tinygrad runtime docs: https://docs.tinygrad.org/developer/runtime/
- tinygrad overview: https://docs.tinygrad.org/
- Hugging Face Gemma4 docs: https://huggingface.co/docs/transformers/model_doc/gemma4
- Hugging Face cache explanation: https://huggingface.co/docs/transformers/en/cache_explanation
- Local performance envelope: `benchmarks/gemma4-metal-speed-targets.md`
