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
22 tok/s 1000-token note is stale for the current code path; the latest accepted
fused rowwise-int8 gate/up row measures 10.863932 warmup-excluded tok/s and
still shows a large gap to the 50 tok/s target.

Update on 2026-04-24: a raw Metal rowwise-int8 gate/up primitive measured
0.123437 ms median in isolation. Direct TinyJit decode wiring was initially
rejected because the raw `MetalProgram` launch replayed stale capture-time
output. The primitive now appends a repo-local custom `Runner` `ExecItem` during
capture. `benchmarks/gemma4-metal-runtime-bridge-tinyjit-diagnostic-current.json`
records `safe_for_decode_tinyjit_replay` for both `float32` and `bfloat16`
inputs under `JIT=1` and `JIT=2`. This still does not update the Gemma tok/s
row until it is wired into the MLP path.

Second update on 2026-04-24: wiring that replay-safe raw Metal primitive into
the real Gemma MLP path was correct but slower. The focused MLP diagnostic
passes against the existing fused rowwise-int8 path, but the real E2B int8
`METAL` 200/20 row regressed to 6.609982 tok/s with `decode_fallback=false`.
The raw path is now opt-in only; default decode remains on the existing
tinygrad fused rowwise-int8 matmul path.

Update on 2026-04-25: the raw Metal gate/up regression is now explained. Under
`JIT=1`, the default post-window decode capture has 4807 `CompiledRunner` items
and graph-batches into 8 `MetalGraph` launches, measuring 67.716500 ms for the
profiled token. With the opt-in raw gate/up path, the capture grows to 13657
`CompiledRunner` items plus 35 custom `RowwiseInt8DecodeLinearRunner` items and
fragments into 37 `MetalGraph` batches plus 35 raw runner launches, measuring
197.940124 ms. The raw kernel is replay-safe, but the Python custom `Runner` is
not graphable in tinygrad's Metal graph path.

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
| Runtime int8 matmul, beam=0, 1000 tokens, 20-token warmup | 10.721864 measured tok/s, `rollout_jit_count=999`, `decode_fallback=false` |
| Fused rowwise-int8 gate/up, beam=0, 1000 tokens, 20-token warmup | 10.863932 measured tok/s, `rollout_jit_count=999`, `decode_fallback=false` |
| Diagnostic: raw Metal runtime TinyJit bridge | `safe_for_decode_tinyjit_replay`; no throughput row superseded |
| Rejected raw Metal MLP gate/up, beam=0, 200 tokens, 20-token warmup | 74.828046 s total, 6.609982 measured tok/s, `rollout_jit_count=199`, `decode_fallback=false` |
| Diagnostic: raw gate/up graph fragmentation, post-window token at context 703 | default 8 `MetalGraph` batches / 67.716500 ms; raw 37 `MetalGraph` batches + 35 raw runners / 197.940124 ms |
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
   - Fusing rowwise-int8 MLP gate/up was a small real-row win despite a worse
     synthetic profile. The 200/20 row moved to 16.025808 tok/s and the
     1000/20 row moved to 10.863932 tok/s, both with `decode_fallback=false`.
     The synthetic profile rose to 97.738 ms, so keep benchmark truth above
     profiler intuition when they disagree.
   - The Metal int8 lowering diagnostic is now explicit in
     `benchmarks/gemma4-metal-int8-matmul-lowering-diagnostic-current.json`.
     For the exact one-token E2B gate/up shape, rowwise int8 captured one
     kernel, did not emit `simdgroup_multiply_accumulate`, and measured
     0.185354 ms median over 20 diagnostic replays versus 0.146021 ms for the
     bf16 dequantized-weight control. tinygrad's visible Metal tensor-core
     registry contains float/half/bfloat16 entries and no int8 entry. This
     makes a lower-level kernel path, not another Python reshape, the credible
     route to a step change.
   - The first raw Metal prototype is in
     `benchmarks/gemma4-metal-rowwise-int8-linear-prototype-current.json`.
     It compiles through tinygrad's Metal runtime and runs against tinygrad
     buffers. The threadgroup-x variant stages the decode hidden vector in
     threadgroup memory and measured 0.087417 ms median over 100 replays for
     the `(1536) @ (12288, 1536).T` gate/up shape, with
     `max_abs_error=0.00023651123046875`. This is a viable lower-level path,
     but not a Gemma throughput row yet.
   - The raw Metal path is now a reusable repo primitive in
     `tinygrad_gemma/metal_int8.py`. The refreshed prototype artifact includes
     a tensor-wrapper module check with output shape `[1, 1, 12288]`,
     `max_abs_error=0.00023651123046875`, and `passed=true`. The current
     threadgroup-x timing is 0.123437 ms median over 100 replays. The next
     useful row must come from wiring this guarded primitive into decode.

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
  - Current fused-rowwise-int8-gate-up proofs:
    `--max-new-tokens 200 --decode-warmup-tokens 20` measured 16.025808 tok/s
    with `decode_fallback=false` and `rollout_jit_count=199`;
    `--max-new-tokens 1000 --decode-warmup-tokens 20` measured 10.863932
    tok/s with `decode_fallback=false` and `rollout_jit_count=999`.
  - Current Metal int8 lowering diagnostic:
    `scripts/diagnose_metal_int8_matmul.py --repeats 20` writes
    `benchmarks/gemma4-metal-int8-matmul-lowering-diagnostic-current.json` and
    records that the exact one-token rowwise-int8 decode shape does not use
    Metal simdgroup MMA.
  - Current raw Metal rowwise-int8 prototype:
    `scripts/prototype_metal_rowwise_int8_linear.py --repeats 100` writes
    `benchmarks/gemma4-metal-rowwise-int8-linear-prototype-current.json` and
    records a correct threadgroup-x kernel at 0.123437 ms median for the E2B
    gate/up shape. This does not update the accepted tok/s row until it is
    wired into decode replay.
  - Current Metal rowwise-int8 runtime bridge:
    the same prototype artifact now verifies
    `tinygrad_gemma.metal_int8.metal_rowwise_int8_decode_linear` with
    `module_check.passed=true` and output shape `[1, 1, 12288]`.
  - Current Metal runtime TinyJit bridge diagnostic:
    `scripts/diagnose_metal_runtime_tinyjit_bridge.py --calls 4` writes
    `benchmarks/gemma4-metal-runtime-bridge-tinyjit-diagnostic-current.json`
    and records `summary.status=safe_for_decode_tinyjit_replay`. The bridge
    appends a repo-local custom `Runner` `ExecItem` during capture. Under both
    `JIT=1` and `JIT=2`, `float32` captures one runner item and `bfloat16`
    captures the normal cast plus the runner item.
  - Current raw Metal MLP gate/up diagnostic and rejected row:
    `scripts/diagnose_metal_mlp_runtime_gate_up.py` writes
    `benchmarks/gemma4-metal-mlp-runtime-gate-up-diagnostic-current.json` and
    records `summary.status=safe_for_decode_tinyjit_replay` against the
    existing fused rowwise-int8 path, but the real E2B int8 200/20 row in
    `benchmarks/gemma4-metal-runtime-gate-up-200-current.csv` measured only
    6.609982 tok/s. Keep `TINYGRAD_GEMMA_METAL_INT8_GATE_UP` opt-in until the
    custom runner is graph-compatible or replaced.
  - Current raw gate/up graph diagnostic:
    `scripts/profile_decode_jit.py --jit-mode 1 --metal-int8-gate-up default`
    writes `benchmarks/gemma4-metal-decode-graph-default-current.json`; the
    default capture condenses from 4807 `CompiledRunner` items to 8
    `MetalGraph` batches and measures 67.716500 ms. The matching
    `--metal-int8-gate-up raw` artifact
    `benchmarks/gemma4-metal-decode-graph-raw-gate-up-current.json` captures
    13657 `CompiledRunner` items plus 35 raw runners, executes as 37
    `MetalGraph` batches plus 35 raw launches, and measures 197.940124 ms.
    This rejects the custom Python `Runner` as a default decode route.
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
