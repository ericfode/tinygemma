# Gemma 4 Metal Long-Sample Notes

Date: 2026-04-23

Machine target: local Apple Metal through tinygrad `METAL`.

Completed evidence:

- `benchmarks/gemma4-metal-preflight.csv` passed for all four official Gemma 4 sizes and both native formats (`bf16`, repo-native `int8`) with `beam=1` and `max_new_tokens=1`.
- A short E2B int8 Metal comparison in `/tmp/tinygemma-beam-compare.csv` completed 10 generated tokens with identical output hashes for `beam=0` and `beam=1`.
- `beam=0`: 10 tokens in 86.676913 seconds, 0.115371 tokens/sec after a 32.885662 second load.
- `beam=1`: 10 tokens in 89.059974 seconds, 0.112284 tokens/sec after the same load.

Long-sample attempts:

- E2B int8, `METAL`, `beam=1`, `max_new_tokens=1000`, `progress_every=10` reached 10 tokens in 154.896696 seconds before being stopped for code instrumentation.
- E2B int8, `METAL`, `beam=1`, `max_new_tokens=1000`, `progress_every=100` ran for about 47 minutes without reaching the first 100-token progress marker and grew to roughly 21 GB RSS. It was stopped because the run was not producing durable benchmark progress on an interactive timescale.
- After adding a real TinyJit rollout path, E2B int8, `METAL`, `beam=1`, `max_new_tokens=1000` completed in 90.487042 seconds at 11.051306 tokens/sec. The row is in `benchmarks/gemma4-metal-1000.csv` and reports `rollout_jit_count=999`.

Rejected hot-path experiments:

- Fixed-size static KV cache was correct on toy tests but failed to reach 5 tokens in eight minutes on the real E2B int8 Metal path.
- Realizing/detaching cached K/V tensors inside attention was correct on toy tests but failed to reach 5 tokens in eight minutes on the real E2B int8 Metal path.
- Realizing/detaching cached K/V tensors after token realization was also slower than the plain dynamic cache path.

Current conclusion:

The native model loads and generates on Metal across the full official Gemma 4 size/native-format matrix, and the E2B int8 1000-token Metal gate now completes with TinyJit-backed rollout. The next production increment should separate JIT warmup from measured generation, run the full beam/format matrix, and reduce the remaining beam compile cliff.
