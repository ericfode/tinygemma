# Gemma 4 Metal Speed Targets

Date: 2026-04-23

Machine target: local Apple M5 Max MacBook Pro, 18-core CPU, 40-core GPU, 128 GB unified memory. Apple lists the 40-core M5 Max configuration at 614 GB/s memory bandwidth.

Purpose: define the performance envelope this repo is trying to reach. These are not claims about the current tinygrad implementation. They are external optimized-runtime targets plus local repo gates.

## External Optimized-Runtime Envelope

These are the useful comparison numbers for this machine class when Gemma 4 is run through mature Apple Silicon paths such as MLX, oMLX, Ollama, or llama.cpp with Metal.

| Model and runtime class | Expected generation speed | Evidence |
| --- | ---: | --- |
| Gemma 4 E4B, Q4-class MLX | 100-130 tok/s | LLMCheck reports 128 tok/s on M5 Max 128 GB. |
| Gemma 4 26B-A4B MoE, Q4-class Ollama/llama.cpp Metal | 75-85 tok/s | Ollama issue report shows about 75 tok/s on M5 Max 128 GB. |
| Gemma 4 31B dense, 4-bit oMLX/Ollama/llama.cpp Metal | 15-25 tok/s | oMLX reports 21.1 tok/s at 4k context; the Ollama issue reports about 15 tok/s. |
| Gemma 4 31B dense, bf16/full precision class | single-digit tok/s | Treat as a soft planning number until this repo records a real long-run row. |

Sources:

- Apple M5 Max technical specs: https://support.apple.com/en-euro/126318
- Google Gemma 4 model card and architecture sizes: https://huggingface.co/google/gemma-4-E4B
- Ollama M5 Max Gemma 4 issue report: https://github.com/ollama/ollama/issues/15368
- oMLX Gemma 4 31B M5 Max benchmark: https://omlx.ai/benchmarks/ad20mcss
- LLMCheck Gemma 4 E4B M5 Max benchmark: https://llmcheck.net/models/gemma-4-e4b-on-m5-max/

## Current Repo Evidence

The repo has already proven load/generate coverage, not competitive decode speed.

| Local evidence | Current result |
| --- | --- |
| Full size/format Metal preflight | All four official Gemma 4 sizes and both native formats generate one token successfully. |
| E2B int8 Metal, 10-token beam comparison | About 0.11 tok/s for `beam=0` and `beam=1`. |
| E2B int8 Metal, 1000-token long attempt | About 47 minutes without reaching the first 100-token progress marker. |

The current bottleneck is therefore the repo's autoregressive tinygrad decode path, not the machine's ability to host the models.

## Local Gates

Use these gates to keep optimization work honest.

| Gate | Target | Why it matters |
| --- | ---: | --- |
| Load/generate preflight | All sizes, `bf16` and repo-native `int8`, `METAL`, one token | Correctness and device coverage. This is already passing in `gemma4-metal-preflight.csv`. |
| First usable E2B row | E2B int8, `METAL`, `beam=1`, 1000 tokens at >=10 tok/s | Establishes an interactive floor and proves the long-run harness can complete. |
| E2B competitive floor | E2B/E4B class, `METAL`, sustained >=50 tok/s | Shows the decode path is in the right order of magnitude for Apple Silicon. |
| 26B-A4B practical target | Sustained >=25 tok/s in the repo runtime | Conservative midpoint before comparing against the 75-85 tok/s optimized-runtime envelope. |
| 31B practical target | Sustained >=15 tok/s for a 4-bit-equivalent optimized path | Matches the lower bound of current external 31B Apple Silicon evidence. |
| Memory stability | 1000-token row completes without runaway RSS growth | Prevents a nominal tok/s win from hiding cache or graph-retention failure. |

Do not treat the external Q4 numbers as direct pass/fail gates for the current repo-native `int8` format. This repo's quantized checkpoint path currently reloads into ordinary tinygrad tensors, so it is a storage/checkpoint path first, not yet a fused low-bit inference path.

## Next Performance Increment

The next meaningful production increment is to make E2B int8 `METAL`, `beam=1`, `max_new_tokens=1000` complete with durable progress and at least 10 tok/s sustained generation. Anything below that is still proof-of-life, not usable local inference.
