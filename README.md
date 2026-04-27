# tinygrad-gemma

Native tinygrad implementation of Gemma 4.

It loads standard Hugging Face `config.json` plus `.safetensors` checkpoints from a local model directory and runs Gemma 4 generation without GGUF or a PyTorch runtime. Full Gemma 4 text checkpoints load directly, and multimodal checkpoints now run through the native tinygrad text, vision, and audio towers.

## Scope

Implemented:

- Gemma 4 only
- Official Gemma 4 size configs: E2B, E4B, 26B A4B, and 31B
- Gemma 4 text decoder stack, including per-layer embeddings, Q/K/V norms, mixed sliding/full attention, KV sharing, and MoE-capable config parsing
- Native Gemma 4 multimodal loading for text, vision, and audio towers
- Hugging Face safetensor loading
- Fine-tuning surfaces for shifted-label loss, optimizer wiring, selective freezing, resumable checkpoint save/load, and quantized-checkpoint reload into trainable weights
- KV-cache generation
- Hugging Face `tokenizer.json` and SentencePiece tokenizer support
- Multimodal prompt preprocessing with image placeholder expansion and WAV audio placeholder expansion
- tinygrad `METAL` execution path on Apple Silicon when this process can open a Metal device
- tinygrad `BEAM` control from the CLI, including `--beam max` as the widest production beam this package supports by default

Non-goals for this package:

- distributed training orchestration

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .[dev,tokenizer,multimodal]
```

## Run

Text-only prompt:

```bash
. .venv/bin/activate
tinygrad-gemma \
  --model-dir /path/to/google-gemma-4-checkpoint \
  --prompt "Write a haiku about tinygrad." \
  --device auto \
  --beam max \
  --max-new-tokens 64
```

Image prompt:

```bash
. .venv/bin/activate
tinygrad-gemma \
  --model-dir /path/to/google-gemma-4-checkpoint \
  --prompt "Describe <|image|>." \
  --image /path/to/input.png \
  --max-new-tokens 32
```

Image plus audio prompt:

```bash
. .venv/bin/activate
tinygrad-gemma \
  --model-dir /path/to/google-gemma-4-checkpoint \
  --prompt "Describe <|image|> and transcribe <|audio|>." \
  --image /path/to/input.png \
  --audio /path/to/input.wav \
  --max-new-tokens 32
```

Point `--model-dir` at the normal Gemma 4 Hugging Face checkpoint directory. `--audio` currently expects WAV input. The multimodal CLI expands `<|image|>` and `<|audio|>` into the soft-token spans required by the native towers.

`--device auto` resolves to the first usable tinygrad backend and prefers `METAL` on Apple when this process can actually open a Metal device. This repo does not use Apple's separate MLX runtime. `--beam max` maps to this package's widest production beam by default.

If tokenizer support is not installed, you can still drive the model with raw token ids:

```bash
. .venv/bin/activate
tinygrad-gemma \
  --model-dir /path/to/google-gemma-4-checkpoint \
  --prompt-tokens 2,106,1234 \
  --device auto \
  --beam max \
  --max-new-tokens 16
```

On CPU and Python backends, nonzero beam settings automatically set tinygrad `PARALLEL` to the local CPU count if it was unset. The default `--beam max` value is `4`; override it with `TINYGRAD_GEMMA_MAX_BEAM`.

## Gemma 4 Matrix Workflow

Download the official Hugging Face checkpoints for every supported Gemma 4 size:

```bash
. .venv/bin/activate
python scripts/download_gemma4_matrix.py --sizes E2B E4B 26B-A4B 31B
```

Create repo-native row-wise int8 checkpoints beside the downloaded bf16 checkpoints:

```bash
. .venv/bin/activate
python scripts/quantize_gemma4_matrix.py --sizes E2B E4B 26B-A4B 31B
```

Run a cheap Metal load/generate preflight across every size and native format:

```bash
. .venv/bin/activate
python scripts/benchmark_gemma4_matrix.py \
  --sizes E2B E4B 26B-A4B 31B \
  --formats bf16 int8 \
  --devices METAL \
  --beams 1 \
  --max-new-tokens 1 \
  --out benchmarks/gemma4-metal-preflight.csv
```

Run a long Metal sample. This generates exactly 1000 new tokens because the benchmark disables EOS stopping, and it writes progress to `benchmarks/gemma4-metal-1000.csv.progress.jsonl` while the row is running:

```bash
. .venv/bin/activate
python scripts/benchmark_gemma4_matrix.py \
  --sizes E2B \
  --formats int8 \
  --devices METAL \
  --beams 1 \
  --max-new-tokens 1000 \
  --out benchmarks/gemma4-metal-1000.csv \
  --progress-every 100
```

The machine-specific speed envelope and local optimization gates are recorded in `benchmarks/gemma4-metal-speed-targets.md`. The short version: optimized Apple Silicon runtimes should reach tens to 100+ tokens/sec on this M5 Max depending on model size, while this repo's current tinygrad decode path is still below the first usable long-run target.

For same-session baseline/candidate comparisons, use the paired decode benchmark helper. It accepts a baseline target, a candidate target, and forwards benchmark arguments after `--`; when the default evo benchmark is absent from main, it resolves it from the baseline evo worktree. Replace `exp_0013` with the candidate experiment worktree you want to compare:

```bash
. .venv/bin/activate
python scripts/paired_e2b_decode_benchmark.py \
  --baseline-target .evo/run_0000/worktrees/exp_0005/tinygrad_gemma/model.py \
  --candidate-target .evo/run_0000/worktrees/exp_0013/tinygrad_gemma/model.py \
  --out benchmarks/paired-exp0005-exp0013-hash16.json \
  --label exp0005-vs-exp0013-hash16 \
  -- \
  --max-new-tokens 16 \
  --decode-warmup-tokens 4 \
  --min-score 5.0
```

The helper writes both child benchmark payloads plus absolute/relative score deltas. The child benchmark stdout must be JSON containing a finite numeric `score`; `NaN` and infinities are rejected rather than recorded. Add `--min-delta <float>` before `--` to make the helper exit nonzero after writing the JSON artifact when the candidate delta is below a required floor. Treat very short paired runs as smoke tests; throughput claims still require durable benchmark artifacts under `benchmarks/` and the longer evo gates.

When local benchmark/profile artifacts accumulate, inventory them before cleanup:

```bash
python scripts/inventory_untracked_artifacts.py \
  --output docs/plans/$(date +%F)-untracked-artifact-inventory.md
```

The inventory helper is read-only: it reports untracked paths, sizes, categories, tracked-doc references, and suggested dispositions. Its main table is reference-ranked: artifacts cited by tracked docs/state/config appear before uncited generated files, and higher reference counts sort first. Use that ranking to review likely evidence-to-keep candidates deliberately; it is not a license to auto-commit them. For downstream scripts, add `--format json`; stdout remains machine-readable JSON and the human `inventoried N untracked path(s)` summary stays on stderr. The JSON payload includes `untracked_count`, `total_size`, `referenced_count`, `counts_by_category`, `counts_by_suffix`, and ranked `rows` with per-row `reference_count`. Use repeatable `--only-category <category>` for focused review, for example `--format json --only-category benchmark-progress-log` to inspect progress logs without shell-side filtering. It does not delete artifacts, edit ignore files, stage files, or decide which benchmark evidence should travel with the repo. A small tool that knows it is not a broom is a civilized thing.

The full beam/format matrix is intentionally resumable because the large checkpoints and higher beams can take a long time on local Metal:

```bash
. .venv/bin/activate
python scripts/benchmark_gemma4_matrix.py \
  --sizes E2B E4B 26B-A4B 31B \
  --formats bf16 int8 \
  --devices METAL \
  --beams 1 2 3 4 \
  --max-new-tokens 1000 \
  --out benchmarks/gemma4-metal-matrix-1000.csv \
  --resume \
  --progress-every 100
```

## API

```python
from tinygrad_gemma import GemmaCache, GemmaMultimodalProcessor, load_pretrained

model = load_pretrained("/path/to/model", device="CPU")
processor = GemmaMultimodalProcessor.from_pretrained("/path/to/model")
prepared = processor.prepare_inputs("Describe <|image|>.", images=["/path/to/image.png"])
cache = GemmaCache.empty(model.config.text_config.num_hidden_layers if hasattr(model.config, "text_config") else model.config.num_hidden_layers)
logits, cache = model.forward_ids(
  prepared.input_ids,
  cache=cache,
  pixel_values=prepared.pixel_values,
  image_position_ids=prepared.image_position_ids,
  input_features=prepared.input_features,
  input_features_mask=prepared.input_features_mask,
)
```

Fine-tuning:

```python
from tinygrad_gemma import (
  GemmaTrainingBatch,
  build_optimizer,
  load_pretrained,
  save_training_checkpoint,
  train_step,
)

model = load_pretrained("/path/to/model", device="CPU")
optimizer = build_optimizer(model, optimizer="adamw", lr=1e-5, weight_decay=0.01)
loss = train_step(model, optimizer, GemmaTrainingBatch(input_ids=[2, 106, 1234, 99]))
save_training_checkpoint(model, "/tmp/gemma4-finetune-step", optimizer=optimizer, training_metadata={"step": 1})
```

For multimodal fine-tuning, pass the processor outputs directly into `GemmaTrainingBatch`. If labels are omitted, the training helpers build shifted next-token labels automatically and ignore `pad`, `<|image|>`, and `<|audio|>` target tokens by default.

Quantized checkpoints:

```python
from tinygrad_gemma import build_optimizer, load_pretrained, save_training_checkpoint

save_training_checkpoint(model, "/tmp/gemma4-int8", quantize="int8")
reloaded = load_pretrained("/tmp/gemma4-int8", device="CPU")
optimizer = build_optimizer(reloaded, optimizer="adamw", lr=1e-5, weight_decay=0.0)
```

The quantized checkpoint format is repo-native and intentionally narrow: floating-point matrix-like weights are stored as symmetric row-wise `int8` plus scales in safetensors, and `load_pretrained` dequantizes them back into ordinary tinygrad tensors so the same optimizer and training path keeps working.

The implementation is intentionally narrow and Gemma 4 only. E2B and E4B support text, image, and audio towers; 26B A4B and 31B support text plus image towers and use the Gemma 4 large-model vision attention mask. The strong gate here is correctness against deterministic reference tests, cache/full-forward equivalence on tiny configs, training-step/save-load roundtrips, followed by real-checkpoint loader, tokenizer, and CLI smoke runs.
