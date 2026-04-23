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
