# tinygrad-gemma Optimization Survey & Research

Date: 2026-04-23
Author: Meridian (Hermes Agent)
Context: Survey of `/Users/ericfode/Downloads/tinygrad-gemma` after model switch to moonshotai/kimi-k2.6

---

## 1. Repo Survey

**Path**: `/Users/ericfode/Downloads/tinygrad-gemma`

**What it is**: A native tinygrad implementation of Google Gemma 4 (E2B, E4B, 26B-A4B MoE, 31B). It loads Hugging Face `safetensors` directly and runs text, vision, and audio towers without PyTorch or GGUF.

**Core source**:
- `model.py` (662 lines) — TextScaledEmbedding, RMSNorm, GemmaMLP, GemmaAttention, GemmaDecoderLayer, GemmaModel, GemmaForCausalLM
- `multimodal.py` (805 lines) — Vision/audio towers, processor
- `config.py` (370 lines) — Config dataclasses with Gemma 4 specifics
- `loader.py` / `quantization.py` — HF checkpoint loading, row-wise int8 storage
- `runtime.py` — METAL device detection and patching
- `scripts/benchmark_gemma4_matrix.py` — Matrix benchmark harness

**Implemented architecture features**:
- Per-Layer Embeddings (PLE) for edge models
- Shared KV cache (last N layers reuse KV from earlier layers)
- Mixed sliding-window / full attention
- GQA (2:1 local, 8:1 global)
- K=V for global attention
- p-RoPE (partial rotary: 25% of dims for global, 100% for local)
- MoE router + experts

**Current performance** (M5 Max, METAL):
- E2B int8 decode: ~0.11 tok/s
- 1000-token long-run: ~47 min without reaching 100 tokens
- Target gates: >=10 tok/s (first usable), >=50 tok/s (competitive floor)

The gap between proof-of-life and usable inference is roughly **two orders of magnitude**.

---

## 2. Found Bottlenecks (from code inspection)

### A. TinyJit is recreated on every `generate()` call
In `GemmaForCausalLM.generate()`:
```python
rollout_jit = TinyJit(lambda token, start_pos: self._rollout_next_token(token, start_pos, cache, temperature))
```
This is instantiated *inside* the method. Every call retraces and recompiles the entire decode graph. JIT compilation overhead is paid per generation, not once per model.

### B. Synchronous `.realize()` inside the attention hot path
```python
entry.key[:, :, past_seen_tokens:end_pos, :].assign(k).realize()
entry.value[:, :, past_seen_tokens:end_pos, :].assign(v).realize()
```
`.realize()` forces device synchronization and materialization on every single token during autoregressive decode. This flattens the scheduler and destroys pipelining.

### C. Quantization is storage-only, not inference-optimized
`int8` checkpoints are dequantized back to float at load time. The model runs full-precision matmuls. On a memory-bandwidth-bound machine like Apple Silicon, this wastes ~50% of effective weight bandwidth.

### D. `repeat_kv` materializes duplicated heads
```python
key_states = repeat_kv(k, self.num_key_value_groups)
```
`repeat_interleave` physically copies KV head data. For GQA with 8:1 grouping, this multiplies KV-cache read bandwidth by 8x during attention.

### E. Attention mask rebuilt every forward pass
Even for decode-step `query_len=1` with sliding window, the mask is reconstructed from `Tensor.arange` each time. It should be pre-allocated and sliced.

### F. `start_pos` Variable + manual cache length tracking
The JIT boundary passes a `Variable`, but then Python manually mutates `cache.past_seen_tokens` and calls `cache.set_active_length()`. Python-side state mutations outside the JITted graph prevent the scheduler from seeing the full decode step as a single fused unit.

### G. `.contiguous()` inside the token loop
```python
next_token = rollout_jit(next_token.reshape(1, 1).contiguous(), start_var)
```
A memory copy per token to satisfy JIT buffer requirements.

### H. RMSNorm upcasts to float32 every call
```python
output = x.float()
output = output * (output.square().mean(-1, keepdim=True) + self.eps).rsqrt()
```
On Metal with bf16 weights, this is extra memory traffic and compute that could stay in lower precision.

---

## 3. Novel Optimization Directions

### 1. Persistent JIT with Pre-allocated KV Ring Buffers
Create the `TinyJit` **once** at model initialization, not per `generate()`. Pre-allocate KV caches to `max_length` as contiguous buffers. Use `Tensor.assign()` **without** `.realize()` so the scheduler treats cache updates as internal graph edges. This alone should account for a **10-50x** reduction in per-token Python and compile overhead.

### 2. Weight-Only INT8 Fused Matmul
Keep weights in int8 + scales at runtime. Instead of dequantizing at load, express matmuls as:
```python
# conceptually; tinygrad scheduler can fuse this
dequant_weight = qweight.cast("float") * scale.reshape(-1, 1)
output = input @ dequant_weight
```
If the scheduler does not fuse this automatically, a custom tinygrad op or explicit kernel fusion can force a single kernel that loads int8 weights, expands to float, and multiplies. This doubles effective weight bandwidth on Metal.

### 3. GQA Broadcasted Attention (avoid `repeat_interleave`)
Do not materialize repeated KV heads. Reshape `k` to `(batch, kv_heads, 1, seq_len, head_dim)` and `q` to `(batch, kv_heads, num_groups, seq_len, head_dim)`, then perform a batched matmul that broadcasts over the group dimension. This avoids duplicating the KV cache and saves both memory and bandwidth.

### 4. Fused p-RoPE Kernel
Gemma 4 global layers only rotate 25% of head dimensions. The current code splits, rotates, and concatenates. A custom kernel (or even a tinygrad expression that only touches the rotated slice) would save ~75% of RoPE memory traffic on global layers.

### 5. Pre-computed Banded Mask + Sliced Lookup
Build the causal+sliding-window mask once as a `(max_length, max_length)` tensor. During decode with `query_len=1`, slice `mask[:, past_seen_tokens:end_pos]` instead of rebuilding from arange. This removes several small kernel launches per token.

### 6. KV-Cache Quantization / Compression
Recent work (TurboQuant, FoveatedKV) shows 2-4x KV-cache compression on Apple Silicon with minimal quality loss. Since Gemma 4 already shares KV across layers, compressing the shared cache is disproportionately valuable. Even symmetric int8 per-channel KV cache would halve KV memory traffic.

### 7. Speculative Decoding
Use a tiny draft model (or even the same model with a shallow head) to predict 2-4 tokens ahead, then verify in parallel with the main model. For Gemma 4 E2B, a smaller n-gram or bigram draft could be sufficient because the target model itself is small.

### 8. JIT the Prefill Path Separately
The initial prompt processing (`forward_ids` on the full prompt) is not JITted. Prefill is compute-bound and benefits from kernel fusion. A separate `TinyJit` with a symbolic `seq_len` Variable would accelerate long multimodal prompts.

### 9. `BEAM` Cache Warm-up + `DEBUG=2` Profiling
Run `DEBUG=2` and `PROFILE=1` on a 10-token decode to see actual kernel times. tinygrad's BEAM cache lives in `~/.cache/tinygrad/`. Warm it with `BEAM=2` or `BEAM=3` overnight, then run production inference with cached kernels. For small-batch decode on Metal, `BEAM=0` is sometimes faster than `BEAM>0` because search overhead dominates; measure rather than assume.

### 10. Fused MLP + Norm via Graph Scheduling
Eliminate intermediate `.realize()` calls between attention, layernorm, and MLP. In tinygrad, returning a single expression tree from `GemmaDecoderLayer.__call__` (without internal realizes) lets the scheduler fuse norms, activations, and projections into fewer kernels. The current code is already fairly fused, but any `.float()` or `.cast()` inside `RMSNorm` or `apply_activation` can split the graph.

---

## 4. Prioritized Impact Estimate

| Optimization | Difficulty | Expected Speedup | Notes |
|---|---|---|---|
| Persistent JIT (fix #1) | Low | **10-50x** | One-line structural fix; biggest immediate win |
| Remove `.realize()` in KV update | Low | **2-5x** | Lets scheduler pipeline |
| Pre-allocate contiguous token buffer | Low | **1.2-1.5x** | Removes `.contiguous()` copy |
| Weight-only int8 inference | Medium | **1.5-2x** | Critical for Apple Silicon bandwidth |
| GQA broadcast (no repeat_interleave) | Medium | **1.3-2x** | Reduces KV bandwidth, esp. on global layers |
| Fused p-RoPE / banded mask | Low-Medium | **1.1-1.3x** | Removes small kernel overhead |
| KV-cache quantization | Medium | **1.2-1.5x** | More impact at long context |
| Speculative decoding | High | **2-3x** | Requires draft model infrastructure |
| Prefill JIT | Low | **2-5x** on long prompts | Separate path, easy to add |

The first four items alone are likely sufficient to move E2B from **0.11 tok/s** into the **10-50 tok/s** range.

---

## 5. References

- tinygrad JIT docs: https://mintlify.wiki/tinygrad/tinygrad/concepts/jit
- tinygrad Speed docs: https://docs.tinygrad.org/developer/speed/
- tinygrad Performance Optimization: https://mintlify.wiki/tinygrad/tinygrad/advanced/performance
- tinygrad Environment Variables: https://docs.tinygrad.org/env_vars/
- Gemma 4 Visual Guide: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
- Gemma 4 Architecture Deep Dive: https://machine-learning-made-simple.medium.com/googles-gemma-4-is-weirder-than-you-realize-17d00d95b0d5
- TurboQuant / KV Cache Compression on Apple Silicon: https://medium.com/data-science-collective/turboquant-compressing-kv-cache-4x-on-apple-silicon-how-i-doubled-the-usable-context-length-5a8bce975fe2
- FoveatedKV (2x KV compression): https://www.reddit.com/r/LocalLLaMA/comments/1s1xbv6/foveatedkv_2x_kv_cache_compression_on_apple/
- Speculative Decoding paper: https://openreview.net/forum?id=H-VlwsYvVi
