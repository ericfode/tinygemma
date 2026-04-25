from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
from tinygrad import Tensor, dtypes, nn

from tinygrad_gemma import (
  DEFAULT_IGNORE_INDEX,
  GemmaAudioConfig,
  GemmaCache,
  GemmaConditionalConfig,
  GemmaConfig,
  GemmaForCausalLM,
  GemmaForConditionalGeneration,
  GemmaMultimodalProcessor,
  GemmaTrainingBatch,
  GemmaVisionConfig,
  build_causal_labels,
  build_optimizer,
  causal_language_model_loss,
  load_pretrained,
  load_optimizer_state,
  load_quantization_manifest,
  load_text_config,
  save_training_checkpoint,
  set_trainable,
  supported_optimizers,
  supported_quantizations,
  train_step,
)
from tinygrad_gemma.cli import DEFAULT_MAX_BEAM, resolve_beam
from tinygrad_gemma.metal_int8 import metal_rowwise_int8_decode_linear
from tinygrad_gemma.model import GemmaMLP, build_attention_mask
from tinygrad_gemma.quantization import RowwiseInt8Linear, quantize_state_dict
from tinygrad_gemma.runtime import temporary_default_device
from tinygrad_gemma.tokenizer import GemmaTokenizer


def gelu_pytorch_tanh_np(x: np.ndarray) -> np.ndarray:
  return x * 0.5 * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def apply_activation_np(name: str, x: np.ndarray) -> np.ndarray:
  if name in ("gelu", "gelu_new"):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
  if name == "gelu_pytorch_tanh":
    return gelu_pytorch_tanh_np(x)
  if name == "silu":
    return x / (1.0 + np.exp(-x))
  raise ValueError(name)


def rms_norm_np(
  x: np.ndarray,
  weight: np.ndarray | None,
  eps: float,
  *,
  with_scale: bool = True,
  plus_one_scale: bool = False,
) -> np.ndarray:
  normed = x.astype(np.float32) * (np.mean(np.square(x.astype(np.float32)), axis=-1, keepdims=True) + eps) ** -0.5
  if not with_scale:
    return normed
  scale = (1.0 + weight.astype(np.float32)) if plus_one_scale else weight.astype(np.float32)
  return normed * scale


def rotate_half_np(x: np.ndarray) -> np.ndarray:
  half = x.shape[-1] // 2
  return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary_np(
  x: np.ndarray,
  position_ids: np.ndarray,
  head_dim: int,
  rope_theta: float,
  partial_rotary_factor: float = 1.0,
) -> np.ndarray:
  rot_dim = max(0, int(head_dim * partial_rotary_factor))
  rot_dim -= rot_dim % 2
  if rot_dim == 0:
    return x
  inv_freq = 1.0 / (rope_theta ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
  freqs = position_ids[..., None].astype(np.float32) * inv_freq.reshape(1, 1, -1)
  emb = np.concatenate([freqs, freqs], axis=-1)
  cos = emb[:, :, None, :]
  sin = emb[:, :, None, :]
  rot = x[..., :rot_dim]
  rest = x[..., rot_dim:]
  rotated = rot * np.cos(cos) + rotate_half_np(rot) * np.sin(sin)
  return np.concatenate([rotated, rest], axis=-1)


def linear_np(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
  return x @ weight.T


def build_mask_np(
  query_len: int,
  key_len: int,
  past_seen_tokens: int,
  sliding_window: int | None,
  *,
  causal: bool = True,
) -> np.ndarray | None:
  if query_len == 1 and sliding_window is None and causal:
    return None
  q_pos = np.arange(past_seen_tokens, past_seen_tokens + query_len, dtype=np.int32).reshape(query_len, 1)
  key_start = past_seen_tokens + query_len - key_len
  k_pos = np.arange(key_start, past_seen_tokens + query_len, dtype=np.int32).reshape(1, key_len)
  allowed = (k_pos <= q_pos) if causal else np.ones((query_len, key_len), dtype=bool)
  if sliding_window is not None:
    if causal:
      allowed &= k_pos >= (q_pos - sliding_window + 1)
    else:
      allowed &= np.abs(q_pos - k_pos) < sliding_window
  mask = np.where(allowed, 0.0, -np.inf).astype(np.float32)
  return mask.reshape(1, 1, query_len, key_len)


def repeat_kv_np(x: np.ndarray, n_rep: int) -> np.ndarray:
  if n_rep == 1:
    return x
  return np.repeat(x, n_rep, axis=1)


def attention_params_for(config: GemmaConfig, layer_idx: int) -> dict[str, float | int | bool]:
  assert config.layer_types is not None
  layer_type = config.layer_types[layer_idx]
  rope_params = (config.rope_parameters or {})[layer_type]
  is_sliding = layer_type == "sliding_attention"
  head_dim = config.head_dim if is_sliding else (config.global_head_dim or config.head_dim)
  use_k_eq_v = bool(config.attention_k_eq_v and not is_sliding)
  num_kv_heads = config.num_key_value_heads if (is_sliding or not use_k_eq_v) else (config.num_global_key_value_heads or config.num_key_value_heads)
  return {
    "layer_type": layer_type,
    "head_dim": head_dim,
    "num_kv_heads": num_kv_heads,
    "num_kv_groups": config.num_attention_heads // num_kv_heads,
    "rope_theta": float(rope_params.get("rope_theta", 10000.0)),
    "partial_rotary_factor": float(rope_params.get("partial_rotary_factor", 1.0)),
    "sliding_window": config.sliding_window if is_sliding else None,
    "scaling": 1.0,
    "use_k_eq_v": use_k_eq_v,
  }


def numpy_forward(config: GemmaConfig, params: dict[str, np.ndarray], input_ids: list[int]) -> np.ndarray:
  token_ids = np.array(input_ids, dtype=np.int32).reshape(1, -1)
  x = params["model.embed_tokens.weight"][token_ids] * (config.hidden_size ** 0.5)

  per_layer_inputs = None
  if config.hidden_size_per_layer_input:
    token_identity = params["model.embed_tokens_per_layer.weight"][token_ids].reshape(
      1, len(input_ids), config.num_hidden_layers, config.hidden_size_per_layer_input
    ) * (config.hidden_size_per_layer_input ** 0.5)
    context_projection = linear_np(x, params["model.per_layer_model_projection.weight"]) * (config.hidden_size ** -0.5)
    context_projection = context_projection.reshape(1, len(input_ids), config.num_hidden_layers, config.hidden_size_per_layer_input)
    context_projection = rms_norm_np(context_projection, params["model.per_layer_projection_norm.weight"], config.rms_norm_eps, plus_one_scale=False)
    per_layer_inputs = (context_projection + token_identity) * (2.0 ** -0.5)

  position_ids = np.arange(len(input_ids), dtype=np.int32).reshape(1, -1)
  for layer_idx in range(config.num_hidden_layers):
    prefix = f"model.layers.{layer_idx}."
    attn_params = attention_params_for(config, layer_idx)

    residual = x
    attn_in = rms_norm_np(x, params[prefix + "input_layernorm.weight"], config.rms_norm_eps, plus_one_scale=False)
    q = linear_np(attn_in, params[prefix + "self_attn.q_proj.weight"]).reshape(
      1, len(input_ids), config.num_attention_heads, int(attn_params["head_dim"])
    )
    q = rms_norm_np(q, params[prefix + "self_attn.q_norm.weight"], config.rms_norm_eps, plus_one_scale=False)
    q = apply_rotary_np(q, position_ids, int(attn_params["head_dim"]), float(attn_params["rope_theta"]), float(attn_params["partial_rotary_factor"]))
    q = np.transpose(q, (0, 2, 1, 3))

    raw_k = linear_np(attn_in, params[prefix + "self_attn.k_proj.weight"]).reshape(
      1, len(input_ids), int(attn_params["num_kv_heads"]), int(attn_params["head_dim"])
    )
    raw_v = raw_k if attn_params["use_k_eq_v"] else linear_np(
      attn_in, params[prefix + "self_attn.v_proj.weight"]
    ).reshape(1, len(input_ids), int(attn_params["num_kv_heads"]), int(attn_params["head_dim"]))
    k = rms_norm_np(raw_k, params[prefix + "self_attn.k_norm.weight"], config.rms_norm_eps, plus_one_scale=False)
    k = apply_rotary_np(k, position_ids, int(attn_params["head_dim"]), float(attn_params["rope_theta"]), float(attn_params["partial_rotary_factor"]))
    v = rms_norm_np(raw_v, None, config.rms_norm_eps, with_scale=False)
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    key = repeat_kv_np(k, int(attn_params["num_kv_groups"]))
    value = repeat_kv_np(v, int(attn_params["num_kv_groups"]))
    scores = np.matmul(q.astype(np.float32), np.swapaxes(key.astype(np.float32), -1, -2)) * float(attn_params["scaling"])
    mask = build_mask_np(
      len(input_ids),
      len(input_ids),
      0,
      attn_params["sliding_window"],
      causal=config.use_bidirectional_attention != "all",
    )
    if mask is not None:
      scores = scores + mask
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    attn_out = np.matmul(weights, value).transpose(0, 2, 1, 3).reshape(1, len(input_ids), -1)
    attn_out = linear_np(attn_out, params[prefix + "self_attn.o_proj.weight"])
    attn_out = rms_norm_np(attn_out, params[prefix + "post_attention_layernorm.weight"], config.rms_norm_eps, plus_one_scale=False)
    x = residual + attn_out

    residual = x
    mlp_in = rms_norm_np(x, params[prefix + "pre_feedforward_layernorm.weight"], config.rms_norm_eps, plus_one_scale=False)
    mlp = linear_np(
      apply_activation_np(config.activation_name, linear_np(mlp_in, params[prefix + "mlp.gate_proj.weight"])) *
      linear_np(mlp_in, params[prefix + "mlp.up_proj.weight"]),
      params[prefix + "mlp.down_proj.weight"],
    )
    mlp = rms_norm_np(mlp, params[prefix + "post_feedforward_layernorm.weight"], config.rms_norm_eps, plus_one_scale=False)
    x = residual + mlp

    if per_layer_inputs is not None:
      residual = x
      ple = linear_np(x, params[prefix + "per_layer_input_gate.weight"])
      ple = apply_activation_np(config.activation_name, ple)
      ple = ple * per_layer_inputs[:, :, layer_idx, :]
      ple = linear_np(ple, params[prefix + "per_layer_projection.weight"])
      ple = rms_norm_np(ple, params[prefix + "post_per_layer_input_norm.weight"], config.rms_norm_eps, plus_one_scale=False)
      x = residual + ple

    x = x * params[prefix + "layer_scalar"]

  x = rms_norm_np(x, params["model.norm.weight"], config.rms_norm_eps, plus_one_scale=False)
  logits = linear_np(x, params["lm_head.weight"])
  if config.final_logit_softcapping is not None:
    logits = np.tanh(logits / config.final_logit_softcapping) * config.final_logit_softcapping
  return logits


def randomize_model(model, seed: int = 0) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  shared: dict[int, np.ndarray] = {}
  params = {}
  for name, tensor in nn.state.get_state_dict(model).items():
    if tensor.shape:
      arr = shared.setdefault(id(tensor), rng.standard_normal(tensor.shape, dtype=np.float32) * 0.05)
    else:
      arr = shared.setdefault(id(tensor), np.array(rng.standard_normal() * 0.05, dtype=np.float32))
    params[name] = arr.astype(np.float32)
  nn.state.load_state_dict(
    model,
    {name: Tensor(value.item()) if value.shape == () else Tensor(value) for name, value in params.items()},
    strict=True,
    verbose=False,
  )
  return params


def make_config() -> GemmaConfig:
  return GemmaConfig(
    model_type="gemma4",
    vocab_size=48,
    hidden_size=12,
    intermediate_size=24,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    num_global_key_value_heads=1,
    head_dim=4,
    global_head_dim=6,
    hidden_activation="gelu_pytorch_tanh",
    hidden_size_per_layer_input=2,
    vocab_size_per_layer_input=48,
    max_position_embeddings=64,
    sliding_window=3,
    layer_types=["sliding_attention", "full_attention"],
    rope_parameters={
      "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
      "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.5, "rope_theta": 1000000.0},
    },
    attention_k_eq_v=True,
    final_logit_softcapping=4.0,
  )


def make_conditional_config() -> GemmaConditionalConfig:
  return GemmaConditionalConfig(
    text_config=GemmaConfig(
      model_type="gemma4_text",
      vocab_size=64,
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=1,
      num_attention_heads=4,
      num_key_value_heads=2,
      num_global_key_value_heads=2,
      head_dim=4,
      global_head_dim=4,
      hidden_activation="gelu_pytorch_tanh",
      hidden_size_per_layer_input=0,
      max_position_embeddings=64,
      sliding_window=3,
      layer_types=["full_attention"],
      rope_parameters={
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 1.0, "rope_theta": 1000000.0},
      },
      attention_k_eq_v=False,
      final_logit_softcapping=None,
    ),
    vision_config=GemmaVisionConfig(
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=1,
      num_attention_heads=4,
      num_key_value_heads=4,
      head_dim=4,
      patch_size=2,
      pooling_kernel_size=1,
      position_embedding_size=16,
      default_output_length=70,
    ),
    audio_config=GemmaAudioConfig(
      hidden_size=16,
      num_hidden_layers=1,
      num_attention_heads=4,
      subsampling_conv_channels=[4, 4],
      output_proj_dims=16,
    ),
    tie_word_embeddings=True,
  )


def official_size_config(
  *,
  hidden_size: int,
  intermediate_size: int,
  num_hidden_layers: int,
  num_attention_heads: int,
  num_key_value_heads: int,
  max_position_embeddings: int,
  sliding_window: int,
  audio: bool,
  moe: bool = False,
  num_experts: int | None = None,
  top_k_experts: int | None = None,
  moe_intermediate_size: int | None = None,
  attention_k_eq_v: bool = False,
  num_global_key_value_heads: int | None = None,
) -> dict:
  return {
    "model_type": "gemma4",
    "tie_word_embeddings": True,
    "audio_config": {"model_type": "gemma4_audio"} if audio else None,
    "vision_config": {"model_type": "gemma4_vision", "hidden_size": 768 if hidden_size < 2816 else 1152, "head_dim": 64 if hidden_size < 2816 else 72, "num_attention_heads": 12 if hidden_size < 2816 else 16, "num_key_value_heads": 12 if hidden_size < 2816 else 16, "standardize": hidden_size >= 2816, "use_clipped_linears": hidden_size < 2816},
    "text_config": {
      "model_type": "gemma4_text",
      "vocab_size": 262144,
      "hidden_size": hidden_size,
      "intermediate_size": intermediate_size,
      "num_hidden_layers": num_hidden_layers,
      "num_attention_heads": num_attention_heads,
      "num_key_value_heads": num_key_value_heads,
      "head_dim": 256,
      "global_head_dim": 512,
      "hidden_size_per_layer_input": 256 if hidden_size < 2816 else 0,
      "max_position_embeddings": max_position_embeddings,
      "sliding_window": sliding_window,
      "attention_k_eq_v": attention_k_eq_v,
      "num_global_key_value_heads": num_global_key_value_heads,
      "use_bidirectional_attention": None if hidden_size < 2816 else "vision",
      "enable_moe_block": moe,
      "num_experts": num_experts,
      "top_k_experts": top_k_experts,
      "moe_intermediate_size": moe_intermediate_size,
    },
  }


def test_forward_matches_numpy_reference_for_gemma4():
  config = make_config()
  model = GemmaForCausalLM(config)
  params = randomize_model(model, seed=13)
  input_ids = [2, 5, 7, 11]
  logits, _ = model.forward_ids(input_ids)
  expected = numpy_forward(config, params, input_ids)
  np.testing.assert_allclose(logits.numpy(), expected, rtol=1e-4, atol=1e-4)


def test_cache_matches_full_forward_for_gemma4():
  config = make_config()
  model = GemmaForCausalLM(config)
  randomize_model(model, seed=29)
  prompt = [2, 4, 6]
  cache = GemmaCache.empty(config.num_hidden_layers)
  _, cache = model.forward_ids(prompt, cache=cache)
  step_logits, _ = model.forward_ids([9], cache=cache)
  full_logits, _ = model.forward_ids(prompt + [9])
  np.testing.assert_allclose(step_logits.numpy(), full_logits.numpy()[:, -1:, :], rtol=1e-4, atol=1e-4)


def test_next_logits_ids_matches_forward_last_logits_for_gemma4():
  config = make_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(config)
    randomize_model(model, seed=30)
    full_logits, _ = model.forward_ids([2, 4, 6, 9])
    next_logits, _ = model.next_logits_ids([2, 4, 6, 9])
  assert next_logits.shape == (1, config.vocab_size)
  np.testing.assert_allclose(next_logits.numpy(), full_logits.numpy()[:, -1, :], rtol=1e-4, atol=1e-4)


def test_preallocated_cache_matches_full_forward_for_gemma4():
  config = make_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(config)
    randomize_model(model, seed=31)
    prompt = [2, 4, 6]
    cache = GemmaCache.empty(config.num_hidden_layers, max_length=8)
    _, cache = model.forward_ids(prompt, cache=cache)
    step_logits, _ = model.forward_ids([9], cache=cache)
    full_logits, _ = model.forward_ids(prompt + [9])
  np.testing.assert_allclose(step_logits.numpy(), full_logits.numpy()[:, -1:, :], rtol=1e-4, atol=1e-4)


def test_preallocated_sliding_cache_matches_full_forward_after_window_for_gemma4():
  config = make_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(config)
    randomize_model(model, seed=34)
    prompt = [2, 4, 6, 8, 10, 12]
    cache = GemmaCache.empty(config.num_hidden_layers, max_length=10)
    _, cache = model.forward_ids(prompt, cache=cache)
    step_logits, _ = model.forward_ids([14], cache=cache)
    full_logits, _ = model.forward_ids(prompt + [14])
  np.testing.assert_allclose(step_logits.numpy(), full_logits.numpy()[:, -1:, :], rtol=1e-4, atol=1e-4)


def test_dynamic_sliding_cache_keeps_suffix_and_matches_full_forward_for_gemma4():
  config = make_config()
  assert config.layer_types == ["sliding_attention", "full_attention"]
  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(config)
    randomize_model(model, seed=32)
    prompt = [2, 4, 6, 8, 10, 12]
    cache = GemmaCache.empty(config.num_hidden_layers)
    _, cache = model.forward_ids(prompt, cache=cache)
    sliding_entry = cache.entries[0]
    full_entry = cache.entries[1]
    assert sliding_entry is not None
    assert full_entry is not None
    assert sliding_entry.key.shape[2] == config.sliding_window - 1
    assert full_entry.key.shape[2] == len(prompt)

    step_logits, _ = model.forward_ids([14, 16], cache=cache)
    full_logits, _ = model.forward_ids(prompt + [14, 16])
  np.testing.assert_allclose(step_logits.numpy(), full_logits.numpy()[:, -2:, :], rtol=1e-4, atol=1e-4)


def test_preallocated_generate_matches_dynamic_cache_for_gemma4():
  config = make_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(config)
    randomize_model(model, seed=33)
    dynamic_cache = GemmaCache.empty(config.num_hidden_layers)
    logits, dynamic_cache = model.forward_ids([2, 4, 6], cache=dynamic_cache)
    dynamic_tokens = []
    for _ in range(4):
      next_token = model.sample_next(logits[:, -1, :])
      dynamic_tokens.append(int(next_token.item()))
      logits, dynamic_cache = model(next_token.reshape(1, 1), cache=dynamic_cache)
  preallocated_tokens = list(model.generate([2, 4, 6], max_new_tokens=4, stop_token_ids=None))
  assert preallocated_tokens == dynamic_tokens
  assert model._last_rollout_jit is None


def test_loader_roundtrip_for_nested_gemma4(tmp_path: Path):
  config = make_config()
  model = GemmaForCausalLM(config)
  randomize_model(model, seed=37)
  model_dir = tmp_path / "gemma4"
  model_dir.mkdir()

  (model_dir / "config.json").write_text(json.dumps({
    "model_type": "gemma4",
    "text_config": config.to_dict(),
    "tie_word_embeddings": config.tie_word_embeddings,
  }))

  state_dict = nn.state.get_state_dict(model)
  nested = {}
  for name, value in state_dict.items():
    nested["model.language_model." + name[len("model."):]] = value if name.startswith("model.") else value
    if not name.startswith("model."):
      nested[name] = value
  nn.state.safe_save(nested, str(model_dir / "model.safetensors"))

  reloaded = load_pretrained(model_dir, verbose=False)
  input_ids = [2, 3, 5]
  logits_a, _ = model.forward_ids(input_ids)
  logits_b, _ = reloaded.forward_ids(input_ids)
  np.testing.assert_allclose(logits_a.numpy(), logits_b.numpy(), rtol=1e-5, atol=1e-5)


def test_loader_roundtrip_for_conditional_gemma4(tmp_path: Path):
  config = make_conditional_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForConditionalGeneration(config)
    randomize_model(model, seed=47)
    model_dir = tmp_path / "gemma4-conditional"
    model_dir.mkdir()

    (model_dir / "config.json").write_text(json.dumps(config.to_dict()))

    state_dict = dict(nn.state.get_state_dict(model))
    state_dict.pop("lm_head.weight", None)
    nn.state.safe_save(state_dict, str(model_dir / "model.safetensors"))

    reloaded = load_pretrained(model_dir, verbose=False)
    assert isinstance(reloaded, GemmaForConditionalGeneration)
    assert load_text_config(model_dir).family == "gemma4"

    input_ids = [2, config.image_token_id, 5] + [config.audio_token_id] * 5 + [7]
    pixel_values = np.random.default_rng(1).random((1, 1, 12), dtype=np.float32)
    image_position_ids = np.array([[[0, 0]]], dtype=np.int32)
    input_features = np.random.default_rng(2).random((1, 20, 4), dtype=np.float32)
    input_features_mask = np.ones((1, 20), dtype=bool)
    logits_a, _ = model.forward_ids(
      input_ids,
      pixel_values=pixel_values,
      image_position_ids=image_position_ids,
      input_features=input_features,
      input_features_mask=input_features_mask,
    )
    logits_b, _ = reloaded.forward_ids(
      input_ids,
      pixel_values=pixel_values,
      image_position_ids=image_position_ids,
      input_features=input_features,
      input_features_mask=input_features_mask,
    )
  np.testing.assert_allclose(logits_a.numpy(), logits_b.numpy(), rtol=1e-5, atol=1e-5)


def test_conditional_forward_handles_multimodal_placeholders():
  config = make_conditional_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForConditionalGeneration(config)
    randomize_model(model, seed=53)
    input_ids = [2, config.image_token_id, 5] + [config.audio_token_id] * 5 + [7]
    pixel_values = np.random.default_rng(3).random((1, 1, 12), dtype=np.float32)
    image_position_ids = np.array([[[0, 0]]], dtype=np.int32)
    input_features = np.random.default_rng(4).random((1, 20, 4), dtype=np.float32)
    input_features_mask = np.ones((1, 20), dtype=bool)
    logits, cache = model.forward_ids(
      input_ids,
      pixel_values=pixel_values,
      image_position_ids=image_position_ids,
      input_features=input_features,
      input_features_mask=input_features_mask,
    )
    assert logits.shape == (1, len(input_ids), config.text_config.vocab_size)
    assert cache is None


def test_forward_loss_ids_matches_manual_shifted_loss():
  config = make_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(config)
    randomize_model(model, seed=59)
    input_ids = [2, 5, 0, 11]
    logits, _ = model.forward_ids(input_ids)
    labels = build_causal_labels(input_ids, device=logits.device, ignore_index=DEFAULT_IGNORE_INDEX, ignore_token_ids={0})
    expected = causal_language_model_loss(logits, labels)
    actual = model.forward_loss_ids(input_ids)
  np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


def test_conditional_forward_loss_ignores_multimodal_targets():
  config = make_conditional_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForConditionalGeneration(config)
    randomize_model(model, seed=61)
    input_ids = [2, config.image_token_id, 5] + [config.audio_token_id] * 5 + [7]
    pixel_values = np.random.default_rng(5).random((1, 1, 12), dtype=np.float32)
    image_position_ids = np.array([[[0, 0]]], dtype=np.int32)
    input_features = np.random.default_rng(6).random((1, 20, 4), dtype=np.float32)
    input_features_mask = np.ones((1, 20), dtype=bool)
    logits, _ = model.forward_ids(
      input_ids,
      pixel_values=pixel_values,
      image_position_ids=image_position_ids,
      input_features=input_features,
      input_features_mask=input_features_mask,
    )
    labels = build_causal_labels(
      input_ids,
      device=logits.device,
      ignore_index=DEFAULT_IGNORE_INDEX,
      ignore_token_ids={config.text_config.pad_token_id, config.image_token_id, config.audio_token_id},
    )
    expected = causal_language_model_loss(logits, labels)
    actual = model.forward_loss_ids(
      input_ids,
      pixel_values=pixel_values,
      image_position_ids=image_position_ids,
      input_features=input_features,
      input_features_mask=input_features_mask,
    )
  np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


def test_gemma4_official_config_aliases_and_defaults():
  config = GemmaConfig.from_dict({
    "model_type": "gemma4",
    "text_config": {
      "model_type": "gemma4_text",
      "vocab_size": 64,
      "hidden_size": 16,
      "intermediate_size": 32,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 4,
      "hidden_activation": "gelu_pytorch_tanh",
      "use_bidirectional_attention": "all",
      "enable_moe_block": True,
      "num_experts": 8,
      "top_k_experts": 2,
      "expert_intermediate_size": 48,
    },
    "tie_word_embeddings": True,
  })
  assert config.family == "gemma4"
  assert config.max_position_embeddings == 131072
  assert config.sliding_window == 257
  assert config.moe_intermediate_size == 48
  assert config.tie_word_embeddings is True


@pytest.mark.parametrize(
  ("size_name", "raw_config", "expected"),
  [
    (
      "E2B",
      official_size_config(hidden_size=1536, intermediate_size=6144, num_hidden_layers=35, num_attention_heads=8, num_key_value_heads=1, max_position_embeddings=131072, sliding_window=512, audio=True),
      {"audio": True, "moe": False, "context": 131072, "vision_attention": None},
    ),
    (
      "E4B",
      official_size_config(hidden_size=2560, intermediate_size=10240, num_hidden_layers=42, num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=131072, sliding_window=512, audio=True),
      {"audio": True, "moe": False, "context": 131072, "vision_attention": None},
    ),
    (
      "26B-A4B",
      official_size_config(hidden_size=2816, intermediate_size=2112, num_hidden_layers=30, num_attention_heads=16, num_key_value_heads=8, max_position_embeddings=262144, sliding_window=1024, audio=False, moe=True, num_experts=128, top_k_experts=8, moe_intermediate_size=704, attention_k_eq_v=True, num_global_key_value_heads=2),
      {"audio": False, "moe": True, "context": 262144, "vision_attention": "vision"},
    ),
    (
      "31B",
      official_size_config(hidden_size=5376, intermediate_size=21504, num_hidden_layers=60, num_attention_heads=32, num_key_value_heads=16, max_position_embeddings=262144, sliding_window=1024, audio=False, attention_k_eq_v=True, num_global_key_value_heads=4),
      {"audio": False, "moe": False, "context": 262144, "vision_attention": "vision"},
    ),
  ],
)
def test_official_gemma4_size_configs_parse(size_name: str, raw_config: dict, expected: dict):
  config = GemmaConditionalConfig.from_dict(raw_config)
  assert config.vision_config is not None
  assert (config.audio_config is not None) is expected["audio"]
  assert config.text_config.enable_moe_block is expected["moe"]
  assert config.text_config.max_position_embeddings == expected["context"]
  assert config.text_config.use_bidirectional_attention == expected["vision_attention"]
  assert len(config.text_config.layer_types or []) == config.text_config.num_hidden_layers
  assert config.text_config.layer_types[-1] == "full_attention"


def test_vision_bidirectional_sliding_mask_allows_same_image_group():
  with temporary_default_device("PYTHON"):
    group_ids = Tensor([[-1, 0, 0, -1]], dtype="int32", device="PYTHON")
    mask = build_attention_mask(
      query_len=4,
      key_len=4,
      past_seen_tokens=0,
      sliding_window=3,
      dtype="float",
      device="PYTHON",
      causal=True,
      bidirectional_group_ids=group_ids,
    )
  mask_np = mask.numpy()[0, 0]
  assert mask_np[1, 2] == 0.0
  assert np.isneginf(mask_np[1, 3])
  assert np.isneginf(mask_np[3, 0])


def test_single_token_cropped_sliding_mask_is_elided():
  with temporary_default_device("PYTHON"):
    cropped = build_attention_mask(
      query_len=1,
      key_len=3,
      past_seen_tokens=6,
      sliding_window=3,
      dtype="float",
      device="PYTHON",
      causal=True,
    )
    uncropped = build_attention_mask(
      query_len=1,
      key_len=4,
      past_seen_tokens=6,
      sliding_window=3,
      dtype="float",
      device="PYTHON",
      causal=True,
    )
  assert cropped is None
  assert uncropped is not None
  uncropped_np = uncropped.numpy()[0, 0, 0]
  assert np.isneginf(uncropped_np[0])
  np.testing.assert_allclose(uncropped_np[1:], np.zeros(3, dtype=np.float32))


def test_conditional_forward_handles_large_model_vision_attention_mode():
  config = make_conditional_config()
  config.text_config.layer_types = ["sliding_attention"]
  config.text_config.sliding_window = 2
  config.text_config.use_bidirectional_attention = "vision"
  with temporary_default_device("PYTHON"):
    model = GemmaForConditionalGeneration(config)
    randomize_model(model, seed=83)
    input_ids = [2, config.image_token_id, 5]
    pixel_values = np.random.default_rng(9).random((1, 1, 12), dtype=np.float32)
    image_position_ids = np.array([[[0, 0]]], dtype=np.int32)
    logits, _ = model.forward_ids(input_ids, pixel_values=pixel_values, image_position_ids=image_position_ids)
    assert logits.shape == (1, len(input_ids), config.text_config.vocab_size)


def test_conditional_sliding_decode_start_is_metal_only():
  config = make_conditional_config()
  config.text_config.layer_types = ["sliding_attention"]
  config.text_config.sliding_window = 5
  with temporary_default_device("PYTHON"):
    model = GemmaForConditionalGeneration(config)
  assert model._sliding_decode_start() is None
  model.device = "METAL"
  assert model._sliding_decode_start() == 4


def test_gemma4_full_attention_uses_regular_kv_heads_without_k_eq_v():
  config = GemmaConfig(
    model_type="gemma4",
    vocab_size=48,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_global_key_value_heads=1,
    head_dim=4,
    global_head_dim=6,
    hidden_activation="gelu_pytorch_tanh",
    hidden_size_per_layer_input=0,
    max_position_embeddings=64,
    sliding_window=3,
    layer_types=["full_attention"],
    rope_parameters={
      "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
      "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.5, "rope_theta": 1000000.0},
    },
    attention_k_eq_v=False,
    final_logit_softcapping=4.0,
  )
  model = GemmaForCausalLM(config)
  params = randomize_model(model, seed=41)
  input_ids = [2, 5, 7, 11]
  logits, _ = model.forward_ids(input_ids)
  expected = numpy_forward(config, params, input_ids)
  np.testing.assert_allclose(logits.numpy(), expected, rtol=1e-4, atol=1e-4)


def test_model_forward_ids_respects_model_device():
  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(make_config())
    randomize_model(model, seed=43)
    assert model.device == "PYTHON"
    logits, _ = model.forward_ids([2, 5, 7])
    assert logits.device == "PYTHON"


def test_tokenizer_json_support(tmp_path: Path):
  tokenizers = pytest.importorskip("tokenizers")

  tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
    {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "hello": 4, "world": 5},
    unk_token="<unk>",
  ))
  tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
  tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
    single="<bos> $A",
    special_tokens=[("<bos>", 2)],
  )
  model_dir = tmp_path / "tokenizer-json"
  model_dir.mkdir()
  tokenizer.save(str(model_dir / "tokenizer.json"))
  (model_dir / "tokenizer_config.json").write_text(json.dumps({
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
  }))

  loaded = GemmaTokenizer.from_pretrained(model_dir)
  assert loaded.bos_id == 2
  assert loaded.eos_id == 1
  assert loaded.encode("hello world", add_bos=True) == [2, 4, 5]
  assert loaded.encode("hello world", add_bos=False) == [4, 5]
  assert loaded.decode([4, 5]) == "hello world"


def test_multimodal_processor_expands_placeholders(tmp_path: Path):
  pytest.importorskip("transformers")
  pytest.importorskip("PIL")
  model_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "gemma-4-E2B"
  if not model_dir.exists():
    pytest.skip("real Gemma 4 checkpoint is not available in this workspace")

  try:
    from PIL import Image
  except ImportError:  # pragma: no cover - importorskip above
    raise AssertionError("PIL should be available")

  image_path = tmp_path / "sample.png"
  Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8)).save(image_path)

  audio_path = tmp_path / "sample.wav"
  waveform = (0.2 * np.sin(2 * np.pi * 440 * np.arange(1600, dtype=np.float32) / 16000.0) * 32767.0).astype(np.int16)
  with wave.open(str(audio_path), "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(16000)
    wav_file.writeframes(waveform.tobytes())

  processor = GemmaMultimodalProcessor.from_pretrained(model_dir)
  prepared = processor.prepare_inputs("describe <|image|> <|audio|>", images=[image_path], audio=[audio_path])
  assert prepared.pixel_values is not None
  assert prepared.image_position_ids is not None
  assert prepared.input_features is not None
  assert prepared.input_features_mask is not None
  assert prepared.input_ids[0] == processor.tokenizer.bos_id
  valid_patches = int((prepared.image_position_ids[0, :, 0] != -1).sum())
  pooling_area = processor.config.vision_config.pooling_kernel_size ** 2
  assert sum(token == processor.tokenizer.image_token_id for token in prepared.input_ids) == valid_patches // pooling_area
  expected_audio_tokens = int(prepared.input_features_mask[0, ::2][::2].sum())
  assert sum(token == processor.tokenizer.audio_token_id for token in prepared.input_ids) == expected_audio_tokens


def test_training_helpers_step_checkpoint_and_optimizer_roundtrip(tmp_path: Path):
  config = make_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(config)
    randomize_model(model, seed=67)
    set_trainable(model, False)
    selected = set_trainable(model, True, include_prefixes=["model.layers.0"])
    assert selected
    optimizer = build_optimizer(model, optimizer="adamw", include_prefixes=["model.layers.0"], freeze_unselected=True, lr=1e-2, weight_decay=0.0)
    assert optimizer.params
    frozen_names = [name for name, tensor in nn.state.get_state_dict(model).items() if tensor.requires_grad is False]
    assert any(name.startswith("model.layers.1") for name in frozen_names)

    set_trainable(model, True)
    optimizer = build_optimizer(model, optimizer="adamw", lr=5e-2, weight_decay=0.0)

    before_logits, _ = model.forward_ids([2, 5, 7, 11])
    before_logits_np = before_logits.numpy().copy()
    loss = train_step(model, optimizer, GemmaTrainingBatch(input_ids=[2, 5, 7, 11]))
    after_logits, _ = model.forward_ids([2, 5, 7, 11])
    after_logits_np = after_logits.numpy().copy()
    assert float(loss.item()) > 0.0
    assert not np.allclose(before_logits_np, after_logits_np)

    checkpoint_dir = tmp_path / "train-ckpt"
    save_training_checkpoint(model, checkpoint_dir, optimizer=optimizer, training_metadata={"step": 1, "phase": "finetune"})
    assert (checkpoint_dir / "config.json").exists()
    assert (checkpoint_dir / "model.safetensors").exists()
    assert (checkpoint_dir / "optimizer.safetensors").exists()
    assert json.loads((checkpoint_dir / "training.json").read_text())["step"] == 1

    reloaded = load_pretrained(checkpoint_dir)
    reload_logits, _ = reloaded.forward_ids([2, 5, 7, 11])
    np.testing.assert_allclose(after_logits_np, reload_logits.numpy(), rtol=1e-5, atol=1e-5)

    reloaded_optimizer = build_optimizer(reloaded, optimizer="adamw", lr=5e-2, weight_decay=0.0)
    load_optimizer_state(reloaded_optimizer, checkpoint_dir / "optimizer.safetensors")
    original_opt_state = nn.state.get_state_dict(optimizer)
    reloaded_opt_state = nn.state.get_state_dict(reloaded_optimizer)
    assert set(original_opt_state) == set(reloaded_opt_state)
    for key in original_opt_state:
      np.testing.assert_allclose(original_opt_state[key].numpy(), reloaded_opt_state[key].numpy(), rtol=1e-5, atol=1e-5)


def test_quantized_text_checkpoint_reloads_and_trains(tmp_path: Path):
  config = make_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(config)
    randomize_model(model, seed=71)
    baseline_logits, _ = model.forward_ids([2, 5, 7, 11])
    baseline_logits_np = baseline_logits.numpy().copy()

    checkpoint_dir = tmp_path / "quantized-text"
    save_training_checkpoint(model, checkpoint_dir, quantize="int8", training_metadata={"step": 0})

    manifest = load_quantization_manifest(checkpoint_dir)
    assert manifest is not None
    assert manifest["method"] == "int8"
    assert manifest["tensors"]

    reloaded = load_pretrained(checkpoint_dir, device="PYTHON", runtime_quantization=False)
    reloaded_logits, _ = reloaded.forward_ids([2, 5, 7, 11])
    np.testing.assert_allclose(baseline_logits_np, reloaded_logits.numpy(), rtol=0.12, atol=0.12)

    optimizer = build_optimizer(reloaded, optimizer="adamw", lr=5e-2, weight_decay=0.0)
    before_step = reloaded_logits.numpy().copy()
    loss = train_step(reloaded, optimizer, GemmaTrainingBatch(input_ids=[2, 5, 7, 11]))
    after_step, _ = reloaded.forward_ids([2, 5, 7, 11])
    assert float(loss.item()) > 0.0
    assert not np.allclose(before_step, after_step.numpy())


def test_quantized_checkpoint_supports_dequantized_and_runtime_int8_loads(tmp_path: Path):
  config = make_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(config)
    randomize_model(model, seed=72)
    checkpoint_dir = tmp_path / "quantized-contract"
    save_training_checkpoint(model, checkpoint_dir, quantize="int8")

    manifest = load_quantization_manifest(checkpoint_dir)
    assert manifest is not None
    tensor_name, tensor_info = next(iter(manifest["tensors"].items()))
    raw_state = nn.state.safe_load(checkpoint_dir / "model.safetensors")
    assert raw_state[tensor_name].dtype == dtypes.int8
    assert tensor_info["scale_key"] in raw_state

    dequantized = load_pretrained(checkpoint_dir, device="PYTHON", runtime_quantization=False)
    dequantized_state = nn.state.get_state_dict(dequantized)
    runtime = load_pretrained(checkpoint_dir, device="PYTHON")
    runtime_state = nn.state.get_state_dict(runtime)
    linear_name = "model.layers.0.mlp.gate_proj.weight"
    dequantized_logits, _ = dequantized.forward_ids([2, 5, 7, 11])
    runtime_logits, _ = runtime.forward_ids([2, 5, 7, 11])

  assert dequantized_state[tensor_name].dtype != dtypes.int8
  assert not any(name.startswith("_quant_scale.") for name in dequantized_state)
  assert linear_name in manifest["tensors"]
  assert isinstance(runtime.model.layers[0].mlp.gate_proj, RowwiseInt8Linear)
  assert runtime_state[linear_name].dtype == dtypes.int8
  assert runtime_state["model.layers.0.mlp.gate_proj.scale"].shape == (config.intermediate_size,)
  assert not any(name.startswith("_quant_scale.") for name in runtime_state)
  np.testing.assert_allclose(dequantized_logits.numpy(), runtime_logits.numpy(), rtol=1e-4, atol=1e-4)


def test_quantization_includes_bfloat16_matrices():
  state_dict = {
    "matrix": Tensor.ones(2, 3, dtype=dtypes.bfloat16),
    "vector": Tensor.ones(3, dtype=dtypes.bfloat16),
  }
  quantized, manifest = quantize_state_dict(state_dict, quantize="int8")

  assert quantized["matrix"].dtype == dtypes.int8
  assert manifest["tensors"]["matrix"]["dtype"] == "bfloat16"
  assert manifest["tensors"]["matrix"]["scale_key"] in quantized
  assert quantized["vector"].dtype == dtypes.bfloat16
  assert "vector" not in manifest["tensors"]


def test_runtime_int8_mlp_fused_gate_up_matches_separate_path(tmp_path: Path):
  config = make_config()
  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(config)
    randomize_model(model, seed=74)
    checkpoint_dir = tmp_path / "quantized-fused-mlp"
    save_training_checkpoint(model, checkpoint_dir, quantize="int8")
    runtime = load_pretrained(checkpoint_dir, device="PYTHON")
    mlp = runtime.model.layers[0].mlp
    assert isinstance(mlp.gate_proj, RowwiseInt8Linear)
    assert isinstance(mlp.up_proj, RowwiseInt8Linear)
    x = Tensor.randn(1, 3, config.hidden_size)

    fused = mlp(x)
    can_fuse = mlp._can_use_fused_int8_gate_up
    mlp._can_use_fused_int8_gate_up = lambda: False
    try:
      separate = mlp(x)
    finally:
      mlp._can_use_fused_int8_gate_up = can_fuse

  np.testing.assert_allclose(fused.numpy(), separate.numpy(), rtol=1e-5, atol=1e-5)


def test_metal_rowwise_int8_decode_linear_rejects_non_metal():
  x = Tensor.ones(1, 1, 4, device="PYTHON")
  qweight = Tensor.ones(8, 4, dtype=dtypes.int8, device="PYTHON")
  scale = Tensor.ones(8, dtype=dtypes.float32, device="PYTHON")

  with pytest.raises(RuntimeError, match="expects x on METAL"):
    metal_rowwise_int8_decode_linear(x, qweight, scale)


def test_gemma_mlp_raw_metal_gate_up_path_is_disabled_until_graphable():
  mlp = GemmaMLP(make_config(), 0)
  mlp._force_metal_fused_int8_gate_up = True
  fake_metal_decode_row = type("FakeTensor", (), {"device": "METAL", "shape": (1, 1, 4)})()

  assert mlp._can_use_metal_fused_int8_gate_up(fake_metal_decode_row) is False


def test_quantized_multimodal_checkpoint_reloads_and_runs(tmp_path: Path):
  config = make_conditional_config()
  input_ids = [2, config.image_token_id, 5] + [config.audio_token_id] * 5 + [7]
  pixel_values = np.random.default_rng(7).random((1, 1, 12), dtype=np.float32)
  image_position_ids = np.array([[[0, 0]]], dtype=np.int32)
  input_features = np.random.default_rng(8).random((1, 20, 4), dtype=np.float32)
  input_features_mask = np.ones((1, 20), dtype=bool)

  with temporary_default_device("PYTHON"):
    model = GemmaForConditionalGeneration(config)
    randomize_model(model, seed=73)
    baseline_logits, _ = model.forward_ids(
      input_ids,
      pixel_values=pixel_values,
      image_position_ids=image_position_ids,
      input_features=input_features,
      input_features_mask=input_features_mask,
    )
    checkpoint_dir = tmp_path / "quantized-conditional"
    save_training_checkpoint(model, checkpoint_dir, quantize="int8")

    manifest = load_quantization_manifest(checkpoint_dir)
    assert manifest is not None
    assert manifest["method"] == "int8"

    reloaded = load_pretrained(checkpoint_dir, device="PYTHON")
    assert isinstance(reloaded, GemmaForConditionalGeneration)
    assert isinstance(reloaded.model.language_model.layers[0].mlp.gate_proj, RowwiseInt8Linear)
    reloaded_logits, _ = reloaded.forward_ids(
      input_ids,
      pixel_values=pixel_values,
      image_position_ids=image_position_ids,
      input_features=input_features,
      input_features_mask=input_features_mask,
    )
    np.testing.assert_allclose(baseline_logits.numpy(), reloaded_logits.numpy(), rtol=0.12, atol=0.12)


def test_supported_optimizer_and_quantization_surfaces_are_honest():
  assert "muon" in supported_optimizers()
  assert supported_quantizations() == ("int8",)

  with temporary_default_device("PYTHON"):
    model = GemmaForCausalLM(make_config())
    randomize_model(model, seed=79)
    optimizer = build_optimizer(model, optimizer="muon", lr=1e-3, weight_decay=0.0)
    loss = train_step(model, optimizer, GemmaTrainingBatch(input_ids=[2, 5, 7, 11]))
    assert float(loss.item()) > 0.0


def test_gemma4_only_repo_rejects_older_models():
  with pytest.raises(ValueError):
    GemmaConfig(model_type="gemma")
  with pytest.raises(ValueError):
    GemmaConfig.from_dict({"model_type": "gemma2"})


def test_resolve_beam_supports_max():
  assert resolve_beam("0") == 0
  assert resolve_beam("7") == 7
  assert resolve_beam("max") == DEFAULT_MAX_BEAM
  with pytest.raises(ValueError):
    resolve_beam("-1")
