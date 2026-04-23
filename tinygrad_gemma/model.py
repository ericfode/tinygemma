from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tinygrad import Tensor, nn

from .config import GemmaConfig

DEFAULT_IGNORE_INDEX = -100


def token_rows(input_ids: Sequence[int] | Sequence[Sequence[int]]) -> list[list[int]]:
  if len(input_ids) == 0:
    raise ValueError("input_ids must not be empty")
  first = input_ids[0]
  if isinstance(first, (list, tuple)):
    return [list(row) for row in input_ids]  # type: ignore[arg-type]
  return [list(input_ids)]  # type: ignore[arg-type]


def input_ids_tensor(input_ids: Sequence[int] | Sequence[Sequence[int]], *, device: str) -> Tensor:
  return Tensor(token_rows(input_ids), dtype="int32", device=device)


def label_tensor(labels: Tensor | Sequence[int] | Sequence[Sequence[int]], *, device: str) -> Tensor:
  if isinstance(labels, Tensor):
    return labels if labels.ndim == 2 else labels.reshape(1, -1)
  return Tensor(token_rows(labels), dtype="int32", device=device)


def build_causal_labels(
  input_ids: Tensor | Sequence[int] | Sequence[Sequence[int]],
  *,
  device: str | None = None,
  ignore_index: int = DEFAULT_IGNORE_INDEX,
  ignore_token_ids: set[int] | None = None,
) -> Tensor:
  if isinstance(input_ids, Tensor):
    labels_device = input_ids.device if device is None else device
    input_array = input_ids.numpy().astype(np.int32, copy=False)
  else:
    labels_device = "CPU" if device is None else device
    input_array = np.array(token_rows(input_ids), dtype=np.int32)
  labels = np.full(input_array.shape, ignore_index, dtype=np.int32)
  if input_array.shape[1] > 1:
    labels[:, :-1] = input_array[:, 1:]
  if ignore_token_ids:
    labels[np.isin(labels, np.array(sorted(ignore_token_ids), dtype=np.int32))] = ignore_index
  return Tensor(labels, dtype="int32", device=labels_device)


def causal_language_model_loss(logits: Tensor, labels: Tensor | Sequence[int] | Sequence[Sequence[int]], *, ignore_index: int = DEFAULT_IGNORE_INDEX) -> Tensor:
  labels_tensor = label_tensor(labels, device=logits.device)
  return logits.reshape(-1, logits.shape[-1]).sparse_categorical_crossentropy(labels_tensor.reshape(-1), ignore_index=ignore_index)


def gelu_pytorch_tanh(x: Tensor) -> Tensor:
  return x * 0.5 * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())


def apply_activation(name: str, x: Tensor) -> Tensor:
  if name in ("gelu", "gelu_new"):
    return x.gelu()
  if name == "gelu_pytorch_tanh":
    return gelu_pytorch_tanh(x)
  if name == "silu":
    return x.silu()
  raise ValueError(f"unsupported activation {name!r}")


def rotate_half(x: Tensor) -> Tensor:
  half = x.shape[-1] // 2
  return (-x[..., half:]).cat(x[..., :half], dim=-1)


def rotary_embedding(
  position_ids: Tensor,
  head_dim: int,
  rope_theta: float,
  dtype,
  device,
  partial_rotary_factor: float = 1.0,
) -> tuple[Tensor, Tensor]:
  rot_dim = max(0, int(head_dim * partial_rotary_factor))
  rot_dim -= rot_dim % 2
  if rot_dim == 0:
    shape = (*position_ids.shape, 0)
    return Tensor.empty(*shape, device=device, dtype=dtype), Tensor.empty(*shape, device=device, dtype=dtype)
  inv_freq = 1.0 / (rope_theta ** (Tensor.arange(0, rot_dim, 2, device=device).float() / rot_dim))
  freqs = position_ids.float().unsqueeze(-1) * inv_freq.reshape(1, 1, -1)
  emb = freqs.cat(freqs, dim=-1)
  return emb.cos().cast(dtype), emb.sin().cast(dtype)


def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1) -> Tensor:
  if cos.shape[-1] == 0:
    return x
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  if cos.shape[-1] == x.shape[-1]:
    return (x * cos) + (rotate_half(x) * sin)
  rot, rest = x[..., :cos.shape[-1]], x[..., cos.shape[-1]:]
  return ((rot * cos) + (rotate_half(rot) * sin)).cat(rest, dim=-1)


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
  if n_rep == 1:
    return hidden_states
  return hidden_states.repeat_interleave(n_rep, dim=1)


def build_attention_mask(
  query_len: int,
  key_len: int,
  past_seen_tokens: int,
  sliding_window: int | None,
  dtype,
  device,
  causal: bool = True,
  bidirectional_group_ids: Tensor | None = None,
) -> Tensor | None:
  if query_len == 1 and sliding_window is None and causal and bidirectional_group_ids is None:
    return None
  query_positions = Tensor.arange(past_seen_tokens, past_seen_tokens + query_len, device=device, dtype="int32").reshape(query_len, 1)
  key_start = past_seen_tokens + query_len - key_len
  key_positions = Tensor.arange(key_start, past_seen_tokens + query_len, device=device, dtype="int32").reshape(1, key_len)
  allowed = (key_positions <= query_positions) if causal else Tensor.ones(query_len, key_len, device=device, dtype="bool")
  if sliding_window is not None:
    if causal:
      allowed = allowed & (key_positions >= (query_positions - sliding_window + 1))
    else:
      allowed = allowed & ((query_positions - key_positions).abs() < sliding_window)
  if bidirectional_group_ids is not None:
    if bidirectional_group_ids.shape[1] != key_len:
      raise ValueError("bidirectional_group_ids must span the current key length")
    batch_size = bidirectional_group_ids.shape[0]
    q_groups = bidirectional_group_ids[:, key_len - query_len : key_len].reshape(bidirectional_group_ids.shape[0], query_len, 1)
    k_groups = bidirectional_group_ids.reshape(bidirectional_group_ids.shape[0], 1, key_len)
    same_group = (q_groups == k_groups) & (q_groups >= 0)
    allowed = allowed.reshape(1, query_len, key_len) | same_group
    return allowed.where(0.0, float("-inf")).cast(dtype).reshape(batch_size, 1, query_len, key_len)
  return allowed.where(0.0, float("-inf")).cast(dtype).reshape(1, 1, query_len, key_len)


@dataclass
class GemmaCacheEntry:
  key: Tensor
  value: Tensor
  length: int | None = None


@dataclass
class GemmaCache:
  entries: list[GemmaCacheEntry | None]
  past_seen_tokens: int = 0
  max_length: int | None = None

  @classmethod
  def empty(cls, num_layers: int, *, max_length: int | None = None) -> "GemmaCache":
    return cls(entries=[None] * num_layers, past_seen_tokens=0, max_length=max_length)


def active_cache_tensors(entry: GemmaCacheEntry) -> tuple[Tensor, Tensor]:
  if entry.length is None:
    return entry.key, entry.value
  return entry.key[:, :, : entry.length, :], entry.value[:, :, : entry.length, :]


class TextScaledEmbedding:
  def __init__(self, vocab_size: int, embedding_dim: int, embed_scale: float):
    self.weight = Tensor.glorot_uniform(vocab_size, embedding_dim)
    self.embed_scale = float(embed_scale)

  def __call__(self, input_ids: Tensor) -> Tensor:
    return self.weight[input_ids] * self.embed_scale


class RMSNorm:
  def __init__(self, dim: int, eps: float = 1e-6, *, with_scale: bool = True, plus_one_scale: bool = False):
    self.eps = eps
    self.with_scale = with_scale
    self.plus_one_scale = plus_one_scale
    self.weight = Tensor.zeros(dim) if with_scale and plus_one_scale else (Tensor.ones(dim) if with_scale else None)

  def __call__(self, x: Tensor) -> Tensor:
    output = x.float()
    output = output * (output.square().mean(-1, keepdim=True) + self.eps).rsqrt()
    if not self.with_scale:
      return output.cast(x.dtype)
    scale = (1.0 + self.weight.float()) if self.plus_one_scale else self.weight.float()
    return (output * scale).cast(x.dtype)


class GemmaMLP:
  def __init__(self, config: GemmaConfig, layer_idx: int):
    self.config = config
    first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
    is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
    use_double_wide = config.use_double_wide_mlp and is_kv_shared_layer
    self.intermediate_size = config.intermediate_size * (2 if use_double_wide else 1)
    self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    return self.down_proj(apply_activation(self.config.activation_name, self.gate_proj(x)) * self.up_proj(x))


class GemmaRouter:
  def __init__(self, config: GemmaConfig):
    self.config = config
    self.hidden_size = config.hidden_size
    self.scalar_root_size = self.hidden_size ** -0.5
    self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps, with_scale=False)
    self.proj = nn.Linear(config.hidden_size, config.num_experts or 0, bias=False)
    self.scale = Tensor.ones(self.hidden_size)
    self.per_expert_scale = Tensor.ones(config.num_experts or 0)

  def __call__(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    hidden_states = self.norm(hidden_states)
    hidden_states = hidden_states * self.scale * self.scalar_root_size
    router_probabilities = self.proj(hidden_states).softmax(-1)
    top_k_weights, top_k_index = router_probabilities.topk(self.config.top_k_experts or 0, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(axis=-1, keepdim=True)
    top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
    return router_probabilities, top_k_weights, top_k_index


class GemmaExperts:
  def __init__(self, config: GemmaConfig):
    self.config = config
    num_experts = config.num_experts or 0
    intermediate = config.moe_intermediate_size or config.intermediate_size
    self.gate_up_proj = Tensor.zeros(num_experts, 2 * intermediate, config.hidden_size)
    self.down_proj = Tensor.zeros(num_experts, config.hidden_size, intermediate)

  def __call__(self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
    h = hidden_states.unsqueeze(1)
    gate_up = (h.unsqueeze(-2) @ self.gate_up_proj[top_k_index].transpose(-1, -2)).squeeze(-2)
    gate, up = gate_up.chunk(2, dim=-1)
    current = apply_activation(self.config.activation_name, gate) * up
    down = (current.unsqueeze(-2) @ self.down_proj[top_k_index].transpose(-1, -2)).squeeze(-2)
    return (down * top_k_weights.unsqueeze(-1)).sum(axis=1)


class GemmaAttention:
  def __init__(self, config: GemmaConfig, layer_idx: int):
    self.config = config
    self.layer_idx = layer_idx
    self.layer_type = config.layer_types[layer_idx] if config.layer_types is not None else "full_attention"
    self.is_sliding = self.layer_type == "sliding_attention"
    self.sliding_window = config.sliding_window if self.is_sliding else None
    self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
    self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
    self.num_key_value_heads = (
      config.num_global_key_value_heads
      if self.use_alternative_attention and config.num_global_key_value_heads is not None
      else config.num_key_value_heads
    )
    self.num_key_value_groups = config.num_attention_heads // self.num_key_value_heads
    self.scaling = 1.0
    self.causal = config.use_bidirectional_attention != "all"

    first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
    self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
    prev_layers = config.layer_types[:first_kv_shared_layer_idx] if config.layer_types else []
    if self.is_kv_shared_layer:
      self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
      self.store_full_length_kv = False
    else:
      self.kv_shared_layer_index = None
      self.store_full_length_kv = (
        bool(prev_layers)
        and self.layer_type in prev_layers
        and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
      )

    self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
    self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    if not self.is_kv_shared_layer:
      self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
      self.v_proj = None if self.use_alternative_attention else nn.Linear(
        config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
      )
      self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
      self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
    self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

  def rope_parameters(self) -> tuple[float, float]:
    params = (self.config.rope_parameters or {}).get(self.layer_type, {})
    return float(params.get("rope_theta", 10000.0)), float(params.get("partial_rotary_factor", 1.0))

  def shared_state(
    self,
    cache: GemmaCache | None,
    shared_kv_states: dict[int, GemmaCacheEntry] | None,
  ) -> GemmaCacheEntry:
    if shared_kv_states is not None and self.kv_shared_layer_index in shared_kv_states:
      return shared_kv_states[self.kv_shared_layer_index]
    if cache is not None and cache.entries[self.kv_shared_layer_index] is not None:
      return cache.entries[self.kv_shared_layer_index]  # type: ignore[return-value]
    raise RuntimeError(f"missing shared KV state for layer {self.layer_idx}")

  def __call__(
    self,
    hidden_states: Tensor,
    position_ids: Tensor,
    cache: GemmaCache | None = None,
    attention_mask: Tensor | None = None,
    bidirectional_group_ids: Tensor | None = None,
    shared_kv_states: dict[int, GemmaCacheEntry] | None = None,
  ) -> Tensor:
    batch, query_len, _ = hidden_states.shape
    hidden_shape = (batch, query_len, -1, self.head_dim)
    rope_theta, partial_rotary_factor = self.rope_parameters()

    q = self.q_proj(hidden_states).reshape(*hidden_shape)
    q = self.q_norm(q)
    cos, sin = rotary_embedding(position_ids, self.head_dim, rope_theta, q.dtype, hidden_states.device, partial_rotary_factor)
    q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)

    past_seen_tokens = 0 if cache is None else cache.past_seen_tokens
    if self.is_kv_shared_layer:
      entry = self.shared_state(cache, shared_kv_states)
      k, v = active_cache_tensors(entry)
    else:
      raw_k = self.k_proj(hidden_states).reshape(*hidden_shape[:-2], self.num_key_value_heads, self.head_dim)
      raw_v = self.v_proj(hidden_states).reshape(*hidden_shape[:-2], self.num_key_value_heads, self.head_dim) if self.v_proj is not None else raw_k
      k = self.k_norm(raw_k)
      k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
      v = self.v_norm(raw_v).transpose(1, 2)

      if cache is not None:
        end_pos = past_seen_tokens + query_len
        if cache.max_length is not None:
          if end_pos > cache.max_length:
            raise ValueError(f"cache capacity exceeded: need {end_pos}, capacity {cache.max_length}")
          entry = cache.entries[self.layer_idx]
          if entry is None:
            key_cache = Tensor.zeros(batch, self.num_key_value_heads, cache.max_length, self.head_dim, device=k.device, dtype=k.dtype).contiguous().realize()
            value_cache = Tensor.zeros(batch, self.num_key_value_heads, cache.max_length, self.head_dim, device=v.device, dtype=v.dtype).contiguous().realize()
            entry = GemmaCacheEntry(key=key_cache, value=value_cache, length=0)
            cache.entries[self.layer_idx] = entry
          entry.key[:, :, past_seen_tokens:end_pos, :].assign(k).realize()
          entry.value[:, :, past_seen_tokens:end_pos, :].assign(v).realize()
          entry.length = end_pos
          k, v = active_cache_tensors(entry)
        else:
          if (entry := cache.entries[self.layer_idx]) is not None:
            k = entry.key.cat(k, dim=2)
            v = entry.value.cat(v, dim=2)
          cache.entries[self.layer_idx] = GemmaCacheEntry(key=k, value=v)
      if shared_kv_states is not None and self.store_full_length_kv:
        shared_kv_states[self.layer_idx] = cache.entries[self.layer_idx] if cache is not None else GemmaCacheEntry(key=k, value=v)

    key_states = repeat_kv(k, self.num_key_value_groups)
    value_states = repeat_kv(v, self.num_key_value_groups)
    scores = q.matmul(key_states.transpose(-2, -1)).float() * self.scaling

    mask = build_attention_mask(
      query_len,
      key_states.shape[2],
      past_seen_tokens,
      self.sliding_window,
      scores.dtype,
      hidden_states.device,
      causal=self.causal,
      bidirectional_group_ids=bidirectional_group_ids if self.is_sliding else None,
    )
    if attention_mask is not None:
      mask = attention_mask if mask is None else mask + attention_mask
    if mask is not None:
      scores = scores + mask.float()

    probs = scores.softmax(-1).cast(q.dtype)
    attn_output = (probs @ value_states).transpose(1, 2).reshape(batch, query_len, -1)
    return self.o_proj(attn_output)


class GemmaDecoderLayer:
  def __init__(self, config: GemmaConfig, layer_idx: int):
    self.config = config
    self.self_attn = GemmaAttention(config, layer_idx)
    self.mlp = GemmaMLP(config, layer_idx)
    self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.post_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.layer_scalar = Tensor.ones(1)
    self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
    if self.hidden_size_per_layer_input:
      self.per_layer_input_gate = nn.Linear(config.hidden_size, self.hidden_size_per_layer_input, bias=False)
      self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, config.hidden_size, bias=False)
      self.post_per_layer_input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.enable_moe_block = config.enable_moe_block
    if self.enable_moe_block:
      self.router = GemmaRouter(config)
      self.experts = GemmaExperts(config)
      self.post_feedforward_layernorm_1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
      self.post_feedforward_layernorm_2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
      self.pre_feedforward_layernorm_2 = RMSNorm(config.hidden_size, config.rms_norm_eps)

  def __call__(
    self,
    hidden_states: Tensor,
    position_ids: Tensor,
    cache: GemmaCache | None = None,
    attention_mask: Tensor | None = None,
    per_layer_input: Tensor | None = None,
    bidirectional_group_ids: Tensor | None = None,
    shared_kv_states: dict[int, GemmaCacheEntry] | None = None,
  ) -> Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
      hidden_states,
      position_ids,
      cache=cache,
      attention_mask=attention_mask,
      bidirectional_group_ids=bidirectional_group_ids,
      shared_kv_states=shared_kv_states,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if self.enable_moe_block:
      hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
      hidden_states_flat = residual.reshape(-1, residual.shape[-1])
      _, top_k_weights, top_k_index = self.router(hidden_states_flat)
      hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
      hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights).reshape(residual.shape)
      hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
      hidden_states = hidden_states_1 + hidden_states_2
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    if self.hidden_size_per_layer_input:
      if per_layer_input is None:
        raise RuntimeError(f"missing per-layer input for Gemma 4 layer {self.self_attn.layer_idx}")
      residual = hidden_states
      hidden_states = apply_activation(self.config.activation_name, self.per_layer_input_gate(hidden_states))
      hidden_states = hidden_states * per_layer_input
      hidden_states = self.per_layer_projection(hidden_states)
      hidden_states = self.post_per_layer_input_norm(hidden_states)
      hidden_states = residual + hidden_states
    return hidden_states * self.layer_scalar


class GemmaModel:
  def __init__(self, config: GemmaConfig):
    self.config = config
    self.embed_tokens = TextScaledEmbedding(config.vocab_size, config.hidden_size, config.hidden_size ** 0.5)
    self.layers = [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
    self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
    if self.hidden_size_per_layer_input:
      self.embed_tokens_per_layer = TextScaledEmbedding(
        config.vocab_size_per_layer_input or config.vocab_size,
        config.num_hidden_layers * self.hidden_size_per_layer_input,
        self.hidden_size_per_layer_input ** 0.5,
      )
      self.per_layer_input_scale = 2.0 ** -0.5
      self.per_layer_model_projection = nn.Linear(
        config.hidden_size,
        config.num_hidden_layers * self.hidden_size_per_layer_input,
        bias=False,
      )
      self.per_layer_model_projection_scale = config.hidden_size ** -0.5
      self.per_layer_projection_norm = RMSNorm(self.hidden_size_per_layer_input, config.rms_norm_eps)

  def get_per_layer_inputs(self, input_ids: Tensor) -> Tensor:
    return self.embed_tokens_per_layer(input_ids).reshape(
      *input_ids.shape,
      self.config.num_hidden_layers,
      self.hidden_size_per_layer_input,
    )

  def project_per_layer_inputs(self, inputs_embeds: Tensor, per_layer_inputs: Tensor | None = None) -> Tensor:
    projected = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
    projected = projected.reshape(
      *inputs_embeds.shape[:-1],
      self.config.num_hidden_layers,
      self.hidden_size_per_layer_input,
    )
    projected = self.per_layer_projection_norm(projected)
    if per_layer_inputs is None:
      return projected
    return (projected + per_layer_inputs) * self.per_layer_input_scale

  def __call__(
    self,
    input_ids: Tensor | None = None,
    cache: GemmaCache | None = None,
    attention_mask: Tensor | None = None,
    *,
    inputs_embeds: Tensor | None = None,
    per_layer_inputs: Tensor | None = None,
    bidirectional_group_ids: Tensor | None = None,
  ) -> Tensor:
    if (input_ids is None) == (inputs_embeds is None):
      raise ValueError("provide exactly one of input_ids or inputs_embeds")

    if input_ids is not None:
      if input_ids.ndim == 1:
        input_ids = input_ids.reshape(1, -1)
      hidden_states = self.embed_tokens(input_ids)
      batch, seq_len = input_ids.shape
      if self.hidden_size_per_layer_input:
        token_identity = self.get_per_layer_inputs(input_ids)
        per_layer_inputs = self.project_per_layer_inputs(hidden_states, token_identity if per_layer_inputs is None else per_layer_inputs)
    else:
      assert inputs_embeds is not None
      hidden_states = inputs_embeds if inputs_embeds.ndim == 3 else inputs_embeds.reshape(1, *inputs_embeds.shape)
      batch, seq_len, _ = hidden_states.shape
      if self.hidden_size_per_layer_input and per_layer_inputs is None:
        per_layer_inputs = self.project_per_layer_inputs(hidden_states)

    past_seen_tokens = 0 if cache is None else cache.past_seen_tokens
    position_ids = Tensor.arange(past_seen_tokens, past_seen_tokens + seq_len, device=hidden_states.device, dtype="int32").reshape(1, seq_len).expand(batch, seq_len)

    shared_kv_states: dict[int, GemmaCacheEntry] | None = {}
    for i, layer in enumerate(self.layers):
      layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
      hidden_states = layer(
        hidden_states,
        position_ids,
        cache=cache,
        attention_mask=attention_mask,
        per_layer_input=layer_input,
        bidirectional_group_ids=bidirectional_group_ids,
        shared_kv_states=shared_kv_states,
      )
    hidden_states = self.norm(hidden_states)
    if cache is not None:
      cache.past_seen_tokens += seq_len
    return hidden_states


class GemmaForCausalLM:
  def __init__(self, config: GemmaConfig):
    self.config = config
    if config.enable_moe_block and (config.num_experts is None or config.top_k_experts is None):
      raise ValueError("Gemma 4 MoE requires num_experts and top_k_experts")
    self.model = GemmaModel(config)
    if config.tie_word_embeddings:
      self.lm_head = {"weight": self.model.embed_tokens.weight}
    else:
      self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.device = self.model.embed_tokens.weight.device

  def logits(self, hidden_states: Tensor) -> Tensor:
    logits = hidden_states.linear(self.lm_head["weight"].transpose()) if self.config.tie_word_embeddings else self.lm_head(hidden_states)
    if self.config.final_logit_softcapping is not None:
      logits = (logits / self.config.final_logit_softcapping).tanh() * self.config.final_logit_softcapping
    return logits

  def __call__(
    self,
    input_ids: Tensor | None = None,
    cache: GemmaCache | None = None,
    attention_mask: Tensor | None = None,
    *,
    inputs_embeds: Tensor | None = None,
    per_layer_inputs: Tensor | None = None,
  ) -> tuple[Tensor, GemmaCache | None]:
    hidden_states = self.model(input_ids, cache=cache, attention_mask=attention_mask, inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs)
    return self.logits(hidden_states), cache

  def default_ignore_token_ids(self) -> set[int]:
    return set() if self.config.pad_token_id is None else {int(self.config.pad_token_id)}

  def prepare_labels(
    self,
    input_ids: Tensor | Sequence[int] | Sequence[Sequence[int]],
    *,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    ignore_token_ids: set[int] | None = None,
  ) -> Tensor:
    return build_causal_labels(
      input_ids,
      device=self.device if not isinstance(input_ids, Tensor) else input_ids.device,
      ignore_index=ignore_index,
      ignore_token_ids=self.default_ignore_token_ids() | (ignore_token_ids or set()),
    )

  def forward_ids(self, input_ids: Sequence[int] | Sequence[Sequence[int]], cache: GemmaCache | None = None) -> tuple[Tensor, GemmaCache | None]:
    return self(input_ids_tensor(input_ids, device=self.device), cache=cache)

  def loss(self, logits: Tensor, labels: Tensor | Sequence[int] | Sequence[Sequence[int]], *, ignore_index: int = DEFAULT_IGNORE_INDEX) -> Tensor:
    return causal_language_model_loss(logits, labels, ignore_index=ignore_index)

  def forward_loss_ids(
    self,
    input_ids: Sequence[int] | Sequence[Sequence[int]],
    *,
    labels: Tensor | Sequence[int] | Sequence[Sequence[int]] | None = None,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    ignore_token_ids: set[int] | None = None,
  ) -> Tensor:
    logits, _ = self.forward_ids(input_ids)
    labels_tensor = self.prepare_labels(input_ids, ignore_index=ignore_index, ignore_token_ids=ignore_token_ids) if labels is None else label_tensor(labels, device=logits.device)
    return self.loss(logits, labels_tensor, ignore_index=ignore_index)

  def sample_next(self, logits: Tensor, temperature: float = 0.0) -> Tensor:
    if temperature <= 0.0:
      return logits.argmax(axis=-1, keepdim=True).cast("int32")
    probs = (logits / temperature).softmax(-1)
    return probs.multinomial().cast("int32")

  def generate(
    self,
    input_ids: list[int],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    stop_token_ids: set[int] | None = None,
  ):
    cache = GemmaCache.empty(self.config.num_hidden_layers, max_length=len(input_ids) + max_new_tokens)
    logits, cache = self.forward_ids(input_ids, cache=cache)
    for _ in range(max_new_tokens):
      next_token = self.sample_next(logits[:, -1, :], temperature=temperature)
      token_id = int(next_token.item())
      yield token_id
      if stop_token_ids is not None and token_id in stop_token_ids:
        break
      logits, cache = self(next_token.reshape(1, 1), cache=cache)
