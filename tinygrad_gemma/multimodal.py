from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor, TinyJit, Variable, nn

from .config import GemmaAudioConfig, GemmaConditionalConfig, GemmaVisionConfig
from .model import (
  DEFAULT_IGNORE_INDEX,
  GemmaCache,
  GemmaForCausalLM,
  GemmaModel,
  RMSNorm,
  apply_activation,
  build_causal_labels,
  causal_language_model_loss,
  input_ids_tensor,
  label_tensor,
  repeat_kv,
  rotate_half,
  token_rows,
)


def cat_tensors(tensors: list[Tensor], dim: int) -> Tensor:
  if not tensors:
    raise ValueError("expected at least one tensor to concatenate")
  head, *rest = tensors
  return head if not rest else head.cat(*rest, dim=dim)


def tensor_from_numpy(value: np.ndarray, *, device: str, dtype: str | None = None) -> Tensor:
  return Tensor(value, device=device, dtype=dtype) if dtype is not None else Tensor(value, device=device)


def multimodal_bidirectional_group_ids(rows: list[list[int]], *, image_token_id: int | None, video_token_id: int | None) -> np.ndarray:
  group_rows = []
  vision_ids = {token_id for token_id in (image_token_id, video_token_id) if token_id is not None}
  for row in rows:
    current_group = -1
    previous_was_vision = False
    group_row = []
    for token_id in row:
      is_vision = token_id in vision_ids
      if is_vision and not previous_was_vision:
        current_group += 1
      group_row.append(current_group if is_vision else -1)
      previous_was_vision = is_vision
    group_rows.append(group_row)
  return np.array(group_rows, dtype=np.int32)


def flatten_valid_tokens(hidden_states: Tensor, valid_mask: Tensor) -> Tensor:
  pieces = [
    hidden_states[batch_idx : batch_idx + 1, token_idx : token_idx + 1, :].reshape(1, hidden_states.shape[-1])
    for batch_idx, batch_mask in enumerate(valid_mask.numpy().astype(bool).tolist())
    for token_idx, is_valid in enumerate(batch_mask)
    if is_valid
  ]
  if not pieces:
    return Tensor.empty(0, hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)
  return cat_tensors(pieces, dim=0)


def build_vision_key_mask(valid_mask: Tensor, dtype) -> Tensor:
  return valid_mask.reshape(valid_mask.shape[0], 1, 1, valid_mask.shape[1]).where(0.0, float("-inf")).cast(dtype)


def build_audio_block_mask(valid_mask: Tensor, config: GemmaAudioConfig, device: str) -> Tensor:
  valid = valid_mask.numpy().astype(bool)
  batch, seq_len = valid.shape
  chunk = config.attention_chunk_size
  left = config.attention_context_left - 1
  right = config.attention_context_right
  context = chunk + left + right
  num_blocks = (seq_len + chunk - 1) // chunk
  mask = np.zeros((batch, 1, num_blocks, chunk, context), dtype=np.bool_)
  for b in range(batch):
    for block in range(num_blocks):
      context_start = block * chunk - left
      for q_idx in range(chunk):
        q_abs = block * chunk + q_idx
        if q_abs >= seq_len or not valid[b, q_abs]:
          continue
        for ctx_idx in range(context):
          k_abs = context_start + ctx_idx
          if 0 <= k_abs < seq_len and valid[b, k_abs]:
            mask[b, 0, block, q_idx, ctx_idx] = True
  return tensor_from_numpy(mask, device=device, dtype="bool")


def vision_rotary_embedding(position_ids: Tensor, head_dim: int, rope_theta: float, dtype, device) -> tuple[Tensor, Tensor]:
  spatial_dim = head_dim // 2
  inv_freq = 1.0 / (rope_theta ** (Tensor.arange(0, spatial_dim, 2, device=device).float() / spatial_dim))
  cos_parts, sin_parts = [], []
  for axis in range(2):
    freqs = position_ids[:, :, axis].float().unsqueeze(-1) * inv_freq.reshape(1, 1, -1)
    emb = freqs.cat(freqs, dim=-1)
    cos_parts.append(emb.cos().cast(dtype))
    sin_parts.append(emb.sin().cast(dtype))
  return cat_tensors(cos_parts, dim=-1), cat_tensors(sin_parts, dim=-1)


def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1) -> Tensor:
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  return (x * cos) + (rotate_half(x) * sin)


def apply_multidimensional_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
  x0, x1 = x.chunk(2, dim=-1)
  cos0, cos1 = cos.chunk(2, dim=-1)
  sin0, sin1 = sin.chunk(2, dim=-1)
  return apply_rotary_pos_emb(x0, cos0, sin0, unsqueeze_dim=2).cat(
    apply_rotary_pos_emb(x1, cos1, sin1, unsqueeze_dim=2), dim=-1
  )


class LayerNormNoBias:
  def __init__(self, dim: int, eps: float):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x: Tensor) -> Tensor:
    return x.layernorm(eps=self.eps) * self.weight


class ClippableLinear:
  def __init__(self, config: GemmaVisionConfig | GemmaAudioConfig, in_features: int, out_features: int):
    self.use_clipped_linears = config.use_clipped_linears
    self.linear = nn.Linear(in_features, out_features, bias=False)
    if self.use_clipped_linears:
      self.input_min = Tensor(float("-inf"), requires_grad=False)
      self.input_max = Tensor(float("inf"), requires_grad=False)
      self.output_min = Tensor(float("-inf"), requires_grad=False)
      self.output_max = Tensor(float("inf"), requires_grad=False)

  def __call__(self, hidden_states: Tensor) -> Tensor:
    if self.use_clipped_linears:
      hidden_states = hidden_states.clamp(self.input_min, self.input_max)
    hidden_states = self.linear(hidden_states)
    if self.use_clipped_linears:
      hidden_states = hidden_states.clamp(self.output_min, self.output_max)
    return hidden_states


class Gemma4AudioRelPositionalEncoding:
  def __init__(self, config: GemmaAudioConfig):
    self.hidden_size = config.hidden_size
    self.context_size = config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right
    min_timescale = 1.0
    max_timescale = 10000.0
    num_timescales = self.hidden_size // 2
    log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales, dtype=np.float32) * -log_timescale_increment)
    self.inv_timescales_np = inv_timescales.reshape(1, 1, -1)

  def __call__(self, hidden_states: Tensor) -> Tensor:
    position_ids = Tensor(np.arange(self.context_size - 1, -1, -1, dtype=np.float32).reshape(-1, 1), device=hidden_states.device)
    inv_timescales = tensor_from_numpy(self.inv_timescales_np, device=hidden_states.device, dtype=hidden_states.dtype)
    scaled_time = position_ids * inv_timescales
    return scaled_time.sin().cat(scaled_time.cos(), dim=-1).cast(hidden_states.dtype)


class Gemma4AudioAttention:
  def __init__(self, config: GemmaAudioConfig, layer_idx: int):
    self.config = config
    self.layer_idx = layer_idx
    self.attention_logits_soft_cap = config.attention_logit_cap
    self.head_dim = config.hidden_size // config.num_attention_heads
    self.num_heads = config.num_attention_heads
    self.q_scale = (self.head_dim**-0.5) / math.log(2)
    self.k_scale = math.log(1 + math.e) / math.log(2)
    self.chunk_size = config.attention_chunk_size
    self.max_past_horizon = config.attention_context_left - 1
    self.max_future_horizon = config.attention_context_right
    self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon
    self.q_proj = ClippableLinear(config, config.hidden_size, self.num_heads * self.head_dim)
    self.k_proj = ClippableLinear(config, config.hidden_size, self.num_heads * self.head_dim)
    self.v_proj = ClippableLinear(config, config.hidden_size, self.num_heads * self.head_dim)
    self.post = ClippableLinear(config, config.hidden_size, config.hidden_size)
    self.relative_k_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
    self.per_dim_scale = Tensor.zeros(self.head_dim)
    self.softcap = float(self.attention_logits_soft_cap)

  def _convert_to_block(self, hidden_states: Tensor) -> Tensor:
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
    pad = num_blocks * self.chunk_size - seq_len
    hidden_states = hidden_states.pad(((0, 0), (0, pad), (0, 0), (0, 0)))
    return hidden_states.reshape(batch_size, num_blocks, self.chunk_size, num_heads, head_dim)

  def _extract_block_context(self, hidden_states: Tensor) -> Tensor:
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    pad_right = self.max_future_horizon + self.chunk_size - 1
    padded = hidden_states.pad(((0, 0), (self.max_past_horizon, pad_right), (0, 0), (0, 0)))
    num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
    windows = [padded[:, block * self.chunk_size : block * self.chunk_size + self.context_size, :, :].unsqueeze(1) for block in range(num_blocks)]
    return cat_tensors(windows, dim=1)

  def _rel_shift(self, x: Tensor) -> Tensor:
    batch_size, num_heads, num_blocks, block_size, position_length = x.shape
    x = x.pad(((0, 0), (0, 0), (0, 0), (0, 0), (0, self.context_size + 1 - position_length)))
    x = x.reshape(batch_size, num_heads, num_blocks, block_size * (self.context_size + 1))
    x = x[:, :, :, : block_size * self.context_size]
    return x.reshape(batch_size, num_heads, num_blocks, block_size, self.context_size)

  def __call__(self, hidden_states: Tensor, position_embeddings: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    batch_size, seq_length, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)
    query_states = self.q_proj(hidden_states).float().reshape(*hidden_shape)
    key_states = self.k_proj(hidden_states).float().reshape(*hidden_shape)
    value_states = self.v_proj(hidden_states).float().reshape(*hidden_shape)

    query_states = query_states * self.q_scale * self.per_dim_scale.softplus()
    key_states = key_states * self.k_scale

    query_states = self._convert_to_block(query_states)
    key_states = self._extract_block_context(key_states)
    value_states = self._extract_block_context(value_states)
    num_blocks = query_states.shape[1]

    relative_key_states = self.relative_k_proj(position_embeddings).reshape(-1, self.num_heads, self.head_dim).cast(query_states.dtype)

    queries = query_states.permute(0, 3, 1, 2, 4)
    matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)

    queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
    matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
    matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
    matrix_bd = self._rel_shift(matrix_bd)

    attn_weights = matrix_ac + matrix_bd
    attn_weights = (attn_weights / self.softcap).tanh() * self.softcap
    if attention_mask is not None:
      attn_weights = attention_mask.where(attn_weights, self.config.attention_invalid_logits_value)
    attn_weights = attn_weights.softmax(-1).cast(value_states.dtype)
    attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
    attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, num_blocks * self.chunk_size, -1)
    attn_output = attn_output[:, :seq_length, :]
    attn_output = self.post(attn_output.cast(self.post.linear.weight.dtype))
    return attn_output, attn_weights


class Gemma4AudioSubSampleConvProjectionLayer:
  def __init__(self, in_channels: int, out_channels: int, norm_eps: float):
    self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), stride=(2, 2), padding=1, bias=False)
    self.norm = LayerNormNoBias(out_channels, norm_eps)

  def __call__(self, hidden_states: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
    if mask is not None:
      hidden_states = hidden_states * mask.unsqueeze(1).unsqueeze(-1).cast(hidden_states.dtype)
    hidden_states = self.conv(hidden_states.cast(self.conv.weight.dtype))
    hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).relu().permute(0, 3, 1, 2)
    if mask is not None:
      mask = mask[:, ::2]
    return hidden_states, mask


class Gemma4AudioSubSampleConvProjection:
  def __init__(self, config: GemmaAudioConfig):
    self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(1, config.subsampling_conv_channels[0], config.rms_norm_eps)
    self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
      config.subsampling_conv_channels[0], config.subsampling_conv_channels[1], config.rms_norm_eps
    )
    proj_input_dim = (config.subsampling_conv_channels[0] // 4) * config.subsampling_conv_channels[1]
    self.input_proj_linear = nn.Linear(proj_input_dim, config.hidden_size, bias=False)

  def __call__(self, input_features: Tensor, input_features_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
    hidden_states = input_features.unsqueeze(1)
    hidden_states, mask = self.layer0(hidden_states, input_features_mask)
    hidden_states, mask = self.layer1(hidden_states, mask)
    batch_size, _, seq_len, _ = hidden_states.shape
    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, seq_len, -1)
    return self.input_proj_linear(hidden_states), mask


class Gemma4AudioFeedForward:
  def __init__(self, config: GemmaAudioConfig):
    self.config = config
    self.ffw_layer_1 = ClippableLinear(config, config.hidden_size, config.hidden_size * 4)
    self.ffw_layer_2 = ClippableLinear(config, config.hidden_size * 4, config.hidden_size)
    self.pre_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_layer_scale = config.residual_weight

  def __call__(self, hidden_states: Tensor) -> Tensor:
    residual = hidden_states
    hidden_states = hidden_states.clamp(-self.config.gradient_clipping, self.config.gradient_clipping)
    hidden_states = self.pre_layer_norm(hidden_states)
    hidden_states = self.ffw_layer_1(hidden_states)
    hidden_states = apply_activation(self.config.activation_name, hidden_states)
    hidden_states = self.ffw_layer_2(hidden_states)
    hidden_states = hidden_states.clamp(-self.config.gradient_clipping, self.config.gradient_clipping)
    hidden_states = self.post_layer_norm(hidden_states)
    return hidden_states * self.post_layer_scale + residual


class Gemma4AudioCausalConv1d:
  def __init__(self, config: GemmaAudioConfig):
    self.kernel_size = config.conv_kernel_size
    self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, config.conv_kernel_size, groups=config.hidden_size, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    x = x.pad(((0, 0), (0, 0), (self.kernel_size - 1, 0)))
    return self.conv(x)


class Gemma4AudioLightConv1d:
  def __init__(self, config: GemmaAudioConfig):
    self.config = config
    self.linear_start = ClippableLinear(config, config.hidden_size, config.hidden_size * 2)
    self.linear_end = ClippableLinear(config, config.hidden_size, config.hidden_size)
    self.depthwise_conv1d = Gemma4AudioCausalConv1d(config)
    self.pre_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.conv_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def __call__(self, hidden_states: Tensor) -> Tensor:
    residual = hidden_states
    hidden_states = self.pre_layer_norm(hidden_states)
    hidden_states = self.linear_start(hidden_states)
    left, right = hidden_states.chunk(2, dim=-1)
    hidden_states = left * right.sigmoid()
    hidden_states = self.depthwise_conv1d(hidden_states.transpose(1, 2)).transpose(1, 2)
    hidden_states = hidden_states.clamp(-self.config.gradient_clipping, self.config.gradient_clipping)
    hidden_states = self.conv_norm(hidden_states)
    hidden_states = apply_activation(self.config.activation_name, hidden_states)
    hidden_states = self.linear_end(hidden_states)
    return hidden_states + residual


class Gemma4AudioLayer:
  def __init__(self, config: GemmaAudioConfig, layer_idx: int):
    self.config = config
    self.feed_forward1 = Gemma4AudioFeedForward(config)
    self.feed_forward2 = Gemma4AudioFeedForward(config)
    self.self_attn = Gemma4AudioAttention(config, layer_idx)
    self.lconv1d = Gemma4AudioLightConv1d(config)
    self.norm_pre_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.norm_post_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.norm_out = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def __call__(self, hidden_states: Tensor, attention_mask: Tensor | None, position_embeddings: Tensor) -> Tensor:
    hidden_states = self.feed_forward1(hidden_states)
    residual = hidden_states
    hidden_states = hidden_states.clamp(-self.config.gradient_clipping, self.config.gradient_clipping)
    hidden_states = self.norm_pre_attn(hidden_states)
    hidden_states, _ = self.self_attn(hidden_states, position_embeddings, attention_mask)
    hidden_states = hidden_states.clamp(-self.config.gradient_clipping, self.config.gradient_clipping)
    hidden_states = self.norm_post_attn(hidden_states)
    hidden_states = hidden_states + residual
    hidden_states = self.lconv1d(hidden_states)
    hidden_states = self.feed_forward2(hidden_states)
    hidden_states = hidden_states.clamp(-self.config.gradient_clipping, self.config.gradient_clipping)
    return self.norm_out(hidden_states)


class Gemma4AudioModel:
  def __init__(self, config: GemmaAudioConfig):
    self.config = config
    self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(config)
    self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config)
    self.layers = [Gemma4AudioLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
    self.output_proj = nn.Linear(config.hidden_size, config.output_proj_dims, bias=True)

  def __call__(self, input_features: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
    hidden_states, output_mask = self.subsample_conv_projection(input_features, attention_mask)
    position_embeddings = self.rel_pos_enc(hidden_states)
    blocked_mask = build_audio_block_mask(output_mask if output_mask is not None else Tensor.ones(*hidden_states.shape[:2], dtype="bool"), self.config, hidden_states.device)
    for encoder_layer in self.layers:
      hidden_states = encoder_layer(hidden_states, blocked_mask, position_embeddings)
    return self.output_proj(hidden_states), output_mask


class Gemma4VisionPatchEmbedder:
  def __init__(self, config: GemmaVisionConfig):
    self.config = config
    self.hidden_size = config.hidden_size
    self.patch_size = config.patch_size
    self.position_embedding_size = config.position_embedding_size
    self.input_proj = nn.Linear(3 * self.patch_size**2, self.hidden_size, bias=False)
    self.position_embedding_table = Tensor.ones(2, self.position_embedding_size, self.hidden_size)

  def __call__(self, pixel_values: Tensor, pixel_position_ids: Tensor, padding_positions: Tensor) -> Tensor:
    pixel_values = 2 * (pixel_values - 0.5)
    hidden_states = self.input_proj(pixel_values.cast(self.input_proj.weight.dtype))
    clamped = pixel_position_ids.clamp(min_=0).cast("int32")
    x_pos = self.position_embedding_table[0][clamped[:, :, 0]]
    y_pos = self.position_embedding_table[1][clamped[:, :, 1]]
    position_embeddings = padding_positions.unsqueeze(-1).where(0.0, x_pos + y_pos)
    return hidden_states + position_embeddings


class Gemma4VisionPooler:
  def __init__(self, config: GemmaVisionConfig):
    self.hidden_size = config.hidden_size
    self.root_hidden_size = self.hidden_size**0.5

  def _avg_pool_by_positions(self, hidden_states: Tensor, pixel_position_ids: Tensor, length: int) -> tuple[Tensor, Tensor]:
    batch_size, input_seq_len, hidden_size = hidden_states.shape
    k = int((input_seq_len // length) ** 0.5)
    if k * k * length != input_seq_len:
      raise ValueError(f"Cannot pool {hidden_states.shape} to {length}")

    pooled_batches, mask_batches = [], []
    positions = pixel_position_ids.numpy()
    for b in range(batch_size):
      clamped = np.clip(positions[b], 0, None)
      max_x = int(clamped[:, 0].max()) + 1
      kernel_idxs = (clamped[:, 0] // k) + (max_x // k) * (clamped[:, 1] // k)
      batch_out, batch_mask = [], []
      for out_idx in range(length):
        selected = np.nonzero(kernel_idxs == out_idx)[0].tolist()
        if selected:
          pieces = [hidden_states[b : b + 1, idx : idx + 1, :] for idx in selected]
          batch_out.append(cat_tensors(pieces, dim=1).sum(axis=1, keepdim=True) / float(k * k))
          batch_mask.append(True)
        else:
          batch_out.append(Tensor.zeros(1, 1, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype))
          batch_mask.append(False)
      pooled_batches.append(cat_tensors(batch_out, dim=1))
      mask_batches.append(batch_mask)
    pooled = cat_tensors(pooled_batches, dim=0)
    mask = tensor_from_numpy(np.array(mask_batches, dtype=np.bool_), device=hidden_states.device, dtype="bool")
    return pooled, mask

  def __call__(self, hidden_states: Tensor, pixel_position_ids: Tensor, padding_positions: Tensor, output_length: int) -> tuple[Tensor, Tensor]:
    if output_length > hidden_states.shape[1]:
      raise ValueError(f"Cannot output more soft tokens ({output_length}) than patches ({hidden_states.shape[1]})")
    hidden_states = padding_positions.unsqueeze(-1).where(0.0, hidden_states)
    if hidden_states.shape[1] != output_length:
      hidden_states, padding_positions = self._avg_pool_by_positions(hidden_states, pixel_position_ids, output_length)
    else:
      padding_positions = padding_positions.logical_not()
    return hidden_states * self.root_hidden_size, padding_positions


class Gemma4VisionMLP:
  def __init__(self, config: GemmaVisionConfig):
    self.config = config
    self.gate_proj = ClippableLinear(config, config.hidden_size, config.intermediate_size)
    self.up_proj = ClippableLinear(config, config.hidden_size, config.intermediate_size)
    self.down_proj = ClippableLinear(config, config.intermediate_size, config.hidden_size)

  def __call__(self, x: Tensor) -> Tensor:
    return self.down_proj(apply_activation(self.config.activation_name, self.gate_proj(x)) * self.up_proj(x))


class Gemma4VisionAttention:
  def __init__(self, config: GemmaVisionConfig, layer_idx: int):
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = config.head_dim
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.attention_dropout = config.attention_dropout
    self.q_proj = ClippableLinear(config, config.hidden_size, config.num_attention_heads * self.head_dim)
    self.k_proj = ClippableLinear(config, config.hidden_size, config.num_key_value_heads * self.head_dim)
    self.v_proj = ClippableLinear(config, config.hidden_size, config.num_key_value_heads * self.head_dim)
    self.o_proj = ClippableLinear(config, config.num_attention_heads * self.head_dim, config.hidden_size)
    self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
    self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
    self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

  def __call__(self, hidden_states: Tensor, position_embeddings: tuple[Tensor, Tensor], attention_mask: Tensor, position_ids: Tensor) -> Tensor:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings
    query_states = self.q_norm(self.q_proj(hidden_states).reshape(*hidden_shape))
    query_states = apply_multidimensional_rope(query_states, cos, sin).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).reshape(*hidden_shape))
    key_states = apply_multidimensional_rope(key_states, cos, sin).transpose(1, 2)
    value_states = self.v_norm(self.v_proj(hidden_states).reshape(*hidden_shape)).transpose(1, 2)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = query_states.matmul(key_states.transpose(-2, -1)).float()
    attn_weights = attn_weights + attention_mask.float()
    attn_output = attn_weights.softmax(-1).cast(query_states.dtype) @ value_states
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
    return self.o_proj(attn_output)


class Gemma4VisionEncoderLayer:
  def __init__(self, config: GemmaVisionConfig, layer_idx: int):
    self.self_attn = Gemma4VisionAttention(config, layer_idx)
    self.mlp = Gemma4VisionMLP(config)
    self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def __call__(self, hidden_states: Tensor, position_embeddings: tuple[Tensor, Tensor], attention_mask: Tensor, position_ids: Tensor) -> Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask, position_ids)
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    return residual + hidden_states


class Gemma4VisionEncoder:
  def __init__(self, config: GemmaVisionConfig):
    self.config = config
    self.layers = [Gemma4VisionEncoderLayer(config, i) for i in range(config.num_hidden_layers)]

  def __call__(self, inputs_embeds: Tensor, attention_mask: Tensor, pixel_position_ids: Tensor) -> Tensor:
    hidden_states = inputs_embeds
    position_embeddings = vision_rotary_embedding(
      pixel_position_ids.clamp(min_=0),
      self.config.head_dim,
      float(self.config.rope_parameters["rope_theta"]),
      hidden_states.dtype,
      hidden_states.device,
    )
    key_mask = build_vision_key_mask(attention_mask, hidden_states.dtype)
    for layer in self.layers:
      hidden_states = layer(hidden_states, position_embeddings, key_mask, pixel_position_ids.clamp(min_=0))
      hidden_states = attention_mask.unsqueeze(-1).where(hidden_states, 0.0)
    return hidden_states


class Gemma4VisionModel:
  def __init__(self, config: GemmaVisionConfig):
    self.config = config
    self.patch_embedder = Gemma4VisionPatchEmbedder(config)
    self.encoder = Gemma4VisionEncoder(config)
    self.pooler = Gemma4VisionPooler(config)
    if self.config.standardize:
      self.std_bias = Tensor.empty(self.config.hidden_size)
      self.std_scale = Tensor.empty(self.config.hidden_size)

  def __call__(self, pixel_values: Tensor, pixel_position_ids: Tensor) -> Tensor:
    output_length = pixel_values.shape[1] // (self.config.pooling_kernel_size * self.config.pooling_kernel_size)
    padding_positions = (pixel_position_ids == -1).all(axis=-1)
    inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
    hidden_states = self.encoder(inputs_embeds, padding_positions.logical_not(), pixel_position_ids)
    hidden_states, pooler_mask = self.pooler(hidden_states, pixel_position_ids, padding_positions, output_length)
    hidden_states = flatten_valid_tokens(hidden_states, pooler_mask)
    if self.config.standardize:
      hidden_states = (hidden_states - self.std_bias) * self.std_scale
    return hidden_states


class Gemma4MultimodalEmbedder:
  def __init__(self, multimodal_config: GemmaVisionConfig | GemmaAudioConfig, text_config):
    self.multimodal_hidden_size = getattr(multimodal_config, "output_proj_dims", multimodal_config.hidden_size)
    self.embedding_projection = nn.Linear(self.multimodal_hidden_size, text_config.hidden_size, bias=False)
    self.embedding_pre_projection_norm = RMSNorm(self.multimodal_hidden_size, eps=multimodal_config.rms_norm_eps, with_scale=False)

  def __call__(self, inputs_embeds: Tensor) -> Tensor:
    return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))


class GemmaConditionalModel:
  def __init__(self, config: GemmaConditionalConfig):
    self.config = config
    self.language_model = GemmaModel(config.text_config)
    self.vision_tower = Gemma4VisionModel(config.vision_config) if config.vision_config is not None else None
    self.audio_tower = Gemma4AudioModel(config.audio_config) if config.audio_config is not None else None
    self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config) if config.vision_config is not None else None
    self.embed_audio = Gemma4MultimodalEmbedder(config.audio_config, config.text_config) if config.audio_config is not None else None

  def get_image_features(self, pixel_values: Tensor, image_position_ids: Tensor) -> Tensor:
    if self.vision_tower is None or self.embed_vision is None:
      raise ValueError("image features requested but this checkpoint has no vision tower")
    return self.embed_vision(self.vision_tower(pixel_values, image_position_ids))

  def get_audio_features(self, input_features: Tensor, input_features_mask: Tensor) -> Tensor:
    if self.audio_tower is None or self.embed_audio is None:
      raise ValueError("audio features requested but this checkpoint has no audio tower")
    audio_hidden, output_mask = self.audio_tower(input_features, input_features_mask)
    projected = self.embed_audio(audio_hidden)
    if output_mask is None:
      return projected.reshape(-1, projected.shape[-1])
    return flatten_valid_tokens(projected, output_mask)

  def merge_multimodal_embeddings(
    self,
    input_ids: list[int] | list[list[int]],
    base_embeds: Tensor,
    *,
    image_features: Tensor | None = None,
    audio_features: Tensor | None = None,
  ) -> Tensor:
    image_index = 0
    audio_index = 0
    batch_pieces: list[Tensor] = []
    rows = token_rows(input_ids)
    if len(rows) != base_embeds.shape[0]:
      raise ValueError(f"input row count {len(rows)} does not match embedding batch {base_embeds.shape[0]}")
    image_total = 0 if image_features is None else image_features.shape[0]
    audio_total = 0 if audio_features is None else audio_features.shape[0]
    for batch_idx, row in enumerate(rows):
      row_pieces: list[Tensor] = []
      for pos, token_id in enumerate(row):
        if token_id == self.config.image_token_id:
          if image_features is None or image_index >= image_total:
            raise ValueError("image placeholder count does not match image feature count")
          row_pieces.append(image_features[image_index : image_index + 1].reshape(1, 1, -1))
          image_index += 1
        elif token_id == self.config.audio_token_id:
          if audio_features is None or audio_index >= audio_total:
            raise ValueError("audio placeholder count does not match audio feature count")
          row_pieces.append(audio_features[audio_index : audio_index + 1].reshape(1, 1, -1))
          audio_index += 1
        else:
          row_pieces.append(base_embeds[batch_idx : batch_idx + 1, pos : pos + 1, :])
      batch_pieces.append(cat_tensors(row_pieces, dim=1))
    if image_index != image_total or audio_index != audio_total:
      raise ValueError("unused multimodal features remain after prompt merge")
    return cat_tensors(batch_pieces, dim=0)


class GemmaForConditionalGeneration:
  def __init__(self, config: GemmaConditionalConfig):
    self.config = config
    self.model = GemmaConditionalModel(config)
    if config.tie_word_embeddings:
      self.lm_head = {"weight": self.model.language_model.embed_tokens.weight}
    else:
      self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
    self.device = self.model.language_model.embed_tokens.weight.device
    self._last_rollout_jit: TinyJit | None = None

  def logits(self, hidden_states: Tensor) -> Tensor:
    if isinstance(self.lm_head, dict):
      logits = hidden_states.linear(self.lm_head["weight"].transpose())
    else:
      logits = self.lm_head(hidden_states)
    if self.config.text_config.final_logit_softcapping is not None:
      logits = (logits / self.config.text_config.final_logit_softcapping).tanh() * self.config.text_config.final_logit_softcapping
    return logits

  def __call__(
    self,
    input_ids: Tensor | None = None,
    cache: GemmaCache | None = None,
    *,
    inputs_embeds: Tensor | None = None,
    per_layer_inputs: Tensor | None = None,
    bidirectional_group_ids: Tensor | None = None,
  ) -> tuple[Tensor, GemmaCache | None]:
    hidden_states = self.model.language_model(
      input_ids=input_ids,
      cache=cache,
      inputs_embeds=inputs_embeds,
      per_layer_inputs=per_layer_inputs,
      bidirectional_group_ids=bidirectional_group_ids,
    )
    return self.logits(hidden_states), cache

  def forward_ids(
    self,
    input_ids: list[int] | list[list[int]],
    cache: GemmaCache | None = None,
    *,
    pixel_values: np.ndarray | None = None,
    image_position_ids: np.ndarray | None = None,
    input_features: np.ndarray | None = None,
    input_features_mask: np.ndarray | None = None,
  ) -> tuple[Tensor, GemmaCache | None]:
    if cache is not None and cache.past_seen_tokens > 0 and any(v is not None for v in (pixel_values, image_position_ids, input_features, input_features_mask)):
      raise ValueError("multimodal features are only valid on the first decoding step")
    if cache is not None and cache.past_seen_tokens > 0:
      return self(input_ids_tensor(input_ids, device=self.device), cache=cache)

    rows = token_rows(input_ids)
    llm_input_ids = [
      [self.config.text_config.pad_token_id if tok in (self.config.image_token_id, self.config.audio_token_id) else tok for tok in row]
      for row in rows
    ]
    input_ids_tt = input_ids_tensor(llm_input_ids, device=self.device)
    base_embeds = self.model.language_model.embed_tokens(input_ids_tt)
    per_layer_inputs = None
    if self.model.language_model.hidden_size_per_layer_input:
      token_identity = self.model.language_model.get_per_layer_inputs(input_ids_tt)
      per_layer_inputs = self.model.language_model.project_per_layer_inputs(base_embeds, token_identity)

    image_features = None
    if pixel_values is not None and image_position_ids is not None:
      image_features = self.model.get_image_features(
        tensor_from_numpy(pixel_values, device=self.device),
        tensor_from_numpy(image_position_ids, device=self.device, dtype="int32"),
      )
    audio_features = None
    if input_features is not None and input_features_mask is not None:
      audio_features = self.model.get_audio_features(
        tensor_from_numpy(input_features, device=self.device),
        tensor_from_numpy(input_features_mask, device=self.device, dtype="bool"),
      )

    inputs_embeds = self.model.merge_multimodal_embeddings(rows, base_embeds, image_features=image_features, audio_features=audio_features)
    bidirectional_group_ids = None
    if self.config.text_config.use_bidirectional_attention == "vision":
      bidirectional_group_ids = tensor_from_numpy(
        multimodal_bidirectional_group_ids(rows, image_token_id=self.config.image_token_id, video_token_id=self.config.video_token_id),
        device=self.device,
        dtype="int32",
      )
    return self(None, cache=cache, inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs, bidirectional_group_ids=bidirectional_group_ids)

  def sample_next(self, logits: Tensor, temperature: float = 0.0) -> Tensor:
    if temperature <= 0.0:
      return logits.argmax(axis=-1, keepdim=True).cast("int32")
    return (logits / temperature).softmax(-1).multinomial().cast("int32")

  def _rollout_next_token(self, token: Tensor, start_pos, cache: GemmaCache, temperature: float) -> Tensor:
    concrete_start = cache.past_seen_tokens
    cache.past_seen_tokens = start_pos
    logits, _ = self(token.reshape(1, 1), cache=cache)
    cache.past_seen_tokens = concrete_start
    return self.sample_next(logits[:, -1, :], temperature=temperature)

  def generate(
    self,
    input_ids: list[int],
    *,
    pixel_values: np.ndarray | None = None,
    image_position_ids: np.ndarray | None = None,
    input_features: np.ndarray | None = None,
    input_features_mask: np.ndarray | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    stop_token_ids: set[int] | None = None,
  ):
    if max_new_tokens <= 0:
      return
    cache = GemmaCache.empty(self.config.text_config.num_hidden_layers, max_length=len(input_ids) + max_new_tokens)
    logits, cache = self.forward_ids(
      input_ids,
      cache=cache,
      pixel_values=pixel_values,
      image_position_ids=image_position_ids,
      input_features=input_features,
      input_features_mask=input_features_mask,
    )
    next_token = self.sample_next(logits[:, -1, :], temperature=temperature)
    max_start_pos = max(1, (cache.max_length or len(input_ids) + max_new_tokens) - 1)
    rollout_jit = TinyJit(lambda token, start_pos: self._rollout_next_token(token, start_pos, cache, temperature))
    self._last_rollout_jit = rollout_jit
    for idx in range(max_new_tokens):
      token_id = int(next_token.item())
      yield token_id
      if stop_token_ids is not None and token_id in stop_token_ids:
        break
      if idx == max_new_tokens - 1:
        break
      start_pos = cache.past_seen_tokens
      start_var = Variable("gemma_start_pos", 0, max_start_pos).bind(start_pos)
      next_token = rollout_jit(next_token.reshape(1, 1).contiguous(), start_var)
      cache.set_active_length(start_pos + 1)

  def default_ignore_token_ids(self) -> set[int]:
    ignore_ids = set()
    if self.config.text_config.pad_token_id is not None:
      ignore_ids.add(int(self.config.text_config.pad_token_id))
    if self.config.image_token_id is not None:
      ignore_ids.add(int(self.config.image_token_id))
    if self.config.audio_token_id is not None:
      ignore_ids.add(int(self.config.audio_token_id))
    return ignore_ids

  def prepare_labels(
    self,
    input_ids: Tensor | list[int] | list[list[int]],
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

  def loss(self, logits: Tensor, labels: Tensor | list[int] | list[list[int]], *, ignore_index: int = DEFAULT_IGNORE_INDEX) -> Tensor:
    return causal_language_model_loss(logits, labels, ignore_index=ignore_index)

  def forward_loss_ids(
    self,
    input_ids: list[int] | list[list[int]],
    *,
    labels: Tensor | list[int] | list[list[int]] | None = None,
    pixel_values: np.ndarray | None = None,
    image_position_ids: np.ndarray | None = None,
    input_features: np.ndarray | None = None,
    input_features_mask: np.ndarray | None = None,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    ignore_token_ids: set[int] | None = None,
  ) -> Tensor:
    logits, _ = self.forward_ids(
      input_ids,
      pixel_values=pixel_values,
      image_position_ids=image_position_ids,
      input_features=input_features,
      input_features_mask=input_features_mask,
    )
    labels_tensor = self.prepare_labels(input_ids, ignore_index=ignore_index, ignore_token_ids=ignore_token_ids) if labels is None else label_tensor(labels, device=logits.device)
    return self.loss(logits, labels_tensor, ignore_index=ignore_index)
