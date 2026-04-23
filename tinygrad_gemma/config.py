from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def load_config_dict(path: str | Path) -> dict[str, Any]:
  return json.loads(Path(path).read_text())


@dataclass(slots=True)
class GemmaConfig:
  model_type: str = "gemma4"
  vocab_size: int = 262144
  hidden_size: int = 2304
  intermediate_size: int = 9216
  num_hidden_layers: int = 30
  num_attention_heads: int = 8
  num_key_value_heads: int = 4
  head_dim: int = 256
  hidden_act: str = "gelu_pytorch_tanh"
  hidden_activation: str | None = None
  max_position_embeddings: int = 131072
  rms_norm_eps: float = 1e-6
  rope_theta: float = 10000.0
  attention_bias: bool = False
  attention_dropout: float = 0.0
  tie_word_embeddings: bool = True
  query_pre_attn_scalar: int | None = None
  sliding_window: int | None = 512
  layer_types: list[str] | None = None
  final_logit_softcapping: float | None = 30.0
  attn_logit_softcapping: float | None = None
  bos_token_id: int | None = 2
  eos_token_id: int | list[int] | None = 1
  pad_token_id: int | None = 0
  use_cache: bool = True
  rope_parameters: dict[str, Any] | None = None
  use_bidirectional_attention: str | None = None
  vocab_size_per_layer_input: int | None = 262144
  hidden_size_per_layer_input: int = 256
  num_global_key_value_heads: int | None = None
  global_head_dim: int | None = 512
  attention_k_eq_v: bool = False
  num_kv_shared_layers: int = 0
  enable_moe_block: bool = False
  use_double_wide_mlp: bool = False
  num_experts: int | None = None
  top_k_experts: int | None = None
  moe_intermediate_size: int | None = None

  def __post_init__(self):
    if self.model_type not in ("gemma4", "gemma4_text"):
      raise ValueError(f"tinygrad-gemma only supports Gemma 4 text checkpoints, got {self.model_type!r}")
    if self.num_attention_heads % self.num_key_value_heads != 0:
      raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if self.use_bidirectional_attention == "all" and self.sliding_window is not None:
      self.sliding_window = (self.sliding_window // 2) + 1
    if self.vocab_size_per_layer_input is None:
      self.vocab_size_per_layer_input = self.vocab_size
    if self.num_global_key_value_heads is None:
      self.num_global_key_value_heads = self.num_key_value_heads
    if self.global_head_dim is None:
      self.global_head_dim = self.head_dim
    if self.layer_types is None:
      sliding_window_pattern = 6
      self.layer_types = [
        "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
        for i in range(self.num_hidden_layers)
      ]
    if self.layer_types and self.layer_types[-1] != "full_attention":
      self.layer_types[-1] = "full_attention"
    if self.rope_parameters is None:
      self.rope_parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
      }
    if self.hidden_activation is None:
      self.hidden_activation = self.hidden_act
    if self.query_pre_attn_scalar is None:
      self.query_pre_attn_scalar = self.head_dim

  @property
  def activation_name(self) -> str:
    return self.hidden_activation or self.hidden_act

  @property
  def family(self) -> str:
    return "gemma4"

  @property
  def num_key_value_groups(self) -> int:
    return self.num_attention_heads // self.num_key_value_heads

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "GemmaConfig":
    model_type = data.get("model_type", "gemma4")
    if model_type == "gemma4":
      text_config = data.get("text_config") or {}
      data = {
        **text_config,
        "model_type": "gemma4",
        "tie_word_embeddings": data.get("tie_word_embeddings", text_config.get("tie_word_embeddings", True)),
      }
      model_type = "gemma4"
    if model_type not in ("gemma4", "gemma4_text"):
      raise ValueError(f"tinygrad-gemma only supports Gemma 4 text checkpoints, got {model_type!r}")
    rope_parameters = data.get("rope_parameters") or {}
    rope_theta = data.get("rope_theta", rope_parameters.get("rope_theta", 10000.0))
    hidden_activation = data.get("hidden_activation", data.get("hidden_act", "gelu_pytorch_tanh"))
    moe_intermediate_size = data.get("moe_intermediate_size", data.get("expert_intermediate_size"))
    defaults = {
      "vocab_size": 262144,
      "hidden_size": 2304,
      "intermediate_size": 9216,
      "num_hidden_layers": 30,
      "num_attention_heads": 8,
      "num_key_value_heads": 4,
      "head_dim": 256,
      "hidden_act": "gelu_pytorch_tanh",
      "max_position_embeddings": 131072,
      "sliding_window": 512,
      "hidden_size_per_layer_input": 256,
      "vocab_size_per_layer_input": 262144,
      "global_head_dim": 512,
      "final_logit_softcapping": 30.0,
    }
    return cls(
      model_type=model_type,
      vocab_size=data.get("vocab_size", defaults["vocab_size"]),
      hidden_size=data.get("hidden_size", defaults["hidden_size"]),
      intermediate_size=data.get("intermediate_size", defaults["intermediate_size"]),
      num_hidden_layers=data.get("num_hidden_layers", defaults["num_hidden_layers"]),
      num_attention_heads=data.get("num_attention_heads", defaults["num_attention_heads"]),
      num_key_value_heads=data.get("num_key_value_heads", defaults["num_key_value_heads"]),
      head_dim=data.get("head_dim", defaults["head_dim"]),
      hidden_act=data.get("hidden_act", hidden_activation),
      hidden_activation=hidden_activation,
      max_position_embeddings=data.get("max_position_embeddings", defaults["max_position_embeddings"]),
      rms_norm_eps=data.get("rms_norm_eps", 1e-6),
      rope_theta=rope_theta,
      attention_bias=data.get("attention_bias", False),
      attention_dropout=float(data.get("attention_dropout", 0.0)),
      tie_word_embeddings=data.get("tie_word_embeddings", True),
      query_pre_attn_scalar=data.get("query_pre_attn_scalar"),
      sliding_window=data.get("sliding_window", defaults["sliding_window"]),
      layer_types=data.get("layer_types"),
      final_logit_softcapping=data.get("final_logit_softcapping", defaults["final_logit_softcapping"]),
      attn_logit_softcapping=data.get("attn_logit_softcapping"),
      bos_token_id=data.get("bos_token_id", 2),
      eos_token_id=data.get("eos_token_id", 1),
      pad_token_id=data.get("pad_token_id", 0),
      use_cache=data.get("use_cache", True),
      rope_parameters=data.get("rope_parameters"),
      use_bidirectional_attention=data.get("use_bidirectional_attention"),
      vocab_size_per_layer_input=data.get("vocab_size_per_layer_input", defaults["vocab_size_per_layer_input"]),
      hidden_size_per_layer_input=data.get("hidden_size_per_layer_input", defaults["hidden_size_per_layer_input"]),
      num_global_key_value_heads=data.get("num_global_key_value_heads"),
      global_head_dim=data.get("global_head_dim", defaults["global_head_dim"]),
      attention_k_eq_v=data.get("attention_k_eq_v", False),
      num_kv_shared_layers=data.get("num_kv_shared_layers", 0),
      enable_moe_block=data.get("enable_moe_block", False),
      use_double_wide_mlp=data.get("use_double_wide_mlp", False),
      num_experts=data.get("num_experts"),
      top_k_experts=data.get("top_k_experts"),
      moe_intermediate_size=moe_intermediate_size,
    )

  @classmethod
  def from_path(cls, path: str | Path) -> "GemmaConfig":
    return cls.from_dict(load_config_dict(path))

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(slots=True)
class GemmaVisionConfig:
  model_type: str = "gemma4_vision"
  hidden_size: int = 768
  intermediate_size: int = 3072
  num_hidden_layers: int = 16
  num_attention_heads: int = 12
  num_key_value_heads: int = 12
  head_dim: int = 64
  hidden_activation: str = "gelu_pytorch_tanh"
  rms_norm_eps: float = 1e-6
  max_position_embeddings: int = 131072
  attention_bias: bool = False
  attention_dropout: float = 0.0
  rope_parameters: dict[str, Any] | None = None
  pooling_kernel_size: int = 3
  patch_size: int = 16
  position_embedding_size: int = 10240
  default_output_length: int = 280
  use_clipped_linears: bool = False
  standardize: bool = False
  initializer_range: float = 0.02

  def __post_init__(self):
    if self.model_type != "gemma4_vision":
      raise ValueError(f"unsupported Gemma 4 vision config type {self.model_type!r}")
    if self.rope_parameters is None:
      self.rope_parameters = {"rope_type": "default", "rope_theta": 100.0}

  @property
  def activation_name(self) -> str:
    return self.hidden_activation

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "GemmaVisionConfig":
    return cls(
      model_type=data.get("model_type", "gemma4_vision"),
      hidden_size=data.get("hidden_size", 768),
      intermediate_size=data.get("intermediate_size", 3072),
      num_hidden_layers=data.get("num_hidden_layers", 16),
      num_attention_heads=data.get("num_attention_heads", 12),
      num_key_value_heads=data.get("num_key_value_heads", 12),
      head_dim=data.get("head_dim", 64),
      hidden_activation=data.get("hidden_activation", "gelu_pytorch_tanh"),
      rms_norm_eps=data.get("rms_norm_eps", 1e-6),
      max_position_embeddings=data.get("max_position_embeddings", 131072),
      attention_bias=data.get("attention_bias", False),
      attention_dropout=float(data.get("attention_dropout", 0.0)),
      rope_parameters=data.get("rope_parameters"),
      pooling_kernel_size=data.get("pooling_kernel_size", 3),
      patch_size=data.get("patch_size", 16),
      position_embedding_size=data.get("position_embedding_size", 10240),
      default_output_length=data.get("default_output_length", 280),
      use_clipped_linears=data.get("use_clipped_linears", False),
      standardize=data.get("standardize", False),
      initializer_range=float(data.get("initializer_range", 0.02)),
    )

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(slots=True)
class GemmaAudioConfig:
  model_type: str = "gemma4_audio"
  hidden_size: int = 1024
  num_hidden_layers: int = 12
  num_attention_heads: int = 8
  hidden_act: str = "silu"
  subsampling_conv_channels: list[int] = field(default_factory=lambda: [128, 32])
  conv_kernel_size: int = 5
  residual_weight: float = 0.5
  attention_chunk_size: int = 12
  attention_context_left: int = 13
  attention_context_right: int = 0
  attention_logit_cap: float = 50.0
  attention_invalid_logits_value: float = -1.0e9
  use_clipped_linears: bool = True
  rms_norm_eps: float = 1e-6
  gradient_clipping: float = 1e10
  output_proj_dims: int = 1536
  initializer_range: float = 0.02

  def __post_init__(self):
    if self.model_type != "gemma4_audio":
      raise ValueError(f"unsupported Gemma 4 audio config type {self.model_type!r}")
    self.subsampling_conv_channels = list(self.subsampling_conv_channels)

  @property
  def activation_name(self) -> str:
    return self.hidden_act

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "GemmaAudioConfig":
    return cls(
      model_type=data.get("model_type", "gemma4_audio"),
      hidden_size=data.get("hidden_size", 1024),
      num_hidden_layers=data.get("num_hidden_layers", 12),
      num_attention_heads=data.get("num_attention_heads", 8),
      hidden_act=data.get("hidden_act", "silu"),
      subsampling_conv_channels=list(data.get("subsampling_conv_channels", [128, 32])),
      conv_kernel_size=data.get("conv_kernel_size", 5),
      residual_weight=float(data.get("residual_weight", 0.5)),
      attention_chunk_size=data.get("attention_chunk_size", 12),
      attention_context_left=data.get("attention_context_left", 13),
      attention_context_right=data.get("attention_context_right", 0),
      attention_logit_cap=float(data.get("attention_logit_cap", 50.0)),
      attention_invalid_logits_value=float(data.get("attention_invalid_logits_value", -1.0e9)),
      use_clipped_linears=data.get("use_clipped_linears", True),
      rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
      gradient_clipping=float(data.get("gradient_clipping", 1e10)),
      output_proj_dims=data.get("output_proj_dims", 1536),
      initializer_range=float(data.get("initializer_range", 0.02)),
    )

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(slots=True)
class GemmaConditionalConfig:
  model_type: str = "gemma4"
  text_config: GemmaConfig = field(default_factory=GemmaConfig)
  vision_config: GemmaVisionConfig | None = None
  audio_config: GemmaAudioConfig | None = None
  boi_token_id: int | None = 255999
  eoi_token_id: int | None = 258882
  image_token_id: int | None = 258880
  video_token_id: int | None = 258884
  boa_token_id: int | None = 256000
  eoa_token_id: int | None = 258883
  audio_token_id: int | None = 258881
  initializer_range: float | None = 0.02
  tie_word_embeddings: bool = True

  def __post_init__(self):
    if self.model_type != "gemma4":
      raise ValueError(f"tinygrad-gemma only supports Gemma 4 checkpoints, got {self.model_type!r}")

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "GemmaConditionalConfig":
    if data.get("model_type", "gemma4") != "gemma4":
      raise ValueError(f"tinygrad-gemma only supports Gemma 4 checkpoints, got {data.get('model_type')!r}")
    text_dict = data.get("text_config") or {}
    text_model_type = text_dict.get("model_type", "gemma4_text")
    if text_model_type == "gemma4":
      text_model_type = "gemma4_text"
    vision_dict = data.get("vision_config")
    audio_dict = data.get("audio_config")
    text_config = GemmaConfig.from_dict({
      **text_dict,
      "model_type": text_model_type,
      "tie_word_embeddings": data.get("tie_word_embeddings", text_dict.get("tie_word_embeddings", True)),
    })
    return cls(
      text_config=text_config,
      vision_config=GemmaVisionConfig.from_dict(vision_dict) if vision_dict is not None else None,
      audio_config=GemmaAudioConfig.from_dict(audio_dict) if audio_dict is not None else None,
      boi_token_id=data.get("boi_token_id", 255999),
      eoi_token_id=data.get("eoi_token_id", 258882),
      image_token_id=data.get("image_token_id", 258880),
      video_token_id=data.get("video_token_id", 258884),
      boa_token_id=data.get("boa_token_id", 256000),
      eoa_token_id=data.get("eoa_token_id", data.get("eoa_token_index", 258883)),
      audio_token_id=data.get("audio_token_id", 258881),
      initializer_range=data.get("initializer_range", 0.02),
      tie_word_embeddings=data.get("tie_word_embeddings", True),
    )

  @classmethod
  def from_path(cls, path: str | Path) -> "GemmaConditionalConfig":
    return cls.from_dict(load_config_dict(path))

  def to_dict(self) -> dict[str, Any]:
    text_config = self.text_config.to_dict()
    if text_config.get("model_type") == "gemma4":
      text_config["model_type"] = "gemma4_text"
    return {
      "model_type": self.model_type,
      "text_config": text_config,
      "vision_config": None if self.vision_config is None else self.vision_config.to_dict(),
      "audio_config": None if self.audio_config is None else self.audio_config.to_dict(),
      "boi_token_id": self.boi_token_id,
      "eoi_token_id": self.eoi_token_id,
      "image_token_id": self.image_token_id,
      "video_token_id": self.video_token_id,
      "boa_token_id": self.boa_token_id,
      "eoa_token_id": self.eoa_token_id,
      "audio_token_id": self.audio_token_id,
      "initializer_range": self.initializer_range,
      "tie_word_embeddings": self.tie_word_embeddings,
    }
