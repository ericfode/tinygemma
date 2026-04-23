from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tinygrad import Tensor, nn
import tinygrad.nn.optim as optim

from .model import DEFAULT_IGNORE_INDEX, GemmaForCausalLM
from .multimodal import GemmaForConditionalGeneration
from .quantization import QUANTIZATION_MANIFEST, quantize_state_dict

GemmaTrainableModel = GemmaForCausalLM | GemmaForConditionalGeneration
SUPPORTED_OPTIMIZERS = ("sgd", "adam", "adamw", "lamb", "lars", "muon")


@dataclass(slots=True)
class GemmaTrainingBatch:
  input_ids: Sequence[int] | Sequence[Sequence[int]]
  labels: Tensor | Sequence[int] | Sequence[Sequence[int]] | None = None
  pixel_values: np.ndarray | None = None
  image_position_ids: np.ndarray | None = None
  input_features: np.ndarray | None = None
  input_features_mask: np.ndarray | None = None


def _prefix_match(name: str, prefixes: Sequence[str] | None) -> bool:
  return True if not prefixes else any(name.startswith(prefix) for prefix in prefixes)


def named_parameters(
  model: GemmaTrainableModel,
  *,
  include_prefixes: Sequence[str] | None = None,
  exclude_prefixes: Sequence[str] | None = None,
) -> list[tuple[str, Tensor]]:
  parameter_ids = {id(tensor) for tensor in nn.state.get_parameters(model)}
  seen: set[int] = set()
  selected = []
  for name, tensor in nn.state.get_state_dict(model).items():
    tensor_id = id(tensor)
    if tensor_id not in parameter_ids or tensor_id in seen:
      continue
    if not _prefix_match(name, include_prefixes):
      continue
    if exclude_prefixes and _prefix_match(name, exclude_prefixes):
      continue
    selected.append((name, tensor))
    seen.add(tensor_id)
  return selected


def parameters(
  model: GemmaTrainableModel,
  *,
  include_prefixes: Sequence[str] | None = None,
  exclude_prefixes: Sequence[str] | None = None,
) -> list[Tensor]:
  return [tensor for _, tensor in named_parameters(model, include_prefixes=include_prefixes, exclude_prefixes=exclude_prefixes)]


def set_trainable(
  model: GemmaTrainableModel,
  enabled: bool,
  *,
  include_prefixes: Sequence[str] | None = None,
  exclude_prefixes: Sequence[str] | None = None,
) -> list[Tensor]:
  selected = parameters(model, include_prefixes=include_prefixes, exclude_prefixes=exclude_prefixes)
  for tensor in selected:
    tensor.requires_grad_(enabled)
  return selected


def supported_optimizers() -> tuple[str, ...]:
  return SUPPORTED_OPTIMIZERS


def build_optimizer(
  model: GemmaTrainableModel,
  *,
  optimizer: str = "adamw",
  include_prefixes: Sequence[str] | None = None,
  exclude_prefixes: Sequence[str] | None = None,
  freeze_unselected: bool = False,
  **kwargs: Any,
):
  selected = parameters(model, include_prefixes=include_prefixes, exclude_prefixes=exclude_prefixes)
  selected_ids = {id(tensor) for tensor in selected}
  if freeze_unselected:
    for tensor in parameters(model):
      tensor.requires_grad_(id(tensor) in selected_ids)
  else:
    for tensor in selected:
      tensor.requires_grad_(True)

  optimizer_name = optimizer.strip().lower()
  optimizer_map = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "lamb": optim.LAMB,
    "lars": optim.LARS,
    "muon": optim.Muon,
  }
  if optimizer_name not in optimizer_map:
    raise ValueError(f"unsupported optimizer {optimizer!r}; supported optimizers: {', '.join(SUPPORTED_OPTIMIZERS)}")
  if optimizer_name == "muon" and "fused" not in kwargs:
    kwargs["fused"] = False
  return optimizer_map[optimizer_name](selected, **kwargs)


def _pruned_state_dict(model: GemmaTrainableModel) -> dict[str, Tensor]:
  state_dict = dict(nn.state.get_state_dict(model))
  if getattr(model.config, "tie_word_embeddings", False):
    state_dict.pop("lm_head.weight", None)
  return state_dict


def save_pretrained(model: GemmaTrainableModel, model_dir: str | Path, *, quantize: str | None = None) -> None:
  model_dir = Path(model_dir)
  model_dir.mkdir(parents=True, exist_ok=True)
  config_dict = model.config.to_dict()
  if isinstance(model, GemmaForCausalLM) and config_dict.get("model_type") == "gemma4":
    config_dict["model_type"] = "gemma4_text"
  (model_dir / "config.json").write_text(json.dumps(config_dict, indent=2, sort_keys=True))
  state_dict = _pruned_state_dict(model)
  manifest_path = model_dir / QUANTIZATION_MANIFEST
  if quantize is not None:
    state_dict, manifest = quantize_state_dict(state_dict, quantize=quantize)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
  elif manifest_path.exists():
    manifest_path.unlink()
  nn.state.safe_save(state_dict, str(model_dir / "model.safetensors"))


def save_training_checkpoint(
  model: GemmaTrainableModel,
  output_dir: str | Path,
  *,
  optimizer=None,
  training_metadata: dict[str, Any] | None = None,
  quantize: str | None = None,
) -> None:
  output_dir = Path(output_dir)
  save_pretrained(model, output_dir, quantize=quantize)
  if optimizer is not None:
    nn.state.safe_save(dict(nn.state.get_state_dict(optimizer)), str(output_dir / "optimizer.safetensors"))
  if training_metadata is not None:
    (output_dir / "training.json").write_text(json.dumps(training_metadata, indent=2, sort_keys=True))


def load_optimizer_state(optimizer, path: str | Path, *, strict: bool = True, verbose: bool = False) -> None:
  nn.state.load_state_dict(optimizer, nn.state.safe_load(path), strict=strict, verbose=verbose)


def batch_loss(
  model: GemmaTrainableModel,
  batch: GemmaTrainingBatch,
  *,
  ignore_index: int = DEFAULT_IGNORE_INDEX,
  ignore_token_ids: set[int] | None = None,
) -> Tensor:
  if isinstance(model, GemmaForConditionalGeneration):
    return model.forward_loss_ids(
      batch.input_ids,
      labels=batch.labels,
      pixel_values=batch.pixel_values,
      image_position_ids=batch.image_position_ids,
      input_features=batch.input_features,
      input_features_mask=batch.input_features_mask,
      ignore_index=ignore_index,
      ignore_token_ids=ignore_token_ids,
    )
  return model.forward_loss_ids(batch.input_ids, labels=batch.labels, ignore_index=ignore_index, ignore_token_ids=ignore_token_ids)


def train_step(
  model: GemmaTrainableModel,
  optimizer,
  batch: GemmaTrainingBatch,
  *,
  ignore_index: int = DEFAULT_IGNORE_INDEX,
  ignore_token_ids: set[int] | None = None,
) -> Tensor:
  optimizer.zero_grad()
  with Tensor.train():
    loss = batch_loss(model, batch, ignore_index=ignore_index, ignore_token_ids=ignore_token_ids)
    loss.backward()
    optimizer.step()
  return loss.realize()
