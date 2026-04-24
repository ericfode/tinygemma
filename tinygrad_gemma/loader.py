from __future__ import annotations

import json
from pathlib import Path

from tinygrad import nn
from tinygrad import Tensor

from .config import GemmaConditionalConfig, GemmaConfig, load_config_dict
from .model import GemmaForCausalLM
from .multimodal import GemmaForConditionalGeneration
from .quantization import RowwiseInt8Linear, dequantize_rowwise_int8_tensor, dequantize_state_dict, load_quantization_manifest
from .runtime import temporary_default_device


GemmaAnyConfig = GemmaConfig | GemmaConditionalConfig
GemmaAnyModel = GemmaForCausalLM | GemmaForConditionalGeneration


def _has_multimodal_towers(config: GemmaAnyConfig) -> bool:
  return isinstance(config, GemmaConditionalConfig) and (config.vision_config is not None or config.audio_config is not None)


def load_config(model_dir: str | Path) -> GemmaAnyConfig:
  model_dir = Path(model_dir)
  raw_config = load_config_dict(model_dir / "config.json")
  if raw_config.get("vision_config") is not None or raw_config.get("audio_config") is not None:
    return GemmaConditionalConfig.from_dict(raw_config)
  return GemmaConfig.from_dict(raw_config)


def load_text_config(model_dir: str | Path) -> GemmaConfig:
  config = load_config(model_dir)
  return config.text_config if isinstance(config, GemmaConditionalConfig) else config


def resolve_weight_files(model_dir: str | Path) -> list[Path]:
  model_dir = Path(model_dir)
  index_path = model_dir / "model.safetensors.index.json"
  if index_path.exists():
    index_data = json.loads(index_path.read_text())
    files = list(dict.fromkeys(index_data["weight_map"].values()))
    return [model_dir / name for name in files]
  direct = model_dir / "model.safetensors"
  if direct.exists():
    return [direct]
  files = sorted(model_dir.glob("*.safetensors"))
  if files:
    return files
  raise FileNotFoundError(f"no .safetensors checkpoint found under {model_dir}")


def load_state_dict(model_dir: str | Path) -> dict:
  state_dict = {}
  for path in resolve_weight_files(model_dir):
    state_dict.update(nn.state.safe_load(path))
  manifest = load_quantization_manifest(model_dir)
  return dequantize_state_dict(state_dict, manifest) if manifest is not None else state_dict


def load_raw_state_dict(model_dir: str | Path) -> dict:
  state_dict = {}
  for path in resolve_weight_files(model_dir):
    state_dict.update(nn.state.safe_load(path))
  return state_dict


def normalize_state_dict_keys(state_dict: dict, config: GemmaAnyConfig) -> dict:
  normalized = {}
  multimodal = _has_multimodal_towers(config)
  for key, value in state_dict.items():
    if key.startswith("_quant_scale."):
      normalized["_quant_scale." + normalize_state_dict_key(key[len("_quant_scale."):], config)] = value
      continue
    if multimodal and ".depthwise_conv1d.weight" in key:
      key = key.replace(".depthwise_conv1d.weight", ".depthwise_conv1d.conv.weight")
    if key.startswith("model.language_model."):
      mapped = ("model.language_model." if multimodal else "model.") + key[len("model.language_model."):]
      normalized[mapped] = value
    elif key.startswith("language_model."):
      mapped = ("model.language_model." if multimodal else "model.") + key[len("language_model."):]
      normalized[mapped] = value
    elif key.startswith("model.text_model."):
      mapped = ("model.language_model." if multimodal else "model.") + key[len("model.text_model."):]
      normalized[mapped] = value
    else:
      normalized[key] = value
  if config.tie_word_embeddings:
    embed_key = "model.language_model.embed_tokens.weight" if multimodal else "model.embed_tokens.weight"
    if "lm_head.weight" not in normalized and embed_key in normalized:
      normalized["lm_head.weight"] = normalized[embed_key]
  return normalized


def normalize_state_dict_key(key: str, config: GemmaAnyConfig) -> str:
  return next(iter(normalize_state_dict_keys({key: None}, config).keys()))


def normalize_scale_key(key: str, config: GemmaAnyConfig) -> str:
  prefix = "_quant_scale."
  if not key.startswith(prefix):
    return normalize_state_dict_key(key, config)
  return prefix + normalize_state_dict_key(key[len(prefix):], config)


def normalize_quantization_manifest(manifest: dict, config: GemmaAnyConfig) -> dict:
  normalized = {**manifest, "tensors": {}}
  for key, entry in manifest.get("tensors", {}).items():
    mapped = normalize_state_dict_key(key, config)
    normalized["tensors"][mapped] = {**entry, "scale_key": normalize_scale_key(entry["scale_key"], config)}
  if config.tie_word_embeddings:
    multimodal = _has_multimodal_towers(config)
    embed_key = "model.language_model.embed_tokens.weight" if multimodal else "model.embed_tokens.weight"
    if "lm_head.weight" not in normalized["tensors"] and embed_key in normalized["tensors"]:
      normalized["tensors"]["lm_head.weight"] = normalized["tensors"][embed_key]
  return normalized


def module_at_path(root, path: str):
  current = root
  for part in path.split("."):
    current = current[int(part)] if isinstance(current, list) else getattr(current, part)
  return current


def set_module_at_path(root, path: str, value) -> None:
  if "." not in path:
    setattr(root, path, value)
    return
  parent_path, attr = path.rsplit(".", 1)
  parent = module_at_path(root, parent_path)
  if isinstance(parent, list):
    parent[int(attr)] = value
  else:
    setattr(parent, attr, value)


def _is_runtime_int8_linear(model, key: str, tensor: Tensor, scale: Tensor) -> bool:
  if not key.endswith(".weight") or tensor.ndim != 2 or scale.ndim != 1 or tensor.shape[0] != scale.shape[0]:
    return False
  module_path = key[: -len(".weight")]
  try:
    module = module_at_path(model, module_path)
  except (AttributeError, IndexError, ValueError):
    return False
  return module.__class__.__name__ == "Linear" and getattr(module, "weight", None) is not None and module.weight.shape == tensor.shape


def install_runtime_int8_linears(model, state_dict: dict, manifest: dict, *, strict: bool = True, device: str | None = None) -> tuple[dict, list[str]]:
  tensors = manifest.get("tensors", {})
  loadable = {}
  runtime_linear_keys: list[str] = []
  consumed_scale_keys = {entry["scale_key"] for entry in tensors.values()}
  for key, tensor in state_dict.items():
    if key in consumed_scale_keys:
      continue
    if key not in tensors:
      loadable[key] = tensor
      continue
    entry = tensors[key]
    scale_key = entry["scale_key"]
    if scale_key not in state_dict:
      raise KeyError(f"missing quantization scale tensor {scale_key!r} for {key!r}")
    scale = state_dict[scale_key]
    if _is_runtime_int8_linear(model, key, tensor, scale):
      runtime_linear_keys.append(key)
    else:
      loadable[key] = dequantize_rowwise_int8_tensor(tensor, scale, entry.get("dtype", "float"))

  if strict:
    model_keys = set(nn.state.get_state_dict(model))
    missing = model_keys - set(loadable) - set(runtime_linear_keys)
    if missing:
      raise KeyError(f"missing weights in state_dict: {sorted(missing)}")

  nn.state.load_state_dict(model, loadable, strict=False, verbose=False, consume=True)
  installed = []
  for key in runtime_linear_keys:
    entry = tensors[key]
    module_path = key[: -len(".weight")]
    module = module_at_path(model, module_path)
    qweight = state_dict[key].to(device or module.weight.device).contiguous().realize()
    scale = state_dict[entry["scale_key"]].to(device or module.weight.device).contiguous().realize()
    bias = None if getattr(module, "bias", None) is None else module.bias
    set_module_at_path(model, module_path, RowwiseInt8Linear(qweight, scale, original_dtype=entry.get("dtype", "float"), bias=bias))
    installed.append(key)
  return loadable, installed


def load_pretrained(
  model_dir: str | Path,
  *,
  strict: bool = True,
  verbose: bool = False,
  device: str | None = None,
  runtime_quantization: bool = True,
) -> GemmaAnyModel:
  config = load_config(model_dir)
  manifest = load_quantization_manifest(model_dir)
  raw_state_dict = load_raw_state_dict(model_dir)
  state_dict = normalize_state_dict_keys(raw_state_dict, config)
  manifest = normalize_quantization_manifest(manifest, config) if manifest is not None else None
  model_type = GemmaForConditionalGeneration if _has_multimodal_towers(config) else GemmaForCausalLM
  if device is None:
    model = model_type(config)  # type: ignore[arg-type]
    if runtime_quantization and manifest is not None:
      _, installed = install_runtime_int8_linears(model, state_dict, manifest, strict=strict)
      if verbose:
        print(f"installed {len(installed)} runtime int8 linear weights")
    else:
      state_dict = dequantize_state_dict(state_dict, manifest) if manifest is not None else state_dict
      nn.state.load_state_dict(model, state_dict, strict=strict, verbose=verbose, consume=True)
    return model
  with temporary_default_device(device):
    model = model_type(config)  # type: ignore[arg-type]
    if runtime_quantization and manifest is not None:
      _, installed = install_runtime_int8_linears(model, state_dict, manifest, strict=strict, device=device)
      if verbose:
        print(f"installed {len(installed)} runtime int8 linear weights")
    else:
      state_dict = dequantize_state_dict(state_dict, manifest) if manifest is not None else state_dict
      nn.state.load_state_dict(model, state_dict, strict=strict, verbose=verbose, consume=True)
  return model
