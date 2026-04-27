from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable

from tinygrad import Device, Tensor, TinyJit, Variable
from tinygrad.device import MultiBuffer
from tinygrad.engine.jit import TinyJit as TinyJitClass
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import CompiledRunner, ExecItem
try:
  from tinygrad.engine.realize import ExecContext, resolve_params
except ImportError:  # pragma: no cover - exercised by plain repo test collection on stock tinygrad.
  ExecContext = None  # type: ignore[assignment]
  resolve_params = None  # type: ignore[assignment]
from tinygrad.helpers import Context, GlobalCounters, Metadata, flatten
from tinygrad.nn import state as nn_state
try:
  from tinygrad.schedule import linear_to_schedule, pm_post_sched_cache
except ImportError:  # pragma: no cover - stock tinygrad lacks captured-JIT profiling lowerer helpers.
  linear_to_schedule = None  # type: ignore[assignment]
  pm_post_sched_cache = None  # type: ignore[assignment]
from tinygrad.uop.ops import Ops, UOp, UOpMetaClass, all_metadata, graph_rewrite, sym_infer

import tinygrad
import tinygrad_gemma.model as gemma_model
from tinygrad_gemma import load_pretrained
from tinygrad_gemma.loader import load_quantization_manifest, resolve_weight_files
from tinygrad_gemma.model import (
  GemmaAttention,
  GemmaCache,
  GemmaCacheEntry,
  GemmaDecoderLayer,
  GemmaMLP,
  GemmaModel,
  RMSNorm,
  TextScaledEmbedding,
  gelu_pytorch_tanh,
)
from tinygrad_gemma.model import make_packed_cache_entry
from tinygrad_gemma.multimodal import GemmaForConditionalGeneration
from tinygrad_gemma.runtime import prepare_device


DEFAULT_MODEL_DIR = Path("/Users/ericfode/Downloads/tinygrad-gemma/checkpoints/gemma-4-E2B-int8")
DEFAULT_OUT = Path("benchmarks/gemma4-metal-postwindow-jit-profile-current.json")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BATCHED_DISPLAY_RE = re.compile(r"^<batched (?P<count>\d+)>$")
CACHE_WRITE_CATEGORY_RE = re.compile(
  r"^(?P<base>attention_(?P<kind>key|value|packed)_cache_write)"
  r"__role_(?P<role>local|shared_source|shared_consumer)"
  r"__layer_(?P<layer>\d+|unknown)"
  r"__type_(?P<layer_type>[A-Za-z0-9_]+?)"
  r"(?:__phase_(?P<phase>kv_projection|rmsnorm_rope|rhs_pack|store))?$"
)
CACHE_WRITE_ROLES = ("shared_source", "shared_consumer", "local")
CACHE_WRITE_KINDS = ("key", "value", "packed")
CACHE_WRITE_PHASES = ("kv_projection", "rmsnorm_rope", "rhs_pack", "store")
CACHE_WRITE_PHASE_TARGET_PRESETS = ("layer13", "all-shared-source", "all-local", "all-packed")
CACHE_WRITE_PHASE_TARGET_EXACT_RE = re.compile(r"^(?P<role>local|shared-source|shared-consumer)-layer(?P<layer>\d+)$")


class CacheWritePhaseTarget:
  __slots__ = ("kind", "role", "layer_idx", "layer_type")

  def __init__(self, *, kind: str | None = None, role: str | None = None, layer_idx: int | None = None, layer_type: str | None = None):
    self.kind = kind
    self.role = role
    self.layer_idx = layer_idx
    self.layer_type = layer_type

  def matches(self, kind: str, role: str, layer_idx: int | None, layer_type: str | None) -> bool:
    return (
      (self.kind is None or self.kind == kind)
      and (self.role is None or self.role == role)
      and (self.layer_idx is None or self.layer_idx == layer_idx)
      and (self.layer_type is None or self.layer_type == layer_type)
    )


DEFAULT_CACHE_WRITE_PHASE_TARGETS = (
  CacheWritePhaseTarget(kind="packed", role="shared_source", layer_idx=13, layer_type="sliding_attention"),
)
_CACHE_WRITE_PHASE_TARGETS: tuple[CacheWritePhaseTarget, ...] = DEFAULT_CACHE_WRITE_PHASE_TARGETS


def _source_range(obj) -> tuple[int, int]:
  lines, start = inspect.getsourcelines(obj)
  return start, start + len(lines) - 1


CATEGORY_RANGES = {
  "tinygrad_gemma.model": [
    ("mlp", _source_range(GemmaMLP.__call__)),
    ("mlp", _source_range(gelu_pytorch_tanh)),
    ("attention", _source_range(GemmaAttention.__call__)),
    ("embedding_per_layer", _source_range(TextScaledEmbedding.__call__)),
    ("embedding_per_layer", _source_range(GemmaModel.project_per_layer_inputs)),
    ("decoder_residual", _source_range(GemmaDecoderLayer.__call__)),
    ("norm", _source_range(RMSNorm.__call__)),
  ],
  "tinygrad_gemma.multimodal": [
    ("logits_argmax", _source_range(GemmaForConditionalGeneration.logits)),
    ("logits_argmax", _source_range(GemmaForConditionalGeneration.sample_next)),
  ],
}

PROFILE_CATEGORIES = (
  "logits_argmax",
  "mlp",
  "attention_key_cache_write_shared_source",
  "attention_value_cache_write_shared_source",
  "attention_key_cache_write_shared_consumer",
  "attention_value_cache_write_shared_consumer",
  "attention_key_cache_write_local",
  "attention_value_cache_write_local",
  "attention_packed_cache_write_shared_source",
  "attention_packed_cache_write_shared_consumer",
  "attention_packed_cache_write_local",
  "attention_key_cache_write",
  "attention_value_cache_write",
  "attention_cache_write",
  "attention",
  "embedding_per_layer",
  "decoder_residual",
  "norm",
)
_SIDECAR_STACK: list[str] = []
_ATTENTION_CONTEXT_STACK: list[dict[str, Any]] = []
_CACHE_WRITE_PHASE_CUTPOINTS = False
UOP_SIDECAR_CATEGORY_BASIS = "repo_sidecar_uop_creation_metadata"
UOP_SIDECAR_IGNORED_OPS = {
  op
  for name in (
    "CONST",
    "VCONST",
    "DEVICE",
    "UNIQUE",
    "LUNIQUE",
    "DEFINE_VAR",
    "BIND",
    "NOOP",
    "BUFFER",
    "PARAM",
  )
  if (op := getattr(Ops, name, None)) is not None
}
STORE_EFFECT_OPS = {
  op
  for name in ("STORE", "ASSIGN", "AFTER")
  if (op := getattr(Ops, name, None)) is not None
}


def _line_from_caller(caller: str) -> int | None:
  parts = caller.split(":")
  if len(parts) < 2:
    return None
  try:
    return int(parts[1])
  except ValueError:
    return None


def _metadata_payload(metadata) -> list[dict[str, str]]:
  counts: dict[tuple[str, str], int] = defaultdict(int)
  for item in metadata:
    counts[(str(getattr(item, "name", "")), str(getattr(item, "caller", "")))] += 1
  return [
    {"name": name, "caller": caller, "count": str(count)}
    for (name, caller), count in sorted(counts.items())
  ]


def cache_write_layer_label(layer_idx: int | None) -> str:
  if layer_idx is None:
    return "unknown"
  try:
    return f"{int(layer_idx):02d}"
  except (TypeError, ValueError):
    return "unknown"


def cache_write_layer_type_label(layer_type: str | None) -> str:
  if layer_type is None:
    return "unknown"
  label = re.sub(r"[^A-Za-z0-9_]+", "_", str(layer_type)).strip("_")
  return label or "unknown"


def cache_write_sidecar_category(
  kind: str,
  role: str,
  layer_idx: int | None,
  layer_type: str | None,
  *,
  phase: str | None = None,
) -> str:
  if kind not in CACHE_WRITE_KINDS:
    raise ValueError(f"unknown cache write kind {kind!r}")
  if role not in CACHE_WRITE_ROLES:
    raise ValueError(f"unknown cache write role {role!r}")
  if phase is not None and phase not in CACHE_WRITE_PHASES:
    raise ValueError(f"unknown cache write phase {phase!r}")
  category = (
    f"attention_{kind}_cache_write"
    f"__role_{role}"
    f"__layer_{cache_write_layer_label(layer_idx)}"
    f"__type_{cache_write_layer_type_label(layer_type)}"
  )
  return category if phase is None else f"{category}__phase_{phase}"


@lru_cache(maxsize=None)
def cache_write_category_metadata(category: str) -> dict[str, Any] | None:
  match = CACHE_WRITE_CATEGORY_RE.match(category)
  if match is not None:
    layer_label = match.group("layer")
    kind = match.group("kind")
    role = match.group("role")
    layer_idx = None if layer_label == "unknown" else int(layer_label)
    phase = match.group("phase")
    metadata = {
      "kind": kind,
      "layout": "packed" if kind == "packed" else "split",
      "role": role,
      "layer_idx": layer_idx,
      "layer": layer_label,
      "layer_type": match.group("layer_type"),
      "rollup": f"{match.group('base')}_{role}",
    }
    if phase is not None:
      metadata["phase"] = phase
      metadata["parent"] = cache_write_sidecar_category(kind, role, layer_idx, match.group("layer_type"))
    return metadata
  prefix = "attention_"
  suffix = "_cache_write_"
  if category.startswith(prefix) and suffix in category:
    kind_and_role = category.removeprefix(prefix).split(suffix, 1)
    if len(kind_and_role) == 2:
      kind, role = kind_and_role
      if kind in CACHE_WRITE_KINDS and role in CACHE_WRITE_ROLES:
        return {
          "kind": kind,
          "layout": "packed" if kind == "packed" else "split",
          "role": role,
          "layer_idx": None,
          "layer": "unknown",
          "layer_type": "unknown",
          "rollup": category,
        }
  return None


@lru_cache(maxsize=None)
def is_cache_write_phase_category(category: str) -> bool:
  metadata = cache_write_category_metadata(category)
  return metadata is not None and "phase" in metadata


@lru_cache(maxsize=None)
def parent_cache_write_category(category: str) -> str:
  metadata = cache_write_category_metadata(category)
  return category if metadata is None else str(metadata.get("parent", category))


@lru_cache(maxsize=None)
def cache_write_phase_name(category: str) -> str | None:
  metadata = cache_write_category_metadata(category)
  return None if metadata is None else metadata.get("phase")


@lru_cache(maxsize=None)
def cache_write_phase_order(category: str) -> int:
  phase = cache_write_phase_name(category)
  return CACHE_WRITE_PHASES.index(phase) if phase in CACHE_WRITE_PHASES else len(CACHE_WRITE_PHASES)


def parse_cache_write_phase_targets(raw_targets: list[str] | tuple[str, ...] | None) -> tuple[CacheWritePhaseTarget, ...]:
  if not raw_targets:
    return DEFAULT_CACHE_WRITE_PHASE_TARGETS
  presets = {
    "layer13": DEFAULT_CACHE_WRITE_PHASE_TARGETS,
    "all-shared-source": (CacheWritePhaseTarget(kind="packed", role="shared_source"),),
    "all-local": (CacheWritePhaseTarget(kind="packed", role="local"),),
    "all-packed": (CacheWritePhaseTarget(kind="packed"),),
  }
  targets: list[CacheWritePhaseTarget] = []
  for raw_target in raw_targets:
    if raw_target in presets:
      targets.extend(presets[raw_target])
      continue
    exact_match = CACHE_WRITE_PHASE_TARGET_EXACT_RE.fullmatch(raw_target)
    if exact_match is not None:
      targets.append(CacheWritePhaseTarget(
        kind="packed",
        role=exact_match.group("role").replace("-", "_"),
        layer_idx=int(exact_match.group("layer")),
      ))
      continue
    allowed = ", ".join(CACHE_WRITE_PHASE_TARGET_PRESETS)
    raise ValueError(
      f"unknown cache write phase target {raw_target!r}; expected one of: {allowed}, local-layer<N>, shared-source-layer<N>, shared-consumer-layer<N>"
    )
  return tuple(targets)


@contextmanager
def cache_write_phase_target_scope(targets: tuple[CacheWritePhaseTarget, ...] | None):
  global _CACHE_WRITE_PHASE_TARGETS
  previous = _CACHE_WRITE_PHASE_TARGETS
  _CACHE_WRITE_PHASE_TARGETS = parse_cache_write_phase_targets(None) if targets is None else targets
  try:
    yield
  finally:
    _CACHE_WRITE_PHASE_TARGETS = previous


def cache_write_phase_target(
  kind: str,
  role: str,
  layer_idx: int | None,
  layer_type: str | None,
  *,
  targets: tuple[CacheWritePhaseTarget, ...] | None = None,
) -> bool:
  active_targets = _CACHE_WRITE_PHASE_TARGETS if targets is None else targets
  return any(target.matches(kind, role, layer_idx, layer_type) for target in active_targets)


def cache_write_phase_target_category(category: str) -> bool:
  metadata = cache_write_category_metadata(parent_cache_write_category(category))
  return metadata is not None and cache_write_phase_target(
    str(metadata["kind"]),
    str(metadata["role"]),
    metadata["layer_idx"],
    str(metadata["layer_type"]),
  )


@contextmanager
def cache_write_phase_cutpoint_scope(enabled: bool):
  global _CACHE_WRITE_PHASE_CUTPOINTS
  previous = _CACHE_WRITE_PHASE_CUTPOINTS
  _CACHE_WRITE_PHASE_CUTPOINTS = enabled
  try:
    yield
  finally:
    _CACHE_WRITE_PHASE_CUTPOINTS = previous


@lru_cache(maxsize=None)
def category_rollup(category: str) -> str:
  metadata = cache_write_category_metadata(category)
  return category if metadata is None else str(metadata["rollup"])


@lru_cache(maxsize=None)
def is_profile_category(category: str) -> bool:
  return category in PROFILE_CATEGORIES or cache_write_category_metadata(category) is not None


def category_count_rollups(category_counts: dict[str, int]) -> dict[str, int]:
  rollups: dict[str, int] = defaultdict(int)
  for category, count in category_counts.items():
    rollups[category_rollup(category)] += int(count)
  return dict(sorted(rollups.items()))


def add_category_timing(
  groups: dict[str, dict[str, Any]],
  category: str,
  *,
  count_key: str,
  count: int,
  elapsed_ms: float,
  est_ops: int = 0,
  est_mem: int = 0,
) -> None:
  entry = groups.setdefault(category, {count_key: 0, "elapsed_ms": 0.0, "est_ops": 0, "est_mem": 0})
  entry[count_key] += count
  entry["elapsed_ms"] += elapsed_ms
  entry["est_ops"] += est_ops
  entry["est_mem"] += est_mem


def strip_ansi(value: str) -> str:
  return ANSI_RE.sub("", value)


def priority_category(categories: set[str], *, prefer_cache_write_phase: bool = False) -> str:
  candidates = {category for category in categories if is_profile_category(category)}
  if prefer_cache_write_phase:
    phase_candidates = {category for category in candidates if is_cache_write_phase_category(category)}
    if phase_candidates:
      candidates = phase_candidates
  else:
    candidates = {parent_cache_write_category(category) for category in candidates}
  for profile_category in PROFILE_CATEGORIES:
    matches = sorted(category for category in candidates if category == profile_category or category_rollup(category) == profile_category)
    if matches:
      detailed_matches = [category for category in matches if category != profile_category]
      return detailed_matches[0] if detailed_matches else profile_category
  return "other"


def sidecar_metadata_category(metadata, *, prefer_cache_write_phase: bool = False) -> str:
  categories: set[str] = set()
  for item in metadata:
    name = str(getattr(item, "name", ""))
    caller = str(getattr(item, "caller", ""))
    if is_profile_category(name):
      categories.add(name)
    if caller.startswith(("repo_sidecar:", "repo_uop_sidecar:")) and "::" in caller:
      category = caller.rsplit("::", 1)[-1]
      if is_profile_category(category):
        categories.add(category)
  return priority_category(categories, prefer_cache_write_phase=prefer_cache_write_phase)


def merge_metadata(existing, extra) -> tuple[Metadata, ...]:
  merged = tuple(existing or ())
  extra_tuple = tuple(extra or ())
  seen = {(str(getattr(item, "name", "")), str(getattr(item, "caller", ""))) for item in merged}
  additions = [
    item
    for item in extra_tuple
    if (str(getattr(item, "name", "")), str(getattr(item, "caller", ""))) not in seen
  ]
  return merged + tuple(additions)


def uop_sidecar_metadata(category: str) -> tuple[Metadata, ...]:
  return (Metadata(name=category, caller=f"repo_uop_sidecar:1::{category}"),)


def cache_write_phase_sidecar_metadata(metadata) -> tuple[Metadata, ...]:
  phase_metadata = []
  for item in metadata or ():
    name = str(getattr(item, "name", ""))
    caller = str(getattr(item, "caller", ""))
    caller_category = caller.rsplit("::", 1)[-1] if "::" in caller else ""
    if is_cache_write_phase_category(name) or is_cache_write_phase_category(caller_category):
      phase_metadata.append(item)
  return tuple(phase_metadata)


def uop_cache_write_phase_sidecar_metadata(uop, seen: set[int] | None = None) -> tuple[Metadata, ...]:
  if uop is None:
    return ()
  if seen is None:
    seen = set()
  identity = id(uop)
  if identity in seen:
    return ()
  seen.add(identity)
  metadata = cache_write_phase_sidecar_metadata(
    merge_metadata(getattr(uop, "metadata", None) or (), all_metadata.get(uop, ()))
  )
  for source in getattr(uop, "src", ()) or ():
    metadata = merge_metadata(metadata, uop_cache_write_phase_sidecar_metadata(source, seen))
  return metadata


def direct_uop_cache_write_phase_sidecar_metadata(uop) -> tuple[Metadata, ...]:
  if uop is None:
    return ()
  return cache_write_phase_sidecar_metadata(
    merge_metadata(getattr(uop, "metadata", None) or (), all_metadata.get(uop, ()))
  )


def direct_uop_sources_cache_write_phase_sidecar_metadata(uops) -> tuple[Metadata, ...]:
  metadata: tuple[Metadata, ...] = ()
  for uop in uops or ():
    metadata = merge_metadata(metadata, direct_uop_cache_write_phase_sidecar_metadata(uop))
  return metadata


def ast_uop_sidecar_category_counts(ast, *, prefer_cache_write_phase: bool = False) -> dict[str, int]:
  if ast is None:
    return {}
  counts: dict[str, int] = defaultdict(int)
  try:
    uops = ast.toposort()
  except Exception:
    return {"error": 1}
  for uop in uops:
    if getattr(uop, "op", None) in UOP_SIDECAR_IGNORED_OPS:
      continue
    metadata = merge_metadata(getattr(uop, "metadata", None) or (), all_metadata.get(uop, ()))
    category = sidecar_metadata_category(metadata, prefer_cache_write_phase=prefer_cache_write_phase)
    if category != "other":
      counts[category] += 1
  return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def ast_uop_sidecar_category(ast) -> str:
  counts = ast_uop_sidecar_category_counts(ast)
  return priority_category(set(counts)) if counts else "other"


def is_store_effect_op(op) -> bool:
  return op in STORE_EFFECT_OPS or str(op) in {"Ops.STORE", "Ops.ASSIGN", "Ops.AFTER", "STORE", "ASSIGN", "AFTER"}


def ast_cache_write_phase_category_counts(ast) -> dict[str, int]:
  counts = {
    category: count
    for category, count in ast_uop_sidecar_category_counts(ast, prefer_cache_write_phase=True).items()
    if is_cache_write_phase_category(category)
  }
  return dict(sorted(counts.items(), key=lambda item: (-item[1], cache_write_phase_order(item[0]), item[0])))


def ast_cache_write_phase_category(ast) -> tuple[str | None, bool]:
  phase_counts = ast_cache_write_phase_category_counts(ast)
  if not phase_counts:
    return None, False
  store_effect_categories: set[str] = set()
  try:
    uops = ast.toposort()
  except Exception:
    uops = []
  for uop in uops:
    if is_store_effect_op(getattr(uop, "op", None)):
      metadata = merge_metadata(getattr(uop, "metadata", None) or (), all_metadata.get(uop, ()))
      category = sidecar_metadata_category(metadata, prefer_cache_write_phase=True)
      if is_cache_write_phase_category(category):
        store_effect_categories.add(category)
  candidates = store_effect_categories or set(phase_counts)
  selected = sorted(candidates, key=lambda category: (-phase_counts.get(category, 0), cache_write_phase_order(category), category))[0]
  return selected, len(phase_counts) > 1


def source_item_cache_write_phase_category(item) -> tuple[str | None, bool]:
  metadata_phase_category = sidecar_metadata_category(
    getattr(item, "metadata", None) or (),
    prefer_cache_write_phase=True,
  )
  ast_phase_category, ast_conflict = ast_cache_write_phase_category(getattr(item, "ast", None))
  if is_cache_write_phase_category(metadata_phase_category):
    return metadata_phase_category, ast_conflict or (
      ast_phase_category is not None and ast_phase_category != metadata_phase_category
    )
  return ast_phase_category, ast_conflict


def source_range_metadata_category(metadata) -> str:
  callers = [str(getattr(m, "caller", "")) for m in metadata]
  categories = set()
  for caller in callers:
    caller_module = caller.split(":", 1)[0]
    line = _line_from_caller(caller)
    if line is None:
      continue
    for category, (start, end) in CATEGORY_RANGES.get(caller_module, []):
      if start <= line <= end:
        categories.add(category)
  return priority_category(categories)


def classify_kernel(metadata) -> str:
  sidecar_category = sidecar_metadata_category(metadata)
  if sidecar_category != "other":
    return sidecar_category
  return source_range_metadata_category(metadata)


@contextmanager
def sidecar_scope(category: str):
  if not is_profile_category(category):
    raise ValueError(f"unknown profile sidecar category {category!r}")
  _SIDECAR_STACK.append(category)
  try:
    yield
  finally:
    _SIDECAR_STACK.pop()


def add_sidecar_metadata(linear: UOp, category: str) -> UOp:
  parent_category = parent_cache_write_category(category)
  scope_categories = [parent_category]
  if is_cache_write_phase_category(category):
    scope_categories.append(category)

  def add_call_metadata(call: UOp) -> UOp:
    metadata = tuple(call.arg.metadata or ())
    for scope_category in scope_categories:
      metadata = merge_metadata(
        metadata,
        (Metadata(name=scope_category, caller=f"repo_sidecar:1::{scope_category}"),),
      )
    return call.replace(arg=replace(call.arg, metadata=metadata))

  return linear.replace(src=tuple(add_call_metadata(call) for call in linear.src))


@contextmanager
def uop_creation_sidecar_patch():
  original_call = UOpMetaClass.__call__

  def call_with_uop_sidecar(cls, *args, **kwargs):
    uop = original_call(cls, *args, **kwargs)
    current_metadata = all_metadata.get(uop, ())
    inherited_phase_metadata = direct_uop_sources_cache_write_phase_sidecar_metadata(getattr(uop, "src", ()))
    if inherited_phase_metadata:
      current_metadata = merge_metadata(current_metadata, inherited_phase_metadata)
    if _SIDECAR_STACK:
      current_metadata = merge_metadata(current_metadata, uop_sidecar_metadata(_SIDECAR_STACK[-1]))
    if current_metadata:
      all_metadata[uop] = current_metadata
    return uop

  UOpMetaClass.__call__ = call_with_uop_sidecar
  try:
    yield
  finally:
    UOpMetaClass.__call__ = original_call


@contextmanager
def uop_replace_sidecar_patch():
  original_replace = UOp.replace

  def replace_with_metadata(self, **kwargs):
    previous_metadata = all_metadata.get(self, ())
    new_uop = original_replace(self, **kwargs)
    current_metadata = all_metadata.get(new_uop, ())
    if previous_metadata:
      current_metadata = merge_metadata(current_metadata, previous_metadata)
    inherited_phase_metadata = direct_uop_sources_cache_write_phase_sidecar_metadata(getattr(new_uop, "src", ()))
    if inherited_phase_metadata:
      current_metadata = merge_metadata(current_metadata, inherited_phase_metadata)
    if _SIDECAR_STACK:
      current_metadata = merge_metadata(current_metadata, uop_sidecar_metadata(_SIDECAR_STACK[-1]))
    if current_metadata:
      all_metadata[new_uop] = current_metadata
    return new_uop

  UOp.replace = replace_with_metadata
  try:
    yield
  finally:
    UOp.replace = original_replace


def tensor_has_op(tensor: Tensor, op: Ops) -> bool:
  try:
    return any(uop.op is op for uop in tensor.uop.toposort())
  except Exception:
    return False


def attention_realize_sidecar_category(tensors) -> str | None:
  if not _SIDECAR_STACK or _SIDECAR_STACK[-1] != "attention":
    return None
  if any(tensor_has_op(tensor, Ops.STORE) for tensor in tensors):
    return "attention_cache_write"
  return None


@contextmanager
def tensor_realize_sidecar_patch():
  original_realize = Tensor.realize

  def realize_with_attention_sidecar(self, *lst, **kwargs):
    category = attention_realize_sidecar_category((self, *lst))
    if category is None:
      return original_realize(self, *lst, **kwargs)
    with sidecar_scope(category):
      return original_realize(self, *lst, **kwargs)

  Tensor.realize = realize_with_attention_sidecar
  try:
    yield
  finally:
    Tensor.realize = original_realize


def cache_write_role(is_kv_shared_layer: bool, store_full_length_kv: bool) -> str:
  if is_kv_shared_layer:
    return "shared_consumer"
  if store_full_length_kv:
    return "shared_source"
  return "local"


@contextmanager
def cache_update_sidecar_patch():
  original_realize_cache_update = gemma_model.realize_cache_update

  def realize_cache_update_with_sidecar(
    key_cache: Tensor,
    value_cache: Tensor,
    key: Tensor,
    value: Tensor,
    start,
    end,
    *,
    layer_idx: int | None = None,
    layer_type: str | None = None,
    is_kv_shared_layer: bool = False,
    store_full_length_kv: bool = False,
    single_position: bool = False,
    packed_cache: Tensor | None = None,
  ) -> None:
    role = cache_write_role(is_kv_shared_layer, store_full_length_kv)
    if packed_cache is not None:
      parent_category = cache_write_sidecar_category("packed", role, layer_idx, layer_type)
      if cache_write_phase_target("packed", role, layer_idx, layer_type):
        rmsnorm_rope_category = cache_write_sidecar_category("packed", role, layer_idx, layer_type, phase="rmsnorm_rope")
        rhs_pack_category = cache_write_sidecar_category("packed", role, layer_idx, layer_type, phase="rhs_pack")
        store_category = cache_write_sidecar_category("packed", role, layer_idx, layer_type, phase="store")
        with sidecar_scope(parent_category):
          if _CACHE_WRITE_PHASE_CUTPOINTS:
            with sidecar_scope(rmsnorm_rope_category):
              key = key.realize()
              value = value.realize()
          if single_position:
            with sidecar_scope(rhs_pack_category):
              packed_kv = key.squeeze(2).unsqueeze(-1).cat(value.squeeze(2).unsqueeze(-1), dim=-1)
              if _CACHE_WRITE_PHASE_CUTPOINTS:
                packed_kv = packed_kv.realize()
            with sidecar_scope(store_category):
              assigned = packed_cache[:, :, start, :, :].assign(packed_kv)
              if _CACHE_WRITE_PHASE_CUTPOINTS:
                assigned.realize()
          else:
            with sidecar_scope(rhs_pack_category):
              packed_kv = key.unsqueeze(-1).cat(value.unsqueeze(-1), dim=-1)
              if _CACHE_WRITE_PHASE_CUTPOINTS:
                packed_kv = packed_kv.realize()
            with sidecar_scope(store_category):
              assigned = packed_cache[:, :, start:end, :, :].assign(packed_kv)
              if _CACHE_WRITE_PHASE_CUTPOINTS:
                assigned.realize()
          if not _CACHE_WRITE_PHASE_CUTPOINTS:
            assigned.realize()
        return
      with sidecar_scope(parent_category):
        if single_position:
          packed_kv = key.squeeze(2).unsqueeze(-1).cat(value.squeeze(2).unsqueeze(-1), dim=-1)
          packed_cache[:, :, start, :, :].assign(packed_kv).realize()
        else:
          packed_cache[:, :, start:end, :, :].assign(key.unsqueeze(-1).cat(value.unsqueeze(-1), dim=-1)).realize()
      return
    with sidecar_scope(cache_write_sidecar_category("key", role, layer_idx, layer_type)):
      if single_position:
        key_cache[:, :, start, :].assign(key.squeeze(2)).realize()
      else:
        key_cache[:, :, start:end, :].assign(key).realize()
    with sidecar_scope(cache_write_sidecar_category("value", role, layer_idx, layer_type)):
      if single_position:
        value_cache[:, :, start, :].assign(value.squeeze(2)).realize()
      else:
        value_cache[:, :, start:end, :].assign(value).realize()

  gemma_model.realize_cache_update = realize_cache_update_with_sidecar
  try:
    yield
  finally:
    gemma_model.realize_cache_update = original_realize_cache_update


@contextmanager
def add_linear_sidecar_patch():
  original_add_linear = getattr(TinyJitClass, "add_linear", None)
  if original_add_linear is None:
    yield
    return

  def add_linear_with_sidecar(self, linear: UOp, var_vals: dict[str, int]):
    if _SIDECAR_STACK:
      linear = add_sidecar_metadata(linear, _SIDECAR_STACK[-1])
    return original_add_linear(self, linear, var_vals)

  TinyJitClass.add_linear = add_linear_with_sidecar
  try:
    yield
  finally:
    TinyJitClass.add_linear = original_add_linear


@contextmanager
def method_sidecar_patch(owner: type, name: str, category: str):
  original = getattr(owner, name)

  @wraps(original)
  def wrapped(self, *args, **kwargs):
    with sidecar_scope(category):
      return original(self, *args, **kwargs)

  setattr(owner, name, wrapped)
  try:
    yield
  finally:
    setattr(owner, name, original)


def current_attention_context() -> dict[str, Any] | None:
  return _ATTENTION_CONTEXT_STACK[-1] if _ATTENTION_CONTEXT_STACK else None


def attention_phase_category(attn: GemmaAttention, phase: str) -> str | None:
  role = cache_write_role(attn.is_kv_shared_layer, attn.store_full_length_kv)
  if not cache_write_phase_target("packed", role, attn.layer_idx, attn.layer_type):
    return None
  return cache_write_sidecar_category("packed", role, attn.layer_idx, attn.layer_type, phase=phase)


@contextmanager
def attention_sidecar_patch():
  original = GemmaAttention.__call__

  @wraps(original)
  def wrapped(self, *args, **kwargs):
    context = {"attention": self, "after_kv_project": False}
    _ATTENTION_CONTEXT_STACK.append(context)
    try:
      with sidecar_scope("attention"):
        return original(self, *args, **kwargs)
    finally:
      _ATTENTION_CONTEXT_STACK.pop()

  GemmaAttention.__call__ = wrapped
  try:
    yield
  finally:
    GemmaAttention.__call__ = original


@contextmanager
def kv_projection_phase_sidecar_patch():
  original = GemmaAttention._project_kv

  @wraps(original)
  def wrapped(self, hidden_states, hidden_shape):
    category = attention_phase_category(self, "kv_projection")
    if category is None:
      result = original(self, hidden_states, hidden_shape)
    else:
      with sidecar_scope(category):
        result = original(self, hidden_states, hidden_shape)
        if _CACHE_WRITE_PHASE_CUTPOINTS:
          result = tuple(tensor.realize() for tensor in result)
    context = current_attention_context()
    if context is not None and context.get("attention") is self:
      context["after_kv_project"] = True
    return result

  GemmaAttention._project_kv = wrapped
  try:
    yield
  finally:
    GemmaAttention._project_kv = original


@contextmanager
def rmsnorm_phase_sidecar_patch():
  original = RMSNorm.__call__

  @wraps(original)
  def wrapped(self, *args, **kwargs):
    category = "norm"
    context = current_attention_context()
    if context is not None and context.get("after_kv_project"):
      attn = context["attention"]
      phase_category = attention_phase_category(attn, "rmsnorm_rope")
      if phase_category is not None and (self is getattr(attn, "k_norm", None) or self is getattr(attn, "v_norm", None)):
        category = phase_category
    with sidecar_scope(category):
      return original(self, *args, **kwargs)

  RMSNorm.__call__ = wrapped
  try:
    yield
  finally:
    RMSNorm.__call__ = original


@contextmanager
def rope_phase_sidecar_patch():
  original = gemma_model.apply_rotary_pos_emb

  @wraps(original)
  def wrapped(*args, **kwargs):
    category = None
    context = current_attention_context()
    if context is not None and context.get("after_kv_project"):
      category = attention_phase_category(context["attention"], "rmsnorm_rope")
    if category is None:
      return original(*args, **kwargs)
    with sidecar_scope(category):
      return original(*args, **kwargs)

  gemma_model.apply_rotary_pos_emb = wrapped
  try:
    yield
  finally:
    gemma_model.apply_rotary_pos_emb = original


@contextmanager
def gemma_profile_sidecars(
  *,
  cache_write_phase_cutpoints: bool = False,
  cache_write_phase_targets: tuple[CacheWritePhaseTarget, ...] | None = None,
):
  patches: list[tuple[type, str, str]] = [
    (TextScaledEmbedding, "__call__", "embedding_per_layer"),
    (GemmaModel, "project_per_layer_inputs", "embedding_per_layer"),
    (GemmaMLP, "__call__", "mlp"),
    (GemmaDecoderLayer, "__call__", "decoder_residual"),
    (GemmaForConditionalGeneration, "logits", "logits_argmax"),
    (GemmaForConditionalGeneration, "sample_next", "logits_argmax"),
  ]
  with ExitStack() as stack:
    stack.enter_context(cache_write_phase_target_scope(cache_write_phase_targets))
    stack.enter_context(cache_write_phase_cutpoint_scope(cache_write_phase_cutpoints))
    stack.enter_context(uop_creation_sidecar_patch())
    stack.enter_context(uop_replace_sidecar_patch())
    stack.enter_context(add_linear_sidecar_patch())
    stack.enter_context(tensor_realize_sidecar_patch())
    stack.enter_context(cache_update_sidecar_patch())
    stack.enter_context(attention_sidecar_patch())
    stack.enter_context(kv_projection_phase_sidecar_patch())
    stack.enter_context(rmsnorm_phase_sidecar_patch())
    stack.enter_context(rope_phase_sidecar_patch())
    for owner, name, category in patches:
      stack.enter_context(method_sidecar_patch(owner, name, category))
    yield


def classify_exec_item(prg, metadata) -> str:
  program_type = type(prg).__name__ if prg is not None else ""
  if "Graph" in program_type:
    return "graph_batch"
  if program_type == "RowwiseInt8DecodeLinearRunner":
    return "raw_metal_int8_gate_up"
  return classify_kernel(metadata)


def source_category(item) -> str:
  explicit_category = getattr(item, "category", None)
  if explicit_category is not None and explicit_category != "other":
    return explicit_category
  category = classify_exec_item(item.prg, item.metadata)
  if category != "other":
    return category
  return ast_uop_sidecar_category(getattr(item, "ast", None))


def source_item_category_basis(item) -> str:
  explicit_category = getattr(item, "category", None)
  if explicit_category is not None and explicit_category != "other":
    return "source_item_metadata"
  if sidecar_metadata_category(item.metadata) != "other":
    return "repo_sidecar_realize_scope_metadata"
  if source_range_metadata_category(item.metadata) != "other":
    return "source_item_metadata"
  if ast_uop_sidecar_category(getattr(item, "ast", None)) != "other":
    return UOP_SIDECAR_CATEGORY_BASIS
  return "unclassified_source_item_metadata"


def count_map(values, *, limit: int = 20) -> dict[str, int]:
  counts: dict[str, int] = defaultdict(int)
  for value in values:
    counts[str(value)] += 1
  return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def source_item_program_type(item) -> str:
  return type(item.prg).__name__ if item.prg is not None else ""


def source_item_display_name(item) -> str:
  return "" if item.prg is None else strip_ansi(getattr(item.prg, "display_name", ""))


def source_item_ast_root(item) -> str:
  ast = getattr(item, "ast", None)
  return "" if ast is None else str(getattr(ast, "op", ""))


def source_item_op_signature(item, *, limit: int = 8) -> str:
  ast = getattr(item, "ast", None)
  if ast is None:
    return ""
  try:
    op_counts = count_map((getattr(uop, "op", "") for uop in ast.toposort()), limit=limit)
  except Exception as exc:
    return f"error:{type(exc).__name__}"
  return ",".join(f"{op}:{count}" for op, count in op_counts.items())


def graph_source_count(row: dict[str, Any]) -> int | None:
  if "Graph" not in row["program_type"]:
    return 1
  match = BATCHED_DISPLAY_RE.match(row["display_name"])
  return None if match is None else int(match.group("count"))


def source_slice_summary(items) -> dict[str, Any]:
  category_counts: dict[str, int] = defaultdict(int)
  cache_write_phase_category_counts: dict[str, int] = defaultdict(int)
  program_type_counts: dict[str, int] = defaultdict(int)
  category_basis_counts: dict[str, int] = defaultdict(int)
  unclassified_items = []
  cache_write_phase_conflict_count = 0
  cache_write_phase_unclassified_count = 0
  for item in items:
    category = source_category(item)
    program_type_counts[source_item_program_type(item)] += 1
    category_counts[category] += 1
    category_basis_counts[source_item_category_basis(item)] += 1
    if cache_write_phase_target_category(category):
      phase_category, phase_conflict = source_item_cache_write_phase_category(item)
      if phase_category is None:
        cache_write_phase_unclassified_count += 1
      else:
        cache_write_phase_category_counts[phase_category] += 1
        if phase_conflict:
          cache_write_phase_conflict_count += 1
    if category == "other":
      unclassified_items.append(item)
  return {
    "category_counts": dict(sorted(category_counts.items())),
    "category_counts_rollup": category_count_rollups(category_counts),
    "cache_write_phase_category_counts": dict(sorted(cache_write_phase_category_counts.items())),
    "cache_write_phase_conflict_count": cache_write_phase_conflict_count,
    "cache_write_phase_unclassified_count": cache_write_phase_unclassified_count,
    "program_type_counts": dict(sorted(program_type_counts.items())),
    "category_basis_counts": dict(sorted(category_basis_counts.items())),
    "unclassified_source_summary": {
      "count": len(unclassified_items),
      "program_type_counts": count_map(source_item_program_type(item) for item in unclassified_items),
      "display_name_counts": count_map(source_item_display_name(item) for item in unclassified_items),
      "ast_root_counts": count_map(source_item_ast_root(item) for item in unclassified_items),
      "op_signature_counts": count_map(source_item_op_signature(item) for item in unclassified_items),
      "uop_sidecar_category_counts": count_map(
        json.dumps(ast_uop_sidecar_category_counts(getattr(item, "ast", None)), sort_keys=True)
        for item in unclassified_items
      ),
    },
  }


def category_basis(category_basis_counts: dict[str, int], source_count: int) -> str:
  if source_count == 0:
    return "empty"
  if category_basis_counts.get("repo_sidecar_realize_scope_metadata", 0) > 0:
    return "repo_sidecar_realize_scope_metadata"
  if category_basis_counts.get("source_item_metadata", 0) > 0:
    return "source_item_metadata"
  if category_basis_counts.get(UOP_SIDECAR_CATEGORY_BASIS, 0) > 0:
    return UOP_SIDECAR_CATEGORY_BASIS
  return "unclassified_source_item_metadata"


def execution_source_items(execution_item) -> list[Any]:
  prg = execution_item.prg
  return list(getattr(prg, "jit_cache", None) or [execution_item])


def attribute_execution_source_ranges(execution_items, rows: list[dict[str, Any]]) -> dict[str, Any]:
  cursor = 0
  items = []
  unparsed_graph_batches = 0
  source_count_mismatches = 0
  original_exec_count = 0
  for execution_item, row in zip(execution_items, rows):
    expected_source_count = graph_source_count(row)
    source_items = execution_source_items(execution_item)
    source_count = len(source_items)
    original_exec_count += source_count
    if expected_source_count is None:
      unparsed_graph_batches += 1
    elif expected_source_count != source_count:
      source_count_mismatches += 1

    source_start = cursor
    source_end = cursor + source_count - 1
    summary = source_slice_summary(source_items)
    basis = category_basis(summary["category_basis_counts"], source_count)
    items.append({
      "ordinal": row["ordinal"],
      "program_type": row["program_type"],
      "display_name": row["display_name"],
      "elapsed_ms": row["elapsed_ms"],
      "source_count": source_count,
      "expected_source_count": expected_source_count,
      "source_start": source_start,
      "source_end": source_end,
      "category_basis": basis,
      **summary,
    })
    cursor += source_count

  category_basis_counts: dict[str, int] = defaultdict(int)
  for item in items:
    category_basis_counts[item["category_basis"]] += 1
  return {
    "status": (
      "complete"
      if unparsed_graph_batches == 0 and source_count_mismatches == 0 and len(execution_items) == len(rows)
      else "incomplete"
    ),
    "original_exec_count": original_exec_count,
    "attributed_source_count": cursor,
    "unattributed_tail_count": max(0, original_exec_count - cursor),
    "unparsed_graph_batches": unparsed_graph_batches,
    "source_count_mismatches": source_count_mismatches,
    "execution_row_count": len(rows),
    "execution_item_count": len(execution_items),
    "category_basis_counts": dict(sorted(category_basis_counts.items())),
    "items": items,
  }


def attribute_profile_execution_sources(
  execution_items,
  rows: list[dict[str, Any]],
  phase_targets: tuple[CacheWritePhaseTarget, ...],
) -> dict[str, Any]:
  with cache_write_phase_target_scope(phase_targets):
    return attribute_execution_source_ranges(execution_items, rows)


def attach_source_attribution(rows: list[dict[str, Any]], attribution: dict[str, Any]) -> None:
  by_ordinal = {item["ordinal"]: item for item in attribution["items"]}
  for row in rows:
    item = by_ordinal.get(row["ordinal"])
    if item is None:
      continue
    row["source_start"] = item["source_start"]
    row["source_end"] = item["source_end"]
    row["source_count"] = item["source_count"]
    row["source_category_counts"] = item["category_counts"]
    row["source_category_counts_rollup"] = item["category_counts_rollup"]
    row["cache_write_phase_category_counts"] = item["cache_write_phase_category_counts"]
    row["cache_write_phase_conflict_count"] = item["cache_write_phase_conflict_count"]
    row["cache_write_phase_unclassified_count"] = item["cache_write_phase_unclassified_count"]
    row["source_category_basis"] = item["category_basis"]


def lower_profile_call(call: UOp, ctx: ExecContext, input_uops: tuple[UOp, ...]) -> ExecItem:
  if resolve_params is None or linear_to_schedule is None or pm_post_sched_cache is None:
    raise RuntimeError("profile_decode_jit requires a tinygrad checkout with captured-JIT lowering helpers")
  ast = call.src[0]
  if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "graph":
    inputs = resolve_params(ctx, call)
    bufs = flatten([b.bufs if isinstance(b, MultiBuffer) else [b] for b in (u.buffer for u in inputs)])
    graph_ast = ast.substitute(dict(zip(ast.src[1:], inputs)))
    graph_device = graph_ast.device if isinstance(graph_ast.device, str) else graph_ast.device[0]
    return ExecItem(ast, bufs, call.arg.metadata, prg=Device[graph_device].graph(graph_ast, bufs))

  resolved_linear = graph_rewrite(UOp(Ops.LINEAR, src=(call,)), pm_post_sched_cache, ctx=({}, input_uops), walk=True, name="profile jit call params to buffers")
  items = linear_to_schedule(resolved_linear)
  if len(items) != 1:
    raise RuntimeError(f"expected one lowered item for profile call, got {len(items)}")
  return items[0].lower()


def checkpoint_quantization_summary(model_dir: Path) -> dict[str, Any]:
  manifest = load_quantization_manifest(model_dir)
  tensors = {} if manifest is None else manifest.get("tensors", {})
  dtypes: dict[str, int] = defaultdict(int)
  sample_shapes: dict[str, str] = {}
  for path in resolve_weight_files(model_dir):
    state = nn_state.safe_load(path)
    for name, tensor in state.items():
      dtype_name = str(getattr(tensor.dtype, "name", tensor.dtype))
      dtypes[dtype_name] += 1
      if name in (
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.per_layer_model_projection.weight",
      ):
        sample_shapes[name] = f"{tuple(tensor.shape)} {tensor.dtype}"
  return {
    "manifest_method": None if manifest is None else manifest.get("method"),
    "manifest_quantized_tensor_count": len(tensors),
    "raw_dtype_counts": dict(sorted(dtypes.items())),
    "sample_tensors": sample_shapes,
  }


def language_model(model):
  return getattr(model.model, "language_model", model.model)


def build_zero_cache(model, context_length: int, max_length: int) -> GemmaCache:
  lm = language_model(model)
  dtype = lm.embed_tokens.weight.dtype
  cache = GemmaCache.empty(len(lm.layers), max_length=max_length)
  for idx, layer in enumerate(lm.layers):
    attn = layer.self_attn
    cache.entries[idx] = make_packed_cache_entry(
      1,
      attn.num_key_value_heads,
      max_length,
      attn.head_dim,
      device=model.device,
      dtype=dtype,
      active_length=context_length,
    )
  cache.past_seen_tokens = context_length
  cache.decode_sliding_window = True
  return cache


def run_captured_items(captured, token: Tensor, start_var) -> tuple[list[dict[str, Any]], dict[str, int], list[ExecItem]]:
  if ExecContext is None:
    raise RuntimeError("profile_decode_jit requires a tinygrad checkout with ExecContext for captured JIT profiling")
  input_uops, var_vals, names, expected_info = _prepare_jit_inputs((token, start_var), {})
  if names != captured.expected_names:
    raise RuntimeError(f"JIT input names changed: {names!r} != {captured.expected_names!r}")
  if expected_info != captured.expected_input_info:
    raise RuntimeError("JIT input metadata changed before profiling")

  input_uops_tuple = tuple(input_uops)
  ctx = ExecContext(var_vals, input_uops_tuple, do_update_stats=False, jit=True)
  execution_items = [lower_profile_call(call, ctx, input_uops_tuple) for call in captured.linear.src]
  rows = []
  for ordinal, item in enumerate(execution_items):
    item.lower()
    for buffer in item.bufs:
      if buffer is not None:
        buffer.ensure_allocated()
    prg = item.prg
    metadata = _metadata_payload(item.metadata)
    category = classify_exec_item(prg, item.metadata)
    program_type = type(prg).__name__ if prg is not None else ""
    display_name = "" if prg is None else strip_ansi(prg.display_name)
    device = "" if prg is None else prg.device
    est_ops = sym_infer(prg.estimates.ops, var_vals) if isinstance(prg, CompiledRunner) else 0
    est_mem = sym_infer(prg.estimates.mem, var_vals) if isinstance(prg, CompiledRunner) else 0
    start = time.perf_counter()
    elapsed = item.run(var_vals, wait=True, jit=True, do_update_stats=False)
    wall_elapsed = time.perf_counter() - start
    rows.append({
      "ordinal": ordinal,
      "category": category,
      "program_type": program_type,
      "display_name": display_name,
      "device": device,
      "elapsed_ms": (elapsed if elapsed is not None else wall_elapsed) * 1000.0,
      "est_ops": int(est_ops),
      "est_mem": int(est_mem),
      "metadata": metadata,
    })
  return rows, var_vals, execution_items


def summarize_exec_items(items) -> dict[str, Any]:
  by_program_type: dict[str, int] = defaultdict(int)
  by_category: dict[str, int] = defaultdict(int)
  for item in items:
    prg = item.prg
    program_type = type(prg).__name__ if prg is not None else ""
    by_program_type[program_type] += 1
    by_category[classify_exec_item(prg, item.metadata)] += 1
  return {
    "exec_count": len(items),
    "program_type_counts": dict(sorted(by_program_type.items())),
    "category_counts": dict(sorted(by_category.items())),
    "category_counts_rollup": category_count_rollups(by_category),
    "graph_batch_count": sum(count for name, count in by_program_type.items() if "Graph" in name),
    "compiled_runner_count": by_program_type.get("CompiledRunner", 0),
    "raw_gate_up_runner_count": by_program_type.get("RowwiseInt8DecodeLinearRunner", 0),
  }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
  by_category: dict[str, dict[str, Any]] = {}
  by_category_rollup: dict[str, dict[str, Any]] = {}
  by_program_type: dict[str, dict[str, Any]] = {}
  for row in rows:
    category = row["category"]
    add_category_timing(
      by_category,
      category,
      count_key="kernel_count",
      count=1,
      elapsed_ms=row["elapsed_ms"],
      est_ops=row["est_ops"],
      est_mem=row["est_mem"],
    )
    add_category_timing(
      by_category_rollup,
      category_rollup(category),
      count_key="kernel_count",
      count=1,
      elapsed_ms=row["elapsed_ms"],
      est_ops=row["est_ops"],
      est_mem=row["est_mem"],
    )
    program_type = row["program_type"]
    program_entry = by_program_type.setdefault(program_type, {"kernel_count": 0, "elapsed_ms": 0.0})
    program_entry["kernel_count"] += 1
    program_entry["elapsed_ms"] += row["elapsed_ms"]
  total_elapsed = sum(row["elapsed_ms"] for row in rows)
  total_ops = sum(row["est_ops"] for row in rows)
  total_mem = sum(row["est_mem"] for row in rows)
  for entry in by_category.values():
    entry["elapsed_share"] = entry["elapsed_ms"] / total_elapsed if total_elapsed > 0 else 0.0
    entry["ops_share"] = entry["est_ops"] / total_ops if total_ops else 0.0
  for entry in by_category_rollup.values():
    entry["elapsed_share"] = entry["elapsed_ms"] / total_elapsed if total_elapsed > 0 else 0.0
    entry["ops_share"] = entry["est_ops"] / total_ops if total_ops else 0.0
  for entry in by_program_type.values():
    entry["elapsed_share"] = entry["elapsed_ms"] / total_elapsed if total_elapsed > 0 else 0.0
  return {
    "kernel_count": len(rows),
    "elapsed_ms": total_elapsed,
    "est_ops": total_ops,
    "est_mem": total_mem,
    "by_category": dict(sorted(by_category.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
    "by_category_rollup": dict(sorted(by_category_rollup.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
    "by_program_type": dict(sorted(by_program_type.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
    "top_elapsed_kernels": sorted(rows, key=lambda row: row["elapsed_ms"], reverse=True)[:20],
    "top_est_ops_kernels": sorted(rows, key=lambda row: row["est_ops"], reverse=True)[:20],
  }


def summarize_source_attribution(attribution: dict[str, Any]) -> dict[str, Any]:
  by_category: dict[str, dict[str, Any]] = {}
  by_category_rollup: dict[str, dict[str, Any]] = {}
  by_basis: dict[str, dict[str, Any]] = {}
  total_elapsed = sum(float(item["elapsed_ms"]) for item in attribution["items"])
  total_source_count = sum(int(item["source_count"]) for item in attribution["items"])
  for item in attribution["items"]:
    source_count = max(1, int(item["source_count"]))
    elapsed_ms = float(item["elapsed_ms"])
    basis = item["category_basis"]
    basis_entry = by_basis.setdefault(basis, {"source_count": 0, "execution_count": 0, "elapsed_ms": 0.0})
    basis_entry["source_count"] += source_count
    basis_entry["execution_count"] += 1
    basis_entry["elapsed_ms"] += elapsed_ms
    for category, count in item["category_counts"].items():
      count_int = int(count)
      category_elapsed = elapsed_ms * (count_int / source_count)
      entry = by_category.setdefault(category, {"source_count": 0, "elapsed_ms": 0.0})
      entry["source_count"] += count_int
      entry["elapsed_ms"] += category_elapsed
      rollup = category_rollup(category)
      rollup_entry = by_category_rollup.setdefault(rollup, {"source_count": 0, "elapsed_ms": 0.0})
      rollup_entry["source_count"] += count_int
      rollup_entry["elapsed_ms"] += category_elapsed
  for entry in by_category.values():
    entry["elapsed_share"] = entry["elapsed_ms"] / total_elapsed if total_elapsed else 0.0
  for entry in by_category_rollup.values():
    entry["elapsed_share"] = entry["elapsed_ms"] / total_elapsed if total_elapsed else 0.0
  for entry in by_basis.values():
    entry["elapsed_share"] = entry["elapsed_ms"] / total_elapsed if total_elapsed else 0.0
  return {
    "source_count": total_source_count,
    "elapsed_ms": total_elapsed,
    "by_category": dict(sorted(by_category.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
    "by_category_rollup": dict(sorted(by_category_rollup.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
    "by_category_basis": dict(sorted(by_basis.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
  }


def summarize_cache_write_phase_attribution(attribution: dict[str, Any]) -> dict[str, Any]:
  by_category: dict[str, dict[str, Any]] = {}
  by_parent: dict[str, dict[str, Any]] = {}
  by_phase: dict[str, dict[str, Any]] = {}
  total_elapsed = sum(float(item["elapsed_ms"]) for item in attribution["items"])
  phase_source_count = 0
  conflict_source_count = 0
  unclassified_source_count = 0
  for item in attribution["items"]:
    source_count = max(1, int(item["source_count"]))
    elapsed_ms = float(item["elapsed_ms"])
    phase_counts = item.get("cache_write_phase_category_counts", {})
    phase_item_count = sum(int(count) for count in phase_counts.values())
    phase_source_count += phase_item_count
    conflict_source_count += int(item.get("cache_write_phase_conflict_count", 0))
    unclassified_source_count += int(item.get("cache_write_phase_unclassified_count", 0))
    for category, count in phase_counts.items():
      count_int = int(count)
      category_elapsed = elapsed_ms * (count_int / source_count)
      entry = by_category.setdefault(category, {"source_count": 0, "elapsed_ms": 0.0})
      entry["source_count"] += count_int
      entry["elapsed_ms"] += category_elapsed
      parent = parent_cache_write_category(category)
      parent_entry = by_parent.setdefault(parent, {"source_count": 0, "elapsed_ms": 0.0})
      parent_entry["source_count"] += count_int
      parent_entry["elapsed_ms"] += category_elapsed
      phase = cache_write_phase_name(category) or "unknown"
      phase_entry = by_phase.setdefault(phase, {"source_count": 0, "elapsed_ms": 0.0})
      phase_entry["source_count"] += count_int
      phase_entry["elapsed_ms"] += category_elapsed
  for groups in (by_category, by_parent, by_phase):
    for entry in groups.values():
      entry["elapsed_share"] = entry["elapsed_ms"] / total_elapsed if total_elapsed else 0.0
  return {
    "source_count": phase_source_count,
    "elapsed_ms": sum(entry["elapsed_ms"] for entry in by_category.values()),
    "total_elapsed_ms": total_elapsed,
    "conflict_source_count": conflict_source_count,
    "unclassified_source_count": unclassified_source_count,
    "by_category": dict(sorted(by_category.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
    "by_parent": dict(sorted(by_parent.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
    "by_phase": dict(sorted(by_phase.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
  }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
      "ordinal",
      "category",
      "program_type",
      "display_name",
      "device",
      "elapsed_ms",
      "est_ops",
      "est_mem",
      "source_start",
      "source_end",
      "source_count",
      "source_category_counts",
      "source_category_counts_rollup",
      "cache_write_phase_category_counts",
      "cache_write_phase_conflict_count",
      "cache_write_phase_unclassified_count",
      "source_category_basis",
      "metadata",
    ], lineterminator="\n")
    writer.writeheader()
    for row in rows:
      writer.writerow({
        **{key: row[key] for key in ("ordinal", "category", "display_name", "device", "elapsed_ms", "est_ops", "est_mem")},
        "program_type": row["program_type"],
        "source_start": row.get("source_start", ""),
        "source_end": row.get("source_end", ""),
        "source_count": row.get("source_count", ""),
        "source_category_counts": json.dumps(row.get("source_category_counts", {}), sort_keys=True),
        "source_category_counts_rollup": json.dumps(row.get("source_category_counts_rollup", {}), sort_keys=True),
        "cache_write_phase_category_counts": json.dumps(row.get("cache_write_phase_category_counts", {}), sort_keys=True),
        "cache_write_phase_conflict_count": row.get("cache_write_phase_conflict_count", ""),
        "cache_write_phase_unclassified_count": row.get("cache_write_phase_unclassified_count", ""),
        "source_category_basis": row.get("source_category_basis", ""),
        "metadata": json.dumps(row["metadata"], sort_keys=True),
      })


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile a synthetic post-window Gemma 4 decode TinyJit on tinygrad METAL.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--device", default="METAL")
  parser.add_argument("--context-length", type=int, default=700)
  parser.add_argument("--profile-start", type=int, help="Decode position to profile. Defaults to context length plus 3.")
  parser.add_argument("--jit-mode", type=int, default=2, choices=[1, 2], help="tinygrad JIT mode. JIT=1 applies Metal graph batching; JIT=2 profiles ungraphed items.")
  parser.add_argument(
    "--layer13-phase-cutpoints",
    action="store_true",
    help="Backward-compatible alias for --phase-cutpoints with the default layer13 phase target.",
  )
  parser.add_argument(
    "--phase-cutpoints",
    action="store_true",
    help="Profiler-only diagnostic: force realization cutpoints around selected cache-write phase targets.",
  )
  parser.add_argument(
    "--phase-target",
    action="append",
    dest="phase_targets",
    metavar="TARGET",
    help="Cache-write phase target to instrument. Presets: layer13, all-shared-source, all-local, all-packed. Exact selectors: local-layer<N>, shared-source-layer<N>, shared-consumer-layer<N>. May be repeated. Defaults to layer13.",
  )
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--csv-out", type=Path)
  args = parser.parse_args()
  phase_targets = parse_cache_write_phase_targets(args.phase_targets)
  phase_cutpoints = args.phase_cutpoints or args.layer13_phase_cutpoints

  resolved_device = prepare_device(args.device)
  model = load_pretrained(args.model_dir, device=resolved_device)
  lm = language_model(model)
  sliding_start = model._sliding_decode_start()
  if sliding_start is None:
    raise SystemExit("model does not expose a Metal sliding decode start")
  if args.context_length < sliding_start:
    raise SystemExit(f"--context-length must be >= sliding start {sliding_start}")

  profile_start = args.profile_start if args.profile_start is not None else args.context_length + 3
  max_length = profile_start + 2
  cache = build_zero_cache(model, args.context_length, max_length)
  token = Tensor([[2]], dtype="int32", device=model.device).realize()

  with gemma_profile_sidecars(cache_write_phase_cutpoints=phase_cutpoints, cache_write_phase_targets=phase_targets):
    with Context(JIT=args.jit_mode, BEAM=0, TRACEMETA=2):
      rollout_jit = TinyJit(lambda token, start_pos: model._rollout_next_token(token, start_pos, cache, 0.0, decode_sliding_window=True))
      for offset in range(3):
        out = rollout_jit(token, Variable("gemma_start_pos_window", sliding_start, max_length - 1).bind(args.context_length + offset))
        out.realize()
        cache.set_active_length(args.context_length + offset + 1)
      captured = rollout_jit.captured
      if captured is None:
        raise RuntimeError("decode TinyJit did not capture")
      GlobalCounters.reset()
      rows, var_vals, execution_items = run_captured_items(captured, token, Variable("gemma_start_pos_window", sliding_start, max_length - 1).bind(profile_start))

  source_attribution = attribute_profile_execution_sources(execution_items, rows, phase_targets)
  attach_source_attribution(rows, source_attribution)
  with cache_write_phase_target_scope(phase_targets):
    original_source_summary = source_slice_summary([
      source_item
      for execution_item in execution_items
      for source_item in execution_source_items(execution_item)
    ])
  original_capture_summary = {
    "exec_count": source_attribution["original_exec_count"],
    "program_type_counts": original_source_summary["program_type_counts"],
    "category_counts": original_source_summary["category_counts"],
    "category_counts_rollup": original_source_summary["category_counts_rollup"],
    "cache_write_phase_category_counts": original_source_summary["cache_write_phase_category_counts"],
    "cache_write_phase_conflict_count": original_source_summary["cache_write_phase_conflict_count"],
    "cache_write_phase_unclassified_count": original_source_summary["cache_write_phase_unclassified_count"],
    "category_basis_counts": original_source_summary["category_basis_counts"],
    "unclassified_source_summary": original_source_summary["unclassified_source_summary"],
    "graph_batch_count": 0,
    "compiled_runner_count": sum(
      1
      for execution_item in execution_items
      for source_item in execution_source_items(execution_item)
      if type(source_item.prg).__name__ == "CompiledRunner"
    ),
    "raw_gate_up_runner_count": sum(
      1
      for execution_item in execution_items
      for source_item in execution_source_items(execution_item)
      if type(source_item.prg).__name__ == "RowwiseInt8DecodeLinearRunner"
    ),
  }
  post_graph_summary = summarize_exec_items(execution_items)
  source_attributed_summary = summarize_source_attribution(source_attribution)
  source_attributed_cache_write_phase_summary = summarize_cache_write_phase_attribution(source_attribution)
  csv_out = args.csv_out or args.out.with_suffix(".csv")
  write_csv(csv_out, rows)
  payload = {
    "model_dir": str(args.model_dir),
    "device": resolved_device,
    "tinygrad_path": str(Path(tinygrad.__file__).resolve()),
    "context_length": args.context_length,
    "profile_start": profile_start,
    "sliding_start": sliding_start,
    "max_length": max_length,
    "jit_mode": args.jit_mode,
    "jit_interpretation": "JIT=1 graph-batched execution" if args.jit_mode == 1 else "JIT=2 ungraphed per-kernel timing",
    "metal_int8_gate_up": "default",
    "cache_write_phase_targets": args.phase_targets or ["layer13"],
    "cache_write_phase_cutpoints": phase_cutpoints,
    "tensor_dtype": str(lm.embed_tokens.weight.dtype),
    "text_config": {
      "num_hidden_layers": lm.config.num_hidden_layers,
      "num_attention_heads": lm.config.num_attention_heads,
      "num_key_value_heads": lm.config.num_key_value_heads,
      "hidden_size": lm.config.hidden_size,
      "intermediate_size": lm.config.intermediate_size,
      "vocab_size": lm.config.vocab_size,
      "sliding_window": lm.config.sliding_window,
      "layer_types": lm.config.layer_types,
    },
    "quantization": checkpoint_quantization_summary(args.model_dir),
    "var_vals": var_vals,
    "original_capture": original_capture_summary,
    "post_graph_execution": post_graph_summary,
    "source_attribution": source_attribution,
    "source_attributed_summary": source_attributed_summary,
    "source_attributed_cache_write_phase_summary": source_attributed_cache_write_phase_summary,
    "summary": summarize_rows(rows),
    "csv": str(csv_out),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "out": str(args.out),
    "csv": str(csv_out),
    "kernel_count": payload["summary"]["kernel_count"],
    "elapsed_ms": round(payload["summary"]["elapsed_ms"], 3),
    "jit_mode": args.jit_mode,
    "metal_int8_gate_up": "default",
    "cache_write_phase_targets": args.phase_targets or ["layer13"],
    "cache_write_phase_cutpoints": phase_cutpoints,
    "original_capture": payload["original_capture"],
    "post_graph_execution": payload["post_graph_execution"],
    "source_attribution": {
      "status": source_attribution["status"],
      "original_exec_count": source_attribution["original_exec_count"],
      "attributed_source_count": source_attribution["attributed_source_count"],
      "unattributed_tail_count": source_attribution["unattributed_tail_count"],
      "unparsed_graph_batches": source_attribution["unparsed_graph_batches"],
      "category_basis_counts": source_attribution["category_basis_counts"],
      "top_elapsed_batches": [
        {
          "ordinal": item["ordinal"],
          "elapsed_ms": round(item["elapsed_ms"], 3),
          "source_count": item["source_count"],
          "source_start": item["source_start"],
          "source_end": item["source_end"],
          "category_basis": item["category_basis"],
          "category_counts": item["category_counts"],
          "category_counts_rollup": item["category_counts_rollup"],
          "category_basis_counts": item["category_basis_counts"],
        }
        for item in sorted(source_attribution["items"], key=lambda item: item["elapsed_ms"], reverse=True)[:5]
      ],
    },
    "source_attributed_by_category": {
      key: {
        "source_count": value["source_count"],
        "elapsed_ms": round(value["elapsed_ms"], 3),
        "elapsed_share": round(value["elapsed_share"], 4),
      }
      for key, value in source_attributed_summary["by_category"].items()
    },
    "source_attributed_by_category_rollup": {
      key: {
        "source_count": value["source_count"],
        "elapsed_ms": round(value["elapsed_ms"], 3),
        "elapsed_share": round(value["elapsed_share"], 4),
      }
      for key, value in source_attributed_summary["by_category_rollup"].items()
    },
    "source_attributed_by_basis": {
      key: {
        "source_count": value["source_count"],
        "execution_count": value["execution_count"],
        "elapsed_ms": round(value["elapsed_ms"], 3),
        "elapsed_share": round(value["elapsed_share"], 4),
      }
      for key, value in source_attributed_summary["by_category_basis"].items()
    },
    "by_category": {
      key: {
        "kernel_count": value["kernel_count"],
        "elapsed_ms": round(value["elapsed_ms"], 3),
        "elapsed_share": round(value["elapsed_share"], 4),
      }
      for key, value in payload["summary"]["by_category"].items()
    },
    "by_category_rollup": {
      key: {
        "kernel_count": value["kernel_count"],
        "elapsed_ms": round(value["elapsed_ms"], 3),
        "elapsed_share": round(value["elapsed_share"], 4),
      }
      for key, value in payload["summary"]["by_category_rollup"].items()
    },
    "by_program_type": {
      key: {
        "kernel_count": value["kernel_count"],
        "elapsed_ms": round(value["elapsed_ms"], 3),
        "elapsed_share": round(value["elapsed_share"], 4),
      }
      for key, value in payload["summary"]["by_program_type"].items()
    },
  }, sort_keys=True))


if __name__ == "__main__":
  main()
