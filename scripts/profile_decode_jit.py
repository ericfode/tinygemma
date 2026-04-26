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
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from tinygrad import Device, Tensor, TinyJit, Variable
from tinygrad.device import MultiBuffer
from tinygrad.engine.jit import TinyJit as TinyJitClass
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import CompiledRunner, ExecContext, ExecItem, resolve_params
from tinygrad.helpers import Context, GlobalCounters, Metadata, flatten
from tinygrad.nn import state as nn_state
from tinygrad.schedule import linear_to_schedule, pm_post_sched_cache
from tinygrad.uop.ops import Ops, UOp, graph_rewrite, sym_infer

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
from tinygrad_gemma.multimodal import GemmaForConditionalGeneration
from tinygrad_gemma.runtime import prepare_device


DEFAULT_MODEL_DIR = Path("/Users/ericfode/Downloads/tinygrad-gemma/checkpoints/gemma-4-E2B-int8")
DEFAULT_OUT = Path("benchmarks/gemma4-metal-postwindow-jit-profile-current.json")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BATCHED_DISPLAY_RE = re.compile(r"^<batched (?P<count>\d+)>$")


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
  "attention_key_cache_write",
  "attention_value_cache_write",
  "attention_cache_write",
  "attention",
  "embedding_per_layer",
  "decoder_residual",
  "norm",
)
_SIDECAR_STACK: list[str] = []


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


def strip_ansi(value: str) -> str:
  return ANSI_RE.sub("", value)


def priority_category(categories: set[str]) -> str:
  for category in PROFILE_CATEGORIES:
    if category in categories:
      return category
  return "other"


def sidecar_metadata_category(metadata) -> str:
  categories: set[str] = set()
  for item in metadata:
    name = str(getattr(item, "name", ""))
    caller = str(getattr(item, "caller", ""))
    if name in PROFILE_CATEGORIES:
      categories.add(name)
    if caller.startswith("repo_sidecar:") and "::" in caller:
      category = caller.rsplit("::", 1)[-1]
      if category in PROFILE_CATEGORIES:
        categories.add(category)
  return priority_category(categories)


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
  if category not in PROFILE_CATEGORIES:
    raise ValueError(f"unknown profile sidecar category {category!r}")
  _SIDECAR_STACK.append(category)
  try:
    yield
  finally:
    _SIDECAR_STACK.pop()


def add_sidecar_metadata(linear: UOp, category: str) -> UOp:
  metadata = (Metadata(name=category, caller=f"repo_sidecar:1::{category}"),)
  return linear.replace(src=tuple(
    call.replace(arg=replace(call.arg, metadata=call.arg.metadata or metadata))
    for call in linear.src
  ))


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
  ) -> None:
    del layer_idx, layer_type
    role = cache_write_role(is_kv_shared_layer, store_full_length_kv)
    with sidecar_scope(f"attention_key_cache_write_{role}"):
      key_cache[:, :, start:end, :].assign(key).realize()
    with sidecar_scope(f"attention_value_cache_write_{role}"):
      value_cache[:, :, start:end, :].assign(value).realize()

  gemma_model.realize_cache_update = realize_cache_update_with_sidecar
  try:
    yield
  finally:
    gemma_model.realize_cache_update = original_realize_cache_update


@contextmanager
def add_linear_sidecar_patch():
  original_add_linear = TinyJitClass.add_linear

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


@contextmanager
def gemma_profile_sidecars():
  patches: list[tuple[type, str, str]] = [
    (TextScaledEmbedding, "__call__", "embedding_per_layer"),
    (GemmaModel, "project_per_layer_inputs", "embedding_per_layer"),
    (RMSNorm, "__call__", "norm"),
    (GemmaAttention, "__call__", "attention"),
    (GemmaMLP, "__call__", "mlp"),
    (GemmaDecoderLayer, "__call__", "decoder_residual"),
    (GemmaForConditionalGeneration, "logits", "logits_argmax"),
    (GemmaForConditionalGeneration, "sample_next", "logits_argmax"),
  ]
  with ExitStack() as stack:
    stack.enter_context(add_linear_sidecar_patch())
    stack.enter_context(tensor_realize_sidecar_patch())
    stack.enter_context(cache_update_sidecar_patch())
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
  return getattr(item, "category", None) or classify_exec_item(item.prg, item.metadata)


def source_item_category_basis(item) -> str:
  explicit_category = getattr(item, "category", None)
  if explicit_category is not None:
    return "source_item_metadata" if explicit_category != "other" else "unclassified_source_item_metadata"
  if sidecar_metadata_category(item.metadata) != "other":
    return "repo_sidecar_realize_scope_metadata"
  if source_range_metadata_category(item.metadata) != "other":
    return "source_item_metadata"
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
  program_type_counts: dict[str, int] = defaultdict(int)
  category_basis_counts: dict[str, int] = defaultdict(int)
  unclassified_items = []
  for item in items:
    category = source_category(item)
    program_type_counts[source_item_program_type(item)] += 1
    category_counts[category] += 1
    category_basis_counts[source_item_category_basis(item)] += 1
    if category == "other":
      unclassified_items.append(item)
  return {
    "category_counts": dict(sorted(category_counts.items())),
    "program_type_counts": dict(sorted(program_type_counts.items())),
    "category_basis_counts": dict(sorted(category_basis_counts.items())),
    "unclassified_source_summary": {
      "count": len(unclassified_items),
      "program_type_counts": count_map(source_item_program_type(item) for item in unclassified_items),
      "display_name_counts": count_map(source_item_display_name(item) for item in unclassified_items),
      "ast_root_counts": count_map(source_item_ast_root(item) for item in unclassified_items),
      "op_signature_counts": count_map(source_item_op_signature(item) for item in unclassified_items),
    },
  }


def category_basis(category_basis_counts: dict[str, int], source_count: int) -> str:
  if source_count == 0:
    return "empty"
  if category_basis_counts.get("repo_sidecar_realize_scope_metadata", 0) > 0:
    return "repo_sidecar_realize_scope_metadata"
  if category_basis_counts.get("source_item_metadata", 0) > 0:
    return "source_item_metadata"
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
    row["source_category_basis"] = item["category_basis"]


def lower_profile_call(call: UOp, ctx: ExecContext, input_uops: tuple[UOp, ...]) -> ExecItem:
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


def validate_metal_int8_gate_up_mode(mode: str) -> None:
  if mode == "raw":
    raise SystemExit(
      "--metal-int8-gate-up raw is abandoned: the custom Runner breaks MetalGraph batching and is not a graphable decode optimization path"
    )


def build_zero_cache(model, context_length: int, max_length: int) -> GemmaCache:
  lm = language_model(model)
  dtype = lm.embed_tokens.weight.dtype
  cache = GemmaCache.empty(len(lm.layers), max_length=max_length)
  for idx, layer in enumerate(lm.layers):
    attn = layer.self_attn
    key = Tensor.zeros(
      1,
      attn.num_key_value_heads,
      max_length,
      attn.head_dim,
      device=model.device,
      dtype=dtype,
    ).contiguous().realize()
    value = Tensor.zeros(
      1,
      attn.num_key_value_heads,
      max_length,
      attn.head_dim,
      device=model.device,
      dtype=dtype,
    ).contiguous().realize()
    cache.entries[idx] = GemmaCacheEntry(key=key, value=value, length=context_length)
  cache.past_seen_tokens = context_length
  cache.decode_sliding_window = True
  return cache


def run_captured_items(captured, token: Tensor, start_var) -> tuple[list[dict[str, Any]], dict[str, int], list[ExecItem]]:
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
    "graph_batch_count": sum(count for name, count in by_program_type.items() if "Graph" in name),
    "compiled_runner_count": by_program_type.get("CompiledRunner", 0),
    "raw_gate_up_runner_count": by_program_type.get("RowwiseInt8DecodeLinearRunner", 0),
  }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
  by_category: dict[str, dict[str, Any]] = {}
  by_program_type: dict[str, dict[str, Any]] = {}
  for row in rows:
    category = row["category"]
    entry = by_category.setdefault(category, {"kernel_count": 0, "elapsed_ms": 0.0, "est_ops": 0, "est_mem": 0})
    entry["kernel_count"] += 1
    entry["elapsed_ms"] += row["elapsed_ms"]
    entry["est_ops"] += row["est_ops"]
    entry["est_mem"] += row["est_mem"]
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
  for entry in by_program_type.values():
    entry["elapsed_share"] = entry["elapsed_ms"] / total_elapsed if total_elapsed > 0 else 0.0
  return {
    "kernel_count": len(rows),
    "elapsed_ms": total_elapsed,
    "est_ops": total_ops,
    "est_mem": total_mem,
    "by_category": dict(sorted(by_category.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
    "by_program_type": dict(sorted(by_program_type.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
    "top_elapsed_kernels": sorted(rows, key=lambda row: row["elapsed_ms"], reverse=True)[:20],
    "top_est_ops_kernels": sorted(rows, key=lambda row: row["est_ops"], reverse=True)[:20],
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
  parser.add_argument("--metal-int8-gate-up", choices=["default", "raw"], default="default", help="Use the default tinygrad fused int8 gate/up path or opt into the raw Metal gate/up Runner.")
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--csv-out", type=Path)
  args = parser.parse_args()
  validate_metal_int8_gate_up_mode(args.metal_int8_gate_up)

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

  with gemma_profile_sidecars():
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

  source_attribution = attribute_execution_source_ranges(execution_items, rows)
  attach_source_attribution(rows, source_attribution)
  original_source_summary = source_slice_summary([
    source_item
    for execution_item in execution_items
    for source_item in execution_source_items(execution_item)
  ])
  original_capture_summary = {
    "exec_count": source_attribution["original_exec_count"],
    "program_type_counts": original_source_summary["program_type_counts"],
    "category_counts": original_source_summary["category_counts"],
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
    "metal_int8_gate_up": args.metal_int8_gate_up,
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
    "metal_int8_gate_up": args.metal_int8_gate_up,
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
          "category_basis_counts": item["category_basis_counts"],
        }
        for item in sorted(source_attribution["items"], key=lambda item: item["elapsed_ms"], reverse=True)[:5]
      ],
    },
    "by_category": {
      key: {
        "kernel_count": value["kernel_count"],
        "elapsed_ms": round(value["elapsed_ms"], 3),
        "elapsed_share": round(value["elapsed_share"], 4),
      }
      for key, value in payload["summary"]["by_category"].items()
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
