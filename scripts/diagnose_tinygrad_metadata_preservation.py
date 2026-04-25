from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from tinygrad import Tensor, TinyJit
from tinygrad.engine.jit import TinyJit as TinyJitClass
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import ExecContext
from tinygrad.helpers import Context, Metadata
from tinygrad.schedule import linear_to_schedule, pm_post_sched_cache
from tinygrad.uop.ops import Ops, UOp, graph_rewrite


DEFAULT_OUT = Path("benchmarks/tinygrad-metadata-preservation-current.json")


def metadata_payload(metadata) -> list[dict[str, str]]:
  return [
    {
      "name": str(getattr(item, "name", "")),
      "caller": str(getattr(item, "caller", "")),
      "backward": str(getattr(item, "backward", "")),
    }
    for item in (metadata or ())
  ]


def metadata_count(uop: UOp) -> int:
  return sum(len(item.metadata or ()) for item in uop.toposort())


def metadata_examples(uop: UOp, limit: int = 8) -> list[dict[str, str]]:
  examples: list[dict[str, str]] = []
  for item in uop.toposort():
    examples.extend(metadata_payload(item.metadata))
    if len(examples) >= limit:
      return examples[:limit]
  return examples


def lazy_metadata_probe(device: str) -> dict[str, Any]:
  with Context(TRACEMETA=2):
    source = Tensor.empty(4, device=device).realize()
    added = source + 1
    multiplied = added * 2
  return {
    "added_uop_metadata": metadata_payload(added.uop.metadata),
    "multiplied_uop_metadata": metadata_payload(multiplied.uop.metadata),
    "multiplied_toposort_metadata_count": metadata_count(multiplied.uop),
    "multiplied_toposort_metadata_examples": metadata_examples(multiplied.uop),
  }


def jitted_metadata_function(source: Tensor) -> Tensor:
  added = source + 1
  multiplied = added * 2
  return multiplied.contiguous().realize()


_SIDECAR_STACK: list[str] = []


@contextmanager
def sidecar_scope(name: str):
  _SIDECAR_STACK.append(name)
  try:
    yield
  finally:
    _SIDECAR_STACK.pop()


def jitted_sidecar_metadata_function(source: Tensor) -> Tensor:
  with sidecar_scope("toy_sidecar"):
    added = source + 1
    multiplied = added * 2
    return multiplied.contiguous().realize()


def resolve_call_to_exec_items(call: UOp, input_uops: tuple[UOp, ...]) -> list[Any]:
  resolved_linear = graph_rewrite(
    UOp(Ops.LINEAR, src=(call,)),
    pm_post_sched_cache,
    ctx=({}, input_uops),
    walk=True,
    name="metadata probe params to buffers",
  )
  return [item.lower() for item in linear_to_schedule(resolved_linear)]


def captured_metadata_probe(device: str, jit_mode: int) -> dict[str, Any]:
  with Context(JIT=jit_mode, TRACEMETA=2):
    runner = TinyJit(jitted_metadata_function)
    source = Tensor.empty(4, device=device).realize()
    for _ in range(3):
      runner(source).realize()
    captured = runner.captured

  if captured is None:
    return {"captured": False}

  input_uops, var_vals, names, expected_info = _prepare_jit_inputs((source,), {})
  ctx = ExecContext(var_vals, tuple(input_uops), do_update_stats=False, jit=True)
  del ctx, names, expected_info

  calls = []
  lowered_exec_item_metadata_count = 0
  lowered_exec_item_count = 0
  input_uops_tuple = tuple(input_uops)
  for ordinal, call in enumerate(captured.linear.src):
    call_entry = {
      "ordinal": ordinal,
      "ast_op": str(call.src[0].op),
      "call_metadata_count": len(call.arg.metadata),
      "call_metadata": metadata_payload(call.arg.metadata),
      "ast_toposort_metadata_count": metadata_count(call.src[0]),
      "ast_toposort_metadata_examples": metadata_examples(call.src[0]),
    }
    try:
      exec_items = resolve_call_to_exec_items(call, input_uops_tuple)
      lowered_exec_item_count += len(exec_items)
      item_metadata_counts = [len(item.metadata) for item in exec_items]
      lowered_exec_item_metadata_count += sum(item_metadata_counts)
      call_entry["lowered_exec_item_count"] = len(exec_items)
      call_entry["lowered_exec_item_metadata_counts"] = item_metadata_counts
    except Exception as exc:
      call_entry["lowering_error"] = f"{type(exc).__name__}: {exc}"
    calls.append(call_entry)

  return {
    "captured": True,
    "jit_mode": jit_mode,
    "captured_linear_call_count": len(captured.linear.src),
    "captured_call_metadata_count": sum(item["call_metadata_count"] for item in calls),
    "captured_ast_toposort_metadata_count": sum(item["ast_toposort_metadata_count"] for item in calls),
    "lowered_exec_item_count": lowered_exec_item_count,
    "lowered_exec_item_metadata_count": lowered_exec_item_metadata_count,
    "calls": calls,
  }


def install_add_linear_sidecar_patch():
  original_add_linear = TinyJitClass.add_linear

  def patch_linear_metadata(linear: UOp, category: str) -> UOp:
    metadata = (Metadata(name=category, caller=f"repo_sidecar:1::{category}"),)
    return linear.replace(src=tuple(
      call.replace(arg=replace(call.arg, metadata=call.arg.metadata or metadata))
      for call in linear.src
    ))

  def add_linear_with_sidecar(self, linear: UOp, var_vals: dict[str, int]):
    if _SIDECAR_STACK:
      linear = patch_linear_metadata(linear, _SIDECAR_STACK[-1])
    return original_add_linear(self, linear, var_vals)

  TinyJitClass.add_linear = add_linear_with_sidecar
  return original_add_linear


def captured_sidecar_probe(device: str, jit_mode: int) -> dict[str, Any]:
  original_add_linear = install_add_linear_sidecar_patch()
  try:
    with Context(JIT=jit_mode, TRACEMETA=2):
      runner = TinyJit(jitted_sidecar_metadata_function)
      source = Tensor.empty(4, device=device).realize()
      for _ in range(3):
        runner(source).realize()
      captured = runner.captured
  finally:
    TinyJitClass.add_linear = original_add_linear

  if captured is None:
    return {"captured": False}

  input_uops, var_vals, names, expected_info = _prepare_jit_inputs((source,), {})
  ctx = ExecContext(var_vals, tuple(input_uops), do_update_stats=False, jit=True)
  del ctx, names, expected_info

  calls = []
  lowered_exec_item_metadata_count = 0
  input_uops_tuple = tuple(input_uops)
  for ordinal, call in enumerate(captured.linear.src):
    exec_items = resolve_call_to_exec_items(call, input_uops_tuple)
    item_metadata_counts = [len(item.metadata) for item in exec_items]
    lowered_exec_item_metadata_count += sum(item_metadata_counts)
    calls.append({
      "ordinal": ordinal,
      "call_metadata_count": len(call.arg.metadata),
      "call_metadata": metadata_payload(call.arg.metadata),
      "lowered_exec_item_count": len(exec_items),
      "lowered_exec_item_metadata_counts": item_metadata_counts,
    })

  return {
    "captured": True,
    "jit_mode": jit_mode,
    "captured_linear_call_count": len(captured.linear.src),
    "captured_call_metadata_count": sum(item["call_metadata_count"] for item in calls),
    "lowered_exec_item_metadata_count": lowered_exec_item_metadata_count,
    "calls": calls,
  }


def classify_status(lazy_probe: dict[str, Any], captured_probes: list[dict[str, Any]]) -> str:
  lazy_has_metadata = lazy_probe["multiplied_toposort_metadata_count"] > 0
  captured_has_metadata = any(
    probe.get("captured_call_metadata_count", 0) > 0 or probe.get("captured_ast_toposort_metadata_count", 0) > 0
    for probe in captured_probes
  )
  lowered_has_metadata = any(probe.get("lowered_exec_item_metadata_count", 0) > 0 for probe in captured_probes)
  if lazy_has_metadata and not captured_has_metadata and not lowered_has_metadata:
    return "metadata_lost_before_captured_linear"
  if lazy_has_metadata and captured_has_metadata and not lowered_has_metadata:
    return "metadata_lost_during_schedule_lowering"
  if lazy_has_metadata and lowered_has_metadata:
    return "metadata_preserved_to_exec_items"
  return "metadata_not_created"


def main() -> None:
  parser = argparse.ArgumentParser(description="Diagnose tinygrad Tensor metadata preservation through TinyJit capture/lowering.")
  parser.add_argument("--device", default="METAL")
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = parser.parse_args()

  lazy_probe = lazy_metadata_probe(args.device)
  captured_probes = [captured_metadata_probe(args.device, jit_mode) for jit_mode in (1, 2)]
  sidecar_probes = [captured_sidecar_probe(args.device, jit_mode) for jit_mode in (1, 2)]
  payload = {
    "device": args.device,
    "lazy_probe": lazy_probe,
    "captured_probes": captured_probes,
    "sidecar_probes": sidecar_probes,
    "summary": {
      "status": classify_status(lazy_probe, captured_probes),
      "lazy_metadata_count": lazy_probe["multiplied_toposort_metadata_count"],
      "captured_call_metadata_count": sum(probe.get("captured_call_metadata_count", 0) for probe in captured_probes),
      "captured_ast_toposort_metadata_count": sum(probe.get("captured_ast_toposort_metadata_count", 0) for probe in captured_probes),
      "lowered_exec_item_metadata_count": sum(probe.get("lowered_exec_item_metadata_count", 0) for probe in captured_probes),
      "sidecar_captured_call_metadata_count": sum(probe.get("captured_call_metadata_count", 0) for probe in sidecar_probes),
      "sidecar_lowered_exec_item_metadata_count": sum(probe.get("lowered_exec_item_metadata_count", 0) for probe in sidecar_probes),
    },
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(args.out), **payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
  main()
