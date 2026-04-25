from __future__ import annotations

import importlib.util
from pathlib import Path


def load_profile_module():
  module_path = Path(__file__).resolve().parents[1] / "scripts" / "profile_decode_jit.py"
  spec = importlib.util.spec_from_file_location("profile_decode_jit", module_path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


profile = load_profile_module()


class FakeProgram:
  pass


class FakeGraphProgram:
  def __init__(self, source_items: list["FakeItem"]):
    self.jit_cache = source_items


class FakeItem:
  def __init__(self, category: str, prg=None):
    self.category = category
    self.prg = FakeProgram() if prg is None else prg
    self.metadata = []


class FakeMetadata:
  def __init__(self, name: str, caller: str):
    self.name = name
    self.caller = caller


class FakeUOp:
  def __init__(self, op):
    self.op = op


class FakeTensor:
  def __init__(self, *ops):
    self._ops = ops
    self.uop = self

  def toposort(self):
    return [FakeUOp(op) for op in self._ops]

  def __getitem__(self, key):
    return self

  def assign(self, value):
    del value
    return self

  def realize(self):
    calls = getattr(self, "calls", None)
    if calls is not None:
      calls.append(profile._SIDECAR_STACK[-1])
    return self


def test_profile_raw_gate_up_mode_is_abandoned():
  try:
    profile.validate_metal_int8_gate_up_mode("raw")
  except SystemExit as exc:
    assert "abandoned" in str(exc)
  else:
    raise AssertionError("raw gate/up profiling mode should be rejected")


def test_profile_default_gate_up_mode_is_allowed():
  profile.validate_metal_int8_gate_up_mode("default")


def test_profile_classifies_repo_sidecar_metadata():
  metadata = [
    FakeMetadata("norm", "repo_sidecar:1::norm"),
    FakeMetadata("attention", "repo_sidecar:1::attention"),
    FakeMetadata("attention_cache_write", "repo_sidecar:1::attention_cache_write"),
    FakeMetadata("attention_value_cache_write", "repo_sidecar:1::attention_value_cache_write"),
    FakeMetadata("attention_key_cache_write", "repo_sidecar:1::attention_key_cache_write"),
  ]

  assert profile.classify_kernel(metadata) == "attention_key_cache_write"


def test_profile_method_sidecar_patch_restores_original_method():
  class Target:
    def call(self):
      assert profile._SIDECAR_STACK[-1] == "mlp"
      return "ok"

  original = Target.call
  with profile.method_sidecar_patch(Target, "call", "mlp"):
    assert Target().call() == "ok"

  assert Target.call is original
  assert profile._SIDECAR_STACK == []


def test_profile_refines_attention_store_realize_scope():
  store_tensor = FakeTensor(profile.Ops.STORE)
  compute_tensor = FakeTensor(profile.Ops.ADD)

  assert profile.attention_realize_sidecar_category([store_tensor]) is None
  with profile.sidecar_scope("attention"):
    assert profile.attention_realize_sidecar_category([compute_tensor]) is None
    assert profile.attention_realize_sidecar_category([store_tensor]) == "attention_cache_write"


def test_profile_cache_update_sidecar_patch_splits_key_and_value_scopes():
  original = profile.gemma_model.realize_cache_update
  calls = []
  key_cache = FakeTensor()
  value_cache = FakeTensor()
  key_cache.calls = calls
  value_cache.calls = calls

  try:
    with profile.cache_update_sidecar_patch():
      profile.gemma_model.realize_cache_update(key_cache, value_cache, "key", "value", 0, 1)
  finally:
    profile.gemma_model.realize_cache_update = original

  assert calls == ["attention_key_cache_write", "attention_value_cache_write"]


def test_graph_batch_attribution_maps_batched_display_to_source_ranges():
  first_batch = [
    FakeItem("attention"),
    FakeItem("mlp"),
    FakeItem("mlp"),
  ]
  second_batch = [
    FakeItem("norm"),
    FakeItem("norm"),
  ]
  execution_items = [
    FakeItem("graph_batch", FakeGraphProgram(first_batch)),
    FakeItem("graph_batch", FakeGraphProgram(second_batch)),
  ]
  rows = [
    {
      "ordinal": 0,
      "program_type": "MetalGraph",
      "display_name": "<batched 3>",
      "elapsed_ms": 4.0,
    },
    {
      "ordinal": 1,
      "program_type": "MetalGraph",
      "display_name": "<batched 2>",
      "elapsed_ms": 6.0,
    },
  ]

  attribution = profile.attribute_execution_source_ranges(execution_items, rows)

  assert attribution["status"] == "complete"
  assert attribution["attributed_source_count"] == 5
  assert attribution["items"][0]["source_start"] == 0
  assert attribution["items"][0]["source_end"] == 2
  assert attribution["items"][0]["category_counts"] == {"attention": 1, "mlp": 2}
  assert attribution["items"][0]["category_basis"] == "source_item_metadata"
  assert attribution["items"][1]["source_start"] == 3
  assert attribution["items"][1]["source_end"] == 4
  assert attribution["items"][1]["category_counts"] == {"norm": 2}
  assert attribution["items"][1]["category_basis"] == "source_item_metadata"
  assert attribution["category_basis_counts"] == {"source_item_metadata": 2}


def test_graph_batch_attribution_rejects_unparsed_graph_display():
  attribution = profile.attribute_execution_source_ranges(
    [FakeItem("graph_batch", FakeGraphProgram([FakeItem("attention")]))],
    [{"ordinal": 0, "program_type": "MetalGraph", "display_name": "<unknown>", "elapsed_ms": 1.0}],
  )

  assert attribution["status"] == "incomplete"
  assert attribution["unparsed_graph_batches"] == 1
  assert attribution["attributed_source_count"] == 1


def test_graph_batch_attribution_detects_batch_count_mismatch():
  attribution = profile.attribute_execution_source_ranges(
    [FakeItem("graph_batch", FakeGraphProgram([FakeItem("attention")]))],
    [{"ordinal": 0, "program_type": "MetalGraph", "display_name": "<batched 2>", "elapsed_ms": 1.0}],
  )

  assert attribution["status"] == "incomplete"
  assert attribution["source_count_mismatches"] == 1


def test_graph_batch_attribution_marks_unclassified_source_metadata():
  attribution = profile.attribute_execution_source_ranges(
    [FakeItem("graph_batch", FakeGraphProgram([FakeItem("other"), FakeItem("other")]))],
    [{"ordinal": 0, "program_type": "MetalGraph", "display_name": "<batched 2>", "elapsed_ms": 1.0}],
  )

  assert attribution["status"] == "complete"
  assert attribution["items"][0]["category_counts"] == {"other": 2}
  assert attribution["items"][0]["category_basis"] == "unclassified_source_item_metadata"
  assert attribution["category_basis_counts"] == {"unclassified_source_item_metadata": 1}
