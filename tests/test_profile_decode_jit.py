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
  def __init__(self, display_name="fake_program"):
    self.display_name = display_name


class FakeGraphProgram:
  def __init__(self, source_items: list["FakeItem"]):
    self.jit_cache = source_items


class FakeAst:
  def __init__(self, *ops):
    self.uops = [op if isinstance(op, FakeUOp) else FakeUOp(op) for op in ops]
    self.op = self.uops[0].op if self.uops else ""

  def toposort(self):
    return self.uops


class FakeItem:
  def __init__(self, category: str, prg=None, ast=None):
    self.category = category
    self.prg = FakeProgram() if prg is None else prg
    self.metadata = []
    self.ast = ast


class FakeMetadata:
  def __init__(self, name: str, caller: str):
    self.name = name
    self.caller = caller


class FakeUOp:
  def __init__(self, op, metadata=None):
    self.op = op
    self.metadata = metadata or []


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
    FakeMetadata("attention_value_cache_write_local", "repo_sidecar:1::attention_value_cache_write_local"),
    FakeMetadata("attention_key_cache_write_shared_source", "repo_sidecar:1::attention_key_cache_write_shared_source"),
  ]

  assert profile.classify_kernel(metadata) == "attention_key_cache_write_shared_source"


def test_profile_uop_creation_sidecar_patch_restores_and_stamps_cached_uops():
  original = profile.UOpMetaClass.__call__
  marker = "PROFILE_TEST_UOP_SIDECAR_REUSE"
  base = profile.UOp(profile.Ops.DEVICE, arg=marker)

  with profile.uop_creation_sidecar_patch():
    with profile.sidecar_scope("logits_argmax"):
      reused = profile.UOp(profile.Ops.DEVICE, arg=marker)

  assert reused is base
  assert profile.UOpMetaClass.__call__ is original
  assert profile.sidecar_metadata_category(reused.metadata or ()) == "logits_argmax"


def test_profile_uop_replace_sidecar_patch_preserves_metadata_across_replace():
  original = profile.UOp.replace
  source = profile.UOp(profile.Ops.CONST, arg=1)
  profile.all_metadata[source] = profile.uop_sidecar_metadata("mlp")

  with profile.uop_replace_sidecar_patch():
    replaced = source.replace(arg=2)

  assert profile.UOp.replace is original
  assert replaced is not source
  assert profile.sidecar_metadata_category(replaced.metadata or ()) == "mlp"


def test_profile_classifies_source_item_from_ast_uop_sidecar_metadata():
  ast = FakeAst(
    FakeUOp("Ops.SINK", [FakeMetadata("logits_argmax", "repo_uop_sidecar:1::logits_argmax")]),
    FakeUOp(profile.Ops.CONST, [FakeMetadata("attention", "repo_uop_sidecar:1::attention")]),
  )
  item = FakeItem("other", ast=ast)

  assert profile.source_category(item) == "logits_argmax"
  assert profile.source_item_category_basis(item) == profile.UOP_SIDECAR_CATEGORY_BASIS
  assert profile.ast_uop_sidecar_category_counts(ast) == {"logits_argmax": 1}


def test_profile_batch_category_basis_preserves_uop_sidecar_metadata():
  counts = {profile.UOP_SIDECAR_CATEGORY_BASIS: 1, "unclassified_source_item_metadata": 2}

  assert profile.category_basis(counts, 3) == profile.UOP_SIDECAR_CATEGORY_BASIS


def test_profile_summarizes_source_attribution_by_best_available_category():
  summary = profile.summarize_source_attribution({
    "items": [
      {
        "elapsed_ms": 10.0,
        "source_count": 2,
        "category_basis": "repo_sidecar_uop_creation_metadata",
        "category_counts": {"mlp": 1, "attention_key_cache_write_shared_source": 1},
      },
      {
        "elapsed_ms": 5.0,
        "source_count": 1,
        "category_basis": "repo_sidecar_realize_scope_metadata",
        "category_counts": {"attention_value_cache_write_local": 1},
      },
    ]
  })

  assert summary["source_count"] == 3
  assert summary["elapsed_ms"] == 15.0
  assert summary["by_category"]["mlp"]["elapsed_ms"] == 5.0
  assert summary["by_category"]["attention_key_cache_write_shared_source"]["elapsed_ms"] == 5.0
  assert summary["by_category_basis"]["repo_sidecar_uop_creation_metadata"]["source_count"] == 2


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


def test_profile_classifies_cache_write_role():
  assert profile.cache_write_role(False, False) == "local"
  assert profile.cache_write_role(False, True) == "shared_source"
  assert profile.cache_write_role(True, False) == "shared_consumer"
  assert profile.cache_write_role(True, True) == "shared_consumer"


def test_profile_cache_update_sidecar_patch_splits_key_and_value_scopes():
  original = profile.gemma_model.realize_cache_update
  calls = []
  key_cache = FakeTensor()
  value_cache = FakeTensor()
  key_cache.calls = calls
  value_cache.calls = calls

  try:
    with profile.cache_update_sidecar_patch():
      profile.gemma_model.realize_cache_update(
        key_cache,
        value_cache,
        "key",
        "value",
        0,
        1,
        is_kv_shared_layer=False,
        store_full_length_kv=True,
      )
  finally:
    profile.gemma_model.realize_cache_update = original

  assert calls == ["attention_key_cache_write_shared_source", "attention_value_cache_write_shared_source"]


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
    [FakeItem("graph_batch", FakeGraphProgram([
      FakeItem("other", FakeProgram("copy"), FakeAst("Ops.COPY", "Ops.LOAD")),
      FakeItem("other", FakeProgram("sink"), FakeAst("Ops.SINK", "Ops.LOAD")),
    ]))],
    [{"ordinal": 0, "program_type": "MetalGraph", "display_name": "<batched 2>", "elapsed_ms": 1.0}],
  )

  assert attribution["status"] == "complete"
  assert attribution["items"][0]["category_counts"] == {"other": 2}
  assert attribution["items"][0]["category_basis"] == "unclassified_source_item_metadata"
  assert attribution["items"][0]["unclassified_source_summary"]["count"] == 2
  assert attribution["items"][0]["unclassified_source_summary"]["display_name_counts"] == {"copy": 1, "sink": 1}
  assert attribution["items"][0]["unclassified_source_summary"]["ast_root_counts"] == {"Ops.COPY": 1, "Ops.SINK": 1}
  assert attribution["items"][0]["unclassified_source_summary"]["op_signature_counts"] == {
    "Ops.COPY:1,Ops.LOAD:1": 1,
    "Ops.LOAD:1,Ops.SINK:1": 1,
  }
  assert attribution["category_basis_counts"] == {"unclassified_source_item_metadata": 1}
