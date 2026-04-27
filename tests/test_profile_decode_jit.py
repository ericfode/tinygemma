from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_profile_module():
  module_path = Path(__file__).resolve().parents[1] / "scripts" / "profile_decode_jit.py"
  spec = importlib.util.spec_from_file_location("profile_decode_jit", module_path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


profile = load_profile_module()


def test_profile_add_linear_sidecar_patch_noops_when_tinyjit_hook_is_absent(monkeypatch):
  monkeypatch.delattr(profile.TinyJitClass, "add_linear", raising=False)
  with profile.add_linear_sidecar_patch():
    assert not hasattr(profile.TinyJitClass, "add_linear")


def test_profile_add_linear_sidecar_patch_preserves_existing_tinyjit_hook(monkeypatch):
  calls = []

  def fake_add_linear(self, linear, var_vals):
    calls.append((self, linear, var_vals))
    return linear

  monkeypatch.setattr(profile.TinyJitClass, "add_linear", fake_add_linear, raising=False)
  with profile.add_linear_sidecar_patch():
    result = profile.TinyJitClass.add_linear("jit", "linear", {"x": 1})
  assert result == "linear"
  assert calls == [("jit", "linear", {"x": 1})]


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
  def __init__(self, op, metadata=None, src=()):
    self.op = op
    self.metadata = metadata or []
    self.src = tuple(src)


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
    events = getattr(self, "events", None)
    if events is not None:
      events.append(("assign", profile._SIDECAR_STACK[-1]))
    return self

  def squeeze(self, dim=None):
    del dim
    return self

  def unsqueeze(self, dim):
    del dim
    return self

  def cat(self, other, dim=0):
    del other, dim
    events = getattr(self, "events", None)
    if events is not None:
      events.append(("cat", profile._SIDECAR_STACK[-1]))
    return self

  def realize(self):
    calls = getattr(self, "calls", None)
    if calls is not None:
      calls.append(profile._SIDECAR_STACK[-1])
    events = getattr(self, "events", None)
    if events is not None:
      events.append(("realize", profile._SIDECAR_STACK[-1]))
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


def test_profile_cache_write_category_metadata_is_parseable_and_rolls_up():
  category = profile.cache_write_sidecar_category("packed", "local", 0, "sliding_attention")
  assert category == "attention_packed_cache_write__role_local__layer_00__type_sliding_attention"
  assert profile.cache_write_category_metadata(category) == {
    "kind": "packed",
    "layout": "packed",
    "role": "local",
    "layer_idx": 0,
    "layer": "00",
    "layer_type": "sliding_attention",
    "rollup": "attention_packed_cache_write_local",
  }
  assert profile.category_rollup(category) == "attention_packed_cache_write_local"
  assert profile.sidecar_metadata_category([FakeMetadata(category, f"repo_sidecar:1::{category}")]) == category


def test_profile_cache_write_category_sanitizes_unknown_layer_metadata():
  category = profile.cache_write_sidecar_category("key", "shared_consumer", None, "full-attention")
  assert category == "attention_key_cache_write__role_shared_consumer__layer_unknown__type_full_attention"
  assert profile.cache_write_category_metadata(category)["layout"] == "split"
  assert profile.category_rollup(category) == "attention_key_cache_write_shared_consumer"


def test_profile_cache_write_phase_category_metadata_is_parseable_without_changing_parent():
  parent = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention")
  phase = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention", phase="kv_projection")

  assert phase == f"{parent}__phase_kv_projection"
  assert profile.parent_cache_write_category(phase) == parent
  metadata = profile.cache_write_category_metadata(phase)
  assert metadata["phase"] == "kv_projection"
  assert metadata["parent"] == parent
  assert metadata["rollup"] == "attention_packed_cache_write_shared_source"
  assert profile.sidecar_metadata_category([FakeMetadata(phase, f"repo_uop_sidecar:1::{phase}")]) == parent
  assert profile.sidecar_metadata_category([FakeMetadata(phase, f"repo_uop_sidecar:1::{phase}")], prefer_cache_write_phase=True) == phase


def test_profile_cache_write_phase_category_rejects_unknown_phase():
  with pytest.raises(ValueError):
    profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention", phase="mystery")


def test_profile_cache_write_phase_target_is_layer13_shared_source_only():
  assert profile.cache_write_phase_target("packed", "shared_source", 13, "sliding_attention")
  assert not profile.cache_write_phase_target("packed", "local", 13, "sliding_attention")
  assert not profile.cache_write_phase_target("packed", "shared_source", 14, "full_attention")


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


def test_profile_uop_creation_sidecar_patch_inherits_only_cache_write_phase_metadata_from_sources():
  parent = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention")
  phase = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention", phase="kv_projection")
  source = profile.UOp(profile.Ops.CONST, arg="phase-source-creation")
  derived = None
  profile.all_metadata[source] = profile.merge_metadata(
    profile.uop_sidecar_metadata(phase),
    profile.uop_sidecar_metadata("mlp"),
  )

  try:
    with profile.uop_creation_sidecar_patch():
      with profile.sidecar_scope(parent):
        derived = profile.UOp(profile.Ops.ADD, src=(source, source))

    assert profile.sidecar_metadata_category(derived.metadata or ()) == parent
    assert profile.sidecar_metadata_category(derived.metadata or (), prefer_cache_write_phase=True) == phase
    assert all(str(getattr(item, "name", "")) != "mlp" for item in profile.all_metadata.get(derived, ()))
  finally:
    profile.all_metadata.pop(source, None)
    if derived is not None:
      profile.all_metadata.pop(derived, None)


def test_profile_uop_replace_sidecar_patch_inherits_cache_write_phase_metadata_from_new_sources():
  phase = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention", phase="store")
  base = profile.UOp(profile.Ops.ADD, arg="phase-replace-base")
  source = profile.UOp(profile.Ops.CONST, arg="phase-source-replace")
  replaced = None
  profile.all_metadata[source] = profile.uop_sidecar_metadata(phase)

  try:
    with profile.uop_replace_sidecar_patch():
      replaced = base.replace(src=(source,))

    assert profile.sidecar_metadata_category(replaced.metadata or (), prefer_cache_write_phase=True) == phase
  finally:
    profile.all_metadata.pop(source, None)
    if replaced is not None:
      profile.all_metadata.pop(replaced, None)


def test_profile_ast_cache_write_phase_counts_use_direct_toposort_metadata():
  phase = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention", phase="kv_projection")
  leaf = FakeUOp(
    profile.Ops.ADD,
    [FakeMetadata(phase, f"repo_uop_sidecar:1::{phase}")],
  )
  root = FakeUOp(profile.Ops.ADD, src=(leaf,))

  # The hot summary path intentionally does not recursively rescan source edges;
  # real UOp creation/replace propagation has already made phase metadata direct.
  assert profile.ast_cache_write_phase_category_counts(FakeAst(root)) == {}
  assert profile.ast_cache_write_phase_category(FakeAst(root)) == (None, False)

  ast = FakeAst(root, leaf)
  assert profile.ast_cache_write_phase_category_counts(ast) == {phase: 1}
  assert profile.ast_cache_write_phase_category(ast) == (phase, False)


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
  assert summary["by_category_rollup"]["attention_key_cache_write_shared_source"]["elapsed_ms"] == 5.0
  assert summary["by_category_basis"]["repo_sidecar_uop_creation_metadata"]["source_count"] == 2


def test_profile_summarizes_detailed_cache_categories_with_backward_compatible_rollup():
  detailed = profile.cache_write_sidecar_category("packed", "shared_source", 17, "sliding_attention")
  attribution = {
    "items": [
      {
        "elapsed_ms": 9.0,
        "source_count": 3,
        "category_basis": "repo_sidecar_uop_creation_metadata",
        "category_counts": {detailed: 2, "mlp": 1},
      }
    ]
  }

  summary = profile.summarize_source_attribution(attribution)

  assert summary["source_count"] == 3
  assert summary["by_category"][detailed]["source_count"] == 2
  assert summary["by_category"][detailed]["elapsed_ms"] == 6.0
  assert summary["by_category_rollup"]["attention_packed_cache_write_shared_source"]["source_count"] == 2
  assert summary["by_category_rollup"]["attention_packed_cache_write_shared_source"]["elapsed_ms"] == 6.0


def test_profile_source_slice_summary_keeps_detailed_counts_and_rollups():
  detailed = profile.cache_write_sidecar_category("value", "local", 3, "sliding_attention")

  summary = profile.source_slice_summary([FakeItem(detailed), FakeItem("mlp")])

  assert summary["category_counts"] == {detailed: 1, "mlp": 1}
  assert summary["category_counts_rollup"] == {"attention_value_cache_write_local": 1, "mlp": 1}


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
        layer_idx=0,
        layer_type="sliding_attention",
        is_kv_shared_layer=False,
        store_full_length_kv=True,
      )
  finally:
    profile.gemma_model.realize_cache_update = original

  assert calls == [
    "attention_key_cache_write__role_shared_source__layer_00__type_sliding_attention",
    "attention_value_cache_write__role_shared_source__layer_00__type_sliding_attention",
  ]


def test_profile_cache_update_sidecar_patch_tracks_packed_cache_scope():
  original = profile.gemma_model.realize_cache_update
  calls = []
  key_cache = FakeTensor()
  value_cache = FakeTensor()
  packed_cache = FakeTensor()
  packed_cache.calls = calls

  try:
    with profile.cache_update_sidecar_patch():
      profile.gemma_model.realize_cache_update(
        key_cache,
        value_cache,
        FakeTensor(),
        FakeTensor(),
        0,
        1,
        layer_idx=12,
        layer_type="full_attention",
        is_kv_shared_layer=False,
        store_full_length_kv=False,
        single_position=True,
        packed_cache=packed_cache,
      )
  finally:
    profile.gemma_model.realize_cache_update = original

  assert calls == ["attention_packed_cache_write__role_local__layer_12__type_full_attention"]


def test_profile_cache_update_sidecar_patch_splits_layer13_packed_cache_phases():
  original = profile.gemma_model.realize_cache_update
  calls = []
  events = []
  key_cache = FakeTensor()
  value_cache = FakeTensor()
  key = FakeTensor()
  value = FakeTensor()
  packed_cache = FakeTensor()
  key.events = events
  packed_cache.calls = calls
  packed_cache.events = events
  parent = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention")
  rhs_pack = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention", phase="rhs_pack")
  store = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention", phase="store")

  try:
    with profile.cache_update_sidecar_patch():
      profile.gemma_model.realize_cache_update(
        key_cache,
        value_cache,
        key,
        value,
        0,
        1,
        layer_idx=13,
        layer_type="sliding_attention",
        is_kv_shared_layer=False,
        store_full_length_kv=True,
        single_position=True,
        packed_cache=packed_cache,
      )
  finally:
    profile.gemma_model.realize_cache_update = original

  assert ("cat", rhs_pack) in events
  assert ("assign", store) in events
  assert calls == [parent]
  assert events[-1] == ("realize", parent)


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


def test_profile_source_slice_summary_adds_phase_counts_without_changing_category_counts():
  parent = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention")
  phase = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention", phase="rmsnorm_rope")
  ast_item = FakeItem(parent, ast=FakeAst(FakeUOp(profile.Ops.ADD, [FakeMetadata(phase, f"repo_uop_sidecar:1::{phase}")])))
  metadata_item = FakeItem(parent, ast=FakeAst(profile.Ops.ADD))
  metadata_item.metadata = [FakeMetadata(phase, f"repo_sidecar:1::{phase}")]
  non_target = FakeItem(profile.cache_write_sidecar_category("packed", "local", 12, "sliding_attention"))

  summary = profile.source_slice_summary([ast_item, metadata_item, non_target])

  assert summary["category_counts"] == {
    parent: 2,
    "attention_packed_cache_write__role_local__layer_12__type_sliding_attention": 1,
  }
  assert summary["cache_write_phase_category_counts"] == {phase: 2}
  assert summary["cache_write_phase_unclassified_count"] == 0


def test_profile_summarizes_cache_write_phase_attribution_separately():
  parent = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention")
  kv = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention", phase="kv_projection")
  store = profile.cache_write_sidecar_category("packed", "shared_source", 13, "sliding_attention", phase="store")
  attribution = {
    "items": [{
      "elapsed_ms": 10.0,
      "source_count": 4,
      "category_counts": {parent: 4},
      "category_basis": "repo_sidecar_uop_creation_metadata",
      "cache_write_phase_category_counts": {kv: 3, store: 1},
      "cache_write_phase_conflict_count": 1,
      "cache_write_phase_unclassified_count": 0,
    }]
  }

  summary = profile.summarize_cache_write_phase_attribution(attribution)

  assert summary["source_count"] == 4
  assert summary["conflict_source_count"] == 1
  assert summary["by_category"][kv]["elapsed_ms"] == 7.5
  assert summary["by_category"][store]["elapsed_ms"] == 2.5
  assert summary["by_parent"][parent]["elapsed_ms"] == 10.0
  assert summary["by_phase"]["kv_projection"]["source_count"] == 3
