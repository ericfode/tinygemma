from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_profile_phase_structure.py"


def load_analysis_module():
  spec = importlib.util.spec_from_file_location("analyze_profile_phase_structure", SCRIPT)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_analyze_profile_phase_structure_summarizes_dominant_motifs(tmp_path: Path):
  analysis = load_analysis_module()
  profile = {
    "source_attributed_cache_write_phase_summary": {
      "by_phase": {
        "kv_head_reshape": {"source_count": 10, "elapsed_ms": 5.0, "elapsed_share": 0.5},
        "store": {"source_count": 4, "elapsed_ms": 2.0, "elapsed_share": 0.2},
        "rhs_pack": {"source_count": 1, "elapsed_ms": 0.1, "elapsed_share": 0.01},
      },
      "by_phase_structure": {
        "attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention__phase_kv_head_reshape": {
          "source_count": 10,
          "store_effect_count": 10,
          "store_effect_share": 1.0,
          "display_name_counts": {"r_16_96": 7, "E_16_2_16_4": 3},
          "op_signature_counts": {"Ops.REDUCE:1": 7, "Ops.EXP2:1": 3},
          "category_basis_counts": {"repo_sidecar_uop_creation_metadata": 10},
        },
        "attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention__phase_store": {
          "source_count": 4,
          "store_effect_count": 4,
          "store_effect_share": 1.0,
          "display_name_counts": {"E_16_32_3": 3, "r_128_32_3_384_4": 1},
          "op_signature_counts": {"Ops.INDEX:5": 3, "Ops.REDUCE:1": 1},
          "category_basis_counts": {"repo_sidecar_uop_creation_metadata": 3, "repo_sidecar_realize_scope_metadata": 1},
        },
      },
    }
  }
  profile_path = tmp_path / "profile.json"
  profile_path.write_text(json.dumps(profile))

  result = analysis.analyze_profile(profile_path, phases=["kv_head_reshape", "store"], top_n=2)

  assert [phase["phase"] for phase in result["phases"]] == ["kv_head_reshape", "store"]
  reshape = result["phases"][0]
  assert reshape["source_count"] == 10
  assert reshape["elapsed_ms"] == 5.0
  assert reshape["dominant_display_name"] == "r_16_96"
  assert reshape["top_display_names"][0] == {"name": "r_16_96", "count": 7, "share": 0.7}
  assert reshape["top_op_signatures"][0] == {"signature": "Ops.REDUCE:1", "count": 7, "share": 0.7}
  assert reshape["store_effect_note"] == "all_sources_are_store_effect_roots"
  store = result["phases"][1]
  assert store["dominant_display_name"] == "E_16_32_3"
  assert result["summary"]["total_selected_sources"] == 14
  assert result["summary"]["all_selected_sources_store_effect_roots"] is True


def test_analyze_profile_phase_structure_cli_writes_json(tmp_path: Path):
  analysis = load_analysis_module()
  profile = {
    "source_attributed_cache_write_phase_summary": {
      "by_phase": {"store": {"source_count": 2, "elapsed_ms": 1.0, "elapsed_share": 0.25}},
      "by_phase_structure": {
        "category__phase_store": {
          "source_count": 2,
          "store_effect_count": 1,
          "store_effect_share": 0.5,
          "display_name_counts": {"E_store": 2},
          "op_signature_counts": {"Ops.STORE:1": 1},
          "category_basis_counts": {"basis": 2},
        }
      },
    }
  }
  profile_path = tmp_path / "profile.json"
  out_path = tmp_path / "analysis.json"
  profile_path.write_text(json.dumps(profile))

  rc = analysis.main(["--profile", str(profile_path), "--phase", "store", "--out", str(out_path)])

  assert rc == 0
  payload = json.loads(out_path.read_text())
  assert payload["phases"][0]["phase"] == "store"
  assert payload["phases"][0]["store_effect_note"] == "mixed_store_effect_roots"
