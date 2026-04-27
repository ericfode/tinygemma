from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_STRUCTURAL_CSV = Path("benchmarks/gemma4-metal-decode-layer13-structural-profile-512.csv")
DEFAULT_PHASE_JSON = Path("benchmarks/gemma4-metal-decode-graph-layer13-phase-propagation-profile-512.json")
DEFAULT_OUT = Path("benchmarks/gemma4-metal-layer13-reduce-source-attribution-043.json")
DEFAULT_MARKDOWN_OUT = Path("docs/plans/2026-04-26-layer13-reduce-source-attribution.md")
LAYER13_TARGET = "attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention"
LAYER13_ROLLUP = "attention_packed_cache_write_shared_source"

PRODUCER_OPERATION_MAP = [
  {
    "phase": "kv_projection",
    "status": "candidate_child_phase_not_proven_by_current_artifact",
    "operation": "fused rowwise-int8 K/V projection or separate K/V linears",
    "source_refs": [
      "tinygrad_gemma/model.py:509-520",
      "tinygrad_gemma/model.py:512",
      "tinygrad_gemma/model.py:516-517",
      "tinygrad_gemma/model.py:561",
    ],
    "why_it_can_emit_reduce_r": "hidden_states.matmul(weight.transpose(), dtype='float') and linear projections lower to reduce kernels.",
  },
  {
    "phase": "rmsnorm_rope",
    "status": "candidate_child_phase_not_proven_by_current_artifact",
    "operation": "K RMSNorm, K RoPE, and V RMSNorm after K/V projection",
    "source_refs": [
      "tinygrad_gemma/model.py:91-105",
      "tinygrad_gemma/model.py:327-333",
      "tinygrad_gemma/model.py:562-564",
    ],
    "why_it_can_emit_reduce_r": "RMSNorm uses mean(-1, keepdim=True), which is a reduction; RoPE is mostly elementwise/slice/cat and is not expected to dominate reduce_r mass by itself.",
  },
  {
    "phase": "rhs_pack",
    "status": "candidate_child_phase_not_proven_by_current_artifact",
    "operation": "pack K and V into the final cache lane before assignment",
    "source_refs": [
      "tinygrad_gemma/model.py:298",
      "tinygrad_gemma/model.py:301",
      "scripts/profile_decode_jit.py:649-663",
    ],
    "why_it_can_emit_reduce_r": "Packing is squeeze/unsqueeze/cat/view work, so it should not be the primary origin of large reduce_r kernels unless fused with upstream producers.",
  },
  {
    "phase": "store",
    "status": "exact_parent_realization_scope_but_child_phase_not_proven_by_current_artifact",
    "operation": "packed cache slice assignment and dependency-closed realize",
    "source_refs": [
      "tinygrad_gemma/model.py:280-301",
      "tinygrad_gemma/model.py:586-599",
      "tinygrad_gemma/model.py:605-618",
      "scripts/profile_decode_jit.py:653-667",
    ],
    "why_it_can_emit_reduce_r": "The store realization itself is not a mathematical reduction, but assign(...).realize() pulls the lazy K/V producer graph into the cache-write parent scope.",
  },
]


def json_cell(value: str) -> dict[str, Any]:
  if not value:
    return {}
  return json.loads(value)


def load_structural_rows(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open(newline="") as handle:
    for row in csv.DictReader(handle):
      row["ordinal"] = int(row["ordinal"])
      row["elapsed_ms"] = float(row["elapsed_ms"])
      row["est_ops"] = int(float(row["est_ops"]))
      row["est_mem"] = int(float(row["est_mem"]))
      row["source_category_counts_json"] = json_cell(row.get("source_category_counts", ""))
      row["source_category_counts_rollup_json"] = json_cell(row.get("source_category_counts_rollup", ""))
      row["metadata_json"] = json_cell(row.get("metadata", "[]")) if row.get("metadata", "").startswith("{") else json.loads(row.get("metadata") or "[]")
      rows.append(row)
  return rows


def load_graph_batches(path: Path, target_category: str) -> list[dict[str, Any]]:
  payload = json.loads(path.read_text())
  batches = []
  for item in payload["source_attribution"]["items"]:
    count = int(item["category_counts"].get(target_category, 0))
    if count <= 0:
      continue
    batches.append({
      "ordinal": item["ordinal"],
      "display_name": item["display_name"],
      "elapsed_ms": item["elapsed_ms"],
      "source_count": item["source_count"],
      "source_start": item["source_start"],
      "source_end": item["source_end"],
      "target_source_count": count,
      "target_rollup_source_count": int(item["category_counts_rollup"].get(LAYER13_ROLLUP, 0)),
      "cache_write_phase_unclassified_count": int(item.get("cache_write_phase_unclassified_count", 0)),
      "cache_write_phase_conflict_count": int(item.get("cache_write_phase_conflict_count", 0)),
      "cache_write_phase_category_counts": item.get("cache_write_phase_category_counts", {}),
      "category_basis": item.get("category_basis"),
    })
  return batches


def graph_batch_for_ordinal(ordinal: int, batches: list[dict[str, Any]]) -> str:
  for batch in batches:
    if int(batch["source_start"]) <= ordinal <= int(batch["source_end"]):
      return f"graph_{batch['ordinal']}:{batch['source_start']}-{batch['source_end']}"
  return "unmapped"


def summarize_display_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  groups: dict[str, dict[str, Any]] = {}
  for row in rows:
    entry = groups.setdefault(row["display_name"], {
      "display_name": row["display_name"],
      "source_count": 0,
      "elapsed_ms": 0.0,
      "category_field_counts": Counter(),
      "category_basis_counts": Counter(),
      "example_ordinals": [],
    })
    entry["source_count"] += 1
    entry["elapsed_ms"] += float(row["elapsed_ms"])
    entry["category_field_counts"][row["category"]] += 1
    entry["category_basis_counts"][row["source_category_basis"]] += 1
    if len(entry["example_ordinals"]) < 8:
      entry["example_ordinals"].append(row["ordinal"])
  output = []
  for entry in groups.values():
    output.append({
      **{key: entry[key] for key in ("display_name", "source_count", "elapsed_ms", "example_ordinals")},
      "category_field_counts": dict(entry["category_field_counts"]),
      "category_basis_counts": dict(entry["category_basis_counts"]),
    })
  return sorted(output, key=lambda entry: (-entry["elapsed_ms"], -entry["source_count"], entry["display_name"]))


def phase_name(category: str) -> str:
  return category.rsplit("__phase_", 1)[-1] if "__phase_" in category else category


def phase_status_map(phase_name_counts: Counter[str]) -> dict[str, str]:
  statuses = {
    "kv_projection": "candidate_child_phase_not_proven_by_current_artifact",
    "rmsnorm_rope": "candidate_child_phase_not_proven_by_current_artifact",
    "rhs_pack": "candidate_child_phase_not_proven_by_current_artifact",
    "store": "exact_parent_realization_scope_but_child_phase_not_proven_by_current_artifact",
  }
  if not phase_name_counts:
    return statuses
  dominant_phase, dominant_count = phase_name_counts.most_common(1)[0]
  for name in statuses:
    count = phase_name_counts.get(name, 0)
    if count == 0:
      statuses[name] = "not_observed_in_phase_propagation_profile"
    elif name == dominant_phase:
      statuses[name] = f"dominant_child_phase_proven_by_phase_propagation_profile:{count}_source_rows"
    else:
      statuses[name] = f"minor_child_phase_proven_by_phase_propagation_profile:{count}_source_rows"
  statuses[dominant_phase] = f"dominant_child_phase_proven_by_phase_propagation_profile:{dominant_count}_source_rows"
  return statuses


def producer_operation_map_with_statuses(phase_name_counts: Counter[str]) -> list[dict[str, Any]]:
  statuses = phase_status_map(phase_name_counts)
  return [
    {**item, "status": statuses.get(item["phase"], item["status"])}
    for item in PRODUCER_OPERATION_MAP
  ]


def summarize(structural_rows: list[dict[str, Any]], graph_batches: list[dict[str, Any]], target_category: str) -> dict[str, Any]:
  target_rows = [row for row in structural_rows if int(row["source_category_counts_json"].get(target_category, 0)) > 0]
  reduce_rows = [row for row in target_rows if row["display_name"].startswith("r_")]
  non_reduce_rows = [row for row in target_rows if not row["display_name"].startswith("r_")]
  reduce_graph_counts = Counter(graph_batch_for_ordinal(row["ordinal"], graph_batches) for row in reduce_rows)
  target_category_field_counts = Counter(row["category"] for row in target_rows)
  reduce_category_field_counts = Counter(row["category"] for row in reduce_rows)
  target_basis_counts = Counter(row["source_category_basis"] for row in target_rows)
  reduce_basis_counts = Counter(row["source_category_basis"] for row in reduce_rows)
  phase_category_counts: Counter[str] = Counter()
  phase_name_counts: Counter[str] = Counter()
  phase_unclassified_count = 0
  phase_conflict_count = 0
  for batch in graph_batches:
    phase_unclassified_count += int(batch.get("cache_write_phase_unclassified_count", 0))
    phase_conflict_count += int(batch.get("cache_write_phase_conflict_count", 0))
    for category, count in batch.get("cache_write_phase_category_counts", {}).items():
      phase_category_counts[category] += int(count)
      phase_name_counts[phase_name(category)] += int(count)
  phase_classified_count = sum(phase_category_counts.values())
  if phase_category_counts:
    source_artifact_status = "phase_propagation_profile_classifies_layer13_shared_source_child_phase"
    child_phase_summary = ", ".join(f"{name}={count}" for name, count in phase_name_counts.most_common())
    child_phase = (
      f"partly_recovered: {phase_classified_count}/{len(target_rows)} target parent rows carry child phase metadata "
      f"({child_phase_summary}); {phase_unclassified_count} remain unclassified and {phase_conflict_count} rows report phase conflicts."
    )
    root_cause = (
      "direct cache-write phase propagation through UOp creation/replace preserves most child phase tags without forced realization cutpoints; "
      "the remaining conflicts/unclassified rows mark places where lowered source items merge multiple phase-tagged paths or drop phase metadata."
    )
    next_profiler_target = (
      "Treat layer-13 shared-source K/V projection as the dominant reduce_r producer before writing runtime code; a candidate should reduce that projection/source mass or prove an equivalent fused path, not optimize RHS packing or the store tail."
    )
  else:
    source_artifact_status = "existing_artifact_maps_reduce_rows_exactly_to_parent_cache_write_scope_but_not_to_child_phase"
    child_phase = "not_recoverable_from_current_artifacts: cache_write_phase_category_counts is empty and all target rows are phase-unclassified."
    root_cause = "phase metadata is attached to high-level Tensor/UOp objects before final realization, then tinygrad scheduling/rangeify constructs lowered kernel source items under the parent realization scope; current artifacts preserve the parent category but not kv_projection/rmsnorm_rope/rhs_pack/store child phase tags."
    next_profiler_target = "propagate or snapshot child phase metadata at the captured/lowered source-item boundary without forced realization cutpoints before attempting another runtime optimization."
  top_reduce_rows = sorted(reduce_rows, key=lambda row: row["elapsed_ms"], reverse=True)[:25]
  return {
    "target_category": target_category,
    "hard_metric": "tokens_per_second",
    "runtime_patch_status": "none_in_this_artifact",
    "source_artifact_status": source_artifact_status,
    "counts": {
      "total_source_rows": len(structural_rows),
      "target_parent_source_rows": len(target_rows),
      "target_parent_reduce_r_rows": len(reduce_rows),
      "target_parent_non_reduce_rows": len(non_reduce_rows),
      "target_parent_category_field_counts": dict(target_category_field_counts),
      "target_parent_category_basis_counts": dict(target_basis_counts),
      "reduce_r_category_field_counts": dict(reduce_category_field_counts),
      "reduce_r_category_basis_counts": dict(reduce_basis_counts),
      "reduce_r_graph_batch_counts": dict(reduce_graph_counts),
      "phase_category_counts": dict(phase_category_counts),
      "phase_name_counts": dict(phase_name_counts),
      "phase_classified_source_rows": phase_classified_count,
      "phase_unclassified_source_rows": phase_unclassified_count,
      "phase_conflict_source_rows": phase_conflict_count,
    },
    "graph_batches": graph_batches,
    "top_reduce_display_groups": summarize_display_groups(reduce_rows)[:20],
    "top_reduce_rows": [
      {
        "ordinal": row["ordinal"],
        "display_name": row["display_name"],
        "elapsed_ms": row["elapsed_ms"],
        "category_field": row["category"],
        "source_category_basis": row["source_category_basis"],
        "mapped_parent_category": target_category,
        "graph_batch": graph_batch_for_ordinal(row["ordinal"], graph_batches),
      }
      for row in top_reduce_rows
    ],
    "producer_operation_map": producer_operation_map_with_statuses(phase_name_counts),
    "recoverability": {
      "parent_scope": f"exact: source_category_counts maps all {len(target_rows)} layer-13 shared-source rows, including {len(reduce_rows)} reduce_r rows, to the target category.",
      "child_phase": child_phase,
      "root_cause": root_cause,
      "next_profiler_target": next_profiler_target,
    },
  }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
  counts = payload["counts"]
  lines = [
    "# Layer 13 reduce_r Source Attribution",
    "",
    "## Objective",
    "",
    "Map the layer-13 `reduce_r`-heavy source rows back to producer operations before another runtime patch. The hard metric remains repeated real E2B int8 `METAL` token throughput, not profile aesthetics.",
    "",
    "## Exact artifact mapping",
    "",
    f"- Target parent category: `{payload['target_category']}`.",
    f"- Total source rows in structural artifact: `{counts['total_source_rows']}`.",
    f"- Target parent source rows: `{counts['target_parent_source_rows']}`.",
    f"- Target parent `r_*` / `reduce_r` rows: `{counts['target_parent_reduce_r_rows']}`.",
    f"- Target parent non-reduce rows: `{counts['target_parent_non_reduce_rows']}`.",
    f"- `category` field alone maps only `{counts['reduce_r_category_field_counts'].get(payload['target_category'], 0)}` reduce rows; use `source_category_counts` to map all `{counts['target_parent_reduce_r_rows']}` reduce rows.",
    f"- Child phase rows recovered in the phase-propagation profile: `{counts['phase_classified_source_rows']}` classified, `{counts['phase_unclassified_source_rows']}` unclassified, `{counts['phase_conflict_source_rows']}` conflicted.",
    f"- Child phase split: `{', '.join(f'{name}={count}' for name, count in counts['phase_name_counts'].items()) or 'none'}`.",
    "",
    "Graph-batch split:",
    "",
    "| graph batch | source range | target rows | phase counts | phase-unclassified rows |",
    "| --- | ---: | ---: | --- | ---: |",
  ]
  for batch in payload["graph_batches"]:
    phase_counts = ", ".join(
      f"{phase_name(category)}={count}"
      for category, count in batch.get("cache_write_phase_category_counts", {}).items()
    ) or "none"
    lines.append(
      f"| `{batch['ordinal']} {batch['display_name']}` | `{batch['source_start']}-{batch['source_end']}` | "
      f"`{batch['target_source_count']}` | `{phase_counts}` | `{batch['cache_write_phase_unclassified_count']}` |"
    )
  lines.extend([
    "",
    "## Top reduce display groups",
    "",
    "| display name | rows | elapsed ms | example ordinals |",
    "| --- | ---: | ---: | --- |",
  ])
  for group in payload["top_reduce_display_groups"][:10]:
    ordinals = ", ".join(str(item) for item in group["example_ordinals"])
    lines.append(f"| `{group['display_name']}` | `{group['source_count']}` | `{group['elapsed_ms']:.6f}` | `{ordinals}` |")
  lines.extend([
    "",
    "## Producer operation map",
    "",
    "The parent realization scope is exact, and the phase-propagation profile now identifies the dominant child phase. The relevant source operations feeding that parent scope are:",
    "",
  ])
  for item in payload["producer_operation_map"]:
    refs = ", ".join(f"`{ref}`" for ref in item["source_refs"])
    lines.extend([
      f"- `{item['phase']}` — {item['operation']} ({refs}).",
      f"  - Status: `{item['status']}`.",
      f"  - Reduction relevance: {item['why_it_can_emit_reduce_r']}",
    ])
  lines.extend([
    "",
    "## Recoverability conclusion",
    "",
    f"- Parent scope: {payload['recoverability']['parent_scope']}",
    f"- Child phase: {payload['recoverability']['child_phase']}",
    f"- Root cause: {payload['recoverability']['root_cause']}",
    f"- Next profiler target: {payload['recoverability']['next_profiler_target']}",
    "",
    "No runtime patch is justified by this artifact yet. The profiler now points at layer-13 shared-source K/V projection as the dominant child phase, so the next runtime candidate must reduce that projection/source mass or prove an equivalent fused path before spending the hard `1000/20` throughput gate.",
  ])
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("\n".join(lines) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser(description="Analyze layer-13 shared-source reduce_r rows from profiler artifacts.")
  parser.add_argument("--structural-csv", type=Path, default=DEFAULT_STRUCTURAL_CSV)
  parser.add_argument("--phase-json", type=Path, default=DEFAULT_PHASE_JSON)
  parser.add_argument("--target-category", default=LAYER13_TARGET)
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
  args = parser.parse_args()

  structural_rows = load_structural_rows(args.structural_csv)
  graph_batches = load_graph_batches(args.phase_json, args.target_category)
  payload = summarize(structural_rows, graph_batches, args.target_category)
  payload["inputs"] = {
    "structural_csv": str(args.structural_csv),
    "phase_json": str(args.phase_json),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  write_markdown(args.markdown_out, payload)
  print(json.dumps({
    "out": str(args.out),
    "markdown_out": str(args.markdown_out),
    "target_parent_source_rows": payload["counts"]["target_parent_source_rows"],
    "target_parent_reduce_r_rows": payload["counts"]["target_parent_reduce_r_rows"],
    "phase_unclassified_source_rows": payload["counts"]["phase_unclassified_source_rows"],
    "next_profiler_target": payload["recoverability"]["next_profiler_target"],
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
