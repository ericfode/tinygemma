from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_PHASES = ("kv_head_reshape", "store")


def phase_name_from_category(category: str) -> str:
  marker = "__phase_"
  return category.split(marker, 1)[1] if marker in category else category


def ranked_counts(counts: dict[str, int] | None, *, key_name: str, total: int, top_n: int) -> list[dict[str, Any]]:
  if not counts:
    return []
  ranked = sorted(((str(name), int(count)) for name, count in counts.items()), key=lambda item: (-item[1], item[0]))[:top_n]
  return [{key_name: name, "count": count, "share": count / total if total else 0.0} for name, count in ranked]


def store_effect_note(source_count: int, store_effect_count: int) -> str:
  if source_count <= 0 or store_effect_count <= 0:
    return "no_store_effect_roots"
  if source_count == store_effect_count:
    return "all_sources_are_store_effect_roots"
  return "mixed_store_effect_roots"


def phase_structure_entries(profile: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
  phase_summary = profile.get("source_attributed_cache_write_phase_summary", {})
  structures = phase_summary.get("by_phase_structure", {})
  entries: dict[str, tuple[str, dict[str, Any]]] = {}
  for category, structural in structures.items():
    phase = phase_name_from_category(str(category))
    entries[phase] = (str(category), structural)
  return entries


def summarize_phase(profile: dict[str, Any], phase: str, *, top_n: int) -> dict[str, Any] | None:
  phase_summary = profile.get("source_attributed_cache_write_phase_summary", {})
  phase_rollup = phase_summary.get("by_phase", {}).get(phase, {})
  entries = phase_structure_entries(profile)
  if phase not in entries:
    return None
  category, structural = entries[phase]
  source_count = int(structural.get("source_count", phase_rollup.get("source_count", 0)) or 0)
  store_effect_count = int(structural.get("store_effect_count", 0) or 0)
  top_display_names = ranked_counts(structural.get("display_name_counts"), key_name="name", total=source_count, top_n=top_n)
  top_op_signatures = ranked_counts(structural.get("op_signature_counts"), key_name="signature", total=source_count, top_n=top_n)
  dominant_display_name = top_display_names[0]["name"] if top_display_names else None
  dominant_op_signature = top_op_signatures[0]["signature"] if top_op_signatures else None
  return {
    "phase": phase,
    "category": category,
    "source_count": source_count,
    "elapsed_ms": float(phase_rollup.get("elapsed_ms", 0.0) or 0.0),
    "elapsed_share": float(phase_rollup.get("elapsed_share", 0.0) or 0.0),
    "store_effect_count": store_effect_count,
    "store_effect_share": float(structural.get("store_effect_share", store_effect_count / source_count if source_count else 0.0) or 0.0),
    "store_effect_note": store_effect_note(source_count, store_effect_count),
    "dominant_display_name": dominant_display_name,
    "dominant_op_signature": dominant_op_signature,
    "top_display_names": top_display_names,
    "top_op_signatures": top_op_signatures,
    "category_basis_counts": dict(structural.get("category_basis_counts", {})),
    "ast_root_counts": dict(structural.get("ast_root_counts", {})),
    "effect_kind_counts": dict(structural.get("effect_kind_counts", {})),
  }


def analyze_profile(profile_path: Path | str, *, phases: list[str] | tuple[str, ...] | None = None, top_n: int = 8) -> dict[str, Any]:
  path = Path(profile_path)
  profile = json.loads(path.read_text())
  entries = phase_structure_entries(profile)
  selected_phases = list(phases) if phases else sorted(entries, key=lambda phase: -int(entries[phase][1].get("source_count", 0) or 0))
  phase_rows = [row for phase in selected_phases if (row := summarize_phase(profile, phase, top_n=top_n)) is not None]
  total_selected_sources = sum(int(row["source_count"]) for row in phase_rows)
  total_selected_store_effects = sum(int(row["store_effect_count"]) for row in phase_rows)
  return {
    "profile": str(path),
    "summary": {
      "phase_count": len(phase_rows),
      "total_selected_sources": total_selected_sources,
      "total_selected_store_effects": total_selected_store_effects,
      "all_selected_sources_store_effect_roots": bool(phase_rows) and total_selected_sources == total_selected_store_effects,
    },
    "phases": phase_rows,
  }


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Summarize cache-write phase structural motifs from a profile_decode_jit JSON artifact.")
  parser.add_argument("--profile", type=Path, required=True, help="Path to profile_decode_jit JSON output.")
  parser.add_argument("--phase", action="append", dest="phases", help="Phase to include. May be repeated. Defaults to kv_head_reshape and store.")
  parser.add_argument("--top-n", type=int, default=8, help="Number of display/op motifs to keep per phase.")
  parser.add_argument("--out", type=Path, help="Optional path for JSON output. Defaults to stdout.")
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  payload = analyze_profile(args.profile, phases=args.phases or list(DEFAULT_PHASES), top_n=args.top_n)
  text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
  else:
    sys.stdout.write(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
