#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TEXT_REFERENCE_SUFFIXES = {".md", ".json", ".toml", ".txt", ".rst"}


def git_lines(args: list[str], *, cwd: Path) -> list[str]:
  proc = subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  return proc.stdout.splitlines()


def git_untracked_paths(*, cwd: Path) -> list[str]:
  proc = subprocess.run(
    ["git", "status", "--porcelain", "--untracked-files=all", "-z"],
    cwd=cwd,
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  paths: list[str] = []
  for raw_entry in proc.stdout.split(b"\0"):
    if not raw_entry:
      continue
    entry = raw_entry.decode("utf-8", "replace")
    if entry.startswith("?? "):
      paths.append(entry[3:])
  return sorted(paths)


def tracked_reference_texts(*, cwd: Path) -> list[tuple[str, str]]:
  texts: list[tuple[str, str]] = []
  for rel in git_lines(["ls-files"], cwd=cwd):
    suffix = Path(rel).suffix
    if suffix not in TEXT_REFERENCE_SUFFIXES:
      continue
    try:
      texts.append((rel, (cwd / rel).read_text(errors="ignore")))
    except OSError:
      continue
  return texts


def reference_token_present(token: str, text: str) -> bool:
  escaped = re.escape(token)
  return re.search(rf"(?<![A-Za-z0-9_./-]){escaped}(?![A-Za-z0-9_./-])", text) is not None


def referenced_locations(rel: str, haystacks: list[tuple[str, str]]) -> list[str]:
  name = Path(rel).name
  tokens = tuple(dict.fromkeys((rel, name)))
  found: list[str] = []
  for text_rel, text in haystacks:
    if any(reference_token_present(token, text) for token in tokens):
      found.append(text_rel)
  return found


def artifact_category(rel: str) -> str:
  name = Path(rel).name
  if rel == "uv.lock":
    return "dependency-lockfile"
  if name.endswith(".progress.jsonl"):
    return "benchmark-progress-log"
  if "paired-" in name:
    return "paired-helper-smoke"
  if "phase-overlap" in name or "phase-profile" in name or "decode-graph" in name or "graph-" in name:
    return "decode-profiler-artifact"
  if name.endswith((".csv", ".json", ".jsonl")):
    return "benchmark-result-artifact"
  return "other-untracked"


def suggested_disposition(rel: str, refs: list[str]) -> str:
  if rel == "uv.lock":
    return "hold local; commit only after explicit lockfile policy because repo uses setuptools/pip workflow today"
  if refs:
    return "selective-commit candidate if evidence should travel with docs; otherwise archive externally"
  if rel.endswith(".progress.jsonl"):
    return "archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence"
  return "archive or delete after review unless it becomes cited by a plan/evolution entry"


def collect_inventory(*, cwd: Path) -> list[dict[str, Any]]:
  haystacks = tracked_reference_texts(cwd=cwd)
  rows: list[dict[str, Any]] = []
  for rel in git_untracked_paths(cwd=cwd):
    path = cwd / rel
    size = path.stat().st_size if path.is_file() else 0
    refs = referenced_locations(rel, haystacks)
    rows.append({
      "path": rel,
      "category": artifact_category(rel),
      "size": size,
      "refs": refs,
      "disposition": suggested_disposition(rel, refs),
    })
  return rows


def sort_inventory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return sorted(rows, key=lambda row: (0 if row["refs"] else 1, -len(row["refs"]), row["path"]))


def render_json_payload(rows: list[dict[str, Any]], *, timestamp: str) -> dict[str, Any]:
  rows = sort_inventory_rows(rows)
  total_size = sum(int(row["size"]) for row in rows)
  referenced_count = sum(1 for row in rows if row["refs"])
  return {
    "timestamp": timestamp,
    "untracked_count": len(rows),
    "total_size": total_size,
    "referenced_count": referenced_count,
    "rows": [
      {
        "path": row["path"],
        "category": row["category"],
        "size": row["size"],
        "refs": row["refs"],
        "reference_count": len(row["refs"]),
        "disposition": row["disposition"],
      }
      for row in rows
    ],
  }


def render_markdown(rows: list[dict[str, Any]], *, timestamp: str) -> str:
  rows = sort_inventory_rows(rows)
  by_category = Counter(row["category"] for row in rows)
  by_suffix = Counter(Path(row["path"]).suffix or "<none>" for row in rows)
  total_size = sum(int(row["size"]) for row in rows)
  referenced_count = sum(1 for row in rows if row["refs"])

  lines = [
    "# Untracked Artifact Inventory",
    "",
    "## Summary",
    "",
    f"- Timestamp: `{timestamp}`",
    f"- Untracked paths: `{len(rows)}`",
    f"- Total untracked bytes: `{total_size}` (~{total_size / 1024 / 1024:.2f} MiB)",
    f"- Referenced by tracked docs/state/config: `{referenced_count}`",
    "",
    "## Counts",
    "",
    "By category:",
    "",
  ]
  for key, value in sorted(by_category.items()):
    lines.append(f"- `{key}`: `{value}`")
  lines.extend(["", "By suffix:", ""])
  for key, value in sorted(by_suffix.items()):
    lines.append(f"- `{key}`: `{value}`")
  lines.extend([
    "",
    "## Recommendations",
    "",
    "- Do not blanket-ignore generated benchmark suffixes when the repo intentionally tracks selected benchmark evidence.",
    "- Commit or archive docs-referenced artifacts deliberately; do not let them remain only as terminal folklore.",
    "- Treat uncited progress logs as cleanup candidates after review.",
    "- Hold dependency lockfiles local until the project has an explicit lockfile policy.",
    "",
    "## Reference-ranked candidates",
    "",
    "| Path | Reference count | Referenced by | Category | Size bytes | Suggested disposition |",
    "|---|---:|---|---:|---:|---|",
  ])
  for row in rows:
    refs = "<br>".join(row["refs"]) if row["refs"] else "—"
    lines.append(f"| `{row['path']}` | `{len(row['refs'])}` | {refs} | `{row['category']}` | `{row['size']}` | {row['disposition']} |")
  lines.append("")
  return "\n".join(lines)


def default_timestamp() -> str:
  return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Inventory untracked repo artifacts without mutating git state.")
  parser.add_argument("--output", type=Path, help="Write inventory to this path. Defaults to stdout.")
  parser.add_argument("--format", choices=("markdown", "json"), default="markdown", help="Output format. Defaults to markdown.")
  parser.add_argument("--timestamp", default=default_timestamp())
  args = parser.parse_args(argv)

  cwd = Path.cwd()
  rows = collect_inventory(cwd=cwd)
  if args.format == "json":
    text = json.dumps(render_json_payload(rows, timestamp=args.timestamp), indent=2, sort_keys=True, allow_nan=False) + "\n"
  else:
    text = render_markdown(rows, timestamp=args.timestamp)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
  else:
    print(text, end="")
  print(f"inventoried {len(rows)} untracked path(s)", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
