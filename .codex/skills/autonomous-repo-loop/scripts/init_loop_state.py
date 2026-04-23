#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a repo-loop-state.json file.")
    parser.add_argument("--repo", default=".", help="Target repo root")
    parser.add_argument("--next-target", required=True, help="Name of the next increment")
    parser.add_argument("--plan", default="", help="Relative path to the plan file for the next increment")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo).expanduser().resolve()
    state_path = repo_root / "configs" / "repo-loop-state.json"
    if state_path.exists() and not args.force:
        raise SystemExit(f"{state_path} already exists; rerun with --force to replace it")

    payload = {
        "schema_version": 1,
        "project_root": str(repo_root),
        "current_checkpoint": None,
        "accepted_increments": [],
        "next_target": args.next_target,
        "next_target_plan": args.plan or None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(state_path)


if __name__ == "__main__":
    main()
