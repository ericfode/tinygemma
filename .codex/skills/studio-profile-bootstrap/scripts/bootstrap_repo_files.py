#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


AGENTS_TEMPLATE = """# AGENTS.md

This repo is a Codex workspace with repo-local harness state.

## Working Rules

1. Read repo docs and state files before changing direction.
2. Work in narrow increments.
3. Keep next-step state in the repo, not only in chat.
4. Promote repeated procedures into installable skills or scripts.
"""


EVOLUTION_LOG_TEMPLATE = """# Evolution Log

Use this file for terse, durable checkpoints when a repo change needs a written trail.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a minimal Codex repo harness baseline in a repo.")
    parser.add_argument("--repo", required=True, help="Target repo root")
    parser.add_argument("--next-target", default="define-first-increment")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_text(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_json(path: Path, payload: dict, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo).expanduser().resolve()
    write_text(repo_root / "AGENTS.md", AGENTS_TEMPLATE, args.force)
    write_text(repo_root / "docs" / "plans" / ".gitkeep", "", args.force)
    write_text(repo_root / "state" / "evolution-log.md", EVOLUTION_LOG_TEMPLATE, args.force)
    write_json(repo_root / "state" / "procedure-candidates.json", {"candidates": []}, args.force)
    write_json(
        repo_root / "configs" / "repo-loop-state.json",
        {
            "schema_version": 1,
            "current_checkpoint": None,
            "accepted_increments": [],
            "next_target": args.next_target,
            "next_target_plan": None,
        },
        args.force,
    )
    print(repo_root)


if __name__ == "__main__":
    main()
