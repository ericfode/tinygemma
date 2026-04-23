#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a Hermes profile into Codex-friendly repo artifacts.")
    parser.add_argument("--profile-dir", required=True, help="Path to the Hermes profile directory")
    parser.add_argument("--target-repo", required=True, help="Target repo root")
    parser.add_argument("--profile-name", default="", help="Override output profile name")
    parser.add_argument("--write-agents", action="store_true", help="Write the generated draft to AGENTS.md if it does not already exist")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_text(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> None:
    args = parse_args()
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    target_repo = Path(args.target_repo).expanduser().resolve()
    profile_name = args.profile_name or profile_dir.name

    profile_md = profile_dir / "PROFILE.md"
    soul_md = profile_dir / "SOUL.md"
    if not profile_md.exists() and not soul_md.exists():
        raise SystemExit("expected PROFILE.md or SOUL.md in the Hermes profile directory")

    out_dir = target_repo / "docs" / "hermes-import" / profile_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if profile_md.exists():
        shutil.copyfile(profile_md, out_dir / "PROFILE.md")
    if soul_md.exists():
        shutil.copyfile(soul_md, out_dir / "SOUL.md")

    profile_text = profile_md.read_text() if profile_md.exists() else ""
    soul_text = soul_md.read_text() if soul_md.exists() else ""
    draft = "\n".join(
        [
            "# AGENTS Draft",
            "",
            f"This draft was imported from the Hermes profile `{profile_name}`.",
            "",
            "## Imported Operating Notes",
            "",
            profile_text.strip() or "No PROFILE.md present.",
            "",
            "## Imported Persona Notes",
            "",
            soul_text.strip() or "No SOUL.md present.",
            "",
            "## Codex Translation Notes",
            "",
            "- Keep project-specific rules in repo files.",
            "- Prefer installable skills and scripts over prompt folklore.",
            "- Promote this draft into AGENTS.md only after adapting it to the repo.",
        ]
    ).strip() + "\n"
    draft_path = out_dir / "CODEX-AGENTS-DRAFT.md"
    write_text(draft_path, draft, args.force)

    if args.write_agents:
        agents_path = target_repo / "AGENTS.md"
        if agents_path.exists() and not args.force:
            raise SystemExit(f"{agents_path} already exists; rerun with --force to overwrite it")
        write_text(agents_path, draft, True)

    print(draft_path)


if __name__ == "__main__":
    main()
