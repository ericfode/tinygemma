#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Codex autoloop prompt derived from the Hermes cron pattern.")
    parser.add_argument("--repo", required=True, help="Target repo root")
    parser.add_argument("--recovery-command", default="", help="Command to run first to recover loop state")
    parser.add_argument("--loop-state", default="configs/repo-loop-state.json")
    parser.add_argument("--workflow-doc", default="", help="Workflow document to read before execution")
    parser.add_argument("--plan-reference", default="", help="How to find the current plan")
    parser.add_argument("--gate-command", action="append", default=[], help="Gate command to require (repeatable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    gates = args.gate_command or ["pytest -q"]

    lines = [
        f"Continue autonomous repo development in {repo}.",
        "",
    ]
    if args.recovery_command:
        lines.extend(
            [
                "Start by running:",
                f"- {args.recovery_command}",
                "",
            ]
        )

    lines.append("Then read:")
    lines.append(f"- {args.loop_state}")
    if args.workflow_doc:
        lines.append(f"- {args.workflow_doc}")
    if args.plan_reference:
        lines.append(f"- {args.plan_reference}")
    lines.extend(
        [
            "",
            "Execute exactly ONE next planned increment per run.",
            "",
            "Requirements:",
            "- follow the repo workflow: Research -> Cohere -> Plan -> implement -> review -> loop to fix -> repeat",
            "- use as many deterministic gates as possible",
            "- run at minimum:",
        ]
    )
    lines.extend(f"  - {gate}" for gate in gates)
    lines.extend(
        [
            "- if the increment is accepted:",
            "  - create or update the acceptance note",
            f"  - update {args.loop_state} to mark the increment accepted",
            "  - create the next increment plan file and set it as the next target",
            "  - commit all changes with a clear commit message",
            "  - report the commit hash, tests passed count, and new next target",
            "- if you hit a real blocker or ambiguity that cannot be resolved from the repo, do not guess; report only the blocker clearly",
            "- do not create, modify, or schedule automations from inside the autonomous run",
            "",
            "Stay within the existing repo conventions and keep changes narrow, explicit, and import-safe.",
        ]
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
