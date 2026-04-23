#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path


STOP_MARKERS = (
    "WHOLE PROJECT COMPLETE",
    "ACTUAL QUESTION:",
    "BLOCKED:",
    "UNRESOLVED BLOCKER:",
)

FOLLOWUP_PROMPT = """Continue working autonomously.

Work exactly one next increment.

If the repo has a machine-readable next-step file such as configs/repo-loop-state.json,
read it and execute the next planned increment.

Only stop if one of these is true:
- WHOLE PROJECT COMPLETE
- ACTUAL QUESTION:
- BLOCKED:
- UNRESOLVED BLOCKER:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Codex in a bounded autonomous repo loop.")
    parser.add_argument("--repo", required=True, help="Target repository root")
    parser.add_argument("--prompt", default="", help="Initial prompt")
    parser.add_argument("--prompt-file", help="Path to a file containing the initial prompt")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--model", help="Optional model override")
    parser.add_argument("--skip-git-repo-check", action="store_true")
    parser.add_argument("--dangerous", action="store_true", help="Use Codex without approvals or sandbox")
    return parser.parse_args()


def read_initial_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text()
    if args.prompt:
        return args.prompt
    raise SystemExit("either --prompt or --prompt-file is required")


def build_command(args: argparse.Namespace, message_path: Path) -> list[str]:
    command = ["codex", "exec", "-", "--output-last-message", str(message_path)]
    if args.model:
        command.extend(["--model", args.model])
    if args.skip_git_repo_check:
        command.append("--skip-git-repo-check")
    if args.dangerous:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        command.append("--full-auto")
    return command


def stop_reason(last_message: str) -> str | None:
    upper = last_message.upper()
    for marker in STOP_MARKERS:
        if marker in upper:
            return marker
    return None


def resolve_control_plane_ingest_script() -> Path | None:
    candidates = []
    explicit_root = os.environ.get("CODEX_CONTROL_PLANE_PLUGIN_PATH")
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser().resolve() / "src" / "autoloop-ingest.js")

    candidates.append(Path.home() / "plugins" / "codex-control-plane" / "src" / "autoloop-ingest.js")
    candidates.append(Path(__file__).resolve().parents[2] / "codex-control-plane" / "src" / "autoloop-ingest.js")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def maybe_ingest_iteration(
    *,
    repo_root: Path,
    run_id: str,
    job_id: str | None,
    session_id: str | None,
    session_key: str | None,
    iteration: int,
    stdout_path: Path,
    stderr_path: Path,
    last_message_path: Path,
    reason: str,
) -> None:
    ingest_script = resolve_control_plane_ingest_script()
    bun = shutil.which("bun")
    if ingest_script is None or bun is None:
        return

    command = [
        bun,
        "run",
        str(ingest_script),
        "--repo",
        str(repo_root),
        "--run-id",
        run_id,
        "--iteration",
        str(iteration),
        "--stdout-log",
        str(stdout_path),
        "--stderr-log",
        str(stderr_path),
        "--last-message-path",
        str(last_message_path),
        "--reason",
        reason,
        "--metadata",
        json.dumps({"source": "codex_autoloop.py"}),
    ]
    if job_id:
        command.extend(["--job-id", job_id])
    if session_id:
        command.extend(["--session-id", session_id])
    if session_key:
        command.extend(["--session-key", session_key])

    subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo).expanduser().resolve()
    initial_prompt = read_initial_prompt(args)
    log_dir = repo_root / "state" / "codex-autoloop"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = os.environ.get("CODEX_HARNESS_RUN_ID") or f"autoloop_{uuid.uuid4().hex}"
    job_id = os.environ.get("CODEX_HARNESS_JOB_ID")
    session_id = os.environ.get("CODEX_HARNESS_SESSION_ID")
    session_key = os.environ.get("CODEX_HARNESS_SESSION_KEY")
    prompt = initial_prompt
    for iteration in range(1, args.max_iterations + 1):
        with tempfile.TemporaryDirectory() as temp_dir:
            last_message_path = Path(temp_dir) / "last-message.txt"
            command = build_command(args, last_message_path)
            result = subprocess.run(
                command,
                cwd=repo_root,
                input=prompt,
                text=True,
                capture_output=True,
            )

            stdout_path = log_dir / f"iteration-{iteration:02d}.stdout.log"
            stderr_path = log_dir / f"iteration-{iteration:02d}.stderr.log"
            message_copy_path = log_dir / f"iteration-{iteration:02d}.last-message.txt"
            stdout_path.write_text(result.stdout)
            stderr_path.write_text(result.stderr)

            last_message = ""
            if last_message_path.exists():
                last_message = last_message_path.read_text()
                message_copy_path.write_text(last_message)

            if result.returncode != 0:
                maybe_ingest_iteration(
                    repo_root=repo_root,
                    run_id=run_id,
                    job_id=job_id,
                    session_id=session_id,
                    session_key=session_key,
                    iteration=iteration,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    last_message_path=message_copy_path,
                    reason="UNRESOLVED BLOCKER: codex exec failed",
                )
                raise SystemExit(
                    f"codex exec failed on iteration {iteration}; see {stderr_path}"
                )

            reason = stop_reason(last_message or result.stdout)
            maybe_ingest_iteration(
                repo_root=repo_root,
                run_id=run_id,
                job_id=job_id,
                session_id=session_id,
                session_key=session_key,
                iteration=iteration,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                last_message_path=message_copy_path,
                reason=reason or "continue",
            )
            print(f"iteration={iteration} reason={reason or 'continue'}")
            if reason is not None:
                return

            prompt = FOLLOWUP_PROMPT

    last_iteration = args.max_iterations
    maybe_ingest_iteration(
        repo_root=repo_root,
        run_id=run_id,
        job_id=job_id,
        session_id=session_id,
        session_key=session_key,
        iteration=last_iteration,
        stdout_path=log_dir / f"iteration-{last_iteration:02d}.stdout.log",
        stderr_path=log_dir / f"iteration-{last_iteration:02d}.stderr.log",
        last_message_path=log_dir / f"iteration-{last_iteration:02d}.last-message.txt",
        reason="UNRESOLVED BLOCKER: reached max iterations without explicit stop marker",
    )
    raise SystemExit(
        f"reached max iterations ({args.max_iterations}) without an explicit stop marker"
    )


if __name__ == "__main__":
    main()
