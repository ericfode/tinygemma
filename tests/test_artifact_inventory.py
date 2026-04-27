import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inventory_untracked_artifacts.py"


def run_git(repo: Path, *args: str) -> None:
  subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_inventory_untracked_artifacts_reports_references_without_mutating_repo(tmp_path: Path):
  repo = tmp_path / "repo"
  repo.mkdir()
  run_git(repo, "init")
  (repo / "docs").mkdir()
  (repo / "docs" / "note.md").write_text("Artifact: artifact.csv\n")
  (repo / "tracked.txt").write_text("tracked\n")
  run_git(repo, "add", "docs/note.md", "tracked.txt")

  (repo / "artifact.csv").write_text("score\n1\n")
  (repo / "orphan.progress.jsonl").write_text('{"step": 1}\n')
  (repo / "uv.lock").write_text("version = 1\n")
  out = repo / "inventory.md"

  proc = subprocess.run(
    [sys.executable, str(SCRIPT), "--output", str(out), "--timestamp", "2026-04-27T03:20:00-0700"],
    cwd=repo,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  assert proc.returncode == 0, proc.stderr
  text = out.read_text()
  assert "Untracked paths: `3`" in text
  assert "`artifact.csv`" in text
  assert "docs/note.md" in text
  assert "`benchmark-progress-log`" in text
  assert "dependency-lockfile" in text

  status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
  assert "?? artifact.csv" in status
  assert "?? orphan.progress.jsonl" in status
  assert "?? uv.lock" in status
