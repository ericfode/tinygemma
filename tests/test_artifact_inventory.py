import json
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


def test_inventory_untracked_artifacts_stdout_mode_is_pure_markdown(tmp_path: Path):
  repo = tmp_path / "repo"
  repo.mkdir()
  run_git(repo, "init")
  (repo / "tracked.md").write_text("tracked\n")
  run_git(repo, "add", "tracked.md")
  (repo / "artifact.csv").write_text("score\n1\n")

  proc = subprocess.run(
    [sys.executable, str(SCRIPT), "--timestamp", "2026-04-27T03:21:00-0700"],
    cwd=repo,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  assert proc.returncode == 0, proc.stderr
  assert proc.stdout.startswith("# Untracked Artifact Inventory\n")
  assert "`artifact.csv`" in proc.stdout
  assert "inventoried" not in proc.stdout
  assert "inventoried 1 untracked path(s)" in proc.stderr


def test_inventory_untracked_artifacts_ranks_referenced_artifacts_first(tmp_path: Path):
  repo = tmp_path / "repo"
  repo.mkdir()
  run_git(repo, "init")
  (repo / "docs").mkdir()
  (repo / "docs" / "a.md").write_text("Keep hot.json and also warm.csv\n")
  (repo / "docs" / "b.md").write_text("Second reference to hot.json\n")
  (repo / "tracked.txt").write_text("tracked\n")
  run_git(repo, "add", "docs/a.md", "docs/b.md", "tracked.txt")

  (repo / "cold.csv").write_text("score\n0\n")
  (repo / "hot.json").write_text('{"score": 2}\n')
  (repo / "warm.csv").write_text("score\n1\n")

  proc = subprocess.run(
    [sys.executable, str(SCRIPT), "--timestamp", "2026-04-27T03:30:00-0700"],
    cwd=repo,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  assert proc.returncode == 0, proc.stderr
  inventory_lines = [line for line in proc.stdout.splitlines() if line.startswith("| `")]
  assert [line.split("`", 2)[1] for line in inventory_lines] == ["hot.json", "warm.csv", "cold.csv"]
  assert "## Reference-ranked candidates" in proc.stdout
  assert "| `hot.json` | `2` | docs/a.md<br>docs/b.md |" in proc.stdout


def test_inventory_untracked_artifacts_uses_exact_reference_tokens(tmp_path: Path):
  repo = tmp_path / "repo"
  repo.mkdir()
  run_git(repo, "init")
  (repo / "benchmarks").mkdir()
  (repo / "docs").mkdir()
  (repo / "docs" / "false.md").write_text("Do not count my-artifact.csv or artifact.csv.backup\n")
  (repo / "docs" / "exact_name.md").write_text("Count `artifact.csv` as an exact filename\n")
  (repo / "docs" / "exact_path.md").write_text("Count benchmarks/path-only.json as an exact path\n")
  run_git(repo, "add", "docs/false.md", "docs/exact_name.md", "docs/exact_path.md")

  (repo / "artifact.csv").write_text("score\n1\n")
  (repo / "benchmarks" / "path-only.json").write_text('{"score": 2}\n')
  (repo / "lonely.csv").write_text("score\n0\n")

  proc = subprocess.run(
    [sys.executable, str(SCRIPT), "--timestamp", "2026-04-27T03:40:00-0700"],
    cwd=repo,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  assert proc.returncode == 0, proc.stderr
  artifact_line = next(line for line in proc.stdout.splitlines() if line.startswith("| `artifact.csv`"))
  path_line = next(line for line in proc.stdout.splitlines() if line.startswith("| `benchmarks/path-only.json`"))
  lonely_line = next(line for line in proc.stdout.splitlines() if line.startswith("| `lonely.csv`"))
  assert "| `1` | docs/exact_name.md |" in artifact_line
  assert "false.md" not in artifact_line
  assert "| `1` | docs/exact_path.md |" in path_line
  assert "| `0` | — |" in lonely_line


def test_inventory_untracked_artifacts_json_output_is_machine_readable(tmp_path: Path):
  repo = tmp_path / "repo"
  repo.mkdir()
  run_git(repo, "init")
  (repo / "docs").mkdir()
  (repo / "docs" / "note.md").write_text("Keep artifact.csv\n")
  run_git(repo, "add", "docs/note.md")
  (repo / "artifact.csv").write_text("score\n1\n")
  (repo / "orphan.progress.jsonl").write_text('{"step": 1}\n')

  proc = subprocess.run(
    [sys.executable, str(SCRIPT), "--format", "json", "--timestamp", "2026-04-27T03:50:00-0700"],
    cwd=repo,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  assert proc.returncode == 0, proc.stderr
  payload = json.loads(proc.stdout)
  assert payload["timestamp"] == "2026-04-27T03:50:00-0700"
  assert payload["untracked_count"] == 2
  assert payload["referenced_count"] == 1
  assert [row["path"] for row in payload["rows"]] == ["artifact.csv", "orphan.progress.jsonl"]
  assert payload["rows"][0]["reference_count"] == 1
  assert payload["rows"][0]["refs"] == ["docs/note.md"]
  assert "inventoried" not in proc.stdout
  assert "inventoried 2 untracked path(s)" in proc.stderr
