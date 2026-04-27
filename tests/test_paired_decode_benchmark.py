import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paired_e2b_decode_benchmark.py"


def write_fake_benchmark(path: Path, *, candidate_score: float = 12.5, baseline_score: float = 10.0) -> None:
  path.write_text(
    f"""
import argparse
import json
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--target', required=True)
parser.add_argument('--tag')
args = parser.parse_args()
score = {candidate_score!r} if 'candidate' in args.target else {baseline_score!r}
print(json.dumps({{'score': score, 'tasks': {{'fake': score}}, 'tag': args.tag}}))
print(f'stderr for {{args.target}}', file=sys.stderr)
""".strip()
    + "\n"
  )


def test_paired_decode_benchmark_records_delta_and_commands(tmp_path: Path):
  fake_benchmark = tmp_path / "fake_benchmark.py"
  write_fake_benchmark(fake_benchmark)
  baseline = tmp_path / "baseline_model.py"
  candidate = tmp_path / "candidate_model.py"
  baseline.write_text("# baseline\n")
  candidate.write_text("# candidate\n")
  out = tmp_path / "paired.json"

  proc = subprocess.run(
    [
      sys.executable,
      str(SCRIPT),
      "--benchmark-script",
      str(fake_benchmark),
      "--baseline-target",
      str(baseline),
      "--candidate-target",
      str(candidate),
      "--out",
      str(out),
      "--label",
      "unit-pair",
      "--",
      "--tag",
      "smoke",
    ],
    cwd=ROOT,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=True,
  )

  stdout_payload = json.loads(proc.stdout)
  file_payload = json.loads(out.read_text())
  assert stdout_payload == file_payload
  assert file_payload["label"] == "unit-pair"
  assert file_payload["baseline"]["score"] == 10.0
  assert file_payload["candidate"]["score"] == 12.5
  assert file_payload["delta"] == 2.5
  assert file_payload["relative_delta"] == 0.25
  assert file_payload["candidate_improved"] is True
  assert file_payload["benchmark_args"] == ["--tag", "smoke"]
  assert file_payload["baseline"]["command"][0] == sys.executable
  assert "stderr for" in file_payload["candidate"]["stderr"]


def test_paired_decode_benchmark_finds_default_benchmark_inside_baseline_worktree(tmp_path: Path):
  worktree = tmp_path / "worktree"
  benchmark = worktree / "benchmarks" / "evo_e2b_int8_metal_decode.py"
  model_dir = worktree / "tinygrad_gemma"
  benchmark.parent.mkdir(parents=True)
  model_dir.mkdir(parents=True)
  write_fake_benchmark(benchmark, candidate_score=2.0, baseline_score=1.0)
  baseline = model_dir / "baseline_model.py"
  candidate = model_dir / "candidate_model.py"
  baseline.write_text("# baseline\n")
  candidate.write_text("# candidate\n")

  proc = subprocess.run(
    [
      sys.executable,
      str(SCRIPT),
      "--baseline-target",
      str(baseline),
      "--candidate-target",
      str(candidate),
    ],
    cwd=ROOT,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=True,
  )

  data = json.loads(proc.stdout)
  assert data["benchmark_script"] == str(benchmark)
  assert data["candidate_improved"] is True
  assert data["delta"] == 1.0


def test_paired_decode_benchmark_min_delta_fails_after_writing_payload(tmp_path: Path):
  fake_benchmark = tmp_path / "fake_benchmark.py"
  write_fake_benchmark(fake_benchmark, candidate_score=9.5, baseline_score=10.0)
  baseline = tmp_path / "baseline_model.py"
  candidate = tmp_path / "candidate_model.py"
  baseline.write_text("# baseline\n")
  candidate.write_text("# candidate\n")
  out = tmp_path / "paired-fail.json"

  proc = subprocess.run(
    [
      sys.executable,
      str(SCRIPT),
      "--benchmark-script",
      str(fake_benchmark),
      "--baseline-target",
      str(baseline),
      "--candidate-target",
      str(candidate),
      "--out",
      str(out),
      "--min-delta",
      "0.0",
    ],
    cwd=ROOT,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  assert proc.returncode == 1
  data = json.loads(out.read_text())
  assert data["delta"] == -0.5
  assert data["min_delta"] == 0.0
  assert data["passed_min_delta"] is False
  assert "below --min-delta" in proc.stderr


def test_paired_decode_benchmark_rejects_nonfinite_scores(tmp_path: Path):
  fake_benchmark = tmp_path / "fake_benchmark_nan.py"
  fake_benchmark.write_text(
    """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target', required=True)
parser.parse_args()
print('{"score": NaN}')
""".strip()
    + "\n"
  )
  baseline = tmp_path / "baseline_model.py"
  candidate = tmp_path / "candidate_model.py"
  baseline.write_text("# baseline\n")
  candidate.write_text("# candidate\n")

  proc = subprocess.run(
    [
      sys.executable,
      str(SCRIPT),
      "--benchmark-script",
      str(fake_benchmark),
      "--baseline-target",
      str(baseline),
      "--candidate-target",
      str(candidate),
    ],
    cwd=ROOT,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  assert proc.returncode == 1
  assert "finite" in proc.stderr
