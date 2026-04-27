import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paired_e2b_decode_benchmark.py"


def test_paired_decode_benchmark_records_delta_and_commands(tmp_path: Path):
  fake_benchmark = tmp_path / "fake_benchmark.py"
  fake_benchmark.write_text(
    """
import argparse
import json
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--target', required=True)
parser.add_argument('--tag')
args = parser.parse_args()
score = 12.5 if 'candidate' in args.target else 10.0
print(json.dumps({'score': score, 'tasks': {'fake': score}, 'tag': args.tag}))
print(f'stderr for {args.target}', file=sys.stderr)
""".strip()
    + "\n"
  )
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
