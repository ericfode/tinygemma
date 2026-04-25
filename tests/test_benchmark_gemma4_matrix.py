from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_benchmark_module():
  module_path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_gemma4_matrix.py"
  spec = importlib.util.spec_from_file_location("benchmark_gemma4_matrix", module_path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


benchmark = load_benchmark_module()


def test_decode_warmup_suffix_timing_preserves_generated_sequence():
  source_tokens = [101, 102, 103, 104, 105]
  max_new_tokens = len(source_tokens)
  decode_warmup_tokens = 2
  generated: list[int] = []
  clock_values = iter([12.5])
  timer = benchmark.DecodeSuffixTimer(decode_warmup_tokens=decode_warmup_tokens, started=10.0, clock=lambda: next(clock_values))

  for token_id in source_tokens:
    benchmark.append_generated_token(generated, token_id, timer)

  metrics = timer.finish(len(generated), ended=15.0)

  assert generated == source_tokens
  assert metrics["decode_warmup_tokens"] == decode_warmup_tokens
  assert metrics["measured_decode_tokens"] == max_new_tokens - decode_warmup_tokens
  assert metrics["measured_decode_seconds"] == pytest.approx(2.5)
  assert metrics["measured_decode_tokens_per_second"] == pytest.approx(1.2)


def test_default_beams_include_acceptance_beam_zero():
  args = benchmark.build_parser().parse_args([])

  assert args.beams == [0, 1, 2, 3, 4]


def test_benchmark_checkpoint_uses_current_loader_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  class FakeTokenizer:
    def encode(self, prompt: str, *, add_bos: bool) -> list[int]:
      assert prompt == "hello"
      assert add_bos is True
      return [1, 2]

  class FakeModel:
    device = "PYTHON"
    _last_rollout_jit = None
    _last_decode_fallback = False

    def generate(self, input_ids: list[int], *, max_new_tokens: int, temperature: float, stop_token_ids):
      assert input_ids == [1, 2]
      assert max_new_tokens == 1
      assert temperature == 0.0
      assert stop_token_ids is None
      yield 3

  seen = {}

  def fake_load_pretrained(model_dir: Path, *, device: str):
    seen["model_dir"] = model_dir
    seen["device"] = device
    return FakeModel()

  monkeypatch.setattr(benchmark, "prepare_device", lambda device: device)
  monkeypatch.setattr(benchmark, "load_pretrained", fake_load_pretrained)
  monkeypatch.setattr(benchmark.GemmaTokenizer, "from_pretrained", lambda model_dir: FakeTokenizer())

  rows = benchmark.benchmark_checkpoint(
    tmp_path,
    prompt="hello",
    beams=[0],
    max_new_tokens=1,
    decode_warmup_tokens=0,
    device="PYTHON",
    size="E2B",
    fmt="int8",
    progress_every=0,
    progress_path=None,
  )

  assert seen == {"model_dir": tmp_path, "device": "PYTHON"}
  assert rows[0]["status"] == "ok"
  assert rows[0]["beam"] == 0
  assert rows[0]["generated_tokens"] == 1
  assert rows[0]["decode_fallback"] == "false"


@pytest.mark.parametrize("decode_warmup_tokens", [-1, 5])
def test_decode_warmup_validation_rejects_values_outside_generation_window(decode_warmup_tokens: int):
  with pytest.raises(ValueError):
    benchmark.validate_decode_warmup_tokens(max_new_tokens=5, decode_warmup_tokens=decode_warmup_tokens)
