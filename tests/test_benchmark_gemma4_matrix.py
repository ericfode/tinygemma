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


@pytest.mark.parametrize("decode_warmup_tokens", [-1, 5])
def test_decode_warmup_validation_rejects_values_outside_generation_window(decode_warmup_tokens: int):
  with pytest.raises(ValueError):
    benchmark.validate_decode_warmup_tokens(max_new_tokens=5, decode_warmup_tokens=decode_warmup_tokens)
