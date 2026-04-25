from __future__ import annotations

import importlib.util
from pathlib import Path


def load_profile_module():
  module_path = Path(__file__).resolve().parents[1] / "scripts" / "profile_decode_jit.py"
  spec = importlib.util.spec_from_file_location("profile_decode_jit", module_path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


profile = load_profile_module()


def test_profile_raw_gate_up_mode_is_abandoned():
  try:
    profile.validate_metal_int8_gate_up_mode("raw")
  except SystemExit as exc:
    assert "abandoned" in str(exc)
  else:
    raise AssertionError("raw gate/up profiling mode should be rejected")


def test_profile_default_gate_up_mode_is_allowed():
  profile.validate_metal_int8_gate_up_mode("default")
