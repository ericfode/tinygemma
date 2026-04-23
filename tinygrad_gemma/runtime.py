from __future__ import annotations

import platform

from tinygrad import Device


def metal_is_usable() -> bool:
  if platform.system() != "Darwin":
    return False
  patch_tinygrad_metal_runtime()
  try:
    from tinygrad.runtime.autogen import metal

    sysdevice = metal.MTLCreateSystemDefaultDevice()
    buf = sysdevice.newBufferWithLength_options(4, metal.MTLResourceStorageModeShared)
    return bool(getattr(buf, "length")() >= 4)
  except Exception:
    return False


def available_devices() -> list[str]:
  devices = list(Device.get_available_devices())
  if "METAL" in devices and not metal_is_usable():
    devices = [device for device in devices if device != "METAL"]
  return devices


def default_device() -> str:
  devices = available_devices()
  if "METAL" in devices:
    return "METAL"
  if devices:
    return devices[0]
  return "CPU"


def normalize_device(device: str | None) -> str:
  if device is None or device.strip() == "" or device.lower() == "auto":
    return default_device()
  normalized = device.strip().upper()
  if normalized not in Device._devices and normalized not in ("CPU", "METAL"):
    raise ValueError(f"unsupported tinygrad device {device!r}")
  return normalized


def patch_tinygrad_metal_runtime() -> None:
  if platform.system() != "Darwin":
    return
  try:
    import tinygrad.runtime.ops_metal as ops_metal
  except Exception:
    return
  if getattr(ops_metal.from_ns_str, "_tinygrad_gemma_patched", False):
    return
  original = ops_metal.from_ns_str

  def safe_from_ns_str(s):
    try:
      return original(s)
    except (TypeError, ValueError):
      return ""

  safe_from_ns_str._tinygrad_gemma_patched = True  # type: ignore[attr-defined]
  ops_metal.from_ns_str = safe_from_ns_str


def prepare_device(device: str | None) -> str:
  normalized = normalize_device(device)
  if normalized == "METAL":
    patch_tinygrad_metal_runtime()
    if not metal_is_usable():
      raise RuntimeError("tinygrad METAL backend is unavailable in this process")
  return normalized
