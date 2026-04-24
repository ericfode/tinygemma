from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from tinygrad_gemma import GemmaCache, GemmaConfig, GemmaForCausalLM, load_pretrained, save_pretrained
from tinygrad_gemma.runtime import available_devices, default_device, temporary_default_device


def tiny_config() -> GemmaConfig:
  return GemmaConfig(
    model_type="gemma4",
    vocab_size=48,
    hidden_size=12,
    intermediate_size=24,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    num_global_key_value_heads=1,
    head_dim=4,
    global_head_dim=6,
    hidden_activation="gelu_pytorch_tanh",
    hidden_size_per_layer_input=0,
    max_position_embeddings=64,
    sliding_window=3,
    layer_types=["sliding_attention", "full_attention"],
    rope_parameters={
      "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
      "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.5, "rope_theta": 1000000.0},
    },
    attention_k_eq_v=True,
    final_logit_softcapping=4.0,
  )


def main() -> None:
  devices = available_devices()
  if "METAL" not in devices:
    raise SystemExit(f"METAL unavailable; available tinygrad devices: {devices}")

  with TemporaryDirectory(prefix="tinygrad-gemma-metal-") as tmp_dir:
    checkpoint_dir = Path(tmp_dir) / "checkpoint"
    with temporary_default_device("METAL"):
      source = GemmaForCausalLM(tiny_config())
      save_pretrained(source, checkpoint_dir)

    model = load_pretrained(checkpoint_dir, device="METAL")
    cache = GemmaCache.empty(model.config.num_hidden_layers, max_length=8)
    logits, cache = model.forward_ids([2, 4, 6], cache=cache)
    realized = logits.numpy()
    generated = list(model.generate([2, 4, 6], max_new_tokens=4, stop_token_ids=None))
    rollout_jits = getattr(model, "_last_rollout_jits", [])

  if model.device != "METAL":
    raise SystemExit(f"expected loaded model on METAL, got {model.device}")
  if logits.device != "METAL":
    raise SystemExit(f"expected logits on METAL, got {logits.device}")
  if cache is None or cache.entries[0] is None:
    raise SystemExit("expected populated METAL cache")
  if cache.entries[0].key.device != "METAL" or cache.entries[0].value.device != "METAL":
    raise SystemExit("expected cache tensors on METAL")
  if len(generated) != 4:
    raise SystemExit(f"expected 4 generated tokens on METAL, got {len(generated)}")
  if model._last_decode_fallback:
    raise SystemExit("expected METAL decode TinyJit replay without eager fallback")
  rollout_jit_count = sum(getattr(jit, "cnt", 0) for jit in rollout_jits if jit is not None)
  if rollout_jit_count == 0:
    raise SystemExit("expected METAL decode TinyJit to run during generation")

  print(f"default_device={default_device()}")
  print(f"loaded_model_device={model.device}")
  print(f"logits_device={logits.device}")
  print(f"logits_shape={realized.shape}")
  print(f"cache0_key_device={cache.entries[0].key.device}")
  print(f"generated_tokens={len(generated)}")
  print(f"rollout_jit_count={rollout_jit_count}")
  print(f"decode_fallback={model._last_decode_fallback}")


if __name__ == "__main__":
  main()
