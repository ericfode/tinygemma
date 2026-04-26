import os

os.environ.setdefault("CACHELEVEL", "0")

from tinygrad.helpers import CACHELEVEL

if hasattr(CACHELEVEL, "value"):
    CACHELEVEL.value = int(os.environ["CACHELEVEL"])
else:
    os.environ["CACHELEVEL"] = str(os.environ["CACHELEVEL"])

from .config import GemmaAudioConfig, GemmaConditionalConfig, GemmaConfig, GemmaVisionConfig
from .loader import load_config, load_pretrained, load_state_dict, load_text_config
from .model import DEFAULT_IGNORE_INDEX, GemmaCache, GemmaForCausalLM, build_causal_labels, causal_language_model_loss
from .multimodal import GemmaForConditionalGeneration
from .processing import GemmaMultimodalProcessor, GemmaPreparedInputs
from .quantization import load_quantization_manifest, supported_quantizations
from .runtime import default_device, prepare_device
from .training import GemmaTrainingBatch, build_optimizer, load_optimizer_state, named_parameters, parameters, save_pretrained, save_training_checkpoint, set_trainable, supported_optimizers, train_step

__all__ = [
  "GemmaAudioConfig",
  "GemmaCache",
  "GemmaConditionalConfig",
  "GemmaConfig",
  "GemmaForCausalLM",
  "GemmaForConditionalGeneration",
  "GemmaMultimodalProcessor",
  "GemmaPreparedInputs",
  "GemmaTrainingBatch",
  "GemmaVisionConfig",
  "DEFAULT_IGNORE_INDEX",
  "build_causal_labels",
  "build_optimizer",
  "causal_language_model_loss",
  "default_device",
  "load_optimizer_state",
  "load_config",
  "load_pretrained",
  "load_quantization_manifest",
  "load_state_dict",
  "load_text_config",
  "named_parameters",
  "parameters",
  "prepare_device",
  "save_pretrained",
  "save_training_checkpoint",
  "set_trainable",
  "supported_optimizers",
  "supported_quantizations",
  "train_step",
]
