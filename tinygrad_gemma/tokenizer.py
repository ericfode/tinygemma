from __future__ import annotations

import json
from pathlib import Path

try:
  import sentencepiece as spm
except ImportError:  # pragma: no cover - optional dependency
  spm = None

try:
  from tokenizers import Tokenizer as HFTokenizer
except ImportError:  # pragma: no cover - optional dependency
  HFTokenizer = None


class GemmaTokenizer:
  def __init__(
    self,
    model_path: str | Path | None = None,
    *,
    tokenizer_json_path: str | Path | None = None,
    tokenizer_config_path: str | Path | None = None,
  ):
    self.backend = "sentencepiece"
    self.config: dict[str, str] = {}
    if tokenizer_config_path is not None and Path(tokenizer_config_path).exists():
      self.config = json.loads(Path(tokenizer_config_path).read_text())
    if tokenizer_json_path is not None:
      if HFTokenizer is None:
        raise ImportError("tokenizers is required for tokenizer.json support; install tinygrad-gemma[tokenizer]")
      self.backend = "tokenizer_json"
      self.model_path = Path(tokenizer_json_path)
      self.processor = HFTokenizer.from_file(str(self.model_path))
      return
    if model_path is None:
      raise ValueError("model_path is required for sentencepiece tokenizers")
    if spm is None:
      raise ImportError("sentencepiece is required for text tokenization; install tinygrad-gemma[tokenizer]")
    self.model_path = Path(model_path)
    self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))

  @classmethod
  def from_pretrained(cls, model_dir: str | Path) -> "GemmaTokenizer":
    model_dir = Path(model_dir)
    tokenizer_json = model_dir / "tokenizer.json"
    tokenizer_config = model_dir / "tokenizer_config.json"
    if tokenizer_json.exists():
      return cls(tokenizer_json_path=tokenizer_json, tokenizer_config_path=tokenizer_config if tokenizer_config.exists() else None)
    for name in ("tokenizer.model", "tokenizer.spm"):
      path = model_dir / name
      if path.exists():
        return cls(path, tokenizer_config_path=tokenizer_config if tokenizer_config.exists() else None)
    raise FileNotFoundError(f"no tokenizer.json or tokenizer.model found under {model_dir}")

  def special_token(self, name: str) -> str | None:
    token = self.config.get(name)
    return None if token is None else str(token)

  def token_to_id(self, token: str) -> int:
    if self.backend == "tokenizer_json":
      token_id = self.processor.token_to_id(token)
      return -1 if token_id is None else int(token_id)
    token_id = self.processor.piece_to_id(token)
    return -1 if token_id is None else int(token_id)

  def _special_token_id(self, name: str) -> int:
    token = self.special_token(name)
    if token is None:
      return -1
    return self.token_to_id(token)

  @property
  def bos_id(self) -> int:
    if self.backend == "tokenizer_json":
      return self._special_token_id("bos_token")
    return int(self.processor.bos_id())

  @property
  def eos_id(self) -> int:
    if self.backend == "tokenizer_json":
      return self._special_token_id("eos_token")
    return int(self.processor.eos_id())

  @property
  def image_token_id(self) -> int:
    return self._special_token_id("image_token")

  @property
  def audio_token_id(self) -> int:
    return self._special_token_id("audio_token")

  @property
  def boi_token(self) -> str | None:
    return self.special_token("boi_token")

  @property
  def eoi_token(self) -> str | None:
    return self.special_token("eoi_token")

  @property
  def image_token(self) -> str | None:
    return self.special_token("image_token")

  @property
  def boa_token(self) -> str | None:
    return self.special_token("boa_token")

  @property
  def eoa_token(self) -> str | None:
    return self.special_token("eoa_token")

  @property
  def audio_token(self) -> str | None:
    return self.special_token("audio_token")

  def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
    if self.backend == "tokenizer_json":
      return list(self.processor.encode(text, add_special_tokens=add_bos).ids)
    tokens = list(self.processor.encode(text, out_type=int))
    if add_bos and self.bos_id >= 0:
      return [self.bos_id] + tokens
    return tokens

  def decode(self, token_ids: list[int]) -> str:
    if self.backend == "tokenizer_json":
      return self.processor.decode([int(x) for x in token_ids], skip_special_tokens=True)
    return self.processor.decode([int(x) for x in token_ids])
