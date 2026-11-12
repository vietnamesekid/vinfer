"""Qwen3 / Qwen3.5 / Qwen3.6 models. Read each checkpoint's config.json and modeling code first
(docs/models/)."""

from vinfer.models.qwen.config import QwenConfig
from vinfer.models.qwen.loader import load_qwen
from vinfer.models.qwen.model import QwenModel

__all__ = ["QwenConfig", "QwenModel", "load_qwen"]
