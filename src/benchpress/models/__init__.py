"""Model abstractions for benchpress."""

from .base import BaseModel
from .glhf import GlhfModel
from .openai_compatible import OpenAICompatibleModel

__all__ = ["BaseModel", "OpenAICompatibleModel", "GlhfModel"]
