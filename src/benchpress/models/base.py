"""Base model interface for benchpress."""

from abc import ABC, abstractmethod
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional


class BaseModel(ABC):
    """Abstract base class for all LLM models."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the model.

        Args:
            prompt: The input prompt to send to the model
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that will stop generation when encountered
            **kwargs: Additional model-specific parameters

        Returns:
            The generated text response
        """
        pass

    async def stream_generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream a response from the model one chunk at a time.
        
        Default implementation falls back to non-streaming generate().
        Models that support streaming should override this method.

        Args:
            prompt: The input prompt to send to the model
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that will stop generation when encountered
            **kwargs: Additional model-specific parameters

        Yields:
            Text chunks as they become available
        """
        # Default implementation just yields the full response
        response = await self.generate(
            prompt, temperature, max_tokens, stop_sequences, **kwargs
        )
        yield response

    @abstractmethod
    def get_response_metadata(self) -> Dict[str, Any]:
        """Return metadata from the most recent response.

        Returns:
            A dictionary containing metadata like token usage,
            model specifics, etc.
        """
        pass
