"""Base model interface for benchpress."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional


class BaseModel(ABC):
    """Abstract base class for all LLM models."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier."""
        pass

    def sanitize_params(
        self,
        base_params: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sanitize parameters by removing None values.

        This helps prevent errors with APIs that don't handle None values properly.

        Args:
            base_params: Base parameters for the API call
            **kwargs: Additional parameters

        Returns:
            Dictionary with all parameters, excluding those with None values
        """
        # Combine base parameters with additional kwargs
        all_params = {**base_params, **kwargs}

        # Filter out None values
        return {k: v for k, v in all_params.items() if v is not None}

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
