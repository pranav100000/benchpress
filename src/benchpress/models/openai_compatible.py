"""OpenAI-compatible API model implementation."""

import os
from typing import Any, Dict, List, Optional, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from .base import BaseModel


class OpenAICompatibleModel(BaseModel):
    """Model implementation for OpenAI-compatible APIs."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize OpenAI-compatible model.

        Args:
            model_name: The name of the model to use
            api_key: API key for authentication (if None, uses env var)
            api_base: Base URL for API (if None, uses default or env var)
            system_prompt: Optional system prompt to include in all requests
        """
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._api_base = api_base or os.environ.get("OPENAI_API_BASE")
        self._system_prompt = system_prompt

        if not self._api_key:
            raise ValueError(
                "API key must be provided either directly or via "
                "OPENAI_API_KEY environment variable"
            )

        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._api_base)
        self._last_response: Optional[ChatCompletion] = None

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        return self._model_name

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
        messages: List[ChatCompletionMessageParam] = []

        # Add system message if provided
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # Add user message
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": prompt,
        }
        messages.append(user_message)

        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
            **kwargs,
        )

        self._last_response = response
        message = cast(ChatCompletionMessage, response.choices[0].message)
        return message.content or ""

    def get_response_metadata(self) -> Dict[str, Any]:
        """Return metadata from the most recent response.

        Returns:
            A dictionary containing metadata like token usage, model specifics, etc.
        """
        if not self._last_response or not self._last_response.usage:
            return {}

        return {
            "usage": {
                "prompt_tokens": self._last_response.usage.prompt_tokens,
                "completion_tokens": self._last_response.usage.completion_tokens,
                "total_tokens": self._last_response.usage.total_tokens,
            },
            "model": self._last_response.model,
            "id": self._last_response.id,
        }
