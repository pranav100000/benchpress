"""GLHF.chat API model implementation."""

import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from .base import BaseModel


class GlhfModel(BaseModel):
    """Model implementation for GLHF.chat API."""

    # Default base URL for GLHF.chat API
    DEFAULT_API_BASE = "https://glhf.chat/api/openai/v1"

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize GLHF.chat model.

        Args:
            model_name: The name of the model to use (will be prefixed with 'hf:' if not already)
            api_key: API key for authentication (if None, uses env var GLHF_API_KEY)
            api_base: Base URL for API (if None, uses DEFAULT_API_BASE)
            system_prompt: Optional system prompt to include in all requests
        """
        # Ensure model_name has the 'hf:' prefix
        if not model_name.startswith("hf:"):
            self._model_name = f"hf:{model_name}"
        else:
            self._model_name = model_name

        # Get API key either from parameter or environment variable
        self._api_key = api_key or os.environ.get("GLHF_API_KEY")
        self._api_base = api_base or self.DEFAULT_API_BASE
        self._system_prompt = system_prompt or "You are a helpful AI assistant."

        if not self._api_key:
            raise ValueError(
                "API key must be provided either directly or via "
                "GLHF_API_KEY environment variable"
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
        # Create messages list with system and user messages
        messages: List[ChatCompletionMessageParam] = []

        # Add system message if provided
        if self._system_prompt:
            system_message: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": self._system_prompt,
            }
            messages.append(system_message)

        # Add user message
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": prompt,
        }
        messages.append(user_message)

        # Handle potential timeouts by using streaming for large models
        should_stream = kwargs.pop("stream", False)
        try:
            if should_stream:
                # Stream the response and accumulate it
                content = ""
                stream = await self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                    stream=True,
                    **kwargs,
                )

                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content

                # Create a simulated response object to maintain compatibility
                # with the non-streaming code path
                # This is simplified and doesn't include all fields
                self._last_response = ChatCompletion(
                    id="simulated-from-stream",
                    choices=[{"message": {"content": content}}],
                    created=0,
                    model=self._model_name,
                    object="chat.completion",
                )
                return content
            else:
                # Standard non-streaming request
                response = await self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                    **kwargs,
                )

                self._last_response = response
                return response.choices[0].message.content or ""
        except Exception as e:
            # If a timeout occurs and we're not already streaming,
            # retry with streaming enabled
            if not should_stream and "timeout" in str(e).lower():
                return await self.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_sequences=stop_sequences,
                    stream=True,
                    **kwargs,
                )
            raise  # Re-raise if not a timeout or if already streaming

    def get_response_metadata(self) -> Dict[str, Any]:
        """Return metadata from the most recent response.

        Returns:
            A dictionary containing metadata like token usage, model specifics, etc.
        """
        if (
            not self._last_response
            or not hasattr(self._last_response, "usage")
            or not self._last_response.usage
        ):
            return {
                "model": self._model_name,
                "note": "Limited metadata available - possibly from streaming response",
            }

        return {
            "usage": {
                "prompt_tokens": self._last_response.usage.prompt_tokens,
                "completion_tokens": self._last_response.usage.completion_tokens,
                "total_tokens": self._last_response.usage.total_tokens,
            },
            "model": self._last_response.model,
            "id": self._last_response.id,
        }
