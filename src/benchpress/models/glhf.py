"""GLHF.chat API model implementation."""

import os
from typing import Any, AsyncGenerator, Dict, List, Optional

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
import tiktoken


class GlhfModel(BaseModel):
    """Model implementation for GLHF.chat API."""

    # Default base URL for GLHF.chat API
    DEFAULT_API_BASE = "https://api.glhf.chat/v1"

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize GLHF.chat model.

        Args:
            model_name: The name of the model to use (will be prefixed with 'hf:'
                if not already)
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
        self._streamed_completion_tokens: int = 0
        self._streamed_total_tokens: int = 0

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
                
                # Prepare parameters, removing any None values
                params = self.sanitize_params(
                    {
                        "model": self._model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stop": stop_sequences,
                        "stream": True,
                    },
                    **kwargs,
                )
                
                stream = await self._client.chat.completions.create(**params)

                # Track approximate token counts for metadata
                prompt_tokens = len(tiktoken.encoding_for_model(self._model_name).encode(prompt))
                self._streamed_completion_tokens = 0
                self._streamed_total_tokens = prompt_tokens

                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        content += chunk_content
                        # Roughly estimate tokens for metadata
                        self._streamed_completion_tokens += len(chunk_content) // 4
                        self._streamed_total_tokens = prompt_tokens + self._streamed_completion_tokens

                # Create a simulated response object to maintain compatibility
                # with the non-streaming code path
                # Include all required fields as per the error message
                self._last_response = ChatCompletion(
                    id="simulated-from-stream",
                    choices=[{
                        "message": {
                            "content": content,
                            "role": "assistant",  # Required field
                        },
                        "finish_reason": "stop",  # Required field
                        "index": 0,               # Required field
                    }],
                    created=0,
                    model=self._model_name,
                    object="chat.completion",
                )
                return content
            else:
                # Standard non-streaming request
                # Prepare parameters, removing any None values
                params = self.sanitize_params(
                    {
                        "model": self._model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stop": stop_sequences,
                    },
                    **kwargs,
                )
                
                response = await self._client.chat.completions.create(**params)

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
            
    async def stream_generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream a response from the model one chunk at a time.

        Args:
            prompt: The input prompt to send to the model
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that will stop generation when encountered
            **kwargs: Additional model-specific parameters

        Yields:
            Text chunks as they become available
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

        # Track approximate token counts for metadata
        prompt_tokens = len(prompt) // 4  # Very rough estimate
        self._streamed_completion_tokens = 0
        self._streamed_total_tokens = prompt_tokens
        
        # Create a streaming request
        try:
            # Prepare parameters, removing any None values
            params = self.sanitize_params(
                {
                    "model": self._model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": stop_sequences,
                    "stream": True,
                },
                **kwargs,
            )
            
            stream = await self._client.chat.completions.create(**params)

            full_text = ""
            try:
                async for chunk in stream:
                    # Check if delta exists and has content
                    if (hasattr(chunk, 'choices') and chunk.choices and 
                        hasattr(chunk.choices[0], 'delta') and 
                        hasattr(chunk.choices[0].delta, 'content') and
                        chunk.choices[0].delta.content):
                        
                        content = chunk.choices[0].delta.content
                        full_text += content
                        # Roughly estimate tokens for metadata
                        self._streamed_completion_tokens += len(content) // 4
                        self._streamed_total_tokens = prompt_tokens + self._streamed_completion_tokens
                        yield content
            except Exception as streaming_error:
                # If we had any content before the error, yield what we have so far
                if full_text:
                    # Log the error but continue with what we have
                    print(f"Streaming error occurred but continuing with partial content: {streaming_error}")
                else:
                    # If we have no content, raise the error to be caught by the outer try/except
                    raise streaming_error

            # Create a simulated response object for metadata
            # Include all required fields as per the error message
            self._last_response = ChatCompletion(
                id="simulated-from-stream",
                choices=[{
                    "message": {
                        "content": full_text,
                        "role": "assistant",  # Required field
                    },
                    "finish_reason": "stop",  # Required field
                    "index": 0,               # Required field
                }],
                created=0,
                model=self._model_name,
                object="chat.completion",
            )
            
            # If nothing was yielded (empty response), yield an empty string
            if not full_text:
                yield ""
                
        except Exception as e:
            # If there's an error during streaming, yield the error message
            # and then stop streaming
            error_msg = f"Streaming error: {str(e)}"
            # Just yield an empty string instead of an error message that would be displayed to the user
            yield error_msg
            # Log the actual error for debugging
            print(f"GLHF Streaming error: {str(e)}")
            return

    def get_response_metadata(self) -> Dict[str, Any]:
        """Return metadata from the most recent response.

        Returns:
            A dictionary containing metadata like token usage, model specifics, etc.
        """
        # If we have a regular (non-streamed) response with usage data
        if (
            self._last_response
            and hasattr(self._last_response, "usage")
            and self._last_response.usage
        ):
            return {
                "usage": {
                    "prompt_tokens": self._last_response.usage.prompt_tokens,
                    "completion_tokens": self._last_response.usage.completion_tokens,
                    "total_tokens": self._last_response.usage.total_tokens,
                },
                "model": self._last_response.model,
                "id": self._last_response.id,
            }
        
        # For streamed responses or responses without usage data, provide estimated counts
        if self._streamed_completion_tokens > 0:
            return {
                "usage": {
                    "prompt_tokens": self._streamed_total_tokens - self._streamed_completion_tokens,
                    "completion_tokens": self._streamed_completion_tokens,
                    "total_tokens": self._streamed_total_tokens,
                },
                "model": self._model_name,
                "streamed": True,
            }
            
        # Fallback for other cases
        return {
            "model": self._model_name,
            "note": "Limited metadata available - possibly from streaming response",
        }
