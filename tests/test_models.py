"""Tests for model implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchpress.models import GlhfModel


@patch.dict('os.environ', {}, clear=True)
def test_glhf_model_init():
    """Test that the GLHF model initializes correctly."""
    # Test with explicit hf: prefix
    model = GlhfModel(
        model_name="hf:mistralai/Mistral-7B-Instruct-v0.3",
        api_key="test_key",
        system_prompt="You are a helpful assistant."
    )
    assert model.model_id == "hf:mistralai/Mistral-7B-Instruct-v0.3"

    # Test without hf: prefix (should add it automatically)
    model = GlhfModel(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        api_key="test_key"
    )
    assert model.model_id == "hf:mistralai/Mistral-7B-Instruct-v0.3"

    # Test API key validation
    with pytest.raises(ValueError):
        GlhfModel(model_name="test", api_key=None)


@pytest.mark.asyncio
@patch.dict('os.environ', {}, clear=True)
@patch("openai.AsyncOpenAI")
async def test_glhf_model_generate(mock_async_openai):
    """Test that the GLHF model generate method works correctly."""
    # Setup mock
    mock_client = AsyncMock()
    mock_async_openai.return_value = mock_client

    # Mock the chat.completions.create method
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response

    # Create model and test generate
    model = GlhfModel(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        api_key="test_key",
        system_prompt="You are a helpful assistant."
    )

    response = await model.generate("Test prompt", temperature=0.7)

    # Verify response
    assert response == "Test response"

    # Verify API call was made correctly
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]

    # Check model name has hf: prefix
    assert call_args["model"] == "hf:mistralai/Mistral-7B-Instruct-v0.3"

    # Check messages include system and user messages
    assert len(call_args["messages"]) == 2
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][0]["content"] == "You are a helpful assistant."
    assert call_args["messages"][1]["role"] == "user"
    assert call_args["messages"][1]["content"] == "Test prompt"

    # Check temperature
    assert call_args["temperature"] == 0.7
