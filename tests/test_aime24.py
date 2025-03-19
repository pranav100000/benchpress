"""Tests for the AIME24 benchmark task."""

import pytest
from benchpress.tasks.aime24 import Aime24Task


@pytest.mark.asyncio
async def test_aime24_load_examples():
    """Test loading examples from the AIME24 task."""
    task = Aime24Task()
    examples = await task.load_examples()

    # Check that examples were loaded
    assert len(examples) > 0

    # Check structure of examples
    for example in examples:
        assert example.id is not None
        assert example.question is not None
        assert example.answer is not None
        assert example.year is not None
        assert example.problem_number is not None


@pytest.mark.asyncio
async def test_aime24_evaluate_correct_answer():
    """Test evaluating a correct answer for AIME24."""
    task = Aime24Task()
    examples = await task.load_examples()
    example = examples[0]

    # Test with exact match
    model_output = f"After calculations, the answer is {example.answer}."
    result = await task.evaluate_example(example, model_output)

    assert result.correct is True
    assert result.example_id == example.id
    assert result.metadata["extracted_answer"] == example.answer


@pytest.mark.asyncio
async def test_aime24_evaluate_incorrect_answer():
    """Test evaluating an incorrect answer for AIME24."""
    task = Aime24Task()
    examples = await task.load_examples()
    example = examples[0]

    # Test with wrong answer
    model_output = "The answer is 123."
    result = await task.evaluate_example(example, model_output)

    assert result.correct is False
    assert result.example_id == example.id
    assert result.metadata["extracted_answer"] == "123"
