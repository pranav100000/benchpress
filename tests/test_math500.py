"""Tests for the MATH-500 benchmark task."""

import pytest

from benchpress.tasks.math500 import Math500Task


@pytest.mark.asyncio
async def test_math500_load_examples():
    """Test loading examples from the MATH-500 task."""
    task = Math500Task()
    examples = await task.load_examples()

    # Check that examples were loaded
    assert len(examples) > 0

    # Check structure of examples
    for example in examples:
        assert example.id is not None
        assert example.question is not None
        assert example.answer is not None
        assert example.category is not None
        assert example.difficulty is not None


@pytest.mark.asyncio
async def test_math500_evaluate_correct_answer():
    """Test evaluating a correct answer for MATH-500."""
    task = Math500Task()
    examples = await task.load_examples()
    example = examples[0]

    # Test with exact match
    model_output = f"After computing the steps, I get {example.answer}."
    result = await task.evaluate_example(example, model_output)

    assert result.correct is True
    assert result.example_id == example.id
    assert result.metadata["extracted_answer"] == example.answer


@pytest.mark.asyncio
async def test_math500_evaluate_incorrect_answer():
    """Test evaluating an incorrect answer for MATH-500."""
    task = Math500Task()
    examples = await task.load_examples()
    example = examples[0]

    # Test with wrong answer
    model_output = "After computing the steps, I get 999."
    result = await task.evaluate_example(example, model_output)

    assert result.correct is False
    assert result.example_id == example.id
    assert result.metadata["extracted_answer"] == "999"
