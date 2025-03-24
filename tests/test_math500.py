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
    from benchpress.examples.math500 import Math500Example

    # Create a simple example with a clear answer
    example = Math500Example(
        id="test123",
        question="What is 2+2?",
        answer="4",
        category="Arithmetic",
        difficulty="easy"
    )

    task = Math500Task()

    # Test with exact match
    model_output = "The answer is \\boxed{4}"
    result = await task.evaluate_example(example, model_output)

    assert result.correct is True
    assert result.example_id == example.id
    assert "4" in result.metadata["extracted_answer"]


@pytest.mark.asyncio
async def test_math500_evaluate_incorrect_answer():
    """Test evaluating an incorrect answer for MATH-500."""
    from benchpress.examples.math500 import Math500Example

    # Create a simple example with a clear answer
    example = Math500Example(
        id="test123",
        question="What is 2+2?",
        answer="4",
        category="Arithmetic",
        difficulty="easy"
    )

    task = Math500Task()

    # Test with incorrect answer
    model_output = "After computing the steps, I get 5."
    result = await task.evaluate_example(example, model_output)

    assert result.correct is False
    assert result.example_id == example.id
    assert "5" in result.metadata["extracted_answer"]
