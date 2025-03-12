"""Tests for the GPQA Diamond benchmark task."""

import pytest

from benchpress.tasks import task_registry
from benchpress.tasks.gpqa import GpqaExample, GpqaTask


def test_task_registration():
    """Test that the GPQA task is properly registered."""
    assert "gpqa" in task_registry
    assert task_registry["gpqa"] == GpqaTask


@pytest.mark.asyncio
async def test_gpqa_load_examples():
    """Test loading GPQA examples."""
    task = GpqaTask()
    examples = await task.load_examples()
    
    # Check that we have examples
    assert len(examples) > 0
    
    # Check that all examples are of the correct type
    for example in examples:
        assert isinstance(example, GpqaExample)
        assert example.id.startswith("gpqa_")
        assert hasattr(example, "subject")
        assert hasattr(example, "difficulty")


@pytest.mark.asyncio
async def test_gpqa_evaluate_example():
    """Test evaluating a GPQA example."""
    task = GpqaTask()
    examples = await task.load_examples()
    example = examples[0]  # Use the first example
    
    # Test with correct answer
    correct_output = f"After calculating the radius, I find that the answer is {example.answer}."
    result = await task.evaluate_example(example, correct_output)
    assert result.correct
    
    # Test with incorrect answer
    incorrect_output = "The answer is clearly 42."
    result = await task.evaluate_example(example, incorrect_output)
    assert not result.correct