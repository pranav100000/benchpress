"""Tests for the GPQA Diamond benchmark task."""

import os
import pytest

from benchpress.tasks import task_registry
from benchpress.tasks.gpqa import GpqaTask
from benchpress.tasks.gpqa_example import GpqaExample
from benchpress.datasets.v2 import GpqaDataset


def test_task_registration():
    """Test that the GPQA task is properly registered."""
    assert "gpqa" in task_registry
    assert task_registry["gpqa"] == GpqaTask


@pytest.mark.asyncio
async def test_gpqa_load_examples():
    """Test loading GPQA examples."""
    # Use the specific path to our dataset
    data_path = "/Users/pranavsharan/Developer/benchpress/datasets/gpqa_dataset"
    task = GpqaTask(data_path=data_path)
    examples = await task.load_examples()
    
    # Check that we have examples
    assert len(examples) > 0
    
    # Check that all examples are of the correct type
    for example in examples:
        assert isinstance(example, GpqaExample)
        assert example.id.startswith("gpqa_diamond_")
        assert hasattr(example, "subject")
        assert hasattr(example, "difficulty")


@pytest.mark.asyncio
async def test_gpqa_dataset_integration():
    """Test that the task uses the dataset correctly."""
    # Load examples directly from the dataset
    data_path = "/Users/pranavsharan/Developer/benchpress/datasets/gpqa_dataset"
    dataset = GpqaDataset(data_path=data_path)
    dataset_examples = await dataset.load()
    
    # Load examples from the task
    task = GpqaTask(data_path=data_path)
    task_examples = await task.load_examples()
    
    # Check that we get the same number of examples
    assert len(dataset_examples) == len(task_examples)
    
    # Check that the first few examples match
    # (Not comparing all to avoid long test times)
    for i in range(min(5, len(dataset_examples))):
        assert dataset_examples[i].id == task_examples[i].id
        assert dataset_examples[i].question == task_examples[i].question
        assert dataset_examples[i].answer == task_examples[i].answer


@pytest.mark.asyncio
async def test_gpqa_evaluate_example():
    """Test evaluating a GPQA example."""
    # Use the specific path to our dataset
    data_path = "/Users/pranavsharan/Developer/benchpress/datasets/gpqa_dataset"
    task = GpqaTask(data_path=data_path)
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