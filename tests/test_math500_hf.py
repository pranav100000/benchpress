"""Tests for MATH-500 Hugging Face dataset integration."""

import pytest
import os
import asyncio

from benchpress.datasets.math500_hf_dataset import Math500HfDataset
from benchpress.examples.math500 import Math500Example
from benchpress.tasks.math500 import Math500Task


@pytest.mark.asyncio
async def test_math500_hf_dataset_load():
    """Test loading the MATH-500 dataset from Hugging Face."""
    # Skip if datasets library is not available
    pytest.importorskip("datasets")
    
    # Create dataset
    dataset = Math500HfDataset()
    
    # Load examples
    examples = await dataset.load()
    
    # Check that we got examples
    assert len(examples) > 0
    
    # Check that the first example has the expected structure
    example = examples[0]
    assert example.id.startswith("math500_hf_")
    assert example.question
    assert example.answer
    assert example.category
    assert example.difficulty in ["easy", "medium", "hard"]
    
    # Check metadata
    assert "level" in example.metadata
    assert "solution" in example.metadata
    assert "unique_id" in example.metadata
    assert example.metadata["source"] == "huggingface/MATH-500"


@pytest.mark.asyncio
async def test_math500_task_with_hf_dataset():
    """Test that the MATH-500 task can load examples from the HF dataset."""
    # Skip if datasets library is not available
    pytest.importorskip("datasets")
    
    # Create task
    task = Math500Task()
    
    # Load examples
    examples = await task.load_examples()
    
    # Check that we got examples
    assert len(examples) > 0
    
    # All examples should be from the HF dataset
    assert all(ex.id.startswith("math500_hf_") for ex in examples)
    
    # Test with limit
    limited_task = Math500Task(limit=10)
    limited_examples = await limited_task.load_examples()
    assert len(limited_examples) <= 10


@pytest.mark.asyncio
async def test_math500_evaluation():
    """Test evaluation of MATH-500 examples."""
    # Skip if datasets library is not available
    pytest.importorskip("datasets")
    
    # Create task
    task = Math500Task(limit=1)
    
    # Load one example
    examples = await task.load_examples()
    example = examples[0]
    
    # Test correct answer with boxed format
    result = await task.evaluate_example(example, f"The answer is \\boxed{{{example.answer}}}")
    assert result.correct
    
    # Test incorrect answer
    result = await task.evaluate_example(example, "The answer is \\boxed{wrong}")
    assert not result.correct