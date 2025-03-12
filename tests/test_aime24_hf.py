"""Tests for AIME24 Hugging Face dataset integration."""

import pytest
import os
import asyncio
from datasets import load_dataset

from benchpress.datasets.v2.aime24_hf_dataset import Aime24HfDataset
from benchpress.tasks.aime24 import Aime24Task


@pytest.mark.asyncio
async def test_aime24_hf_dataset_load():
    """Test loading the AIME24 dataset from Hugging Face."""
    # Skip if datasets library is not available
    pytest.importorskip("datasets")
    
    # Create dataset
    dataset = Aime24HfDataset()
    
    # Load examples
    examples = await dataset.load()
    
    # Check that we got examples
    assert len(examples) > 0
    
    # Check that the first example has the expected structure
    example = examples[0]
    assert example.id.startswith("aime24_hf_")
    assert example.question
    assert example.answer
    
    # Check metadata
    assert "solution" in example.metadata
    assert "url" in example.metadata
    assert "source" in example.metadata
    assert example.metadata["source"] == "huggingface/AI-MO/aimo-validation-aime"
    
    # Check that we extracted year and problem number if available
    if "url" in example.metadata and example.metadata["url"]:
        if "year" in example.metadata:
            assert isinstance(example.year, int)
        if "problem_number" in example.metadata:
            assert isinstance(example.problem_number, int)


@pytest.mark.asyncio
async def test_aime24_task_with_hf_dataset():
    """Test that the AIME24 task can load examples from the HF dataset."""
    # Skip if datasets library is not available
    pytest.importorskip("datasets")
    
    # Create task
    task = Aime24Task()
    
    # Load examples
    examples = await task.load_examples()
    
    # Check that we got examples
    assert len(examples) > 0
    
    # All examples should be from the HF dataset
    assert all(ex.id.startswith("aime24_hf_") for ex in examples)
    
    # Test with limit
    limited_task = Aime24Task(limit=10)
    limited_examples = await limited_task.load_examples()
    assert len(limited_examples) <= 10


@pytest.mark.asyncio
async def test_aime24_evaluation():
    """Test evaluation of AIME24 examples."""
    # Skip if datasets library is not available
    pytest.importorskip("datasets")
    
    # Create task
    task = Aime24Task()
    
    # Load one example
    examples = await task.load_examples()
    example = examples[0]
    
    # Test correct answer
    result = await task.evaluate_example(example, f"The answer is {example.answer}")
    assert result.correct
    
    # Test incorrect answer
    result = await task.evaluate_example(example, "The answer is 999")
    assert not result.correct