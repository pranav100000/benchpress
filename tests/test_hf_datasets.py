"""Tests for the Hugging Face dataset integration."""

import pytest
from pathlib import Path

from benchpress.datasets.v2 import (
    dataset_registry,
    HuggingFaceDataset,
    Math500HfDataset
)
from benchpress.tasks.math500_example import Math500Example
from benchpress.tasks.math500 import Math500Task


@pytest.mark.asyncio
async def test_math500_hf_dataset_registry():
    """Test that the MATH-500 HF dataset is properly registered."""
    assert "math500_hf" in dataset_registry
    dataset_class = dataset_registry.get("math500_hf")
    assert dataset_class == Math500HfDataset


@pytest.mark.asyncio
async def test_math500_hf_dataset_properties():
    """Test the MATH-500 HF dataset properties."""
    dataset = Math500HfDataset()
    
    # Check properties
    assert dataset.name == "math500_hf"
    assert dataset.dataset_name == "HuggingFaceH4/MATH-500"


@pytest.mark.asyncio
@pytest.mark.skip(reason="Skip by default as it requires downloading the dataset")
async def test_math500_hf_dataset_load():
    """Test loading the MATH-500 HF dataset.
    
    Note: This test is skipped by default as it requires downloading the dataset.
    Remove the skip decorator to run this test.
    """
    dataset = Math500HfDataset()
    
    # Get size first (faster than loading all examples)
    size = await dataset.get_size()
    assert size > 0
    
    # Test loading a single example
    example = await dataset.get_item(0)
    assert isinstance(example, Math500Example)
    assert example.id.startswith("math500_hf_")
    assert hasattr(example, "category")
    assert hasattr(example, "difficulty")
    
    # Test loading a small batch of examples
    examples = await dataset.sample(5, seed=42)
    assert len(examples) == 5
    assert all(isinstance(ex, Math500Example) for ex in examples)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Skip by default as it requires downloading the dataset")
async def test_math500_task_with_hf():
    """Test the MATH-500 task with HuggingFace dataset.
    
    Note: This test is skipped by default as it requires downloading the dataset.
    Remove the skip decorator to run this test.
    """
    # Create the task
    task = Math500Task()
    
    # Load examples
    examples = await task.load_examples()
    
    # Check that we have examples
    assert len(examples) > 0
    
    # Check that all examples are of the correct type
    for example in examples:
        assert isinstance(example, Math500Example)
        assert example.id.startswith("math500_hf_")


@pytest.mark.asyncio
async def test_math500_task_fallback(monkeypatch):
    """Test the MATH-500 task fallback to sample examples."""
    # Mock HF dataset to trigger the fallback
    async def mock_load_error(*args, **kwargs):
        raise Exception("Simulated HF dataset error")
    
    monkeypatch.setattr(Math500HfDataset, "load", mock_load_error)
    
    # Create the task
    task = Math500Task()
    
    # Load examples
    examples = await task.load_examples()
    
    # Check that we have examples
    assert len(examples) > 0
    
    # Check that all examples are of the correct type
    for example in examples:
        assert isinstance(example, Math500Example)
        # These will have the sample IDs, not HF IDs
        assert example.id.startswith("math500_")
        assert hasattr(example, "category")
        assert hasattr(example, "difficulty")