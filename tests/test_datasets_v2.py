"""Tests for the v2 dataset module."""

import pytest
from pathlib import Path

from benchpress.datasets.v2 import (
    Dataset, 
    CsvDataset, 
    JsonDataset,
    dataset_registry,
    GpqaDataset
)
from benchpress.tasks.gpqa_example import GpqaExample


@pytest.mark.asyncio
async def test_gpqa_dataset():
    """Test the GPQA dataset."""
    # Use the specific dataset path
    data_path = "/Users/pranavsharan/Developer/benchpress/datasets/gpqa_dataset"
    dataset = GpqaDataset(data_path=data_path)
    
    # Check properties
    assert dataset.name == "gpqa"
    assert dataset.file_name == "gpqa_diamond.csv"
    
    # Load examples
    examples = await dataset.load()
    
    # Check that we have examples
    assert len(examples) > 0
    
    # Check that all examples are of the correct type
    for example in examples:
        assert isinstance(example, GpqaExample)
        assert example.id.startswith("gpqa_diamond_")
        assert hasattr(example, "subject")
        assert hasattr(example, "difficulty")


@pytest.mark.asyncio
async def test_dataset_sample():
    """Test the dataset sampling functionality."""
    # Use the specific dataset path
    data_path = "/Users/pranavsharan/Developer/benchpress/datasets/gpqa_dataset"
    dataset = GpqaDataset(data_path=data_path)
    
    # Sample 5 examples
    examples = await dataset.sample(5, seed=42)
    
    # Check that we have the requested number of examples
    assert len(examples) == 5
    
    # Sample with the same seed should give the same examples
    examples2 = await dataset.sample(5, seed=42)
    example_ids = [ex.id for ex in examples]
    example_ids2 = [ex.id for ex in examples2]
    assert example_ids == example_ids2


@pytest.mark.asyncio
async def test_dataset_filter():
    """Test the dataset filtering functionality."""
    # Use the specific dataset path
    data_path = "/Users/pranavsharan/Developer/benchpress/datasets/gpqa_dataset"
    dataset = GpqaDataset(data_path=data_path)
    
    # Get all examples
    all_examples = await dataset.load()
    
    # Filter for a specific subject if available
    if all_examples and hasattr(all_examples[0], "subject"):
        subject = all_examples[0].subject
        filtered = await dataset.filter(lambda ex: ex.subject == subject)
        assert len(filtered) > 0
        assert all(ex.subject == subject for ex in filtered)
    else:
        # If no subject field or no examples, just check the filter works
        filtered = await dataset.filter(lambda ex: True)
        assert len(filtered) == len(all_examples)