"""Tests for the Hugging Face dataset integration."""


import pytest
from benchpress.datasets import Math500HfDataset, dataset_registry
from benchpress.examples.math500 import Math500Example
from benchpress.tasks.math500 import Math500Task


@pytest.mark.asyncio
async def test_math500_hf_dataset_registry():
    """Test that the MATH-500 HF dataset is properly registered."""
    assert "math500_hf" in dataset_registry
    dataset_class = dataset_registry.get("math500_hf")
    assert dataset_class == Math500HfDataset


@pytest.mark.asyncio
async def test_math500_hf_dataset_loading():
    """Test loading the MATH-500 dataset from HuggingFace."""
    # Skip if datasets library is not available
    pytest.importorskip("datasets")

    # Test with a small limit
    limit = 2
    dataset = Math500HfDataset()

    # Sample a few examples for faster testing
    examples = await dataset.sample(limit)

    # Check we have examples
    assert len(examples) == limit

    # Check the data structure
    for example in examples:
        assert isinstance(example, Math500Example)
        # Check for HF dataset ID format
        assert example.id.startswith("math500_hf_")
        # Check for required fields
        assert hasattr(example, "category")
        assert hasattr(example, "difficulty")


@pytest.mark.asyncio
async def test_math500_task_with_hf_dataset():
    """Test that the MATH-500 task properly uses the HF dataset."""
    # Skip if datasets library is not available
    pytest.importorskip("datasets")

    # Create task with a small limit
    task = Math500Task(limit=2)

    # Load examples
    examples = await task.load_examples()

    # Check we have examples
    assert len(examples) == 2

    # Check they're from the HF dataset
    for example in examples:
        assert isinstance(example, Math500Example)
        assert example.id.startswith("math500_hf_")


@pytest.mark.asyncio
async def test_math500_task_error_handling(monkeypatch):
    """Test the MATH-500 task error handling when dataset fails to load."""
    # Mock HF dataset to trigger an error
    async def mock_load_error(*args, **kwargs):
        raise Exception("Simulated HF dataset error")

    monkeypatch.setattr(Math500HfDataset, "load", mock_load_error)

    # Create the task
    task = Math500Task()

    # Since we removed the fallback, loading should fail
    with pytest.raises(RuntimeError) as excinfo:
        await task.load_examples()

    # Check the error message
    assert "Simulated HF dataset error" in str(excinfo.value)


# Skip the dataset test if we don't have the datasets library
pytestmark = pytest.mark.skipif(
    pytest.importorskip("datasets", reason="datasets library not available")
    is None,
    reason="datasets library not available",
)
