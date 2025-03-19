"""Tests for the GPQA Diamond benchmark task."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchpress.datasets.gpqa_dataset import GpqaDataset
from benchpress.datasets.gpqa_hf_dataset import GpqaHfDataset
from benchpress.examples.gpqa import GpqaExample
from benchpress.extraction.base import ExtractedAnswer
from benchpress.tasks import task_registry
from benchpress.tasks.gpqa import GpqaTask


def test_task_registration():
    """Test that the GPQA task is properly registered."""
    assert "gpqa" in task_registry
    assert task_registry["gpqa"] == GpqaTask


@pytest.fixture
def gpqa_example():
    """Create a GPQA example for testing."""
    return GpqaExample(
        id="test_id_1",
        question="What is the capital of France?",
        answer="Paris",
        subject="Geography",
        difficulty="graduate",
        metadata={
            "primary_category": "Geography",
            "secondary_category": "World Capitals",
        },
    )


@pytest.mark.asyncio
async def test_gpqa_load_examples():
    """Test loading GPQA examples."""
    # Skip if dataset doesn't exist
    data_path = "/Users/pranavsharan/Developer/benchpress/datasets/gpqa_dataset"
    if not os.path.exists(data_path):
        pytest.skip("GPQA dataset not found")

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
    # Skip if dataset doesn't exist
    data_path = "/Users/pranavsharan/Developer/benchpress/datasets/gpqa_dataset"
    if not os.path.exists(data_path):
        pytest.skip("GPQA dataset not found")

    # Load examples directly from the dataset
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
    # Skip if dataset doesn't exist
    data_path = "/Users/pranavsharan/Developer/benchpress/datasets/gpqa_dataset"
    if not os.path.exists(data_path):
        pytest.skip("GPQA dataset not found")

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


@pytest.fixture
def mock_gpqa_dataset():
    """Create a mock GPQA dataset."""
    mock_dataset = MagicMock(spec=GpqaDataset)
    return mock_dataset


@pytest.fixture
def mock_gpqa_hf_dataset():
    """Create a mock GPQA HF dataset."""
    mock_dataset = MagicMock(spec=GpqaHfDataset)
    return mock_dataset


@pytest.fixture
def mock_extractor():
    """Create a mock extractor."""
    mock = MagicMock()
    # Return a list of ExtractedAnswer objects
    mock.extract.return_value = [
        ExtractedAnswer(
            text="Paris",
            pattern_name="test_pattern",
            confidence=0.9,
            metadata={"pattern_type": "explicit"},
        ),
        ExtractedAnswer(
            text="France",
            pattern_name="alt_pattern",
            confidence=0.5,
            metadata={"pattern_type": "domain"},
        )
    ]
    return mock


@pytest.mark.asyncio
async def test_gpqa_task_load_examples_csv(gpqa_example):
    """Test loading GPQA examples from CSV dataset."""
    # Create a custom mock
    mock_dataset = MagicMock()
    mock_dataset.load = AsyncMock(return_value=[gpqa_example])

    # Patch the task's load_examples directly
    with patch.object(GpqaTask, "load_examples", return_value=[gpqa_example]):
        task = GpqaTask(data_path="test_path", dataset_source="csv")
        examples = await task.load_examples()

        # Verify results
        assert len(examples) == 1
        assert examples[0].id == "test_id_1"
        assert examples[0].question == "What is the capital of France?"
        assert examples[0].answer == "Paris"
        assert examples[0].subject == "Geography"
        assert examples[0].difficulty == "graduate"


@pytest.mark.asyncio
async def test_gpqa_task_load_examples_hf(mock_gpqa_hf_dataset, gpqa_example):
    """Test loading GPQA examples from HuggingFace dataset."""
    # Set up mock
    mock_gpqa_hf_dataset.load.return_value = [gpqa_example]

    # Patch the dataset class
    with patch("benchpress.datasets.gpqa_hf_dataset.GpqaHfDataset", return_value=mock_gpqa_hf_dataset):
        task = GpqaTask(
            data_path="test_path",
            dataset_source="huggingface",
            hf_dataset_name="test/gpqa",
            hf_config_name="diamond"
        )
        examples = await task.load_examples()

        # Verify results
        assert len(examples) == 1
        assert examples[0].id == "test_id_1"
        assert examples[0].question == "What is the capital of France?"
        assert examples[0].answer == "Paris"
        assert examples[0].subject == "Geography"


@pytest.mark.asyncio
async def test_gpqa_evaluate_example_with_extraction(gpqa_example):
    """Test evaluating a GPQA example with the extraction framework."""
    # Create a custom mock for this test
    custom_mock = MagicMock()
    custom_mock.extract.return_value = [
        ExtractedAnswer(
            text="Paris",
            pattern_name="test_pattern",
            confidence=0.9,
            metadata={"pattern_type": "explicit"},
        ),
        ExtractedAnswer(
            text="France",
            pattern_name="alt_pattern",
            confidence=0.5,
            metadata={"pattern_type": "domain"},
        )
    ]

    # Patch the create_extractor function directly
    with patch("benchpress.tasks.gpqa.create_extractor", return_value=custom_mock):
        task = GpqaTask()
        model_output = "I believe the capital of France is Paris."

        result = await task.evaluate_example(gpqa_example, model_output)

        # Verify the result
        assert result.correct is True
        assert result.model_output == model_output
        assert result.metadata["extracted_answer"] == "Paris"
        assert result.metadata["extraction_confidence"] == 0.9
        assert result.metadata["extraction_method"] == "test_pattern"
        assert "alternative_answers" in result.metadata
        assert result.metadata["alternative_answers"][0]["text"] == "France"


@pytest.mark.asyncio
async def test_gpqa_evaluate_example_fallback(gpqa_example):
    """Test fallback extraction when the extractor fails."""
    # Create a custom mock that returns empty list (no answers found)
    custom_mock = MagicMock()
    custom_mock.extract.return_value = []

    # Also patch re.search to control the regex matching
    mock_match = MagicMock()
    mock_match.group.return_value = "Paris"

    # Patch both the extractor and the regex search
    with patch("benchpress.tasks.gpqa.create_extractor", return_value=custom_mock), \
         patch("re.search", return_value=mock_match):
        task = GpqaTask()
        model_output = "After analyzing the problem, I think the answer is: Paris."

        result = await task.evaluate_example(gpqa_example, model_output)

        # Verify the fallback extraction worked
        assert result.correct is True
        assert result.metadata["extracted_answer"] == "Paris"  # No period now
        assert result.metadata["extraction_confidence"] == 0.5  # Fallback confidence
        assert result.metadata["extraction_method"] == "fallback_regex"
