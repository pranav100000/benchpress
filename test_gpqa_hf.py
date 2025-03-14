"""Test script for GPQA Hugging Face dataset integration."""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env
load_dotenv()

# Login to Hugging Face
hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
if hf_token:
    print("Logging in to Hugging Face Hub with API token...")
    login(token=hf_token)
else:
    print("WARNING: No HUGGINGFACE_API_TOKEN found in .env - may not be able to access gated datasets")

# Add the src directory to the path to import benchpress modules
sys.path.insert(0, str(Path(__file__).parent))

from src.benchpress.datasets.gpqa_hf_dataset import GpqaHfDataset
from src.benchpress.tasks.gpqa import GpqaTask


async def test_gpqa_hf_dataset(dataset_name="Idavidrein/gpqa"):
    """Test loading GPQA examples from Hugging Face."""
    print(f"Testing GPQA HF dataset from {dataset_name}...")
    
    # Create the dataset - don't specify config_name for CSV datasets
    dataset = GpqaHfDataset(
        dataset_name=dataset_name,
        config_name=None  # Let Hugging Face discover the config
    )
    
    # Load examples
    examples = await dataset.load()
    
    # Print stats
    print(f"Loaded {len(examples)} examples")
    
    # Print a few examples
    print("\nSample examples:")
    for i, example in enumerate(examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"ID: {example.id}")
        print(f"Question: {example.question[:100]}...")
        print(f"Answer: {example.answer[:100]}...")
        print(f"Subject: {example.subject}")
        print(f"Difficulty: {example.difficulty}")
        print(f"Metadata: {example.metadata}")
    
    return examples


async def test_gpqa_task(dataset_name="Idavidrein/gpqa"):
    """Test the GPQA task with Hugging Face dataset."""
    print(f"\nTesting GPQA task with HF dataset from {dataset_name}...")
    
    # Create the task - don't specify config_name
    task = GpqaTask(
        dataset_source="huggingface",
        hf_dataset_name=dataset_name,
        hf_config_name=None  # Let Hugging Face discover the config
    )
    
    # Load examples
    examples = await task.load_examples()
    
    # Print stats
    print(f"Loaded {len(examples)} examples via the task")
    
    # Test extraction on a sample example
    if examples:
        example = examples[0]
        print(f"\nTesting extraction on example: {example.id}")
        
        # Create a sample model output
        model_output = f"""
        I've analyzed the problem carefully and arrived at a solution.
        
        The answer is: {example.answer}
        """
        
        # Evaluate
        result = await task.evaluate_example(example, model_output)
        
        # Print result
        print(f"Extraction successful: {result.correct}")
        print(f"Extracted answer: {result.metadata['extracted_answer']}")
        print(f"Extraction confidence: {result.metadata['extraction_confidence']}")
        print(f"Extraction method: {result.metadata['extraction_method']}")
        
        if "alternative_answers" in result.metadata:
            print(f"Alternative answers: {result.metadata['alternative_answers']}")
    
    return examples


async def main():
    """Run the test script."""
    dataset_name = "Idavidrein/gpqa"
    
    # Test dataset
    examples = await test_gpqa_hf_dataset(dataset_name)
    
    # Test task
    task_examples = await test_gpqa_task(dataset_name)
    
    # Verify they are equivalent
    if len(examples) == len(task_examples):
        print("\nVerification successful: Dataset and task return the same number of examples")
    else:
        print(f"\nVerification failed: Dataset returned {len(examples)} examples, but task returned {len(task_examples)}")


if __name__ == "__main__":
    asyncio.run(main())