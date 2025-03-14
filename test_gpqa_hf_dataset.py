"""Test script for the GPQA HF Dataset implementation."""

import asyncio
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

# Load environment variables from .env
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.abspath("."))

# Import our GPQA implementation
from src.benchpress.datasets.gpqa_hf_dataset import GpqaHfDataset
from src.benchpress.tasks.gpqa import GpqaTask
from src.benchpress.utils import get_hf_token

# Get Hugging Face token from environment
hf_token = get_hf_token()
if not hf_token:
    print("No Hugging Face API token found in environment variables.")
    print("Please set HUGGINGFACE_API_TOKEN or HF_TOKEN.")
    sys.exit(1)

# Login to Hugging Face
print("Logging in to Hugging Face...")
login(token=hf_token)


async def test_gpqa_hf_dataset():
    """Test the GPQA HF dataset implementation."""
    print("\nTesting GpqaHfDataset...")
    
    # Create dataset
    dataset = GpqaHfDataset(
        dataset_name="Idavidrein/gpqa",
        config_name="gpqa_diamond",
        token=hf_token,
    )
    
    # Load examples
    print("Loading examples...")
    examples = await dataset.load()
    
    print(f"Loaded {len(examples)} examples")
    
    # Print a few examples
    for i, example in enumerate(examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"ID: {example.id}")
        print(f"Question: {example.question[:100]}...")
        print(f"Answer: {example.answer[:100]}...")
        print(f"Subject: {example.subject}")
        print(f"Subdomain: {example.metadata.get('subdomain', 'N/A')}")
        print(f"Difficulty: {example.difficulty}")
    
    return examples


async def test_gpqa_task():
    """Test the GPQA task with HF dataset."""
    print("\nTesting GpqaTask with HF dataset...")
    
    # Create task
    task = GpqaTask(
        dataset_source="huggingface",
        hf_dataset_name="Idavidrein/gpqa",
        hf_config_name="gpqa_diamond",
        hf_token=hf_token,
    )
    
    # Load examples
    print("Loading examples via task...")
    examples = await task.load_examples()
    
    print(f"Loaded {len(examples)} examples via task")
    
    # Test extraction on an example
    if examples:
        example = examples[0]
        print(f"\nTesting answer extraction on example: {example.id}")
        
        # Create a model output that embeds the answer
        model_output = f"""
        After analyzing the question carefully, I can determine the following:
        
        The answer is: {example.answer}
        
        This is based on the principles discussed in the question.
        """
        
        # Evaluate
        result = await task.evaluate_example(example, model_output)
        
        # Print results
        print(f"Extraction successful: {result.correct}")
        print(f"Extracted answer: '{result.metadata['extracted_answer']}'")
        print(f"Expected answer: '{example.answer}'")
        print(f"Extraction confidence: {result.metadata['extraction_confidence']}")
        print(f"Extraction method: {result.metadata['extraction_method']}")
        
        if "alternative_answers" in result.metadata:
            print("\nAlternative answers:")
            for alt in result.metadata["alternative_answers"]:
                print(f"- '{alt['text']}' (confidence: {alt['confidence']})")


async def main():
    """Run tests."""
    # Test HF dataset implementation
    examples = await test_gpqa_hf_dataset()
    
    # Test task implementation
    await test_gpqa_task()
    
    print("\nTests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())