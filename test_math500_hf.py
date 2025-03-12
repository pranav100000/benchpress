"""Test script for MATH-500 Hugging Face integration."""

import asyncio
import time
from pathlib import Path

from src.benchpress.datasets.v2 import Math500HfDataset
from src.benchpress.tasks.math500 import Math500Task


async def test_math500_hf_dataset():
    """Test the MATH-500 HF dataset."""
    print("\n=== Testing Math500HfDataset ===")
    
    # Create the dataset
    dataset = Math500HfDataset()
    
    # Check dataset information
    print(f"Dataset name: {dataset.name}")
    print(f"Dataset HF name: {dataset.dataset_name}")
    print(f"Split: {dataset.split}")
    
    # Get the dataset size
    size = await dataset.get_size()
    print(f"Dataset size: {size} examples")
    
    # Get available subjects and levels
    subjects = await dataset.get_subjects()
    print(f"Subjects: {subjects}")
    
    levels = await dataset.get_levels()
    print(f"Levels: {levels}")
    
    # Examine the raw dataset format
    print("\n--- Examining raw dataset format ---")
    hf_dataset = await dataset._load_hf_dataset()
    
    # Print dataset info
    print(f"Dataset type: {type(hf_dataset)}")
    print(f"Features: {hf_dataset.features}")
    
    # Get the first example and print its raw format
    raw_example = hf_dataset[0]
    print(f"Raw example type: {type(raw_example)}")
    print(f"Raw example fields: {list(raw_example.keys()) if hasattr(raw_example, 'keys') else 'No keys method'}")
    
    # Try to convert to dict if it's not already
    if not isinstance(raw_example, dict):
        print("Converting raw example to dict...")
        if hasattr(raw_example, 'as_py'):
            raw_example = raw_example.as_py()
        elif hasattr(raw_example, '__dict__'):
            raw_example = raw_example.__dict__
    
    print(f"Raw example after conversion: {type(raw_example)}")
    print(f"Example fields: {list(raw_example.keys()) if hasattr(raw_example, 'keys') else 'No keys method'}")
    
    # Print a few values
    for key in list(raw_example.keys())[:3]:
        print(f"  {key}: {raw_example[key][:50]}..." if isinstance(raw_example[key], str) else f"  {key}: {raw_example[key]}")
    
    # Load a single example
    print("\n--- Loading a single example ---")
    start_time = time.time()
    example = await dataset.get_item(0)
    print(f"Loaded example in {time.time() - start_time:.2f} seconds")
    print(f"Example ID: {example.id}")
    print(f"Question: {example.question[:100]}... (truncated)")
    print(f"Answer: {example.answer}")
    print(f"Category: {example.category}")
    print(f"Difficulty: {example.difficulty}")
    
    # Load a small batch
    print("\n--- Loading a small batch ---")
    start_time = time.time()
    examples = await dataset.sample(5, seed=42)
    print(f"Loaded 5 examples in {time.time() - start_time:.2f} seconds")
    for i, ex in enumerate(examples):
        print(f"Example {i+1} ID: {ex.id}")
    
    # Test filtering
    if subjects:
        print(f"\n--- Testing filtering by subject: {subjects[0]} ---")
        subject_examples = await dataset.filter_by_subject(subjects[0])
        print(f"Found {len(subject_examples)} examples for subject '{subjects[0]}'")
    
    if levels:
        print(f"\n--- Testing filtering by level: {levels[0]} ---")
        level_examples = await dataset.filter_by_level(levels[0])
        print(f"Found {len(level_examples)} examples for level {levels[0]}")
    
    return True


async def test_math500_task():
    """Test the MATH-500 task with HF integration."""
    print("\n=== Testing Math500Task with HF integration ===")
    
    # Create the task
    task = Math500Task()
    
    # Load a limited number of examples for testing
    limit = 10
    print(f"Loading {limit} examples...")
    start_time = time.time()
    examples = await task.load_examples(limit=limit)
    print(f"Loaded {len(examples)} examples in {time.time() - start_time:.2f} seconds")
    
    # Check the first few examples
    print("\n--- Sample examples ---")
    for i, example in enumerate(examples[:3]):
        print(f"Example {i+1}:")
        print(f"  ID: {example.id}")
        print(f"  Question: {example.question[:50]}... (truncated)")
        print(f"  Answer: {example.answer}")
        print(f"  Category: {example.category}")
        print(f"  Difficulty: {example.difficulty}")
    
    return True


async def main():
    """Run the tests."""
    print("Testing MATH-500 Hugging Face integration...")
    
    try:
        # Test the HF dataset
        await test_math500_hf_dataset()
        
        # Test the Math500Task
        await test_math500_task()
        
        print("\n=== All tests passed! ===")
    except Exception as e:
        print(f"\n!!! Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())