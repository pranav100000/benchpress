"""
Standalone script to explore Hugging Face's datasets library with MATH-500.

This script tests basic functionality of the datasets library, 
without any integration with the benchpress system.
"""

import time
import json
from pathlib import Path
import datasets
from datasets import load_dataset


def print_separator(title):
    """Print a section separator with title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def main():
    """Explore the MATH-500 dataset."""
    print_separator("DATASETS LIBRARY INFO")
    print(f"Datasets library version: {datasets.__version__}")
    print(f"Default cache directory: {datasets.config.HF_DATASETS_CACHE}")

    # Time the dataset loading
    print_separator("LOADING DATASET")
    start_time = time.time()
    
    print("Loading MATH-500 dataset...")
    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Print dataset info
    print_separator("DATASET INFO")
    print(f"Dataset: {dataset}")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Get the actual split (should be 'test' for MATH-500)
    split_name = list(dataset.keys())[0]
    split = dataset[split_name]
    
    print(f"Split '{split_name}' size: {len(split)} examples")
    print(f"Features: {split.features}")
    
    # Print a sample example
    print_separator("SAMPLE EXAMPLE")
    example = split[0]
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}... (truncated)")
        else:
            print(f"{key}: {value}")
    
    # Save a few examples to a file for inspection
    print_separator("SAVING SAMPLES")
    samples_file = Path("math500_samples.json")
    samples = [split[i] for i in range(min(5, len(split)))]
    
    # Convert examples to serializable format
    serializable_samples = []
    for sample in samples:
        serializable_sample = {}
        for key, value in sample.items():
            # Convert any non-serializable values to strings
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable_sample[key] = value
            else:
                serializable_sample[key] = str(value)
        serializable_samples.append(serializable_sample)
    
    with open(samples_file, "w") as f:
        json.dump(serializable_samples, f, indent=2)
    print(f"Saved {len(serializable_samples)} sample examples to {samples_file}")
    
    # Test filtering capabilities
    print_separator("FILTERING")
    # This assumes the dataset has fields like 'level' or 'subject'
    # Modify based on actual structure
    if 'level' in split.features:
        levels = split.unique('level')
        print(f"Available levels: {levels}")
        
        # Count examples per level
        for level in levels:
            count = len(split.filter(lambda x: x['level'] == level))
            print(f"Level '{level}': {count} examples")
    
    if 'subject' in split.features:
        subjects = split.unique('subject')
        print(f"Available subjects: {subjects}")
        
        # Count examples per subject
        for subject in subjects:
            count = len(split.filter(lambda x: x['subject'] == subject))
            print(f"Subject '{subject}': {count} examples")
    
    # Test timing for random access
    print_separator("RANDOM ACCESS TIMING")
    iterations = 10
    start_time = time.time()
    for _ in range(iterations):
        # Access a random example
        _ = split[len(split) // 2]
    avg_time = (time.time() - start_time) / iterations
    print(f"Average random access time: {avg_time:.5f} seconds")
    
    # Test batch loading
    print_separator("BATCH LOADING")
    batch_size = 100
    start_time = time.time()
    batch = split[:batch_size]
    print(f"Loaded batch of {batch_size} examples in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()