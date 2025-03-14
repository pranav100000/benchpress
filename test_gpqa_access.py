"""Test script to access the GPQA dataset on Hugging Face."""

import os
from dotenv import load_dotenv
from huggingface_hub import login, list_datasets
from datasets import load_dataset, get_dataset_config_names

# Load environment variables from .env
load_dotenv()

# Get API token from environment
hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
if not hf_token:
    print("ERROR: No HUGGINGFACE_API_TOKEN found in .env")
    print("Please add your Hugging Face API token to your .env file")
    exit(1)

# Login to Hugging Face Hub with token
print(f"Logging in to Hugging Face Hub with API token...")
login(token=hf_token)

# Dataset we want to access
DATASET_NAME = "Idavidrein/gpqa"

# Try to directly access the dataset
print(f"\nAttempting to access dataset: {DATASET_NAME}")

try:
    # Get available configs
    configs = get_dataset_config_names(DATASET_NAME, token=hf_token)
    print(f"Available configurations: {configs}")
except Exception as e:
    print(f"Error getting configurations: {e}")
    configs = [None]  # Try with no config if we can't get configs

# Try each configuration
for config in configs:
    print(f"\nTrying to load with config: {config}")
    try:
        # Load dataset
        dataset = load_dataset(
            DATASET_NAME,
            name=config,
            token=hf_token
        )
        
        print(f"SUCCESS! Loaded dataset with configuration: {config}")
        print(f"Dataset structure: {dataset}")
        
        # Check what splits are available and their sizes
        for split_name, split_data in dataset.items():
            print(f"\nSplit {split_name}: {len(split_data)} examples")
            
            # Print info about first example
            if len(split_data) > 0:
                example = split_data[0]
                print(f"Features available: {list(example.keys())}")
                
                # Display a sample question/answer if available
                if 'question' in example:
                    print(f"\nSample question: {example['question'][:100]}...")
                
                if 'reference_answer' in example:
                    print(f"Sample answer: {example['reference_answer'][:100]}...")
                elif 'answer' in example:
                    print(f"Sample answer: {example['answer'][:100]}...")
        
    except Exception as e:
        print(f"Error loading dataset with config {config}: {e}")

print("\nTest complete.")