"""Explore the GPQA dataset from Hugging Face."""

from datasets import load_dataset
import json

# Load the GPQA dataset
dataset = load_dataset("openai/gpqa")

# Print dataset info
print("Dataset structure:")
print(dataset)

# Check the fields available in the first example
print("\nFields in the first example:")
print(list(dataset["test"][0].keys()))

# Print a few examples
print("\nSample questions:")
for i, example in enumerate(dataset["test"][:5]):
    print(f"\nExample {i+1}:")
    print(f"Question: {example['question']}")
    print(f"Answer: {example['reference_answer']}")
    print(f"Subject: {example['subject']}")
    print(f"Primary category: {example['primary_category']}")
    
# Count examples by subject
subjects = {}
for example in dataset["test"]:
    subject = example["subject"]
    if subject not in subjects:
        subjects[subject] = 0
    subjects[subject] += 1

print("\nNumber of examples by subject:")
for subject, count in subjects.items():
    print(f"{subject}: {count}")

# Save a sample to a file for reference
with open("gpqa_sample.json", "w") as f:
    json.dump(dataset["test"][:10], f, indent=2)
    
print("\nSample saved to gpqa_sample.json")