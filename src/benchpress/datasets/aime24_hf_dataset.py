"""AIME24 Hugging Face dataset implementation for benchpress."""

import re
from typing import Dict, Any, Optional, Union, cast

from ..examples.aime24 import Aime24Example
from .huggingface_dataset import HuggingFaceDataset
from .base import DatasetRegistry


def aime24_hf_mapper(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map an AIME24 HF dataset item to Aime24Example parameters.
    
    Args:
        item: An item from the AI-MO/aimo-validation-aime dataset
        
    Returns:
        Dictionary of parameters for Aime24Example constructor
    """
    # Extract fields from the HF dataset
    problem_id = item.get("id", 0)
    question = item.get("problem", "")
    answer = item.get("answer", "")
    solution = item.get("solution", "")
    url = item.get("url", "")
    
    # Try to extract year and problem number from URL if available
    year = None
    problem_number = None
    
    if url:
        # URLs typically look like: https://artofproblemsolving.com/wiki/index.php/2022_AIME_I_Problems/Problem_1
        year_match = re.search(r'(\d{4})_AIME', url)
        if year_match:
            year = int(year_match.group(1))
            
        problem_match = re.search(r'Problem_(\d+)', url)
        if problem_match:
            problem_number = int(problem_match.group(1))
    
    # Create a unique ID
    unique_id = f"aime24_hf_{problem_id}"
    if year is not None:
        unique_id = f"aime24_hf_{year}_{problem_number}"
    
    return {
        "id": unique_id,
        "question": question,
        "answer": answer,
        "year": year,
        "problem_number": problem_number,
        "metadata": {
            "solution": solution,
            "url": url,
            "source": "huggingface/AI-MO/aimo-validation-aime",
        }
    }


class Aime24HfDataset(HuggingFaceDataset[Aime24Example]):
    """AIME24 HuggingFace dataset implementation."""
    
    def __init__(
        self, 
        version: str = "default", 
        data_path: Optional[str] = None,
        split: str = "train",  # This dataset only has a 'train' split
    ):
        """Initialize the AIME24 HuggingFace dataset.
        
        Args:
            version: Dataset version (default is 'default')
            data_path: Override path to cache the dataset
            split: Dataset split to use (default is 'train')
        """
        super().__init__(
            name="aime24_hf",
            example_class=Aime24Example,
            dataset_name="AI-MO/aimo-validation-aime",
            mapper=aime24_hf_mapper,
            split=split,
            version=version,
            data_path=data_path
        )


# Registration happens in __init__.py