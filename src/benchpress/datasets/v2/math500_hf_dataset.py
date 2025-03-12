"""MATH-500 Hugging Face dataset implementation for benchpress."""

from typing import Dict, Any, Optional

from ...tasks.math500_example import Math500Example
from .huggingface_dataset import HuggingFaceDataset


def math500_hf_mapper(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map a MATH-500 HF dataset item to Math500Example parameters.
    
    Args:
        item: An item from the MATH-500 HF dataset
        
    Returns:
        Dictionary of parameters for Math500Example constructor
    """
    # Extract fields from the HF dataset
    question = item.get("problem", "")
    answer = item.get("answer", "")
    subject = item.get("subject", "")
    
    # Extract level (numeric 1-5)
    level = item.get("level", 3)  # Default to level 3 if missing
    unique_id = item.get("unique_id", "")
    
    # Translate numeric level to difficulty string
    difficulty_map = {
        1: "easy",
        2: "easy",
        3: "medium",
        4: "medium",
        5: "hard",
    }
    difficulty = difficulty_map.get(level, "medium")
    
    return {
        "id": f"math500_hf_{unique_id.replace('/', '_')}",
        "question": question,
        "answer": answer,
        "category": subject,
        "difficulty": difficulty,
        "metadata": {
            "level": level,
            "level_str": f"Level {level}",
            "solution": item.get("solution", ""),
            "unique_id": unique_id,
            "source": "huggingface/MATH-500",
        }
    }


class Math500HfDataset(HuggingFaceDataset[Math500Example]):
    """MATH-500 HuggingFace dataset implementation."""
    
    def __init__(
        self, 
        version: str = "default", 
        data_path: Optional[str] = None,
        split: str = "test",  # MATH-500 only has a 'test' split
    ):
        """Initialize the MATH-500 HuggingFace dataset.
        
        Args:
            version: Dataset version (default is 'default')
            data_path: Override path to cache the dataset
            split: Dataset split to use (default is 'test' for MATH-500)
        """
        super().__init__(
            name="math500_hf",
            example_class=Math500Example,
            dataset_name="HuggingFaceH4/MATH-500",
            mapper=math500_hf_mapper,
            split=split,
            version=version,
            data_path=data_path
        )