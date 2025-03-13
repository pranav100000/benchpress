"""GPQA Hugging Face dataset implementation for benchpress."""

from typing import Dict, Any, Optional

from ..examples.gpqa import GpqaExample
from .huggingface_dataset import HuggingFaceDataset


def gpqa_hf_mapper(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map a GPQA HF dataset item to GpqaExample parameters.
    
    Args:
        item: An item from the GPQA HF dataset
        
    Returns:
        Dictionary of parameters for GpqaExample constructor
    """
    # Extract fields from the HF dataset
    question = item.get("question", "")
    answer = item.get("reference_answer", "")  # GPQA uses 'reference_answer'
    subject = item.get("subject", "")
    
    # Additional fields specific to GPQA
    primary_category = item.get("primary_category", "")
    secondary_category = item.get("secondary_category", "")
    
    # GPQA doesn't have explicit difficulty, use graduate level as default
    difficulty = "graduate"
    
    # Example ID from HF dataset
    example_id = item.get("id", "")
    if not example_id:
        # Fallback if no ID is provided
        example_id = f"gpqa_{hash(question)}"
    
    return {
        "id": f"gpqa_hf_{example_id}",
        "question": question,
        "answer": answer,
        "subject": subject,
        "difficulty": difficulty,
        "metadata": {
            "primary_category": primary_category,
            "secondary_category": secondary_category,
            "source": "huggingface/gpqa",
            "hf_id": example_id,
        }
    }


class GpqaHfDataset(HuggingFaceDataset[GpqaExample]):
    """GPQA HuggingFace dataset implementation."""
    
    def __init__(
        self, 
        version: str = "default", 
        data_path: Optional[str] = None,
        dataset_name: str = "openai/gpqa",  # Support different variants
        config_name: Optional[str] = None,  # For diamond or other configs
        split: str = "test",  # GPQA only has a 'test' split
        use_auth_token: Optional[bool] = None,
    ):
        """Initialize the GPQA HuggingFace dataset.
        
        Args:
            version: Dataset version (default is 'default')
            data_path: Override path to cache the dataset
            dataset_name: HF dataset name to use (default is 'openai/gpqa')
            config_name: Dataset configuration if applicable (e.g., 'diamond')
            split: Dataset split to use (default is 'test')
            use_auth_token: Whether to use the HF auth token for access
        """
        super().__init__(
            name="gpqa_hf",
            example_class=GpqaExample,
            dataset_name=dataset_name,
            config_name=config_name,
            mapper=gpqa_hf_mapper,
            split=split,
            version=version,
            data_path=data_path,
            use_auth_token=use_auth_token
        )