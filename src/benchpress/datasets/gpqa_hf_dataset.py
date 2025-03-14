"""GPQA Hugging Face dataset implementation for benchpress."""

from typing import Dict, Any, Optional

from ..examples.gpqa import GpqaExample
from .huggingface_dataset import HuggingFaceDataset


def gpqa_hf_mapper(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map a GPQA HF dataset item to GpqaExample parameters.
    
    Maps fields from the Idavidrein/gpqa dataset (gpqa_diamond configuration)
    to our internal GpqaExample format.
    
    Args:
        item: An item from the GPQA HF dataset
        
    Returns:
        Dictionary of parameters for GpqaExample constructor
    """
    # Extract fields from the HF dataset (using Idavidrein/gpqa schema)
    question = item.get("Question", "")
    answer = item.get("Correct Answer", "")
    
    # Get domain and difficulty
    subject = item.get("High-level domain", "")
    difficulty = item.get("Writer's Difficulty Estimate", "graduate")
    
    # Get subdomain and record ID
    subdomain = item.get("Subdomain", "")
    record_id = item.get("Record ID", "")
    
    # Get incorrect answers for metadata
    incorrect_answers = [
        item.get("Incorrect Answer 1", ""),
        item.get("Incorrect Answer 2", ""),
        item.get("Incorrect Answer 3", "")
    ]
    
    # Get explanation
    explanation = item.get("Explanation", "")
    
    # Generate ID using record_id if available, otherwise use hash of question
    if record_id:
        example_id = f"gpqa_diamond_{record_id}"
    else:
        example_id = f"gpqa_diamond_{hash(question)}"
    
    return {
        "id": example_id,
        "question": question,
        "answer": answer,
        "subject": subject,
        "difficulty": difficulty,
        "metadata": {
            "subdomain": subdomain,
            "incorrect_answers": incorrect_answers,
            "explanation": explanation,
            "record_id": record_id,
            "source": "huggingface/gpqa_diamond",
        }
    }


class GpqaHfDataset(HuggingFaceDataset[GpqaExample]):
    """GPQA Diamond HuggingFace dataset implementation."""
    
    def __init__(
        self, 
        version: str = "default", 
        data_path: Optional[str] = None,
        dataset_name: str = "Idavidrein/gpqa",  # The correct HF dataset name
        config_name: str = "gpqa_diamond",      # The specific GPQA configuration
        split: str = "train",                   # GPQA Diamond uses 'train' split
        token: Optional[str] = None,            # HF API token for gated dataset access
    ):
        """Initialize the GPQA Diamond HuggingFace dataset.
        
        Args:
            version: Dataset version (default is 'default')
            data_path: Override path to cache the dataset
            dataset_name: HF dataset name (default is 'Idavidrein/gpqa')
            config_name: Dataset configuration (default is 'gpqa_diamond')
            split: Dataset split to use (default is 'train')
            token: HF API token for accessing gated datasets
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
            token=token,  # Use the token parameter for authentication
            use_auth_token=None  # Not needed with token parameter
        )