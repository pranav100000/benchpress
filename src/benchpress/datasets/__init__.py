"""New dataset management for benchpress."""

from .base import Dataset, DatasetRegistry

# Create the registry instance
dataset_registry = DatasetRegistry()

# Import specific dataset types
from .aime24_hf_dataset import Aime24HfDataset
from .csv_dataset import CsvDataset

# Import specific datasets
from .gpqa_dataset import GpqaDataset
from .huggingface_dataset import HuggingFaceDataset
from .json_dataset import JsonDataset
from .math500_hf_dataset import Math500HfDataset

# Register datasets
dataset_registry.register(GpqaDataset)
dataset_registry.register(Math500HfDataset)
dataset_registry.register(Aime24HfDataset)

__all__ = [
    "Dataset",
    "DatasetRegistry",
    "dataset_registry",
    "CsvDataset",
    "JsonDataset",
    "HuggingFaceDataset",
    "GpqaDataset",
    "Math500HfDataset",
    "Aime24HfDataset",
]
