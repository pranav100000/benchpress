"""New dataset management for benchpress."""

from .base import Dataset, DatasetRegistry

# Create the registry instance
dataset_registry = DatasetRegistry()

# Import specific dataset types
from .csv_dataset import CsvDataset
from .json_dataset import JsonDataset
from .huggingface_dataset import HuggingFaceDataset

# Import specific datasets
from .gpqa_dataset import GpqaDataset
from .math500_hf_dataset import Math500HfDataset
from .aime24_hf_dataset import Aime24HfDataset

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