"""Dataset management for benchpress."""

from .base import Dataset, DatasetRegistry, register_dataset, dataset_registry
from .csv_dataset import CsvDataset
from .json_dataset import JsonDataset

# Import and register datasets
from .gpqa_dataset import GpqaDataset
dataset_registry.register(GpqaDataset)

__all__ = [
    "Dataset", 
    "DatasetRegistry", 
    "register_dataset", 
    "dataset_registry",
    "CsvDataset",
    "JsonDataset",
    "GpqaDataset"
]