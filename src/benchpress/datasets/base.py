"""Base dataset interface for benchpress."""

import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generic, List, Optional, Type, TypeVar

# Type variable for dataset example type - without binding to Example class
# to avoid circular imports
T = TypeVar("T")


class Dataset(Generic[T], ABC):
    """Abstract base class for all benchmark datasets."""

    def __init__(
        self,
        name: str,
        version: str = "default",
        data_path: Optional[str] = None,
    ):
        """Initialize a dataset.

        Args:
            name: The dataset name
            version: The dataset version
            data_path: Override path to the dataset files
        """
        self.name = name
        self.version = version

        # Determine the data path with the following priority:
        # 1. Explicitly provided data_path
        # 2. Task-specific environment variable BENCHPRESS_{TASK}_DATA_PATH
        # 3. General environment variable BENCHPRESS_DATA_PATH
        # 4. Default to ./datasets/<name>
        if data_path:
            self._data_path = data_path
        else:
            task_env_var = f"BENCHPRESS_{name.upper()}_DATA_PATH"
            self._data_path = os.environ.get(
                task_env_var,
                os.environ.get(
                    "BENCHPRESS_DATA_PATH",
                    str(Path("./datasets") / name)
                )
            )

        # Convert to Path object for easier manipulation
        self._data_path = Path(self._data_path)

        # If version is specified, append to path
        if version != "default":
            self._data_path = self._data_path / version

    @property
    def data_path(self) -> Path:
        """Get the dataset path.

        Returns:
            Path to the dataset directory
        """
        return self._data_path

    @abstractmethod
    async def load(self) -> List[T]:
        """Load all examples from the dataset.

        Returns:
            List of examples
        """
        pass

    async def sample(self, n: int, seed: Optional[int] = None) -> List[T]:
        """Sample n examples from the dataset.

        Args:
            n: Number of examples to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled examples
        """
        examples = await self.load()
        if seed is not None:
            random.seed(seed)
        return random.sample(examples, min(n, len(examples)))

    async def filter(self, predicate: callable) -> List[T]:
        """Filter examples based on a predicate function.

        Args:
            predicate: Function that takes an example and returns bool

        Returns:
            List of filtered examples
        """
        examples = await self.load()
        return [ex for ex in examples if predicate(ex)]


class DatasetRegistry:
    """Registry for datasets."""

    def __init__(self):
        """Initialize an empty registry."""
        self._datasets: Dict[str, Type[Dataset]] = {}

    def register(self, dataset_class: Type[Dataset]) -> Type[Dataset]:
        """Register a dataset class.

        Args:
            dataset_class: The dataset class to register

        Returns:
            The registered dataset class
        """
        # Create an instance to get the name
        instance = dataset_class("temp")
        self._datasets[instance.name] = dataset_class
        return dataset_class

    def get(self, name: str) -> Type[Dataset]:
        """Get a dataset class by name.

        Args:
            name: The dataset name

        Returns:
            The dataset class

        Raises:
            KeyError: If the dataset is not registered
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found. Available datasets: {list(self._datasets.keys())}")
        return self._datasets[name]

    def list(self) -> List[str]:
        """List all registered datasets.

        Returns:
            List of dataset names
        """
        return list(self._datasets.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a dataset is registered.

        Args:
            name: The dataset name

        Returns:
            True if the dataset is registered, False otherwise
        """
        return name in self._datasets
