"""CSV dataset implementation for benchpress."""

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from ..tasks.base import Example
from .base import Dataset, register_dataset

# Type variable for dataset example type
T = TypeVar("T", bound=Example)


class CsvDataset(Dataset[T]):
    """Dataset implementation for CSV files."""

    def __init__(
        self,
        name: str,
        example_class: Type[T],
        file_name: str,
        mapper: Callable[[Dict[str, str]], Dict[str, Any]],
        version: str = "default",
        data_path: Optional[str] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ):
        """Initialize a CSV dataset.

        Args:
            name: The dataset name
            example_class: The class to use for creating examples
            file_name: The CSV file name (relative to the dataset path)
            mapper: Function to map CSV rows to example constructor parameters
            version: The dataset version
            data_path: Override path to the dataset files
            delimiter: The CSV delimiter
            encoding: The file encoding
        """
        super().__init__(name, version, data_path)
        self.example_class = example_class
        self.file_name = file_name
        self.mapper = mapper
        self.delimiter = delimiter
        self.encoding = encoding

    async def load(self) -> List[T]:
        """Load examples from the CSV file.

        Returns:
            List of examples
        """
        examples = []
        file_path = self.data_path / self.file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            for i, row in enumerate(reader):
                # Skip empty rows
                if not any(row.values()):
                    continue
                
                # Apply the mapper function to get the example parameters
                example_params = self.mapper(row)
                
                # Add a default ID if none is provided
                if "id" not in example_params:
                    example_params["id"] = f"{self.name}_{i}"
                
                # Create the example
                example = self.example_class(**example_params)
                examples.append(example)

        return examples