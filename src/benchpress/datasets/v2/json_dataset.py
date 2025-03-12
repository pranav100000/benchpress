"""JSON dataset implementation for benchpress."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from .base import Dataset

# Type variable for dataset example type
T = TypeVar("T")


class JsonDataset(Dataset[T]):
    """Dataset implementation for JSON files."""

    def __init__(
        self,
        name: str,
        example_class: Type[T],
        file_name: str,
        mapper: Callable[[Dict[str, Any]], Dict[str, Any]],
        version: str = "default",
        data_path: Optional[str] = None,
        encoding: str = "utf-8",
        root_key: Optional[str] = None,
    ):
        """Initialize a JSON dataset.

        Args:
            name: The dataset name
            example_class: The class to use for creating examples
            file_name: The JSON file name (relative to the dataset path)
            mapper: Function to map JSON objects to example constructor parameters
            version: The dataset version
            data_path: Override path to the dataset files
            encoding: The file encoding
            root_key: Key to access the list of examples in the JSON (if not at root)
        """
        super().__init__(name, version, data_path)
        self.example_class = example_class
        self.file_name = file_name
        self.mapper = mapper
        self.encoding = encoding
        self.root_key = root_key

    async def load(self) -> List[T]:
        """Load examples from the JSON file.

        Returns:
            List of examples
        """
        examples = []
        file_path = self.data_path / self.file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding=self.encoding) as f:
            data = json.load(f)
            
        # If a root key is specified, extract the list from that key
        if self.root_key:
            if self.root_key not in data:
                raise KeyError(f"Root key '{self.root_key}' not found in JSON data")
            items = data[self.root_key]
        else:
            # If no root key, assume the JSON is either a list of examples
            # or a dict with a single key containing a list of examples
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and len(data) == 1:
                # Auto-detect a single root key
                items = list(data.values())[0]
                if not isinstance(items, list):
                    items = [data]
            else:
                items = [data]

        for i, item in enumerate(items):
            # Apply the mapper function to get the example parameters
            example_params = self.mapper(item)
            
            # Add a default ID if none is provided
            if "id" not in example_params:
                example_params["id"] = f"{self.name}_{i}"
            
            # Create the example
            example = self.example_class(**example_params)
            examples.append(example)

        return examples