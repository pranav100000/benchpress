"""Hugging Face dataset integration for benchpress."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from datasets import load_dataset, Dataset as HfDataset

from .base import Dataset

# Type variable for dataset example type
T = TypeVar("T")

# Set up logging
logger = logging.getLogger(__name__)


class HuggingFaceDataset(Dataset[T]):
    """Dataset implementation for Hugging Face datasets."""

    def __init__(
        self,
        name: str,
        example_class: Type[T],
        dataset_name: str,
        mapper: Callable[[Dict[str, Any]], Dict[str, Any]],
        config_name: Optional[str] = None,
        split: str = "train",
        revision: Optional[str] = None,
        version: str = "default",
        data_path: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        max_workers: int = 4,
        chunk_size: int = 100,
    ):
        """Initialize a Hugging Face dataset.

        Args:
            name: The dataset name in our system
            example_class: The class to use for creating examples
            dataset_name: The Hugging Face dataset name
            mapper: Function to map dataset items to example constructor parameters
            config_name: Dataset configuration name (if applicable)
            split: Dataset split to use
            revision: Git revision to use
            version: Our version label (different from HF revision)
            data_path: Override path to cache/store the dataset
            use_auth_token: Token for accessing private datasets
            max_workers: Maximum number of threads for concurrent processing
            chunk_size: Size of chunks for processing large datasets
        """
        super().__init__(name, version, data_path)
        self.example_class = example_class
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.revision = revision
        self.mapper = mapper
        self.use_auth_token = use_auth_token
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self._hf_dataset = None
        logger.info(f"Initialized HuggingFace dataset {name} from {dataset_name} (split: {split})")

    async def _load_hf_dataset(self) -> HfDataset:
        """Load the Hugging Face dataset.
        
        Returns:
            The loaded HF Dataset object
            
        Raises:
            ValueError: If the dataset or split cannot be found
            RuntimeError: If there's an error loading the dataset
        """
        if self._hf_dataset is None:
            logger.info(f"Loading dataset {self.dataset_name} (split: {self.split})...")
            try:
                # Load the dataset in a separate thread to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    self._hf_dataset = await loop.run_in_executor(
                        executor,
                        lambda: load_dataset(
                            self.dataset_name,
                            self.config_name,
                            split=self.split,
                            revision=self.revision,
                            cache_dir=str(self._data_path) if self._data_path else None,
                            use_auth_token=self.use_auth_token,
                        )
                    )
                logger.info(f"Successfully loaded dataset with {len(self._hf_dataset)} examples")
            except Exception as e:
                logger.error(f"Error loading dataset {self.dataset_name}: {e}")
                raise RuntimeError(f"Failed to load dataset {self.dataset_name}: {e}") from e
                
        return self._hf_dataset

    async def load(self) -> List[T]:
        """Load examples from the Hugging Face dataset.

        Returns:
            List of examples
            
        Raises:
            RuntimeError: If there's an error processing the dataset
        """
        try:
            # First load the HF dataset
            hf_dataset = await self._load_hf_dataset()
            examples = []
            total_examples = len(hf_dataset)

            # Process the dataset - using a simpler approach to avoid threading issues
            logger.info(f"Processing {total_examples} examples...")
            
            # Process items sequentially (more reliable than concurrent with HF datasets)
            # This might be slower but ensures correct handling
            for i in range(total_examples):
                try:
                    # Get a single item directly - this works correctly
                    item = hf_dataset[i]
                    example = self._process_item(i, item)
                    examples.append(example)
                    
                    # Log progress occasionally
                    if i % 100 == 0 and i > 0:
                        logger.info(f"Processed {i}/{total_examples} examples")
                except Exception as e:
                    logger.error(f"Error processing item {i}: {e}")
                    # Continue with the next item
            
            logger.info(f"Successfully converted {len(examples)} examples")
            return examples
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise RuntimeError(f"Failed to process dataset: {e}") from e

    def _process_item(self, idx: int, item: Any) -> T:
        """Process a single dataset item into an example.
        
        Args:
            idx: The item index in the dataset
            item: The dataset item (could be dict, arrow record, or other format)
            
        Returns:
            An example object
        """
        try:
            # Convert item to dictionary if it's not already
            if not isinstance(item, dict):
                # For arrow records, they can be converted to dict with .as_py()
                if hasattr(item, 'as_py'):
                    item = item.as_py()
                # For other formats, we can try to convert to dict
                elif hasattr(item, '__dict__'):
                    item = item.__dict__
                elif isinstance(item, (list, tuple)) and len(item) > 0 and hasattr(item[0], '__dict__'):
                    # Sometimes, datasets return a list/tuple of items
                    item = item[0].__dict__
                else:
                    # As a last resort, print the item type and create a minimal valid example
                    logger.error(f"Unexpected item type: {type(item)}")
                    raise TypeError(f"Cannot convert item of type {type(item)} to dict")
            
            # Apply the mapper function to get the example parameters
            example_params = self.mapper(item)
            
            # Add a default ID if none is provided
            if "id" not in example_params:
                example_params["id"] = f"{self.name}_{idx}"
            
            # Create the example
            return self.example_class(**example_params)
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            # Create a minimal valid example in case of error
            return self.example_class(
                id=f"{self.name}_error_{idx}",
                question="Error processing example",
                answer="Error",
                category="error",
                difficulty="medium",
                metadata={"error": str(e), "source": "error"}
            )
        
    async def get_item(self, idx: int) -> T:
        """Get a single example by index.
        
        Args:
            idx: Index in the dataset
            
        Returns:
            The example at the specified index
            
        Raises:
            IndexError: If the index is out of range
        """
        hf_dataset = await self._load_hf_dataset()
        if idx < 0 or idx >= len(hf_dataset):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(hf_dataset)}")
        
        item = hf_dataset[idx]
        return self._process_item(idx, item)
    
    async def get_size(self) -> int:
        """Get the size of the dataset.
        
        Returns:
            Number of items in the dataset
        """
        hf_dataset = await self._load_hf_dataset()
        return len(hf_dataset)
        
    @lru_cache(maxsize=32)
    async def get_subjects(self) -> List[str]:
        """Get the list of unique subjects in the dataset.
        
        Returns:
            List of unique subject names
        """
        hf_dataset = await self._load_hf_dataset()
        if 'subject' in hf_dataset.features:
            return sorted(hf_dataset.unique('subject'))
        return []
        
    @lru_cache(maxsize=32)
    async def get_levels(self) -> List[int]:
        """Get the list of unique levels in the dataset.
        
        Returns:
            List of unique level values
        """
        hf_dataset = await self._load_hf_dataset()
        if 'level' in hf_dataset.features:
            return sorted(hf_dataset.unique('level'))
        return []
        
    async def filter_by_subject(self, subject: str) -> List[T]:
        """Filter examples by subject.
        
        Args:
            subject: The subject to filter by
            
        Returns:
            List of examples with the specified subject
        """
        hf_dataset = await self._load_hf_dataset()
        if 'subject' not in hf_dataset.features:
            logger.warning(f"Dataset does not have 'subject' field")
            return []
            
        filtered = hf_dataset.filter(lambda x: x['subject'] == subject)
        
        # Process the filtered items
        examples = []
        for i, item in enumerate(filtered):
            try:
                examples.append(self._process_item(i, item))
            except Exception as e:
                logger.error(f"Error processing filtered item {i}: {e}")
                # Skip problematic items
                continue
        return examples
        
    async def filter_by_level(self, level: int) -> List[T]:
        """Filter examples by level.
        
        Args:
            level: The level to filter by
            
        Returns:
            List of examples with the specified level
        """
        hf_dataset = await self._load_hf_dataset()
        if 'level' not in hf_dataset.features:
            logger.warning(f"Dataset does not have 'level' field")
            return []
            
        filtered = hf_dataset.filter(lambda x: x['level'] == level)
        
        # Process the filtered items - concurrently for better performance
        loop = asyncio.get_event_loop()
        examples = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                loop.run_in_executor(
                    executor,
                    self._process_item,
                    i,
                    item
                )
                for i, item in enumerate(filtered)
            ]
            # Gather results, ignoring errors
            try:
                examples = await asyncio.gather(*futures, return_exceptions=True)
                # Filter out exceptions
                examples = [ex for ex in examples if not isinstance(ex, Exception)]
            except Exception as e:
                logger.error(f"Error gathering filtered examples: {e}")
        
        return examples