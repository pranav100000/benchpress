"""GPQA dataset implementation for benchpress v2."""

from typing import Any, Dict

from ..examples.gpqa import GpqaExample
from .csv_dataset import CsvDataset


def gpqa_mapper(row: Dict[str, str]) -> Dict[str, Any]:
    """Map a GPQA CSV row to GpqaExample parameters.

    Args:
        row: A row from the GPQA CSV file

    Returns:
        Dictionary of parameters for GpqaExample constructor
    """
    return {
        "id": f"gpqa_diamond_{row.get('Record ID', '')}",
        "question": row.get("Question", ""),
        "answer": row.get("Correct Answer", ""),
        "subject": row.get("High-level domain", ""),
        "difficulty": row.get("Writer's Difficulty Estimate", "graduate"),
        "metadata": {
            "subdomain": row.get("Subdomain", ""),
            "incorrect_answers": [
                row.get("Incorrect Answer 1", ""),
                row.get("Incorrect Answer 2", ""),
                row.get("Incorrect Answer 3", "")
            ],
            "explanation": row.get("Explanation", ""),
            "record_id": row.get("Record ID", "")
        }
    }


class GpqaDataset(CsvDataset[GpqaExample]):
    """GPQA dataset implementation."""

    def __init__(self, version: str = "default", data_path: str = None):
        """Initialize the GPQA dataset.

        Args:
            version: Dataset version (default is 'default')
            data_path: Override path to the dataset directory
        """
        super().__init__(
            name="gpqa",
            example_class=GpqaExample,
            file_name="gpqa_diamond.csv",
            mapper=gpqa_mapper,
            version=version,
            data_path=data_path
        )
