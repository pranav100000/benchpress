"""AIME24 example class for the AIME benchmark."""

from typing import Any, Dict, Optional

from .base import Example


class Aime24Example(Example):
    """An example from the AIME24 dataset."""

    def __init__(
        self,
        id: str,
        question: str,
        answer: str,
        year: Optional[int] = None,
        problem_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an AIME24 example.
        
        Args:
            id: Unique identifier for the example
            question: The problem text
            answer: The expected answer
            year: The year of the AIME problem (optional)
            problem_number: The problem number within that year (optional)
            metadata: Additional metadata for the example (optional)
        """
        if metadata is None:
            metadata = {}
            
        # Add year and problem number to metadata if provided
        if year is not None:
            metadata["year"] = year
        if problem_number is not None:
            metadata["problem_number"] = problem_number
            
        super().__init__(id=id, question=question, answer=answer, metadata=metadata)
        self.year = year
        self.problem_number = problem_number