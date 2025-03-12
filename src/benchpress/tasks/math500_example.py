"""MATH-500 example class."""

from typing import Any, Dict, Optional

from .base import Example


class Math500Example(Example):
    """An example from the MATH-500 dataset."""

    def __init__(
        self,
        id: str,
        question: str,
        answer: str,
        category: str,
        difficulty: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a MATH-500 example."""
        if metadata is None:
            metadata = {}
        metadata.update({"category": category, "difficulty": difficulty})
        super().__init__(id=id, question=question, answer=answer, metadata=metadata)
        self.category = category
        self.difficulty = difficulty