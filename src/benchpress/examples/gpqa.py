"""GPQA Diamond example class."""

from typing import Any, Dict, Optional

from .base import Example


class GpqaExample(Example):
    """An example from the GPQA Diamond dataset."""

    def __init__(
        self,
        id: str,
        question: str,
        answer: str,
        subject: str,
        difficulty: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a GPQA Diamond example.

        Args:
            id: Unique identifier for the example
            question: The question text
            answer: The correct answer
            subject: The academic subject (e.g., physics, biology, etc.)
            difficulty: The difficulty level (e.g., easy, medium, hard)
            metadata: Additional metadata about the example
        """
        if metadata is None:
            metadata = {}
        metadata.update({"subject": subject, "difficulty": difficulty})
        super().__init__(id=id, question=question, answer=answer, metadata=metadata)
        self.subject = subject
        self.difficulty = difficulty
