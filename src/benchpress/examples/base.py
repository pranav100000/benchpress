"""Base example class for benchpress framework."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Example:
    """A single example from a benchmark dataset."""

    id: str
    question: str
    answer: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Initialize the example."""
        if self.metadata is None:
            self.metadata = {}
