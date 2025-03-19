"""Base task interface for benchpress."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar

from ..examples.base import Example


@dataclass
class TaskResult:
    """The result of evaluating a model on a task example."""

    example_id: str
    model_id: str
    model_output: str
    correct: bool
    metadata: Optional[Dict[str, Any]] = None

    # Fields for supporting parallel processing display
    raw_output: Optional[str] = None
    example_index: Optional[int] = None
    total_examples: Optional[int] = None


T = TypeVar("T", bound=Example)


class BaseTask(Generic[T], ABC):
    """Abstract base class for all benchmark tasks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the task name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the task description."""
        pass

    @abstractmethod
    async def load_examples(self) -> List[T]:
        """Load all examples for this task.

        Returns:
            A list of examples
        """
        pass

    @abstractmethod
    async def evaluate_example(self, example: T, model_output: str) -> TaskResult:
        """Evaluate a model's output on a single example.

        Args:
            example: The example to evaluate
            model_output: The model's generated output

        Returns:
            A task result containing the evaluation
        """
        pass
