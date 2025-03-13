"""Task implementations for benchpress."""

# Re-export Example classes from examples module
from ..examples import Aime24Example, Example, GpqaExample, Math500Example
from .aime24 import Aime24Task  # Import the AIME24 task
from .base import BaseTask, TaskResult
from .gpqa import GpqaTask  # Import the GPQA Diamond task
from .math500 import Math500Task  # Import the MATH-500 task
from .registry import register_task, task_registry

__all__ = [
    "BaseTask",
    "TaskResult",
    "Example",
    "task_registry",
    "register_task",
    "Math500Task",
    "Math500Example",
    "Aime24Task",
    "Aime24Example",
    "GpqaTask",
    "GpqaExample",
]
