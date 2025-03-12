"""Task implementations for benchpress."""

from .aime24 import Aime24Task  # Import the AIME24 task
from .base import BaseTask, Example, TaskResult
from .math500 import Math500Task  # Import the MATH-500 task
from .registry import register_task, task_registry

__all__ = [
    "BaseTask",
    "Example",
    "TaskResult",
    "task_registry",
    "register_task",
    "Math500Task",
    "Aime24Task",
]
