"""Task implementations for benchpress."""

from .aime24 import Aime24Task  # Import the AIME24 task
from .base import BaseTask, Example, TaskResult
from .gpqa import GpqaTask  # Import the GPQA Diamond task
from .gpqa_example import GpqaExample  # Import the GPQA Example class
from .math500 import Math500Task  # Import the MATH-500 task
from .math500_example import Math500Example  # Import the MATH-500 Example class
from .registry import register_task, task_registry

__all__ = [
    "BaseTask",
    "Example",
    "TaskResult",
    "task_registry",
    "register_task",
    "Math500Task",
    "Math500Example",
    "Aime24Task",
    "GpqaTask",
    "GpqaExample",
]
