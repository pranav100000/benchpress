"""Task registry for benchpress."""

from typing import Dict, Type, TypeVar

from .base import BaseTask

T = TypeVar("T", bound=BaseTask)
task_registry: Dict[str, Type[BaseTask]] = {}


def register_task(task_class: Type[T]) -> Type[T]:
    """Register a task implementation.

    Args:
        task_class: The task class to register

    Returns:
        The same task class (for decorator usage)
    """
    instance = task_class()
    task_registry[instance.name] = task_class
    return task_class
