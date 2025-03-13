"""Example classes used in benchpress benchmark tasks.

This module contains the Example class definitions that represent
benchmark examples across the framework.
"""

from .aime24 import Aime24Example
from .base import Example
from .gpqa import GpqaExample
from .math500 import Math500Example

__all__ = ["Example", "Aime24Example", "GpqaExample", "Math500Example"]
