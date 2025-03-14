"""Benchpress: LLM evaluation framework for standardized benchmarks."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / ".env"
load_dotenv(dotenv_path=env_path)

# Import exception classes for easier access
from benchpress.exceptions import (
    BenchpressError,
    DatasetError,
    ModelError,
    TaskError,
    ExtractionError,
)

__version__ = "0.1.0"
