"""Answer extraction module for benchpress.

This module provides a unified extraction system for extracting answers from model outputs
with support for different types of questions and answer formats.
"""

# Unified extraction system exports
from .core import (
    ExtractedAnswer,
    ExtractionContext,
    extract_answer,
)
from .patterns import get_patterns_for_domain
from .processors import (
    clean_whitespace,
    normalize_answer,
    normalize_gpqa_answer,
    normalize_math_answer,
    remove_latex_formatting,
    remove_markers,
)

__all__ = [
    # Main API
    "ExtractionContext",
    "ExtractedAnswer",
    "extract_answer",
    "normalize_answer",
    "get_patterns_for_domain",

    # Utility functions
    "clean_whitespace",
    "normalize_math_answer",
    "normalize_gpqa_answer",
    "remove_latex_formatting",
    "remove_markers",
]
