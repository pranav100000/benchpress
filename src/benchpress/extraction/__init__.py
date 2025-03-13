"""Answer extraction module for benchpress.

This module provides utilities for extracting answers from model outputs
with support for different types of questions and answer formats.
"""

from .base import BaseExtractor, ExtractionContext, ExtractionPattern, ExtractedAnswer, PatternType
from .general import GeneralExtractor
from .math import MathExtractor
from .patterns import create_domain_pattern_set, get_common_patterns
from .processors import (
    clean_whitespace, normalize_math_answer, remove_latex_formatting, remove_markers
)
from .registry import register_extractor, extractor_registry, get_extractor

__all__ = [
    "BaseExtractor", 
    "ExtractedAnswer",
    "ExtractionContext",
    "ExtractionPattern",
    "PatternType",
    "GeneralExtractor",
    "MathExtractor",
    "register_extractor",
    "extractor_registry",
    "get_extractor",
    "create_domain_pattern_set",
    "get_common_patterns",
    "clean_whitespace",
    "remove_markers",
    "remove_latex_formatting",
    "normalize_math_answer",
]

# Create convenience function for getting an extractor instance
def create_extractor(
    domain: str = "general", 
    extractor_name: str = None
) -> BaseExtractor:
    """Create an appropriate extractor for the given domain.
    
    Args:
        domain: Domain identifier ("math", "general", "gpqa", etc.)
        extractor_name: Optional specific extractor to use
        
    Returns:
        An instance of a BaseExtractor for the domain
    """
    if extractor_name:
        extractor_cls = get_extractor(extractor_name)
        if extractor_cls:
            return extractor_cls()
    
    # Use domain to determine extractor
    if domain in ("math", "math500", "aime24"):
        return MathExtractor()
    elif domain in ("gpqa"):
        # For GPQA we use the GeneralExtractor but with domain context
        extractor = GeneralExtractor()
        # Set the domain so patterns specific to GPQA will be matched
        extractor.domain = "gpqa"
        return extractor
    
    # Default to general extractor
    return GeneralExtractor()