"""Core extraction functionality for benchpress.

This module provides the unified extraction system for extracting answers from
model outputs with support for different types of questions and answer formats.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Pattern as RegexPattern

# Import shared classes from base module to avoid duplication
from .base import ExtractionContext, ExtractionPattern, ExtractedAnswer, PatternType


def extract_answer(
    text: str,
    context: ExtractionContext,
    patterns: Optional[List[Dict[str, Any]]] = None,
    normalize_answers: bool = True
) -> List[ExtractedAnswer]:
    """Extract answers from model output.

    Args:
        text: Model output text
        context: Extraction context
        patterns: Optional list of patterns to use (defaults to domain-specific patterns)
        normalize_answers: Whether to normalize extracted answers (default: True)

    Returns:
        List of extracted answers sorted by confidence
    """
    from .patterns import get_patterns_for_domain
    from .processors import normalize_answer

    # Get appropriate patterns for this domain
    patterns = patterns or get_patterns_for_domain(context.domain)

    candidates = []

    # Apply each pattern to extract candidates
    for pattern in patterns:
        matches = _apply_pattern(pattern, text)
        for match_text, position in matches:
            # Compute confidence
            confidence = _compute_confidence(pattern, position, len(text))
            
            # Apply normalization if requested
            normalized = None
            if normalize_answers:
                try:
                    normalized = normalize_answer(match_text, context.domain)
                except Exception:
                    # If normalization fails, leave as None
                    pass

            # Create extracted answer
            answer = ExtractedAnswer(
                text=match_text,
                pattern_name=pattern['name'],
                confidence=confidence,
                normalized_text=normalized,
                position=position,
                extracted_by="extract_answer",
                metadata={"pattern_type": pattern.get('type', 'unknown')}
            )

            candidates.append(answer)

    # Sort by confidence (highest first)
    candidates.sort(key=lambda x: x.confidence, reverse=True)

    return candidates


def _apply_pattern(pattern: Dict[str, Any], text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Apply a pattern to extract answers."""

    pattern_obj = pattern['pattern']
    matches = []

    if isinstance(pattern_obj, (str, RegexPattern)):
        # It's a regex pattern
        for match in re.finditer(pattern_obj, text, re.MULTILINE | re.DOTALL):
            if match.groups():
                # Use the first capture group
                start, end = match.span(1)
                matches.append((match.group(1), (start, end)))
            else:
                # Use the entire match
                start, end = match.span()
                matches.append((match.group(), (start, end)))

    elif callable(pattern_obj):
        # It's a function that returns extracted text
        result = pattern_obj(text)
        if isinstance(result, str) and result:
            # Use a dummy position for function-based patterns
            matches.append((result, (0, len(text))))

    return matches


def _compute_confidence(pattern: Dict[str, Any], position: Tuple[int, int], text_length: int) -> float:
    """Compute confidence score for a pattern match."""
    # Use the shared utility function
    from .base import compute_confidence_score
    return compute_confidence_score(pattern, position, text_length)
