"""Common patterns for answer extraction."""

import re
from typing import List

from .base import ExtractionPattern, PatternType

# Pattern priority ranges (higher = tried first)
EXPLICIT_PRIORITY_RANGE = (900, 1000)  # Explicit markers
STRUCTURAL_PRIORITY_RANGE = (700, 899)  # Structural patterns
DOMAIN_PRIORITY_RANGE = (500, 699)     # Domain-specific patterns
POSITIONAL_PRIORITY_RANGE = (300, 499)  # Positional patterns
FALLBACK_PRIORITY_RANGE = (100, 299)    # Fallback patterns


def get_common_patterns() -> List[ExtractionPattern]:
    """Get a list of common extraction patterns.

    Returns:
        A list of common extraction patterns
    """
    patterns = []

    # Explicit patterns
    patterns.extend([
        ExtractionPattern(
            name="final_answer_marker",
            pattern=r"(?:FINAL\s+ANSWER|final\s+answer|Final\s+Answer)[:=]\s*(.*?)(?:\n|$)",
            priority=1000,
            base_confidence=0.9,
            pattern_type=PatternType.EXPLICIT,
        ),
        ExtractionPattern(
            name="explicit_answer_marker",
            pattern=r"ANSWER:\s*([^\n]+)(?:\n|$)",
            priority=995,
            base_confidence=0.9,
            pattern_type=PatternType.EXPLICIT,
        ),
        ExtractionPattern(
            name="answer_is_marker",
            pattern=r"(?:The\s+answer\s+is|the\s+answer\s+is|ANSWER:|Answer:)[:=]?\s*(.*?)(?:\n|$)",
            priority=990,
            base_confidence=0.85,
            pattern_type=PatternType.EXPLICIT,
        ),
        ExtractionPattern(
            name="therefore_marker",
            pattern=r"(?:Therefore|Thus|Hence|So),?\s*(.*?)(?:\n|$)",
            priority=980,
            base_confidence=0.8,
            pattern_type=PatternType.EXPLICIT,
        ),
    ])

    # Structural patterns
    patterns.extend([
        ExtractionPattern(
            name="boxed_content",
            pattern=r"\\boxed{((?:[^{}]|{[^{}]*})+)}",
            priority=990,  # Increase priority to be higher than explicit markers
            base_confidence=0.95,  # Increase confidence
            pattern_type=PatternType.STRUCTURAL,
            applies_to={"math", "math500", "aime24", "*"},
        ),
        ExtractionPattern(
            name="answer_section",
            pattern=r"(?:ANSWER|Answer|answer)[:=]?\s*\n*(.*?)(?:\n\n|$)",
            priority=880,
            base_confidence=0.8,
            pattern_type=PatternType.STRUCTURAL,
        ),
    ])

    # Domain-specific patterns - Math
    patterns.extend([
        ExtractionPattern(
            name="math_equals_result",
            pattern=r"=\s*(?:\$)?([^$\n]+?)(?:\$)?(?:\s*$|\s*\.|\n)",  # Modified to better capture decimals
            priority=690,
            base_confidence=0.7,
            pattern_type=PatternType.DOMAIN,
            applies_to={"math", "math500", "aime24"},
        ),
        ExtractionPattern(
            name="we_get_result",
            pattern=r"(?:we\s+get|we\s+have|we\s+find|I\s+get)[:=]?\s*(?:\$)?([^$\n]+?)(?:\$)?(?:\s*$|\s*\.|\n)",  # Modified to better capture decimals
            priority=680,
            base_confidence=0.65,
            pattern_type=PatternType.DOMAIN,
            applies_to={"math", "math500", "aime24"},
        ),
    ])

    # Domain-specific patterns - GPQA
    patterns.extend([
        # Multiple choice answer extraction for GPQA
        ExtractionPattern(
            name="gpqa_multiple_choice_letter",
            pattern=r"(?:the\s+)?(?:answer|option)(?:\s+is)?(?:\s*:+\s*|\s+)(?:option\s+)?([A-E])\b",
            priority=695,
            base_confidence=0.85,
            pattern_type=PatternType.DOMAIN,
            applies_to={"gpqa"},
        ),
        ExtractionPattern(
            name="gpqa_multiple_choice_letter_parentheses",
            pattern=r"(?:the\s+)?(?:correct\s+)?(?:answer|option)(?:\s+is)?(?:\s*:+\s*|\s+)\s*\(?([A-E])\)?",
            priority=694,
            base_confidence=0.85,
            pattern_type=PatternType.DOMAIN,
            applies_to={"gpqa"},
        ),

        # Value with units common in scientific questions
        ExtractionPattern(
            name="gpqa_value_with_units",
            pattern=r"(?:the\s+)?(?:final\s+)?(?:answer|result|value)(?:\s+is)?(?:\s*:+\s*|\s+)\s*(-?\d+\.?\d*\s*(?:[a-zA-Z]+(?:\s*\/\s*[a-zA-Z]+)?))",
            priority=685,
            base_confidence=0.8,
            pattern_type=PatternType.DOMAIN,
            applies_to={"gpqa"},
        ),

        # Chemical formulas or equations
        ExtractionPattern(
            name="gpqa_chemical_formula",
            pattern=r"(?:the\s+)?(?:(?:chemical\s+)?(?:formula|equation|reaction|compound)(?:\s+is)?|answer(?:\s+is)?)\s*(?::+\s*|\s+)([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*(?:\s*(?:\+|\-\>|→|⟶|=)\s*[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)*)",
            priority=675,
            base_confidence=0.75,
            pattern_type=PatternType.DOMAIN,
            applies_to={"gpqa"},
        ),

        # Scientific reasoning pattern
        ExtractionPattern(
            name="gpqa_scientific_conclusion",
            pattern=r"(?:(?:I|we)\s+(?:conclude|determine|find that))\s+(?:that\s+)?([^.\n]{5,200})[.\n]?$",
            priority=670,
            base_confidence=0.7,
            pattern_type=PatternType.DOMAIN,
            applies_to={"gpqa"},
        ),
    ])

    # Positional patterns
    patterns.extend([
        ExtractionPattern(
            name="last_line",
            pattern=lambda text: text.strip().split("\n")[-1] if text.strip() else None,
            priority=490,
            base_confidence=0.5,
            pattern_type=PatternType.POSITIONAL,
        ),
        ExtractionPattern(
            name="last_sentence",
            pattern=lambda text: re.split(r"(?<=[.!?])\s+", text.strip())[-1] if text.strip() else None,
            priority=480,
            base_confidence=0.45,
            pattern_type=PatternType.POSITIONAL,
        ),
    ])

    # Fallback patterns
    patterns.extend([
        ExtractionPattern(
            name="numeric_match",
            pattern=r'(?:\b|^)(\d+(?:\.\d+)?|\d+/\d+)(?:\b|$)',
            priority=290,
            base_confidence=0.3,
            pattern_type=PatternType.FALLBACK,
            applies_to={"math", "math500", "aime24", "number", "*"},
        ),
    ])

    return patterns


def create_domain_pattern_set(domain: str) -> List[ExtractionPattern]:
    """Create a set of patterns specialized for a specific domain.

    Args:
        domain: The domain to create patterns for

    Returns:
        A list of domain-specific extraction patterns
    """
    common_patterns = get_common_patterns()

    # Filter patterns that apply to this domain
    return [p for p in common_patterns if p.matches(domain)]
