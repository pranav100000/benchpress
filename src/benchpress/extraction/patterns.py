"""Extraction patterns for the benchpress system."""

import re
from typing import Any, Dict, List

# Pattern types for categorization
PATTERN_TYPE_EXPLICIT = 'explicit'
PATTERN_TYPE_STRUCTURAL = 'structural'
PATTERN_TYPE_DOMAIN = 'domain'
PATTERN_TYPE_POSITIONAL = 'positional'
PATTERN_TYPE_FALLBACK = 'fallback'

# Base pattern definition format
PatternDefinition = Dict[str, Any]

# Common patterns shared across domains
_COMMON_PATTERNS = [
    # Explicit markers (highest priority, highest confidence)
    {
        'name': 'final_answer_marker',
        'pattern': r"(?:FINAL\s+ANSWER|final\s+answer|Final\s+Answer)[:=]\s*(.*?)(?:\n|$)",
        'type': PATTERN_TYPE_EXPLICIT,
        'base_confidence': 0.9,
        'priority': 1000,
    },
    {
        'name': 'explicit_answer_marker',
        'pattern': r"(?:ANSWER|Answer|answer)[:=]\s*([^\n]+)(?:\n|$)(?!.*(?:ANSWER|Answer|answer)[:=])",
        'type': PATTERN_TYPE_EXPLICIT,
        'base_confidence': 0.95,
        'priority': 995,
    },
    {
        'name': 'answer_is_marker',
        'pattern': r"(?:The\s+answer\s+is|the\s+answer\s+is|ANSWER:|Answer:)[:=]?\s*(.*?)(?:\n|$)",
        'type': PATTERN_TYPE_EXPLICIT,
        'base_confidence': 0.85,
        'priority': 990,
    },
    {
        'name': 'we_get_marker',
        'pattern': r"(?:We\s+get|we\s+get|We\s+have|we\s+have|I\s+get|I\s+find)[:=]?\s+([^.\n]+(?:\.[^.\n]+)?)(?:\n|$|\.)",
        'type': PATTERN_TYPE_EXPLICIT,
        'base_confidence': 0.8,
        'priority': 985,
    },
    {
        'name': 'therefore_marker',
        'pattern': r"(?:Therefore|Thus|Hence|So),?\s+(.*?)(?:\n|$|\.)",
        'type': PATTERN_TYPE_EXPLICIT,
        'base_confidence': 0.8,
        'priority': 980,
    },
    {
        'name': 'answer_section',
        'pattern': r"(?:ANSWER|Answer|answer)[:=]?\s*\n*(.*?)(?:\n\n|$)",
        'type': PATTERN_TYPE_STRUCTURAL,
        'base_confidence': 0.8,
        'priority': 880,
    },

    # Structural patterns
    {
        'name': 'boxed_content',
        'pattern': r"\\boxed{((?:[^{}]|{[^{}]*})+)}",
        'type': PATTERN_TYPE_STRUCTURAL,
        'base_confidence': 0.95,
        'priority': 990,
    },

    # Positional patterns (lowest priority, lowest confidence)
    {
        'name': 'last_line',
        'pattern': lambda text: text.strip().split("\n")[-1] if text.strip() else None,
        'type': PATTERN_TYPE_POSITIONAL,
        'base_confidence': 0.5,
        'priority': 490,
    },
    {
        'name': 'last_sentence',
        'pattern': lambda text: re.split(r"(?<=[.!?])\s+", text.strip())[-1] if text.strip() else None,
        'type': PATTERN_TYPE_POSITIONAL,
        'base_confidence': 0.45,
        'priority': 480,
    },
]

# Math-specific patterns
_MATH_PATTERNS = [
    # Structural patterns
    {
        'name': 'boxed_content',
        'pattern': r"\\boxed{((?:[^{}]|{[^{}]*})+)}",
        'type': PATTERN_TYPE_STRUCTURAL,
        'base_confidence': 0.95,
        'priority': 900,
    },
    {
        'name': 'double_dollar_delimited',
        'pattern': r"\$\$(.*?)\$\$",
        'type': PATTERN_TYPE_STRUCTURAL,
        'base_confidence': 0.8,
        'priority': 855,
    },
    {
        'name': 'dollar_delimited',
        'pattern': r"\$(.*?)\$",
        'type': PATTERN_TYPE_STRUCTURAL,
        'base_confidence': 0.75,
        'priority': 850,
    },
    {
        'name': 'we_get_result',
        'pattern': r"(?:we\s+get|we\s+have|we\s+find|I\s+get)\s+(.+?)(?:\.|\n|$)",
        'type': PATTERN_TYPE_DOMAIN,
        'base_confidence': 0.8,
        'priority': 800,
    },
    {
        'name': 'therefore_result',
        'pattern': r"(?:Therefore|Thus|Hence|So),?\s+(.+?)(?:\.|\n|$)",
        'type': PATTERN_TYPE_DOMAIN,
        'base_confidence': 0.8,
        'priority': 820,
    },
    {
        'name': 'numeric_match',
        'pattern': r'(?:\b|^)(\d+(?:\.\d+)?|\d+/\d+)(?:\b|$)',
        'type': PATTERN_TYPE_FALLBACK,
        'base_confidence': 0.3,
        'priority': 290,
    },
]

# GPQA-specific patterns
_GPQA_PATTERNS = [
    {
        'name': 'gpqa_multiple_choice_letter',
        'pattern': r"(?:the\s+)?(?:answer|option)(?:\s+is)?(?:\s*:+\s*|\s+)(?:option\s+)?([A-E])\b",
        'type': PATTERN_TYPE_DOMAIN,
        'base_confidence': 0.85,
        'priority': 695,
    },
    {
        'name': 'gpqa_multiple_choice_letter_parentheses',
        'pattern': r"(?:the\s+)?(?:correct\s+)?(?:answer|option)(?:\s+is)?(?:\s*:+\s*|\s+)\s*\(?([A-E])\)?",
        'type': PATTERN_TYPE_DOMAIN,
        'base_confidence': 0.85,
        'priority': 694,
    },
    {
        'name': 'gpqa_value_with_units',
        'pattern': r"(?:the\s+)?(?:final\s+)?(?:answer|result|value)(?:\s+is)?(?:\s*:+\s*|\s+)\s*(-?\d+\.?\d*\s*(?:[a-zA-Z]+(?:\s*\/\s*[a-zA-Z]+)?))",
        'type': PATTERN_TYPE_DOMAIN,
        'base_confidence': 0.8,
        'priority': 685,
    },
    {
        'name': 'gpqa_chemical_formula',
        'pattern': r"(?:the\s+)?(?:(?:chemical\s+)?(?:formula|equation|reaction|compound)(?:\s+is)?|answer(?:\s+is)?)\s*(?::+\s*|\s+)([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*(?:\s*(?:\+|\-\>|→|⟶|=)\s*[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)*)",
        'type': PATTERN_TYPE_DOMAIN,
        'base_confidence': 0.75,
        'priority': 675,
    },
    {
        'name': 'gpqa_scientific_conclusion',
        'pattern': r"(?:(?:I|we)\s+(?:conclude|determine|find that))\s+(?:that\s+)?([^.\n]{5,200})[.\n]?$",
        'type': PATTERN_TYPE_DOMAIN,
        'base_confidence': 0.7,
        'priority': 670,
    },
]

# Domain-specific pattern collections
_DOMAIN_PATTERNS = {
    'general': _COMMON_PATTERNS,
    'math': _COMMON_PATTERNS + _MATH_PATTERNS,
    'math500': _COMMON_PATTERNS + _MATH_PATTERNS,
    'aime24': _COMMON_PATTERNS + _MATH_PATTERNS,
    'gpqa': _COMMON_PATTERNS + _GPQA_PATTERNS,
}

def get_patterns_for_domain(domain: str) -> List[PatternDefinition]:
    """Get the appropriate patterns for a specific domain.

    Args:
        domain: The domain identifier

    Returns:
        List of patterns appropriate for the domain, sorted by priority
    """
    # Get domain-specific patterns, fall back to general patterns
    patterns = _DOMAIN_PATTERNS.get(domain.lower(), _COMMON_PATTERNS)

    # Sort by priority (descending)
    return sorted(patterns, key=lambda p: p.get('priority', 0), reverse=True)

# Legacy compatibility functions
def get_common_patterns():
    """Legacy compatibility function."""
    from .base import ExtractionPattern, PatternType

    # Convert to old format
    patterns = []
    for pattern in _COMMON_PATTERNS:
        pattern_type = getattr(PatternType, pattern['type'].upper())
        patterns.append(
            ExtractionPattern(
                name=pattern['name'],
                pattern=pattern['pattern'],
                priority=pattern['priority'],
                base_confidence=pattern['base_confidence'],
                pattern_type=pattern_type,
            )
        )
    return patterns

def create_domain_pattern_set(domain: str):
    """Legacy compatibility function."""
    from .base import ExtractionPattern, PatternType

    patterns = get_patterns_for_domain(domain)
    result = []

    for pattern in patterns:
        pattern_type = getattr(PatternType, pattern['type'].upper())
        result.append(
            ExtractionPattern(
                name=pattern['name'],
                pattern=pattern['pattern'],
                priority=pattern['priority'],
                base_confidence=pattern['base_confidence'],
                pattern_type=pattern_type,
            )
        )

    return result
