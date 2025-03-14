"""Math-specific answer extractor implementation."""

import re
from typing import List

from .base import (
    ExtractedAnswer,
    ExtractionContext,
    ExtractionPattern,
    PatternType,
)
from .general import GeneralExtractor
from .patterns import create_domain_pattern_set
from .processors import normalize_math_answer
from .registry import register_extractor


@register_extractor("math")
class MathExtractor(GeneralExtractor):
    """An extractor specialized for mathematical answers."""

    def __init__(self, name: str = "math"):
        """Initialize the math extractor.

        Args:
            name: The name of the extractor
        """
        super().__init__(name)

        # Override with math-specific patterns
        self.patterns = create_domain_pattern_set("math")

        # Add more math-specific patterns
        self.patterns.extend([
            ExtractionPattern(
                name="latex_fraction",
                pattern=lambda text:
                    # Special handling for LaTeX fractions to capture numerator/denominator
                    f"{m.group(1)}/{m.group(2)}" if (m := re.search(r"\\frac{(.*?)}{(.*?)}", text)) else None,
                priority=670,
                base_confidence=0.7,
                pattern_type=PatternType.DOMAIN,
                applies_to={"math", "math500", "aime24"},
            ),
            ExtractionPattern(
                name="decimal_number",
                pattern=r'(?:\b|^)(\d+\.\d+)(?:\b|$)',
                priority=660,
                base_confidence=0.75,  # Increase confidence for decimal numbers
                pattern_type=PatternType.DOMAIN,
                applies_to={"math", "math500", "aime24"},
            ),
            ExtractionPattern(
                name="numeric_fraction",
                pattern=r'(?:\b|^)(\d+/\d+)(?:\b|$)',
                priority=650,
                base_confidence=0.65,
                pattern_type=PatternType.DOMAIN,
                applies_to={"math", "math500", "aime24"},
            ),
            ExtractionPattern(
                name="symbolic_fraction",
                pattern=r'(?:\b|^)([a-zA-Z]/[a-zA-Z])(?:\b|$)',
                priority=645,
                base_confidence=0.65,
                pattern_type=PatternType.DOMAIN,
                applies_to={"math", "math500", "aime24"},
            ),
        ])

    def normalize(self, text: str, context: ExtractionContext) -> str:
        """Normalize a mathematical answer.

        Args:
            text: The extracted text
            context: Extraction context

        Returns:
            Normalized text
        """
        return normalize_math_answer(text)

    def _validate_math_answer(self, answer: str) -> bool:
        """Validate if a string looks like a valid math answer.

        Args:
            answer: The answer to validate

        Returns:
            True if the answer looks valid
        """
        # Simple validation - check for common math patterns
        if not answer:
            return False

        # Check for numbers, fractions, LaTeX expressions
        if re.search(r'\d', answer):
            return True

        # Check for LaTeX math expressions
        if re.search(r'\\[a-zA-Z]+', answer):
            return True

        # Check for common symbols
        if re.search(r'[+\-*/^()[\]{}]', answer):
            return True

        # Check for symbolic fractions like p/q, n/k
        if re.search(r'^[a-zA-Z]/[a-zA-Z]$', answer):
            return True

        # Check for single letter variables common in math (like x, y, n, k)
        if re.search(r'^[a-zA-Z]$', answer) and len(answer) == 1:
            return True

        return False

    def extract(self, text: str, context: ExtractionContext) -> List[ExtractedAnswer]:
        """Extract mathematical answers from the given text.

        Args:
            text: The text to extract answers from
            context: Extraction context

        Returns:
            A list of extracted answers, ordered by confidence (highest first)
        """
        # Get candidates using the general extraction logic
        candidates = super().extract(text, context)

        # Filter and re-score candidates based on math-specific validation
        for candidate in candidates:
            if not self._validate_math_answer(candidate.normalized_text or ""):
                # Reduce confidence for answers that don't look like math
                candidate.confidence *= 0.5

        # Re-sort candidates
        candidates.sort(key=lambda x: x.confidence, reverse=True)

        return candidates
