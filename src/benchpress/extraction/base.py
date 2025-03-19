"""Base classes for the extraction framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union


class PatternType(Enum):
    """Types of extraction patterns."""

    EXPLICIT = "explicit"  # Explicit markers (FINAL ANSWER:, The answer is:, etc.)
    STRUCTURAL = "structural"  # Structural patterns (boxed, bullet points, etc.)
    POSITIONAL = "positional"  # Positional patterns (last line, after conclusion, etc.)
    DOMAIN = "domain"  # Domain-specific patterns (math, multiple choice, etc.)
    FALLBACK = "fallback"  # Last resort patterns (any numeric/text pattern)


@dataclass
class ExtractionPattern:
    """A pattern for extracting answers from model outputs."""

    name: str
    pattern: Union[str, Pattern, Callable]
    priority: int = 0
    base_confidence: float = 0.5
    pattern_type: PatternType = PatternType.EXPLICIT
    applies_to: Set[str] = field(default_factory=lambda: {"*"})
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, domain: str) -> bool:
        """Check if this pattern applies to the given domain.

        Args:
            domain: The domain to check

        Returns:
            True if the pattern applies to this domain
        """
        return "*" in self.applies_to or domain in self.applies_to


@dataclass
class ExtractionContext:
    """Context for answer extraction."""

    domain: str
    task_name: str
    expected_format: Optional[str] = None
    question_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedAnswer:
    """An answer extracted from a model output."""

    text: str
    pattern_name: str
    confidence: float
    normalized_text: Optional[str] = None
    position: Optional[Tuple[int, int]] = None
    extracted_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExtractor(ABC):
    """Base class for answer extractors."""

    def __init__(self, name: str, default_patterns: Optional[List[ExtractionPattern]] = None):
        """Initialize the extractor.

        Args:
            name: The name of the extractor
            default_patterns: Default patterns to use for extraction
        """
        self.name = name
        self.patterns: List[ExtractionPattern] = []

        # Add default patterns if provided
        if default_patterns:
            for pattern in default_patterns:
                self.add_pattern(pattern)

    def add_pattern(self, pattern: ExtractionPattern) -> None:
        """Add a pattern to the extractor.

        Args:
            pattern: The pattern to add
        """
        self.patterns.append(pattern)
        # Keep patterns sorted by priority (descending)
        self.patterns.sort(key=lambda p: p.priority, reverse=True)

    @abstractmethod
    def extract(self, text: str, context: ExtractionContext) -> List[ExtractedAnswer]:
        """Extract answers from the given text.

        Args:
            text: The text to extract answers from
            context: Extraction context

        Returns:
            A list of extracted answers, ordered by confidence (highest first)
        """
        pass

    def _compute_confidence(
        self,
        match: Any,
        pattern: ExtractionPattern,
        position: Tuple[int, int],
        text_length: int
    ) -> float:
        """Compute confidence score for a match.

        Args:
            match: The pattern match
            pattern: The pattern that matched
            position: Start and end position of match
            text_length: Total length of text

        Returns:
            Confidence score between 0 and 1
        """
        start, end = position

        # Base confidence from pattern
        confidence = pattern.base_confidence

        # Adjust based on pattern type
        type_boost: Dict[str, float] = {
            PatternType.EXPLICIT.value: 0.3,    # Most confident
            PatternType.STRUCTURAL.value: 0.2,  # Very confident
            PatternType.DOMAIN.value: 0.1,      # Domain-specific
            PatternType.POSITIONAL.value: 0.0,  # Position-based
            PatternType.FALLBACK.value: -0.1,   # Last resort
        }
        confidence += type_boost.get(pattern.pattern_type.value, 0.0)

        # Position factor (later in text is better)
        # Scale from 0.0 to 0.1 based on position
        position_factor = start / max(1, text_length)
        confidence += position_factor * 0.1

        # Cap confidence between 0 and 1
        return max(0.0, min(1.0, confidence))

    @abstractmethod
    def normalize(self, text: str, context: ExtractionContext) -> str:
        """Normalize an extracted answer.

        Args:
            text: The extracted text
            context: Extraction context

        Returns:
            Normalized text
        """
        pass


# Utility function for computing confidence scores from dict patterns
def compute_confidence_score(
    pattern: Dict[str, Any],
    position: Tuple[int, int],
    text_length: int
) -> float:
    """Compute confidence score for a pattern match.

    Args:
        pattern: The pattern that matched (dict format)
        position: Start and end position of match
        text_length: Total length of text

    Returns:
        Confidence score between 0 and 1
    """
    start, end = position

    # For dict-based patterns (from core.py)
    base_confidence = pattern.get('base_confidence', 0.5)
    pattern_type = pattern.get('type', 'fallback')

    # Adjust based on pattern type
    type_boost: Dict[str, float] = {
        'explicit': 0.3,    # Most confident
        'structural': 0.2,  # Very confident
        'domain': 0.1,      # Domain-specific
        'positional': 0.0,  # Position-based
        'fallback': -0.1,   # Last resort
    }
    confidence = base_confidence + type_boost.get(pattern_type, 0.0)

    # Position factor (later in text is better)
    # Scale from 0.0 to 0.1 based on position
    position_factor = start / max(1, text_length)
    confidence += position_factor * 0.1

    # Cap confidence between 0 and 1
    return max(0.0, min(1.0, confidence))
