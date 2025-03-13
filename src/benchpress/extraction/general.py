"""General-purpose answer extractor implementation."""

import re
from typing import Any, Callable, List, Optional, Pattern, Tuple, Union

from .base import BaseExtractor, ExtractionContext, ExtractionPattern, ExtractedAnswer
from .patterns import create_domain_pattern_set
from .processors import create_processor_pipeline
from .registry import register_extractor


@register_extractor("general")
class GeneralExtractor(BaseExtractor):
    """A general-purpose answer extractor."""

    def __init__(self, name: str = "general"):
        """Initialize the general extractor.

        Args:
            name: The name of the extractor
        """
        super().__init__(name)
        
        # Add common patterns for general extraction
        self.patterns = create_domain_pattern_set("*")
    
    def _apply_pattern(
        self, pattern: ExtractionPattern, text: str
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """Apply a pattern to extract answers.

        Args:
            pattern: The pattern to apply
            text: The text to extract from

        Returns:
            A list of (extracted_text, (start, end)) tuples
        """
        if isinstance(pattern.pattern, str) or isinstance(pattern.pattern, Pattern):
            # It's a regex pattern
            matches = []
            for match in re.finditer(pattern.pattern, text, re.MULTILINE | re.DOTALL):
                if match.groups():
                    # Use the first capture group
                    start, end = match.span(1)
                    matches.append((match.group(1), (start, end)))
                else:
                    # Use the entire match
                    start, end = match.span()
                    matches.append((match.group(), (start, end)))
            return matches
        
        elif callable(pattern.pattern):
            # It's a function that returns extracted text
            result = pattern.pattern(text)
            if result:
                # Use a dummy position for function-based patterns
                return [(result, (0, len(text)))]
            return []
        
        return []
    
    def extract(self, text: str, context: ExtractionContext) -> List[ExtractedAnswer]:
        """Extract answers from the given text.

        Args:
            text: The text to extract answers from
            context: Extraction context

        Returns:
            A list of extracted answers, ordered by confidence (highest first)
        """
        candidates = []
        
        # Apply each pattern to the text
        for pattern in self.patterns:
            # Check if this pattern applies to the domain
            if not pattern.matches(context.domain):
                continue
            
            matches = self._apply_pattern(pattern, text)
            for match_text, position in matches:
                # Remove trailing punctuation
                clean_text = re.sub(r'[.,;:]+$', '', match_text.strip())
                
                # Compute confidence score
                confidence = self._compute_confidence(
                    clean_text, pattern, position, len(text)
                )
                
                # Normalize the answer
                normalized = self.normalize(clean_text, context)
                
                # Create extracted answer
                answer = ExtractedAnswer(
                    text=clean_text,
                    pattern_name=pattern.name,
                    confidence=confidence,
                    normalized_text=normalized,
                    position=position,
                    extracted_by=self.name,
                    metadata={"pattern_type": pattern.pattern_type.value}
                )
                
                candidates.append(answer)
        
        # Sort candidates by confidence (highest first)
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return candidates
    
    def normalize(self, text: str, context: ExtractionContext) -> str:
        """Normalize an extracted answer.

        Args:
            text: The extracted text
            context: Extraction context

        Returns:
            Normalized text
        """
        # Create a pipeline based on the domain
        if context.domain in ("math", "math500", "aime24"):
            pipeline = create_processor_pipeline([
                "clean_whitespace", 
                "remove_markers", 
                "remove_latex_formatting", 
                "normalize_math_answer"
            ])
        else:
            pipeline = create_processor_pipeline([
                "clean_whitespace", 
                "remove_markers"
            ])
        
        return pipeline(text)