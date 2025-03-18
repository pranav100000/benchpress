"""Core extraction functionality for benchpress."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union, Callable
import re

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
    """An answer extracted from model output."""
    
    text: str
    pattern_name: str
    confidence: float
    position: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

def extract_answer(
    text: str, 
    context: ExtractionContext, 
    patterns: Optional[List[Dict[str, Any]]] = None
) -> List[ExtractedAnswer]:
    """Extract answers from model output.
    
    Args:
        text: Model output text
        context: Extraction context
        patterns: Optional list of patterns to use (defaults to domain-specific patterns)
        
    Returns:
        List of extracted answers sorted by confidence
    """
    from .patterns import get_patterns_for_domain
    
    # Get appropriate patterns for this domain
    patterns = patterns or get_patterns_for_domain(context.domain)
    
    candidates = []
    
    # Apply each pattern to extract candidates
    for pattern in patterns:
        matches = _apply_pattern(pattern, text)
        for match_text, position in matches:
            # Compute confidence
            confidence = _compute_confidence(pattern, position, len(text))
            
            # Create extracted answer
            answer = ExtractedAnswer(
                text=match_text,
                pattern_name=pattern['name'],
                confidence=confidence,
                position=position,
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
    
    if isinstance(pattern_obj, (str, Pattern)):
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
    start, end = position
    
    # Base confidence from pattern
    confidence = pattern.get('base_confidence', 0.5)
    
    # Adjust based on pattern type
    type_boost = {
        'explicit': 0.3,    # Most confident
        'structural': 0.2,  # Very confident
        'domain': 0.1,      # Domain-specific
        'positional': 0.0,  # Position-based
        'fallback': -0.1,   # Last resort
    }
    confidence += type_boost.get(pattern.get('type', 'fallback'), 0.0)
    
    # Position factor (later in text is better)
    # Scale from 0.0 to 0.1 based on position
    position_factor = start / max(1, text_length)
    confidence += position_factor * 0.1
    
    # Cap confidence between 0 and 1
    return max(0.0, min(1.0, confidence))