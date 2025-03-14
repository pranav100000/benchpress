"""Post-processing utilities for extracted answers."""

import re
from typing import Callable, Dict, Optional


def clean_whitespace(text: str) -> str:
    """Clean whitespace in the given text.

    Args:
        text: The text to clean

    Returns:
        Cleaned text
    """
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()


def remove_markers(text: str) -> str:
    """Remove common markers from the given text.

    Args:
        text: The text to process

    Returns:
        Text with markers removed
    """
    # Remove "the answer is", "therefore", etc.
    text = re.sub(
        r'^(?:the\s+answer\s+is|therefore|thus|hence|so|we\s+get|we\s+have|we\s+find|I\s+get|answer[:=])\s*[:=]?\s*',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Remove trailing punctuation
    text = re.sub(r'[.,;:]+$', '', text)

    return text.strip()


def remove_latex_formatting(text: str) -> str:
    """Remove LaTeX formatting from the given text.

    Args:
        text: The text to process

    Returns:
        Text with LaTeX formatting removed
    """
    # Remove \boxed{}
    text = re.sub(r'\\boxed{(.*?)}', r'\1', text)

    # Remove $$ or $ markers
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text)
    text = re.sub(r'\$(.*?)\$', r'\1', text)

    # Convert LaTeX fractions to normal fractions
    text = re.sub(r'\\frac{(.*?)}{(.*?)}', r'\1/\2', text)

    # Remove basic LaTeX commands
    text = re.sub(r'\\text{(.*?)}', r'\1', text)

    return text.strip()


def normalize_math_answer(text: str) -> str:
    """Normalize a mathematical answer.

    Args:
        text: The text to normalize

    Returns:
        Normalized text
    """
    # First clean whitespace and remove markers
    text = clean_whitespace(text)

    # Explicitly look for and remove "ANSWER:" marker
    text = re.sub(r'^ANSWER:\s*', '', text, flags=re.IGNORECASE)

    # Remove other markers
    text = remove_markers(text)
    text = remove_latex_formatting(text)

    # Standardize decimal notation (both 0.5 and .5 become 0.5)
    text = re.sub(r'(\D|^)\.(\d+)', r'\g<1>0.\2', text)

    # Normalize fractions (both numeric and symbolic)
    try:
        # Check for numeric fractions first
        fraction_match = re.match(r'^(\d+)/(\d+)$', text)
        if fraction_match:
            num = int(fraction_match.group(1))
            denom = int(fraction_match.group(2))
            if denom != 0:  # Avoid division by zero
                # Keep the standardized form for numeric fractions
                text = f"{num}/{denom}"
        # Also handle symbolic fractions like p/q, n/k
        elif "/" in text:
            symbolic_match = re.match(r'^([a-zA-Z])/([a-zA-Z])$', text)
            if symbolic_match:
                # Normalize symbolic fractions by preserving the exact form
                numerator = symbolic_match.group(1).lower()
                denominator = symbolic_match.group(2).lower()
                text = f"{numerator}/{denominator}"
    except Exception:
        # If normalization fails, leave as is
        pass

    return text.strip()


# Registry of processor functions
processor_registry: Dict[str, Callable[[str], str]] = {
    "clean_whitespace": clean_whitespace,
    "remove_markers": remove_markers,
    "remove_latex_formatting": remove_latex_formatting,
    "normalize_math_answer": normalize_math_answer,
}


def create_processor_pipeline(
    processors: list[str],
    fallback: Optional[Callable[[str], str]] = None
) -> Callable[[str], str]:
    """Create a pipeline of processors.

    Args:
        processors: List of processor names to apply
        fallback: Fallback processor to use if a processor is not found

    Returns:
        A function that applies all processors in sequence
    """
    def pipeline(text: str) -> str:
        result = text
        for processor_name in processors:
            processor = processor_registry.get(processor_name)
            if processor:
                result = processor(result)
            elif fallback:
                result = fallback(result)
        return result

    return pipeline
