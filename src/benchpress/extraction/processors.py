"""Text processing utilities for answer extraction."""

import re
from typing import Callable, Dict, Optional

def clean_whitespace(text: str) -> str:
    """Clean whitespace in text."""
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()

def remove_markers(text: str) -> str:
    """Remove common answer markers."""
    # Remove "the answer is", "therefore", etc.
    text = re.sub(
        r'^(?:the\s+answer\s+is|therefore|thus|hence|so|we\s+get|we\s+have|we\s+find|I\s+get|answer[:=])\s*[:=]?\s*',
        '',
        text,
        flags=re.IGNORECASE
    )
    
    # Explicitly look for and remove "ANSWER:" marker
    text = re.sub(r'^ANSWER:\s*', '', text, flags=re.IGNORECASE)
    
    # Remove trailing punctuation
    text = re.sub(r'[.,;:]+$', '', text)
    
    return text.strip()

def remove_latex_formatting(text: str) -> str:
    """Remove LaTeX formatting."""
    # Remove \boxed{}
    text = re.sub(r'\\boxed{(.*?)}', r'\1', text)
    
    # Remove $$ markers
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text)
    
    # Remove $ markers
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    
    # Convert LaTeX fractions
    text = re.sub(r'\\frac{(.*?)}{(.*?)}', r'\1/\2', text)
    
    # Remove LaTeX left/right parentheses
    text = re.sub(r'\\left\((.*?)\\right\)', r'(\1)', text)
    
    # Remove basic LaTeX commands
    text = re.sub(r'\\text{(.*?)}', r'\1', text)
    
    # Convert LaTeX Greek letters
    greek_letters = {
        '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ', '\\delta': 'δ', 
        '\\epsilon': 'ε', '\\zeta': 'ζ', '\\eta': 'η', '\\theta': 'θ', 
        '\\iota': 'ι', '\\kappa': 'κ', '\\lambda': 'λ', '\\mu': 'μ', 
        '\\nu': 'ν', '\\xi': 'ξ', '\\pi': 'π', '\\rho': 'ρ', 
        '\\sigma': 'σ', '\\tau': 'τ', '\\upsilon': 'υ', '\\phi': 'φ', 
        '\\chi': 'χ', '\\psi': 'ψ', '\\omega': 'ω'
    }
    
    for latex, unicode in greek_letters.items():
        text = text.replace(latex, unicode)
    
    # Convert LaTeX special symbols
    special_symbols = {
        '\\times': '×', '\\div': '÷', '\\cdot': '·',
        '\\le': '≤', '\\ge': '≥', '\\ne': '≠',
        '\\infty': '∞', '\\pm': '±', '\\rightarrow': '→'
    }
    
    for latex, unicode in special_symbols.items():
        text = text.replace(latex, unicode)
    
    # Convert LaTeX square root
    text = re.sub(r'\\sqrt{(.*?)}', r'√\1', text)
    
    return text.strip()

def normalize_coordinates(text: str) -> str:
    """Normalize coordinate pairs."""
    # First check for LaTeX coordinate pairs: \left( 3, \frac{\pi}{2} \right)
    latex_coord_match = re.search(r'\\left\(\s*(.*?)\s*,\s*(.*?)\s*\\right\)', text)
    if latex_coord_match:
        x_coord = latex_coord_match.group(1).strip()
        y_coord = latex_coord_match.group(2).strip()
        
        # Process coordinates individually
        x_coord = remove_latex_formatting(x_coord)
        y_coord = remove_latex_formatting(y_coord)
        
        return f"({x_coord}, {y_coord})"
    
    # Check for simple coordinate pairs: (3, 4)
    simple_coord_match = re.search(r'\(\s*(.*?)\s*,\s*(.*?)\s*\)', text)
    if simple_coord_match:
        x_coord = simple_coord_match.group(1).strip()
        y_coord = simple_coord_match.group(2).strip()
        
        return f"({x_coord}, {y_coord})"
    
    return text

def normalize_math_answer(text: str) -> str:
    """Normalize a mathematical answer."""
    # First clean whitespace and remove markers
    text = clean_whitespace(text)
    text = remove_markers(text)
    
    # Remove any "the answer is" that might remain
    text = re.sub(r'^(?:the\s+answer\s+is|therefore|thus|hence|so)\s*[:=]?\s*', '', text, flags=re.IGNORECASE)
    
    # Apply general LaTeX formatting removal
    text = remove_latex_formatting(text)
    
    # Standardize decimal notation (both 0.5 and .5 become 0.5)
    text = re.sub(r'(\D|^)\.(\d+)', r'\g<1>0.\2', text)
    
    return text.strip()

def normalize_gpqa_answer(text: str) -> str:
    """Normalize a GPQA answer."""
    # Clean whitespace and remove markers
    text = clean_whitespace(text)
    text = remove_markers(text)
    
    # For multiple choice, extract just the letter
    mc_match = re.search(r'\b([A-E])\b', text)
    if mc_match:
        return mc_match.group(1)
    
    return text

# Simple mapping of domain to normalization function
def normalize_for_domain(domain: str) -> Callable[[str], str]:
    """Get the appropriate normalization function for a domain.
    
    Args:
        domain: The domain identifier
        
    Returns:
        A normalization function appropriate for the domain
    """
    normalizers = {
        'math': normalize_math_answer,
        'math500': normalize_math_answer,
        'aime24': normalize_math_answer,
        'gpqa': normalize_gpqa_answer,
        'general': lambda text: clean_whitespace(remove_markers(text)),
    }
    
    # Return the domain-specific normalizer or a basic one
    return normalizers.get(domain.lower(), lambda text: clean_whitespace(remove_markers(text)))

# Legacy compatibility functions
processor_registry: Dict[str, Callable[[str], str]] = {
    "clean_whitespace": clean_whitespace,
    "remove_markers": remove_markers,
    "remove_latex_formatting": remove_latex_formatting,
    "normalize_math_answer": normalize_math_answer,
    "normalize_gpqa_answer": normalize_gpqa_answer,
    "normalize_coordinates": normalize_coordinates,
}

def create_processor_pipeline(
    processors: list[str],
    fallback: Optional[Callable[[str], str]] = None
) -> Callable[[str], str]:
    """Legacy compatibility function for processor pipelines."""
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