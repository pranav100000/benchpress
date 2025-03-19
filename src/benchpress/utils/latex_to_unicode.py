"""Utility functions for converting LaTeX to Unicode using unicodeit."""

import re
from typing import Dict, Tuple

import unicodeit


def latex_to_unicode(latex_str: str, colorize: bool = True) -> str:
    """Convert LaTeX expressions to Unicode characters using unicodeit.

    This function provides comprehensive LaTeX to Unicode conversion, handling:
    - Math expressions in dollar delimiters ($...$)
    - LaTeX environments like align, matrix, etc.
    - Common math commands and symbols
    - Fractions, superscripts, and subscripts
    - Boxed content

    Args:
        latex_str: String containing LaTeX expressions
        colorize: Whether to add color formatting to math expressions (for terminal)

    Returns:
        String with LaTeX converted to Unicode
    """
    if not latex_str:
        return ""

    # Make a working copy
    result = latex_str

    # Clean up the input for consistent processing
    # result = result.replace('\\\\', '\\')

    # # Step 1: Process LaTeX environments before unicodeit
    # # Handle special environment content - preserve line breaks and strip alignment markers
    # result = re.sub(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
    #               lambda m: m.group(1).replace("\\\\", "\n").replace("&", ""),
    #               result, flags=re.DOTALL)

    # # Handle other environments similarly
    # result = re.sub(r'\\begin\{([^{}]*)\}(.*?)\\end\{\1\}',
    #               lambda m: m.group(2).replace("\\\\", "\n"),
    #               result, flags=re.DOTALL)

    # # Step 2: Process dollar-delimited math expressions
    # # Store math expressions to prevent unicodeit from converting their delimiters
    # math_expressions = {}

    # def store_math_expression(match):
    #     math_expr = match.group(1)
    #     placeholder = f"__MATH_{len(math_expressions)}__"
    #     math_expressions[placeholder] = unicodeit.replace(math_expr)
    #     return placeholder

    # # Process inline math expressions ($...$)
    # result = re.sub(r'\$(.*?)\$', store_math_expression, result)
    # result = re.sub(r'\$\$(.*?)\$\$', store_math_expression, result)

    # Step 3: Process special constructs that unicodeit may not handle well

    # Handle boxed content
    result = re.sub(r"\\boxed\{([^{}]*)\}", r"[\1]", result)

    # Handle degree symbols
    result = re.sub(r'\^\\circ$|\^∘$', '°', result)

    # Handle \dots
    result = re.sub(r'\\dots', '...', result)

    # Handle fractions
    for _ in range(3):  # Multiple passes to handle nesting
        result = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", result)
        result = re.sub(r"\\dfrac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", result)

    # Step 4: Use unicodeit to convert remaining LaTeX commands
    result = unicodeit.replace(result)

    # Step 5: Restore math expressions
    # for placeholder, math_expr in math_expressions.items():
    #     result = result.replace(placeholder, math_expr)

    # Step 6: Post-processing
    # Step 5: Replace arrows

    # Fix alignment markers
    result = result.replace("&=", "=")
    result = result.replace("& =", "=")

    # Improve readability - add spaces after commas and periods if missing
    #result = re.sub(r'([,.])\s*([a-zA-Z])', r'\1 \2', result)

    # Clean up extra spaces
    #result = re.sub(r'\s+', ' ', result).strip()

    # Apply colorization for terminal display if requested
    if colorize:
        result = colorize_latex_for_terminal(result)

    return result


def colorize_latex_for_terminal(text: str) -> str:
    """Add terminal color formatting to highlight LaTeX math expressions.

    Args:
        text: Text to colorize

    Returns:
        Text with color formatting added
    """
    # Look for remaining dollar-delimited expressions
    pattern = r'\$(.*?)\$'

    # Replace with rich's color tags for terminal display
    colorized = re.sub(pattern, r'[magenta]\1[/magenta]', text)

    # We're removing the square bracket highlighting as it causes markup errors
    # when the text contains regular square brackets that aren't from \boxed{}

    return colorized
    # conversion process using a unique marker that doesn't conflict with rich markup

    return colorized

def format_unsimplified_fraction(numerator, denominator):
    """Format a fraction as a Unicode vulgar fraction without simplifying it.

    Args:
        numerator: Integer numerator
        denominator: Integer denominator

    Returns:
        Formatted string with Unicode fraction
    """
    # Dictionary of exact fractions and their Unicode equivalents
    exact_fractions = {
        (1, 4): "¼", (1, 2): "½", (3, 4): "¾",
        (1, 3): "⅓", (2, 3): "⅔",
        (1, 5): "⅕", (2, 5): "⅖", (3, 5): "⅗", (4, 5): "⅘",
        (1, 6): "⅙", (5, 6): "⅚",
        (1, 8): "⅛", (3, 8): "⅜", (5, 8): "⅝", (7, 8): "⅞"
    }

    # Check if it's an exact match for a predefined fraction
    fraction_tuple = (numerator, denominator)
    if fraction_tuple in exact_fractions:
        return exact_fractions[fraction_tuple]

    # Convert to super/subscript digits
    superscript_map = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    numerator_str = str(numerator).translate(superscript_map)
    denominator_str = str(denominator).translate(subscript_map)

    # Combine with fraction slash
    return f"{numerator_str}⁄{denominator_str}"

def simplify_fraction(numerator: int, denominator: int) -> Tuple[int, int]:
    """Simplify a fraction to its lowest terms.

    Args:
        numerator: The fraction numerator
        denominator: The fraction denominator

    Returns:
        Tuple of (simplified numerator, simplified denominator)
    """
    # Handle special cases
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    if numerator == 0:
        return (0, 1)

    # Find greatest common divisor
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    divisor = gcd(abs(numerator), abs(denominator))

    # Normalize sign (always keep denominator positive)
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1

    return (sign * abs(numerator) // divisor, abs(denominator) // divisor)


def format_fraction(numerator: int, denominator: int) -> str:
    """Format a fraction with Unicode fraction characters when possible.

    Args:
        numerator: The fraction numerator
        denominator: The fraction denominator

    Returns:
        Formatted fraction string
    """
    # Simplify the fraction first
    num, den = simplify_fraction(numerator, denominator)

    # Map of common fractions to Unicode characters
    common_fractions: Dict[Tuple[int, int], str] = {
        (1, 4): "¼",
        (1, 2): "½",
        (3, 4): "¾",
        (1, 3): "⅓",
        (2, 3): "⅔",
        (1, 5): "⅕",
        (2, 5): "⅖",
        (3, 5): "⅗",
        (4, 5): "⅘",
        (1, 6): "⅙",
        (5, 6): "⅚",
        (1, 8): "⅛",
        (3, 8): "⅜",
        (5, 8): "⅝",
        (7, 8): "⅞"
    }

    # Return Unicode character if it's a common fraction
    if (num, den) in common_fractions:
        return common_fractions[(num, den)]

    # Otherwise return in standard form
    return f"{num}/{den}"
