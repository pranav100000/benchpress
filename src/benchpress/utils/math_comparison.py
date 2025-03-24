"""Math expression comparison utilities using SymPy."""

import re
from typing import Optional, Tuple

import sympy
from sympy.parsing.latex import parse_latex


def compare_answers(llm_answer: str, reference_answer: str, domain: str = "math") -> bool:
    """Comprehensive comparison of answers using multiple strategies.

    Args:
        llm_answer: Answer extracted from LLM response
        reference_answer: Expected reference answer
        domain: Domain for specialized comparison (default: "math")

    Returns:
        True if answers match according to any strategy
    """
    # Guard against None values
    if llm_answer is None or reference_answer is None:
        return False

    # Convert both to strings for safety
    llm_answer = str(llm_answer).strip()
    reference_answer = str(reference_answer).strip()

    # 1. Raw comparison (exact match)
    if llm_answer == reference_answer:
        return True

    # 2. Basic normalization comparison
    # Use existing normalize_expression function for simple normalization
    normalized_llm_answer = normalize_expression(llm_answer)
    normalized_ref_answer = normalize_expression(reference_answer)
    if normalized_llm_answer == normalized_ref_answer:
        return True
    if normalized_llm_answer == reference_answer:
        return True
    if llm_answer == normalized_ref_answer:
        return True

    # 3. Mathematical comparison using SymPy (if in math domain)
    if domain.lower() in ("math", "math500", "aime24"):
        try:
            # Try mathematical comparison with SymPy
            if compare_math_expressions(llm_answer, reference_answer):
                return True
            if compare_math_expressions(normalized_llm_answer, reference_answer):
                return True
            if compare_math_expressions(llm_answer, normalized_ref_answer):
                return True
        except Exception:
            # If SymPy comparison fails, we already tried normalization above
            pass

    print(f"Failed to compare {llm_answer} and {reference_answer}")
    return False


def parse_coordinate_pair(text: str) -> Optional[Tuple[str, str]]:
    """Parse a coordinate pair from text.

    Args:
        text: Text potentially containing a coordinate pair

    Returns:
        Tuple of (x, y) coordinates or None if not a valid coordinate pair
    """
    # First check for LaTeX coordinate notation with \left( and \right)
    latex_match = re.match(r'^\s*\\left\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\\right\s*\)\s*$', text)
    if latex_match:
        x_coord = latex_match.group(1).strip()
        y_coord = latex_match.group(2).strip()
        return (x_coord, y_coord)

    # Match patterns like (3,π/2), (3, π/2), etc.
    match = re.match(r'^\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*$', text)
    if not match:
        return None

    x_coord = match.group(1).strip()
    y_coord = match.group(2).strip()

    return (x_coord, y_coord)


def compare_math_expressions(expr1: str, expr2: str) -> bool:
    """Compare two mathematical expressions for equivalence using SymPy.

    Args:
        expr1: First expression
        expr2: Second expression

    Returns:
        True if the expressions are mathematically equivalent, False otherwise
    """
    # If they're already string-equal after normalization, we're done
    if expr1 == expr2:
        return True

    # Try to compare as coordinate pairs
    coord1 = parse_coordinate_pair(expr1)
    coord2 = parse_coordinate_pair(expr2)

    if coord1 and coord2:
        # Both are coordinate pairs, compare components
        return compare_coordinate_pairs(coord1, coord2)

    # Try symbolic comparison
    try:
        return compare_with_sympy(expr1, expr2)
    except Exception:
        # If SymPy comparison fails, fall back to string comparison
        return normalize_expression(expr1) == normalize_expression(expr2)


def compare_coordinate_pairs(pair1: Tuple[str, str], pair2: Tuple[str, str]) -> bool:
    """Compare two coordinate pairs for mathematical equivalence.

    Args:
        pair1: First coordinate pair as (x, y)
        pair2: Second coordinate pair as (x, y)

    Returns:
        True if the pairs are mathematically equivalent
    """
    x1, y1 = pair1
    x2, y2 = pair2

    # Compare x-coordinates
    if not compare_expressions(x1, x2):
        return False

    # Compare y-coordinates
    return compare_expressions(y1, y2)


def compare_expressions(expr1: str, expr2: str) -> bool:
    """Compare two individual expressions.

    Args:
        expr1: First expression
        expr2: Second expression

    Returns:
        True if equivalent
    """
    # Direct string comparison
    if expr1 == expr2:
        return True

    # Try symbolic comparison
    try:
        return compare_with_sympy(expr1, expr2)
    except Exception:
        # Fall back to string comparison and basic normalization
        return normalize_expression(expr1) == normalize_expression(expr2)


def parse_fraction(expr: str) -> str:
    """Parse and normalize fraction expressions.
    
    Args:
        expr: Expression potentially containing fractions
        
    Returns:
        Normalized fraction expression
    """
    # Replace first \frac{...}{...} or \dfrac{...}{...} we find with (...)/(...)
    expr = re.sub(r'\\dfrac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', 
                 r'(\1)/(\2)', expr, count=1)
    expr = re.sub(r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', 
                 r'(\1)/(\2)', expr, count=1)
    
    return expr


def normalize_expression(expr: str) -> str:
    """Enhanced normalization for expressions that handles LaTeX and Unicode math."""
    if not expr:
        return ""

    # Remove all whitespace first for consistent processing
    expr = re.sub(r'\s+', '', expr)

    # Only handle pmatrix notation if we detect both \begin{pmatrix} and \end{pmatrix}
    if '\\begin{pmatrix}' in expr and '\\end{pmatrix}' in expr:
        # First replace \\ with commas within the pmatrix environment
        expr = re.sub(r'\\\\', ',', expr)
        # Then convert the pmatrix environment to square brackets
        expr = expr.replace('\\begin{pmatrix}', '[')
        expr = expr.replace('\\end{pmatrix}', ']')
    
    # Remove LaTeX math mode delimiters
    expr = re.sub(r'^\\\((.+?)\\\)$', r'\1', expr)
    
    # If there's an equals sign, take everything after it
    if '=' in expr:
        expr = expr.split('=')[-1]
    
    # Replace infinity with infty
    expr = expr.replace('∞', 'infty')  # unicode infinity
    expr = expr.replace('\\infty', 'infty')  # latex infinity

    # If there's a \in, take everything after it
    if '\\in' in expr:
        expr = expr.split('\\in')[-1]

    # Remove \mbox and everything after it
    expr = re.sub(r'\\mbox.*$', '', expr)
    
    # Remove LaTeX spacing commands with commas
    expr = re.sub(r',\\!', '', expr)
    
    # Parse LaTeX fractions (repeatedly until no more fractions found)
    while '\\frac' in expr or '\\dfrac' in expr:
        print(f"Parsing fraction: {expr}")
        expr = parse_fraction(expr)
    
    # Normalize special symbols first
    expr = expr.replace('∪', ',')  # convert union to comma
    expr = expr.replace('\\cup', ',')  # convert \cup to comma
    
    # Remove parentheses from intervals
    expr = re.sub(r'\((.*?)\)', r'\1', expr)
    expr = re.sub(r'\\left\((.*?)\\right\)', r'\1', expr)
    
    # Handle \text{...} content - remove unnecessary parentheses inside \text
    expr = re.sub(r'\\text\{[(\s]*([^}]*?)[)\s]*\}', r'\1', expr)
    
    # Remove decorative LaTeX commands like \boxed first
    expr = re.sub(r'\\boxed\{([^}]*)\}', r'\1', expr)
    
    # Handle square root notation in different forms first
    # LaTeX \sqrt{...}
    expr = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', expr)
    # LaTeX \sqrt without braces
    expr = re.sub(r'\\sqrt(\d+)', r'sqrt(\1)', expr)
    # Unicode √ - make sure to only capture what's meant to be inside the root
    expr = re.sub(r'√(\d+|{[^}]*})', r'sqrt(\1)', expr)
    expr = re.sub(r'√([a-zA-Z])', r'sqrt(\1)', expr)

    # Handle \pm or ± expansion
    pm_match = re.search(r'(.*?)(?:\\pm|±)(.*)', expr)
    if pm_match:
        base = pm_match.group(1).strip()
        term = pm_match.group(2).strip()
        # Sort the terms to ensure consistent ordering
        terms = sorted([f"{base}+{term}", f"{base}-{term}"])
        expr = ','.join(terms)

    # Remove formatting commas in numbers
    expr = re.sub(r'(\d),(\d)', r'\1\2', expr)
    
    # Handle fractions more flexibly
    expr = re.sub(r'\\dfrac\s*(\{[^}]*\}|\d+)\s*(\{[^}]*\}|\d+)', r'(\1)/(\2)', expr)
    expr = re.sub(r'\\frac\s*(\{[^}]*\}|\d+)\s*(\{[^}]*\}|\d+)', r'(\1)/(\2)', expr)
    
    # Handle comma-separated lists by sorting
    if ',' in expr:
        parts = expr.split(',')
        # Only skip sorting for explicit coordinate pairs like (1,2)
        if not (expr.startswith('(') and expr.endswith(')')):
            parts = sorted(parts)
            expr = ','.join(parts)

    # Remove ) and any ( that comes after it
    expr = re.sub(r'\)([^)]*)\(', r'\1', expr)

    # Remove unnecessary parentheses
    expr = re.sub(r'^\((.*)\)$', r'\1', expr)

    # Remove all whitespace
    expr = re.sub(r'\s+', '', expr)

    # Replace various pi symbols
    expr = expr.replace("π", "pi").replace("\\pi", "pi")

    # Remove LaTeX command markers and braces
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = expr.replace("{", "").replace("}", "")

    # Remove dollar signs
    expr = expr.replace("$", "")

    # Remove degree symbols (both LaTeX and unicode versions)
    expr = re.sub(r'\^\\circ|\^∘', '', expr)
    expr = re.sub(r'\\text\{\s*degrees\s*\}', '', expr)  # LaTeX form with \text
    expr = re.sub(r'\s*degrees\s*', '', expr)  # Plain text form
    expr = re.sub(r'°', '', expr)  # Unicode degree symbol

    # Remove suffixes matching _number pattern
    expr = re.sub(r'_\d+$', '', expr)

    # Convert superscript numbers to ^n form
    superscript_map = {'²': '^2', '³': '^3', '⁴': '^4', '⁵': '^5', '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9'}
    for sup, repl in superscript_map.items():
        expr = expr.replace(sup, repl)

    # Remove all whitespace again (in case any was introduced)
    expr = re.sub(r'\s+', '', expr)

    # Remove backslashes last
    expr = expr.replace("\\", "")

    print(f"Normalized expression: {expr}")
    return expr.lower()


def compare_with_sympy(expr1: str, expr2: str) -> bool:
    """Compare expressions using SymPy.

    Args:
        expr1: First expression
        expr2: Second expression

    Returns:
        True if symbolically equivalent

    Raises:
        Exception: If SymPy comparison fails
    """
    # Try to parse expressions
    try:
        # Prepare for sympy parsing
        expr1 = normalize_for_sympy(expr1)
        expr2 = normalize_for_sympy(expr2)

        # First try parsing as LaTeX
        try:
            sym1 = parse_latex(expr1)
            sym2 = parse_latex(expr2)
        except Exception:
            # If LaTeX parsing fails, try regular SymPy parsing
            sym1 = sympy.sympify(expr1)
            sym2 = sympy.sympify(expr2)

        # Check if the expressions are equal
        return sympy.simplify(sym1 - sym2) == 0
    except Exception as e:
        # If parsing fails, re-raise the exception
        raise Exception(f"Failed to compare with SymPy: {e}") from e


def normalize_for_sympy(expr: str) -> str:
    """Normalize expression for SymPy parsing.

    Args:
        expr: Expression to normalize

    Returns:
        Normalized expression ready for SymPy parsing
    """
    # Replace unicode π with pi
    expr = expr.replace("π", "pi")

    # Replace LaTeX \pi with pi
    expr = expr.replace("\\pi", "pi")

    # Handle LaTeX coordinate pairs - this is a special case since coordinates
    # might not parse well with sympy otherwise
    coord_match = re.match(r'\\left\s*\(\s*(.*?)\s*,\s*(.*?)\s*\\right\s*\)', expr)
    if coord_match:
        x = coord_match.group(1).strip()
        y = coord_match.group(2).strip()
        # Handle fractions in coordinates
        y = re.sub(r'\\frac\s*\{(.*?)\}\s*\{(.*?)\}', r'(\1)/(\2)', y)
        return f"({x},{y})"

    # Replace LaTeX fractions
    expr = re.sub(r"\\frac\s*\{(.*?)\}\s*\{(.*?)\}", r"(\1)/(\2)", expr)

    # Remove LaTeX command markers
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = expr.replace("\\", "")

    # Remove LaTeX braces
    expr = expr.replace("{", "").replace("}", "")

    # Remove spaces - sympy doesn't need them
    expr = re.sub(r'\s+', '', expr)

    return expr
