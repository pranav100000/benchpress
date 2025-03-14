"""Utility functions for converting LaTeX to Unicode using unicodeit."""

import re
import unicodeit


def latex_to_unicode(latex_str: str) -> str:
    """Convert LaTeX expressions to Unicode characters using unicodeit.

    Args:
        latex_str: String containing LaTeX expressions

    Returns:
        String with LaTeX converted to Unicode
    """
    if not latex_str:
        return ""

    # Make a working copy
    result = latex_str
    
    # # Process LaTeX environments (align, matrix, etc.) before unicodeit
    # # Handle special environment content - preserve line breaks and strip alignment markers
    # result = re.sub(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', 
    #               lambda m: m.group(1).replace("\\\\", "\n").replace("&", ""), 
    #               result, flags=re.DOTALL)
    
    # # Handle other environments similarly
    # result = re.sub(r'\\begin\{([^{}]*)\}(.*?)\\end\{\1\}', 
    #               lambda m: m.group(2).replace("\\\\", "\n"), 
    #               result, flags=re.DOTALL)
    
    # # First preserve any inline variables (like $x$, $y$, etc.) 
    # # since these are common in math problems
    # def preserve_inline_vars(match):
    #     # Keep just the variable without $ signs
    #     return match.group(1)
    
    # result = re.sub(r'\$\s*([a-zA-Z])\s*\$', preserve_inline_vars, result)
    
    # Process dollar-delimited math expressions
    # Extract content between $ signs for conversion
    def convert_math_expression(match):
        math_expr = match.group(1)
        # Use unicodeit to convert LaTeX to unicode
        return unicodeit.replace(math_expr)
    
    # # Process inline math expressions ($...$)
    # result = re.sub(r'\$(.*?)\$', convert_math_expression, result)
    
    # # Clean up any remaining dollar signs (in case of unmatched pairs)
    # result = result.replace('$', '')
    
    # Use unicodeit to convert any remaining LaTeX commands
    # This handles common LaTeX commands like \alpha, \infty, etc.
    result = unicodeit.replace(result)
    result = colorize_latex(result)
    
    # # Improve readability - add spaces after commas and periods if missing
    # result = re.sub(r'([,.])\s*([a-zA-Z])', r'\1 \2', result)
    
    # # Clean up extra spaces
    # result = re.sub(r'\s+', ' ', result).strip()
    
    # Fix alignment markers that might remain
    result = result.replace("&=", "=")
    result = result.replace("& =", "=")
    
    return result

def colorize_latex(text) -> str:

    pattern = r'\$(.*?)\$'
    
    # Replace with colored tags
    colorized = re.sub(pattern, r'[magenta]\1[/magenta]', text)
    
    return colorized