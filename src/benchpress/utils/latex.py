"""Utility functions for formatting LaTeX in terminal output."""

import re
import unicodeit
from typing import Dict


def format_latex_for_terminal(latex_str: str) -> str:
    """Convert LaTeX math expressions to more readable terminal-friendly format.

    Args:
        latex_str: String containing LaTeX expressions

    Returns:
        String with LaTeX converted to Unicode where possible
    """
    if not latex_str:
        return ""

    # Make a copy to work with
    result = latex_str

    # Clean up the input for consistent processing
    result = result.replace('\\\\', '\\')

    # Greek letters
    greek_letters: Dict[str, str] = {
        r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
        r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
        r"\theta": "θ", r"\vartheta": "ϑ", r"\iota": "ι", r"\kappa": "κ",
        r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ",
        r"\pi": "π", r"\varpi": "ϖ", r"\rho": "ρ", r"\varrho": "ϱ",
        r"\sigma": "σ", r"\varsigma": "ς", r"\tau": "τ", r"\upsilon": "υ",
        r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
        r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
        r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Upsilon": "Υ",
        r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω"
    }

    # Mathematical symbols
    math_symbols: Dict[str, str] = {
        r"\infty": "∞", r"\pm": "±", r"\mp": "∓", r"\times": "×",
        r"\div": "÷", r"\cdot": "·", r"\ast": "∗", r"\star": "★",
        r"\circ": "○", r"\bullet": "•", r"\oplus": "⊕", r"\otimes": "⊗",
        r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥",
        r"\ne": "≠", r"\neq": "≠", r"\approx": "≈", r"\cong": "≅",
        r"\equiv": "≡", r"\propto": "∝", r"\sim": "∼", r"\simeq": "≃",
        r"\subset": "⊂", r"\supset": "⊃", r"\subseteq": "⊆", r"\supseteq": "⊇",
        r"\in": "∈", r"\notin": "∉", r"\ni": "∋", r"\forall": "∀", r"\exists": "∃",
        r"\rightarrow": "→", r"\leftarrow": "←", r"\Rightarrow": "⇒",
        r"\Leftarrow": "⇐",
        r"\mapsto": "↦", r"\sum": "Σ", r"\prod": "Π", r"\int": "∫",
        r"\partial": "∂", r"\nabla": "∇", r"\sqrt": "√", r"\surd": "√",
        r"\cup": "∪", r"\cap": "∩", r"\emptyset": "∅", r"\therefore": "∴",
        r"\because": "∵", r"\mathbb{R}": "ℝ", r"\mathbb{Z}": "ℤ", r"\mathbb{N}": "ℕ",
        r"\mathbb{Q}": "ℚ", r"\mathbb{C}": "ℂ", r"\ldots": "…", r"\cdots": "⋯",
        r"\vdots": "⋮", r"\ddots": "⋱"
    }

    # Superscripts and subscripts
    superscripts = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
        "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽",
        ")": "⁾", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ",
        "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ", "k": "ᵏ", "l": "ˡ", "m": "ᵐ",
        "n": "ⁿ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ",
        "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ"
    }

    subscripts = {
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
        "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌", "(": "₍",
        ")": "₎", "a": "ₐ", "e": "ₑ", "i": "ᵢ", "j": "ⱼ", "o": "ₒ", "r": "ᵣ",
        "u": "ᵤ", "v": "ᵥ", "x": "ₓ"
    }

    # Special common fractions with unicode replacements
    common_fractions = {
        "\\frac{1}{2}": "½",
        "\\frac{1}{3}": "⅓",
        "\\frac{2}{3}": "⅔",
        "\\frac{1}{4}": "¼",
        "\\frac{3}{4}": "¾",
        "\\frac{17}{21}": "17/21"  # Common example
    }

    # Replace common fractions with unicode symbols (don't remove original yet)
    for tex, unicode_frac in common_fractions.items():
        result = result.replace(tex, unicode_frac)
    
    # Remove common LaTeX control sequences first
    result = result.replace("\\left", "").replace("\\right", "")
    
    # Replace Greek letters and mathematical symbols
    for tex, unicode in {**greek_letters, **math_symbols}.items():
        result = result.replace(tex, unicode)
        
    # Special handling for inline variables in math mode (like $x$, $y$, etc.)
    # These are common in math problems and should be preserved
    single_var_pattern = r'\$([a-zA-Z])[,.]?\$'
    result = re.sub(single_var_pattern, r'\1', result)
    
    # Now handle remaining dollar sign math
    # We use a non-greedy match to avoid capturing too much
    result = re.sub(r'\$(.*?)\$', r'\1', result)  # Simple inline math
    
    # Handle the case where dollar signs span multiple lines
    # This handles multiple dollars on different lines
    while '$' in result:
        result = result.replace('$', '', 1)$', result):
        var = match.group(1)
        placeholder = f"__VAR_{var}__"
        var_replacements[placeholder] = var
        result = result.replace(f"${var}$", placeholder)
        
    # Process LaTeX commands before removing backslashes
    # Handle boxed content first (can contain fractions)
    result = re.sub(r"\\boxed\{([^{}]*)\}", r"[\1]", result)
    
    # Handle fractions - first replace the command, then format with proper division
    for _ in range(3):  # Multiple passes to handle nesting
        result = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", result)
        result = re.sub(r"\\dfrac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", result)

    # Handle sqrt with Unicode square root symbol
    result = re.sub(r"\\sqrt\{([^{}]*)\}", r"√{\1}", result)

    # Handle simple superscripts: x^2 -> x²
    # First handle simple case like x^2
    result = re.sub(
        r"([A-Za-z0-9])\\?\^([0-9A-Za-z])",
        lambda m: m.group(1) + superscripts.get(m.group(2), "^" + m.group(2)),
        result)

    # Then handle more complex case with braces: x^{...}
    def replace_superscript(match):
        base = match.group(1)
        exp = match.group(2)

        # Convert each character in the exponent if possible
        formatted_exp = ""
        for char in exp:
            formatted_exp += superscripts.get(char, char)

        return base + formatted_exp

    result = re.sub(r"([A-Za-z0-9])\\?\^\{([^{}]*)\}", replace_superscript, result)

    # Handle simple subscripts: x_i -> xᵢ
    # First handle simple case like x_1
    result = re.sub(r"([A-Za-z0-9])_([0-9A-Za-z])",
                   lambda m: m.group(1) + subscripts.get(m.group(2), "_" + m.group(2)),
                   result)

    # Then handle more complex case with braces: x_{...}
    def replace_subscript(match):
        base = match.group(1)
        sub = match.group(2)

        # Convert each character in the subscript if possible
        formatted_sub = ""
        for char in sub:
            formatted_sub += subscripts.get(char, char)

        return base + formatted_sub

    result = re.sub(r"([A-Za-z0-9])_\{([^{}]*)\}", replace_subscript, result)

    # Handle integral notation
    result = result.replace(r"\int", "∫")
    # Then format subscripts and superscripts for integrals
    result = re.sub(r"∫_\{([^{}]*)\}\^\{([^{}]*)\}", r"∫_{\1}^{\2}", result)
    result = re.sub(r"∫_([^_{}]+)\^([^_{}]+)", r"∫_\1^\2", result)

    # Handle limits
    result = re.sub(r"\\lim_\{([^{}]*)\}", r"lim_{\1}", result)

    # Handle common operators
    result = re.sub(r"\\sin", "sin", result)
    result = re.sub(r"\\cos", "cos", result)
    result = re.sub(r"\\tan", "tan", result)
    result = re.sub(r"\\log", "log", result)
    result = re.sub(r"\\ln", "ln", result)
    result = re.sub(r"\\exp", "exp", result)
    
    # Handle LaTeX environments (align, matrix, etc.) - convert to plain text
    # Handle special environment content
    result = re.sub(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', 
                   lambda m: m.group(1).replace("\\\\", "\n").replace("&", ""), 
                   result, flags=re.DOTALL)
    
    # Handle other environments
    result = re.sub(r'\\begin\{([^{}]*)\}(.*?)\\end\{\1\}', 
                   lambda m: m.group(2).replace("\\\\", "\n"), 
                   result, flags=re.DOTALL)
                   
    # Fix alignment markers in environments
    result = result.replace("&=", "=")
    result = result.replace("& =", "=")

    # Remove common LaTeX control sequences
    result = result.replace("\\left", "").replace("\\right", "")

    # Remove remaining backslashes, but be careful not to affect our subscripts
    result = re.sub(r"\\([a-zA-Z]+)", r"\1", result)
    result = result.replace("\\", "")

    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    # Fix common text issues - missing spaces after commas and periods
    result = re.sub(r'([,.])\s*([a-zA-Z])', r'\1 \2', result)
    
    # Remove doubled spaces 
    result = re.sub(r'\s{2,}', ' ', result)

    return result
