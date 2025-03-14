"""Test script for the answer extraction logic."""

from benchpress.extraction.base import ExtractionContext
from benchpress.extraction.general import GeneralExtractor
from benchpress.extraction.math import MathExtractor


def test_general_extraction():
    """Test general extraction with explicit ANSWER patterns."""
    extractor = GeneralExtractor()
    context = ExtractionContext(domain="general", task_name="test")

    test_cases = [
        {
            "model_output": "I think the solution is 42. ANSWER: 42",
            "expected": "42",
            "description": "Simple numeric answer with ANSWER: marker"
        },
        {
            "model_output": "After analyzing the problem...\n\nANSWER: 3.14159",
            "expected": "3.14159",
            "description": "Decimal answer with ANSWER: marker"
        },
        {
            "model_output": "The fraction simplifies to p/q where p and q are coprime.\nANSWER: p/q",
            "expected": "p/q",
            "description": "Symbolic fraction with ANSWER: marker"
        },
        {
            "model_output": "Solving for the ratio, I get n/k.\nANSWER: n/k",
            "expected": "n/k",
            "description": "Another symbolic fraction with ANSWER: marker"
        }
    ]

    for case in test_cases:
        model_output = case["model_output"]
        expected = case["expected"]
        description = case["description"]

        results = extractor.extract(model_output, context)

        if results:
            best_result = results[0]
            extracted = best_result.normalized_text

            status = "✓" if extracted == expected else "✗"
            print(f"{status} {description}: extracted '{extracted}' using {best_result.pattern_name} (expected '{expected}')")
        else:
            print(f"✗ {description}: No answer extracted (expected '{expected}')")

def test_math_extraction():
    """Test math-specific extraction with symbolic fractions."""
    extractor = MathExtractor()
    context = ExtractionContext(domain="math", task_name="math500")

    test_cases = [
        {
            "model_output": "The LCM of these numbers is 36. Therefore, the answer is 36.",
            "expected": "36",
            "description": "Simple answer with therefore marker"
        },
        {
            "model_output": "The simplified expression is \\boxed{x^2 + 2x + 1}",
            "expected": "x^2 + 2x + 1",
            "description": "Algebraic expression in LaTeX box"
        },
        {
            "model_output": "The ratio of these values is exactly \\frac{p}{q}.",
            "expected": "p/q",
            "description": "Symbolic fraction in LaTeX"
        },
        {
            "model_output": "After substituting the values, we get n/k.",
            "expected": "n/k",
            "description": "Simple symbolic fraction"
        },
        {
            "model_output": "I will solve this step by step.\n...\nANSWER: p/q",
            "expected": "p/q",
            "description": "Symbolic fraction with ANSWER marker"
        }
    ]

    for case in test_cases:
        model_output = case["model_output"]
        expected = case["expected"]
        description = case["description"]

        results = extractor.extract(model_output, context)

        if results:
            best_result = results[0]
            extracted = best_result.normalized_text

            status = "✓" if extracted == expected else "✗"
            print(f"{status} {description}: extracted '{extracted}' using {best_result.pattern_name} (expected '{expected}')")
        else:
            print(f"✗ {description}: No answer extracted (expected '{expected}')")

if __name__ == "__main__":
    print("Testing general extractor with explicit ANSWER markers:")
    test_general_extraction()

    print("\nTesting math extractor with symbolic fractions:")
    test_math_extraction()
