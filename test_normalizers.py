"""Test script for normalization functions."""

from benchpress.extraction.processors import normalize_math_answer
from benchpress.tasks.math500 import Math500Task


def test_processors_normalizer():
    """Test the normalization function in processors.py."""
    test_cases = [
        {"input": "3/4", "expected": "3/4", "description": "Numeric fraction"},
        {"input": "p/q", "expected": "p/q", "description": "Symbolic fraction"},
        {"input": "n/k", "expected": "n/k", "description": "Another symbolic fraction"},
        {"input": "3.14", "expected": "3.14", "description": "Decimal number"},
        {"input": "\\frac{3}{4}", "expected": "3/4", "description": "LaTeX fraction"},
        {"input": "ANSWER: 5", "expected": "5", "description": "With ANSWER: marker"},
        {"input": "The answer is 5", "expected": "5", "description": "With old-style marker"},
    ]

    for case in test_cases:
        result = normalize_math_answer(case["input"])
        status = "✓" if result == case["expected"] else "✗"
        print(f"{status} {case['description']}: '{case['input']}' -> '{result}' (expected '{case['expected']}')")

def test_math500_normalizer():
    """Test the normalization method in Math500Task."""
    task = Math500Task()

    test_cases = [
        {"input": "3/4", "expected": "3/4", "description": "Numeric fraction"},
        {"input": "p/q", "expected": "p/q", "description": "Symbolic fraction"},
        {"input": "n/k", "expected": "n/k", "description": "Another symbolic fraction"},
        {"input": "3.14", "expected": "3.14", "description": "Decimal number"},
        {"input": "\\frac{3}{4}", "expected": "3/4", "description": "LaTeX fraction"},
        {"input": "ANSWER: 5", "expected": "5", "description": "With ANSWER: marker"},
        {"input": "The answer is 5", "expected": "theansweris5", "description": "With old-style marker (spaces removed in Math500)"},
    ]

    for case in test_cases:
        result = task._normalize_math_answer(case["input"])
        status = "✓" if result == case["expected"] else "✗"
        print(f"{status} {case['description']}: '{case['input']}' -> '{result}' (expected '{case['expected']}')")

if __name__ == "__main__":
    print("Testing normalizer in processors.py:")
    test_processors_normalizer()

    print("\nTesting normalizer in Math500Task:")
    test_math500_normalizer()
