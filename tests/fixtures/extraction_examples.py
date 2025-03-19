"""Test fixtures for extraction tests."""

from typing import List, Tuple

import pytest
from benchpress.extraction.base import ExtractionContext

# Sample math problem responses with expected extractions
MATH_RESPONSES = [
    # Format: (model response, expected extracted answer)
    (
        """I'll solve this step by step.
        First, I need to calculate the value of x.
        x = 5 + 3 = 8
        Therefore, the answer is 8.
        """,
        "8"
    ),
    (
        """To solve this problem:
        1. Find the derivative of f(x) = x^2 + 3x + 2
        2. f'(x) = 2x + 3
        3. Evaluate at x = 4
        4. f'(4) = 2(4) + 3 = 8 + 3 = 11

        ANSWER: 11
        """,
        "11"
    ),
    (
        """The solution is as follows:
        We have to find the area of the triangle.
        Area = (1/2) × base × height
        Area = (1/2) × 6 × 4
        Area = 12

        Therefore, the answer is 12 square units.
        """,
        "12"
    ),
    (
        """Let me solve this step by step:
        The equation 2x + 3 = 9 can be rearranged:
        2x = 6
        x = 3

        FINAL ANSWER: 3
        """,
        "3"
    ),
    (
        """To find the value of n:
        Given expression: n^2 - 14n + 45 = 0
        Using the quadratic formula:
        n = (14 ± √(14^2 - 4×1×45)) / 2
        n = (14 ± √(196 - 180)) / 2
        n = (14 ± √16) / 2
        n = (14 ± 4) / 2
        n = 9 or n = 5

        Since we need the larger value, n = 9.
        The answer is 9.
        """,
        "9"
    ),
    (
        """This problem requires using the volume formula for a sphere.
        V = (4/3)πr^3
        Given r = 5, we have:
        V = (4/3)π(5)^3
        V = (4/3)π(125)
        V = (4/3)(125)π
        V = (500/3)π

        The volume is \\boxed{\\frac{500\\pi}{3}} cubic units.
        """,
        "500π/3"
    ),
]

# Sample math problems with fraction answers
FRACTION_RESPONSES = [
    (
        """The probability is 3/7.

        Therefore, the answer is 3/7.
        """,
        "3/7"
    ),
    (
        """After simplifying the expression:
        x = 5/8

        ANSWER: 5/8
        """,
        "5/8"
    ),
    (
        """The final value is \\frac{17}{42}.

        Therefore, the answer is \\frac{17}{42}.
        """,
        "17/42"
    ),
]

# Responses with LaTeX formatting
LATEX_RESPONSES = [
    (
        """The solution is $\\alpha + \\beta = \\gamma$.

        Therefore, $\\alpha = 42$.
        """,
        "42"
    ),
    (
        """The area of the circle is $A = \\pi r^2 = \\pi \\cdot 3^2 = 9\\pi$.

        ANSWER: $9\\pi$
        """,
        "9π"
    ),
    (
        """The answer is $\\boxed{\\sqrt{2} + 1}$.
        """,
        "√2 + 1"
    ),
]

# Negative examples (should not extract correctly)
NEGATIVE_EXAMPLES = [
    (
        """I'm not sure how to solve this problem completely.
        Let me think about it more...
        """,
        None
    ),
    (
        """This problem involves several steps:
        1. Calculate the value of x
        2. Determine the relationship between variables
        But I need more information to solve it completely.
        """,
        None
    ),
]

# Multi-extract examples (multiple possible answers in response)
MULTI_EXTRACT_EXAMPLES = [
    (
        """First, I calculate x = 5.
        Then, I find y = 10.
        The final answer is z = 15.
        """,
        "15"  # Should extract the final one
    ),
    (
        """I get x = 7 as an intermediate result.
        But after applying the final step, I get ANSWER: 42
        """,
        "42"  # Should prioritize the explicit answer marker
    ),
]


@pytest.fixture
def extraction_context() -> ExtractionContext:
    """Create a basic extraction context for testing."""
    return ExtractionContext(
        domain="math",
        task_name="test_task",
        expected_format=None,
        question_type="math",
    )


@pytest.fixture
def math_responses() -> List[Tuple[str, str]]:
    """Return sample math problem responses with expected answers."""
    return MATH_RESPONSES


@pytest.fixture
def fraction_responses() -> List[Tuple[str, str]]:
    """Return sample responses with fraction answers."""
    return FRACTION_RESPONSES


@pytest.fixture
def latex_responses() -> List[Tuple[str, str]]:
    """Return sample responses with LaTeX formatting."""
    return LATEX_RESPONSES


@pytest.fixture
def negative_examples() -> List[Tuple[str, str]]:
    """Return examples that should not extract correctly."""
    return NEGATIVE_EXAMPLES


@pytest.fixture
def multi_extract_examples() -> List[Tuple[str, str]]:
    """Return examples with multiple possible extractions."""
    return MULTI_EXTRACT_EXAMPLES
