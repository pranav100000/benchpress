"""Tests for the extraction module."""

import pytest

from benchpress.extraction import (
    ExtractionContext,
    GeneralExtractor,
    MathExtractor,
    create_extractor,
)


def test_general_extractor_explicit_markers():
    """Test extraction of answers with explicit markers."""
    extractor = GeneralExtractor()

    # Test with "the answer is" marker
    text = "After solving the equation, the answer is 42."
    context = ExtractionContext(domain="general", task_name="test")

    results = extractor.extract(text, context)
    assert len(results) > 0
    assert results[0].text == "42"
    assert results[0].confidence > 0.8

    # Test with "FINAL ANSWER:" marker
    text = "Working through the problem step by step...\nFINAL ANSWER: Hello, world!"
    results = extractor.extract(text, context)
    assert len(results) > 0
    assert results[0].text == "Hello, world!"
    assert results[0].confidence > 0.9


def test_math_extractor_math_expressions():
    """Test extraction of mathematical expressions."""
    extractor = MathExtractor()

    # Test with LaTeX boxed content
    text = "The solution is \\boxed{\\frac{3}{4}}"
    context = ExtractionContext(domain="math", task_name="test")

    results = extractor.extract(text, context)
    assert len(results) > 0
    assert results[0].text == "\\frac{3}{4}"
    assert results[0].normalized_text == "3/4"

    # Test with decimal numbers
    text = "After calculating, x = 3.14159"
    results = extractor.extract(text, context)
    assert len(results) > 0
    assert float(results[0].normalized_text) == pytest.approx(3.14159)


def test_math_extractor_multiple_candidates():
    """Test that math extractor finds multiple candidates with correct confidence ordering."""
    extractor = MathExtractor()

    # Simple text with a boxed answer
    text = "The answer is \\boxed{42}"

    context = ExtractionContext(domain="math", task_name="test")
    results = extractor.extract(text, context)

    # Check that we found at least one candidate
    assert len(results) >= 1

    # Find the boxed pattern result
    boxed_results = [r for r in results if "boxed" in r.pattern_name]
    assert len(boxed_results) > 0
    assert boxed_results[0].normalized_text == "42"

    # Now test with a different text where multiple patterns match the same answer
    text = """Working on the problem...
    First, I get 12.5 as an intermediate result.
    But upon further calculation, the answer is 42.
    """

    results = extractor.extract(text, context)
    assert len(results) >= 1
    assert results[0].text == "42"  # The answer should still be extracted correctly

    # Check that answer ordering is by confidence
    if len(results) > 1:
        assert results[0].confidence >= results[1].confidence


def test_create_extractor_function():
    """Test the create_extractor convenience function."""
    # Test automatic selection for math domain
    extractor = create_extractor(domain="math500")
    assert isinstance(extractor, MathExtractor)

    # Test automatic selection for general domain
    extractor = create_extractor(domain="general")
    assert isinstance(extractor, GeneralExtractor)

    # Test explicit extractor selection
    extractor = create_extractor(extractor_name="math")
    assert isinstance(extractor, MathExtractor)


def test_normalization():
    """Test answer normalization."""
    math_extractor = MathExtractor()

    # Test LaTeX formatting removal
    assert math_extractor.normalize("\\boxed{42}", ExtractionContext(domain="math", task_name="test")) == "42"

    # Test fraction normalization
    assert math_extractor.normalize("3/4", ExtractionContext(domain="math", task_name="test")) == "3/4"

    # Test decimal standardization
    assert math_extractor.normalize(".5", ExtractionContext(domain="math", task_name="test")) == "0.5"
