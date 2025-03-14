#!/usr/bin/env python
"""Test script for debugging extraction metadata issues."""

import asyncio
from typing import Dict, Any, List

from benchpress.extraction import create_extractor, ExtractionContext
from benchpress.tasks.math500 import Math500Task
from benchpress.examples.math500 import Math500Example
from benchpress.tasks.base import TaskResult


async def test_extraction_bug_fix():
    """Test that the extraction bug fix works correctly."""
    # Create a sample model output with "ANSWER: 108"
    model_output = """Since the leading coefficients of $P(x)$ and $Q(x)$ are $2$ and $-2,$ respectively, we can write $P(x) = 2ax^2 + 2bx + c$ and $Q(x) = -2ax^2 - 2bx + d.$

Substituting the points $(16,54)$ and $(20,53)$ into these equations gives us the system of equations:

$2a(16)^2 + 2b(16) + c = 54$
$2a(20)^2 + 2b(20) + c = 53$
$-2a(16)^2 - 2b(16) + d = 54$
$-2a(20)^2 - 2b(20) + d = 53$

Solving this system of equations gives $a = -\\frac{1}{8},$ $b = 2,$ $c = 38,$ and $d = 70.$

Therefore, $P(0) + Q(0) = c + d = 38 + 70 = 108.$

ANSWER: 108"""

    # Create a mock example
    example = Math500Example(
        id="test_example",
        question="Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. "
                "The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ "
                "Find $P(0) + Q(0).$",
        answer="108",
        category="algebra",
        difficulty="medium",
    )

    # Create the task instance
    task = Math500Task()
    
    # Evaluate example - this should populate metadata correctly
    result = await task.evaluate_example(example, model_output)
    
    # Check that the metadata contains the extraction information
    print(f"Extract answer from task evaluation: {result.metadata.get('extracted_answer', 'MISSING')}")
    print(f"Extraction method: {result.metadata.get('extraction_method', 'MISSING')} / "
          f"alternative key: {result.metadata.get('method', 'MISSING')}")
    print(f"Extraction confidence: {result.metadata.get('extraction_confidence', 'MISSING')} / "
          f"alternative key: {result.metadata.get('confidence', 'MISSING')}")
    
    # Check that we have the new metadata fields
    print(f"Pattern type: {result.metadata.get('pattern_type', 'MISSING')}")
    print(f"Extractor: {result.metadata.get('extractor', 'MISSING')}")
    
    # Check alternative answers
    alt_answers = result.metadata.get('alternative_answers', [])
    print(f"Alternative answers count: {len(alt_answers)}")
    if alt_answers:
        for i, alt in enumerate(alt_answers[:3]):
            print(f"  Alt {i+1}: {alt['text']} (method={alt['method']}, confidence={alt['confidence']})")
    
    # Verify expected values in metadata
    checks = [
        ("extracted_answer", "108", result.metadata.get('extracted_answer') == "108"),
        ("method present", True, 'method' in result.metadata or 'extraction_method' in result.metadata),
        ("confidence present", True, 'confidence' in result.metadata or 'extraction_confidence' in result.metadata),
        ("method is correct", "explicit_answer_marker", 
         result.metadata.get('method') == "explicit_answer_marker" or 
         result.metadata.get('extraction_method') == "explicit_answer_marker"),
        ("pattern_type present", True, 'pattern_type' in result.metadata),
    ]
    
    # Print check results
    print("\nVerification checks:")
    all_passed = True
    for name, expected, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}: expected={expected}")
        if not result:
            all_passed = False
    
    # Final result
    if all_passed:
        print("\n✅ All extraction metadata checks passed - bug is fixed!")
    else:
        print("\n❌ Some extraction metadata checks failed - bug may still exist")

    return all_passed

if __name__ == "__main__":
    print("Testing extraction metadata bug fix...")
    asyncio.run(test_extraction_bug_fix())