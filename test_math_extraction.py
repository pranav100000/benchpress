"""Test the math extraction logic with a real example."""

from benchpress.extraction import create_extractor, ExtractionContext

def test_math_extraction():
    """Test math extraction on a specific problematic case."""
    model_output = """Since the leading coefficients of $P(x)$ and $Q(x)$ are $2$ and $-2,$ respectively, we can write $P(x) = 2ax^2 + 2bx + c$ and $Q(x) = -2ax^2 - 2bx + d.$

Substituting the points $(16,54)$ and $(20,53)$ into these equations gives us the system of equations:

$2a(16)^2 + 2b(16) + c = 54$
$2a(20)^2 + 2b(20) + c = 53$
$-2a(16)^2 - 2b(16) + d = 54$
$-2a(20)^2 - 2b(20) + d = 53$

Solving this system of equations gives $a = -\\frac{1}{8},$ $b = 2,$ $c = 38,$ and $d = 70.$

Therefore, $P(0) + Q(0) = c + d = 38 + 70 = 108.$

ANSWER: 108"""

    # Create extraction context
    context = ExtractionContext(
        domain="math500",
        task_name="math500",
        expected_format=None,
        question_type="math",
        metadata={}
    )
    
    # Create extractor
    extractor = create_extractor(domain="math500")
    
    # Extract answers
    candidates = extractor.extract(model_output, context)
    
    print(f"Found {len(candidates)} extraction candidates:")
    for i, candidate in enumerate(candidates):
        print(f"Candidate {i+1}:")
        print(f"  Text: {candidate.text}")
        print(f"  Normalized: {candidate.normalized_text}")
        print(f"  Pattern: {candidate.pattern_name}")
        print(f"  Confidence: {candidate.confidence}")
        print(f"  Metadata: {candidate.metadata}")
        print()
    
    if candidates:
        best = candidates[0]
        print(f"Best match: {best.normalized_text or best.text} (method={best.pattern_name}, confidence={best.confidence})")
    else:
        print("No answers extracted")

if __name__ == "__main__":
    test_math_extraction()