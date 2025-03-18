import sympy as sp
from benchpress.extraction.math_expr_comparison import verify_equality, batch_verify_equalities

# Example: Verify a single expression from an LLM
llm_expression = "sin(x)^2/(1-cos(x)^2)"
correct_expression = "tan(x)^2/(1+tan(x)^2)"

is_equal, details = verify_equality(
    llm_expression, 
    correct_expression,
    verbose=True,  # Set to True to see detailed verification steps
    complex_check=True  # Important for catching edge cases
)

print(f"LLM expression is correct: {is_equal}")

# For batch processing of multiple problems
llm_answers = [
    "(x+y)^3", 
    "sin(2x)",
    "log(x^2)"
]
correct_answers = [
    "x^3 + 3*x^2*y + 3*x*y^2 + y^3",
    "2*sin(x)*cos(x)",
    "2*log(x)"
]

# Create pairs of expressions
expression_pairs = list(zip(llm_answers, correct_answers))

# Verify all pairs
results = batch_verify_equalities(
    expression_pairs,
    assumptions_list=[
        None,  # No assumptions for first pair
        None,  # No assumptions for second pair
        {"x": {"positive": True}}  # Assumption for third pair
    ],
    numerical_samples=50  # Increase for more thorough testing
)

# Calculate accuracy
accuracy = sum(result[0] for result in results) / len(results)
print(f"LLM accuracy: {accuracy * 100:.2f}%")