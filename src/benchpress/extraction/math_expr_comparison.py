import sympy as sp
import numpy as np
from typing import Union, Tuple, List, Dict, Any, Optional
import random
import warnings

def verify_equality(
    expr1: Union[str, sp.Expr], 
    expr2: Union[str, sp.Expr],
    variables: Optional[List[str]] = None,
    assumptions: Optional[Dict[str, Any]] = None,
    numerical_samples: int = 20,
    tol: float = 1e-10,
    domain_range: Tuple[float, float] = (-10, 10),
    complex_check: bool = False,
    verbose: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensively verifies if two mathematical expressions are equal.
    
    Parameters:
    -----------
    expr1 : Union[str, sp.Expr]
        First expression to compare (string or SymPy expression)
    expr2 : Union[str, sp.Expr]
        Second expression to compare (string or SymPy expression)
    variables : Optional[List[str]]
        List of variable names in the expressions
        If None, will attempt to extract variables automatically
    assumptions : Optional[Dict[str, Any]]
        Dictionary of assumptions for the variables (e.g., {'x': {'positive': True}})
    numerical_samples : int
        Number of random points to sample for numerical verification
    tol : float
        Numerical tolerance for floating-point comparisons
    domain_range : Tuple[float, float]
        Range for random sampling of real values
    complex_check : bool
        Whether to also check equality with complex values
    verbose : bool
        Whether to print detailed verification steps
    
    Returns:
    --------
    Tuple[bool, Dict[str, Any]]
        (is_equal, results_dict)
        is_equal: True if expressions are equivalent, False otherwise
        results_dict: Dictionary containing detailed results of each verification method
    """
    # Convert string expressions to SymPy expressions if needed
    if isinstance(expr1, str):
        try:
            expr1 = sp.sympify(expr1)
        except Exception as e:
            return False, {"error": f"Error parsing first expression: {str(e)}"}
    
    if isinstance(expr2, str):
        try:
            expr2 = sp.sympify(expr2)
        except Exception as e:
            return False, {"error": f"Error parsing second expression: {str(e)}"}
    
    # Automatically extract variables if not provided
    if variables is None:
        all_symbols = set(expr1.free_symbols).union(set(expr2.free_symbols))
        variables = [str(sym) for sym in all_symbols]
    
    # Create symbols with assumptions
    symbols_dict = {}
    for var in variables:
        if assumptions and var in assumptions:
            var_assumptions = assumptions[var]
            symbols_dict[var] = sp.Symbol(var, **var_assumptions)
        else:
            symbols_dict[var] = sp.Symbol(var)
    
    # Substitute symbols with assumptions if needed
    if assumptions:
        for var, symbol in symbols_dict.items():
            expr1 = expr1.subs(sp.Symbol(var), symbol)
            expr2 = expr2.subs(sp.Symbol(var), symbol)
    
    # Store results from different verification methods
    results = {}
    
    # Method 1: Check structural equality
    structural_equal = expr1 == expr2
    results["structural_equality"] = structural_equal
    if verbose and structural_equal:
        print("✓ Expressions are structurally identical.")
    
    # Method 2: Compute difference and check if it simplifies to zero
    diff = expr1 - expr2
    
    # Try various simplification methods
    simple_methods = {
        "simplify": sp.simplify,
        "trigsimp": sp.trigsimp,
        "expand": lambda e: sp.expand(e, complex=True) if complex_check else sp.expand(e),
        "factor": sp.factor,
        "cancel": sp.cancel,
        "apart": sp.apart,
        "nsimplify": sp.nsimplify,
        "radsimp": sp.radsimp,
        "powsimp": lambda e: sp.powsimp(e, force=True),
        "logcombine": sp.logcombine
    }
    
    # Special simplifiers for trig expressions
    trig_methods = {
        "expand_trig": sp.expand_trig,
        "fu": sp.fu,
    }
    
    # Apply simplification methods
    algebraic_equal = False
    simplification_results = {}
    
    # Try basic simplifications
    for name, method in simple_methods.items():
        try:
            result = method(diff)
            is_zero = result == 0
            simplification_results[name] = is_zero
            algebraic_equal = algebraic_equal or is_zero
            if verbose and is_zero:
                print(f"✓ Expressions are equal after {name}.")
        except Exception as e:
            simplification_results[name] = f"Error: {str(e)}"
    
    # Try trigonometric simplifications if expressions contain trig functions
    trig_funcs = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
    has_trig = any(func in str(expr1) or func in str(expr2) for func in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc'])
    
    if has_trig:
        for name, method in trig_methods.items():
            try:
                result = method(diff)
                is_zero = result == 0
                simplification_results[name] = is_zero
                algebraic_equal = algebraic_equal or is_zero
                if verbose and is_zero:
                    print(f"✓ Expressions are equal after {name}.")
            except Exception as e:
                simplification_results[name] = f"Error: {str(e)}"
    
    results["algebraic_methods"] = simplification_results
    results["algebraic_equal"] = algebraic_equal
    
    # Method 3: Check equality using SymPy's equals method
    try:
        sympy_equals = expr1.equals(expr2)
        results["sympy_equals"] = sympy_equals
        if verbose and sympy_equals:
            print("✓ SymPy's equals() confirms expressions are equal.")
    except Exception as e:
        results["sympy_equals"] = f"Error: {str(e)}"
    
    # Method 4: Numerical verification at random points
    numerical_equal = True
    numerical_failures = []
    
    # Prepare symbols list for substitution
    symbols_list = [symbols_dict[var] for var in variables]
    
    # Test with real values
    for _ in range(numerical_samples):
        try:
            # Generate random values for variables
            values = {sym: random.uniform(domain_range[0], domain_range[1]) for sym in symbols_list}
            
            # Evaluate both expressions
            val1 = float(expr1.subs(values).evalf())
            val2 = float(expr2.subs(values).evalf())
            
            # Check if values are approximately equal
            if np.isnan(val1) and np.isnan(val2):
                continue  # Both are NaN, which is consistent
            elif np.isnan(val1) or np.isnan(val2):
                numerical_equal = False
                numerical_failures.append({
                    "values": {str(k): float(v) for k, v in values.items()},
                    "expr1_value": "NaN" if np.isnan(val1) else val1,
                    "expr2_value": "NaN" if np.isnan(val2) else val2
                })
                break
            elif abs(val1 - val2) > tol:
                numerical_equal = False
                numerical_failures.append({
                    "values": {str(k): float(v) for k, v in values.items()},
                    "expr1_value": val1,
                    "expr2_value": val2,
                    "difference": abs(val1 - val2)
                })
                break
        except Exception as e:
            warnings.warn(f"Error in numerical evaluation: {str(e)}")
            continue
    
    # Test with complex values if requested
    if complex_check and numerical_equal:
        for _ in range(numerical_samples):
            try:
                # Generate random complex values for variables
                values = {
                    sym: complex(
                        random.uniform(domain_range[0], domain_range[1]),
                        random.uniform(domain_range[0], domain_range[1])
                    ) for sym in symbols_list
                }
                
                # Evaluate both expressions
                val1 = complex(expr1.subs(values).evalf())
                val2 = complex(expr2.subs(values).evalf())
                
                # Check if values are approximately equal
                if abs(val1 - val2) > tol:
                    numerical_equal = False
                    numerical_failures.append({
                        "values": {str(k): f"{v.real}+{v.imag}j" for k, v in values.items()},
                        "expr1_value": f"{val1.real}+{val1.imag}j",
                        "expr2_value": f"{val2.real}+{val2.imag}j",
                        "difference": abs(val1 - val2)
                    })
                    break
            except Exception as e:
                warnings.warn(f"Error in complex numerical evaluation: {str(e)}")
                continue
    
    results["numerical_equal"] = numerical_equal
    results["numerical_failures"] = numerical_failures
    
    if verbose:
        if numerical_equal:
            print(f"✓ Expressions are numerically equal at {numerical_samples} random points.")
        elif numerical_failures:
            print(f"✗ Expressions differ numerically. Example:")
            failure = numerical_failures[0]
            print(f"  Values: {failure['values']}")
            print(f"  Expr1: {failure['expr1_value']}")
            print(f"  Expr2: {failure['expr2_value']}")
            print(f"  Difference: {failure.get('difference', 'N/A')}")
    
    # Final verdict: Expressions are equal if any method confirms equality
    is_equal = structural_equal or algebraic_equal or results.get("sympy_equals", False) or numerical_equal
    results["is_equal"] = is_equal
    
    if verbose:
        if is_equal:
            print("\nFinal verdict: Expressions are mathematically equivalent.")
        else:
            print("\nFinal verdict: Expressions are NOT mathematically equivalent.")
    
    return is_equal, results


def batch_verify_equalities(
    expression_pairs: List[Tuple[Union[str, sp.Expr], Union[str, sp.Expr]]],
    variables_list: Optional[List[Optional[List[str]]]] = None,
    assumptions_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    **kwargs
) -> List[Tuple[bool, Dict[str, Any]]]:
    """
    Verify equality for multiple pairs of expressions.
    
    Parameters:
    -----------
    expression_pairs : List[Tuple[Union[str, sp.Expr], Union[str, sp.Expr]]]
        List of (expr1, expr2) pairs to verify
    variables_list : Optional[List[Optional[List[str]]]]
        List of variable lists for each expression pair (or None for auto-detection)
    assumptions_list : Optional[List[Optional[Dict[str, Any]]]]
        List of assumption dictionaries for each expression pair
    **kwargs : 
        Additional parameters to pass to verify_equality
    
    Returns:
    --------
    List[Tuple[bool, Dict[str, Any]]]
        List of (is_equal, results_dict) for each expression pair
    """
    results = []
    
    # Prepare variables and assumptions for each pair
    if variables_list is None:
        variables_list = [None] * len(expression_pairs)
    if assumptions_list is None:
        assumptions_list = [None] * len(expression_pairs)
    
    # Ensure lists have the same length
    if len(variables_list) < len(expression_pairs):
        variables_list.extend([None] * (len(expression_pairs) - len(variables_list)))
    if len(assumptions_list) < len(expression_pairs):
        assumptions_list.extend([None] * (len(expression_pairs) - len(assumptions_list)))
    
    # Verify each pair
    for i, (expr_pair, variables, assumptions) in enumerate(zip(expression_pairs, variables_list, assumptions_list)):
        expr1, expr2 = expr_pair
        result = verify_equality(expr1, expr2, variables, assumptions, **kwargs)
        results.append(result)
    
    return results


# Example usage for advanced math verification
def run_examples():
    print("Example 1: Basic algebraic identity")
    result1, details1 = verify_equality("(x + y)^2", "x^2 + 2*x*y + y^2", verbose=True)
    print(f"\nEquality result: {result1}\n")
    
    print("Example 2: Trigonometric identity")
    result2, details2 = verify_equality("sin(x)^2 + cos(x)^2", "1", verbose=True)
    print(f"\nEquality result: {result2}\n")
    
    print("Example 3: Complex identity with domain restrictions")
    result3, details3 = verify_equality(
        "sqrt(x^2)", "x", 
        assumptions={"x": {"positive": True}},
        verbose=True
    )
    print(f"\nEquality result: {result3}\n")
    
    print("Example 4: Advanced trig identity")
    result4, details4 = verify_equality(
        "(sin(x) + sin(y))^2 + (cos(x) - cos(y))^2", 
        "4*sin((x+y)/2)^2",
        verbose=True
    )
    print(f"\nEquality result: {result4}\n")
    
    print("Example 5: Logarithmic identity")
    result5, details5 = verify_equality(
        "log(a*b)", "log(a) + log(b)",
        assumptions={"a": {"positive": True}, "b": {"positive": True}},
        verbose=True
    )
    print(f"\nEquality result: {result5}\n")
    
    print("Example 6: Expression that should be unequal")
    result6, details6 = verify_equality(
        "x^2 + 1", "x^2 - 1",
        verbose=True
    )
    print(f"\nEquality result: {result6}\n")


if __name__ == "__main__":
    run_examples()