import sys
from src.benchpress.utils.math_comparison import normalize_expression

print("\\text{Evelyn} ->", normalize_expression('\\text{Evelyn}'))
print("Evelyn ->", normalize_expression('Evelyn'))
print("3\\sqrt{13} ->", normalize_expression('3\\sqrt{13}'))
print("3√13 ->", normalize_expression('3√13'))
print("3√{13} ->", normalize_expression('3√{13}'))
