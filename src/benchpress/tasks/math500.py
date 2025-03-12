"""MATH-500 benchmark task."""

import os
import re
from typing import Any, Dict, List, Optional

from ..datasets.v2.math500_hf_dataset import Math500HfDataset
from .base import BaseTask, TaskResult 
from .math500_example import Math500Example
from .registry import register_task


@register_task
class Math500Task(BaseTask[Math500Example]):
    """MATH-500 benchmark task implementation."""

    def __init__(self, data_path: Optional[str] = None, limit: Optional[int] = None):
        """Initialize the MATH-500 task.

        Args:
            data_path: Path to the MATH-500 data directory
            limit: Optional limit on the number of examples to load
        """
        self._data_path = data_path
        self._limit = limit

    @property
    def name(self) -> str:
        """Return the task name."""
        return "math500"

    @property
    def description(self) -> str:
        """Return the task description."""
        return "A benchmark of 500 challenging math problems"
        
    @property
    def prompt_template(self) -> str:
        """Return the prompt template for the task."""
        return """You are a mathematical problem solver working on challenging math problems.

Problem:
{question}

Solve this problem step by step, showing your work clearly. Make sure to simplify your answer to its simplest form.

After solving the problem, please clearly indicate your final answer using the format:
\boxed{your_answer_here}

Remember to derive the answer from first principles and verify your work."""
        
    def _normalize_math_answer(self, answer: str) -> str:
        """Normalize a math answer for more robust comparison.
        
        Args:
            answer: The answer string to normalize
            
        Returns:
            Normalized answer string
        """
        if not answer:
            return ""
            
        # Replace LaTeX fractions with division notation
        answer = re.sub(r'\\frac{([^}]+)}{([^}]+)}', r'\1/\2', answer)
        answer = re.sub(r'\\dfrac{([^}]+)}{([^}]+)}', r'\1/\2', answer)
        
        # Remove LaTeX formatting
        answer = answer.replace('\\left', '')
        answer = answer.replace('\\right', '')
        answer = answer.replace('\\', '')
        answer = answer.replace('{', '')
        answer = answer.replace('}', '')
        answer = answer.replace('$', '')
        answer = answer.replace(' ', '')
        
        # Replace LaTeX special symbols
        answer = answer.replace('pi', 'π')
        
        # Convert fractions to decimals for numerical comparison
        try:
            if '/' in answer:
                parts = answer.split('/')
                if len(parts) == 2 and all(part.strip().isdigit() for part in parts):
                    num = int(parts[0].strip())
                    denom = int(parts[1].strip())
                    if denom != 0:  # Avoid division by zero
                        answer = str(num / denom)
        except:
            # If conversion fails, keep the original
            pass
            
        return answer.strip().lower()

    async def load_examples(self) -> List[Math500Example]:
        """Load MATH-500 examples from HuggingFace dataset.

        Returns:
            A list of MATH-500 examples
            
        Raises:
            RuntimeError: If the dataset cannot be loaded
        """
        try:
            # Always use the HuggingFace dataset
            dataset = Math500HfDataset(data_path=self._data_path)
            
            # If limit is specified in constructor, sample that many examples
            if self._limit is not None and self._limit > 0:
                examples = await dataset.sample(self._limit)
            else:
                # Otherwise load all examples
                examples = await dataset.load()
                
            if not examples:
                raise RuntimeError("No examples were loaded from the MATH-500 dataset.")
                
            return examples
            
        except Exception as e:
            # Provide a more detailed error message
            raise RuntimeError(f"Error loading MATH-500 examples from Hugging Face dataset: {e}")

    async def evaluate_example(
        self, example: Math500Example, model_output: str
    ) -> TaskResult:
        """Evaluate a model's output on a MATH-500 example.

        Args:
            example: The MATH-500 example
            model_output: The model's generated output

        Returns:
            A task result containing the evaluation
        """
        # Extract the final answer from the model output with improved pattern matching
        
        # Pattern 1: Look for "boxed" answers, common in LaTeX
        boxed_pattern = r"\\boxed{([^}]+)}"
        match = re.search(boxed_pattern, model_output)
        
        if not match:
            # Pattern 2: Look for "answer/solution/result: X"
            answer_pattern = r"(?:answer|result|solution|final answer):\s*(.+?)(?:\.|$)"
            match = re.search(answer_pattern, model_output.lower(), re.DOTALL)
        
        if not match:
            # Pattern 3: Look for "the answer is X"
            is_pattern = r"(?:the\s+)?(?:answer|result|solution)\s+is\s+(?:\$)?([^.$]+)(?:\$)?(?:\s*$|\s*\.)"
            match = re.search(is_pattern, model_output.lower(), re.DOTALL)
            
        if not match:
            # Pattern 4: Look for "= X" at the end of a line or followed by period
            equals_pattern = r"=\s*(?:\$)?([^.$\n]+)(?:\$)?(?:\s*$|\s*\.|\n)"
            match = re.search(equals_pattern, model_output)
        
        extracted_answer = None
        if match:
            extracted_answer = match.group(1).strip()
            
            # For now, let's keep the LaTeX formatting for exact comparison
            # Clean up any extra whitespace, etc.
            extracted_answer = extracted_answer.replace('$', '').strip()
        else:
            # If all pattern matching fails, try to extract the last number/fraction
            # This is less reliable but serves as a fallback
            number_pattern = r'(?:\d+(?:\.\d+)?)|(?:\d+/\d+)'
            numbers = re.findall(number_pattern, model_output)
            extracted_answer = numbers[-1] if numbers else ""

        # Normalize both answers for comparison
        normalized_extracted = self._normalize_math_answer(extracted_answer)
        normalized_expected = self._normalize_math_answer(example.answer)
        
        # Compare normalized answers
        correct = normalized_extracted == normalized_expected
        
        # If the normalized comparison fails, fall back to exact match
        if not correct:
            correct = extracted_answer == example.answer

        return TaskResult(
            example_id=example.id,
            model_id="",  # Will be filled in by the evaluation engine
            model_output=model_output,
            correct=correct,
            metadata={
                "extracted_answer": extracted_answer,
                "expected_answer": example.answer,
                "category": example.category,
                "difficulty": example.difficulty,
            },
        )