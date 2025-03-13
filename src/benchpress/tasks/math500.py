"""MATH-500 benchmark task."""

import re
from typing import List, Optional

from ..datasets.math500_hf_dataset import Math500HfDataset
from ..examples.math500 import Math500Example
from ..extraction import create_extractor, ExtractionContext
from .base import BaseTask, TaskResult
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
        answer = re.sub(r"\\frac{([^}]+)}{([^}]+)}", r"\1/\2", answer)
        answer = re.sub(r"\\dfrac{([^}]+)}{([^}]+)}", r"\1/\2", answer)

        # Remove LaTeX formatting
        answer = answer.replace("\\left", "")
        answer = answer.replace("\\right", "")
        answer = answer.replace("\\", "")
        answer = answer.replace("{", "")
        answer = answer.replace("}", "")
        answer = answer.replace("$", "")
        answer = answer.replace(" ", "")

        # Replace LaTeX special symbols
        answer = answer.replace("pi", "π")

        # Convert fractions to decimals for numerical comparison
        try:
            if "/" in answer:
                parts = answer.split("/")
                if len(parts) == 2 and all(part.strip().isdigit() for part in parts):
                    num = int(parts[0].strip())
                    denom = int(parts[1].strip())
                    if denom != 0:  # Avoid division by zero
                        answer = str(num / denom)
        except Exception:
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
            raise RuntimeError(
                f"Error loading MATH-500 examples from Hugging Face dataset: {e}"
            )

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
        # Create extraction context
        context = ExtractionContext(
            domain="math500",
            task_name="math500",
            expected_format=None,
            question_type="math",
            metadata={
                "category": example.category,
                "difficulty": example.difficulty,
            }
        )
        
        # Create math extractor
        extractor = create_extractor(domain="math500")
        
        # Extract answers
        candidates = extractor.extract(model_output, context)
        
        # Get the best answer (highest confidence)
        extracted_answer = ""
        if candidates:
            extracted_answer = candidates[0].normalized_text or candidates[0].text
        
        # Get the expected answer (normalize it too)
        normalized_expected = self._normalize_math_answer(example.answer)
        
        # For safety, also normalize the extracted answer again
        normalized_extracted = self._normalize_math_answer(extracted_answer)
        
        # Compare normalized answers
        correct = normalized_extracted == normalized_expected
        
        # If the normalized comparison fails, fall back to exact match
        if not correct:
            correct = extracted_answer == example.answer

        # Create detailed metadata
        metadata = {
            "extracted_answer": extracted_answer,
            "expected_answer": example.answer,
            "category": example.category,
            "difficulty": example.difficulty,
        }
        
        # Add extraction details if available
        if candidates:
            metadata["extraction_method"] = candidates[0].pattern_name
            metadata["extraction_confidence"] = candidates[0].confidence
            
            # Add alternative candidates info if available
            if len(candidates) > 1:
                metadata["alternative_answers"] = [
                    {
                        "text": c.normalized_text or c.text,
                        "method": c.pattern_name,
                        "confidence": c.confidence
                    }
                    for c in candidates[1:3]  # Just include top alternatives
                ]

        return TaskResult(
            example_id=example.id,
            model_id="",  # Will be filled in by the evaluation engine
            model_output=model_output,
            correct=correct,
            metadata=metadata,
        )