"""MATH-500 benchmark task."""

from typing import List, Optional

from ..datasets.math500_hf_dataset import Math500HfDataset
from ..examples.math500 import Math500Example
from ..extraction import ExtractionContext, extract_answer
from ..utils.math_comparison import compare_answers
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
ANSWER: [your final answer]

IMPORTANT: The answer must be ONLY the numeric or algebraic result with:
- No units (don't write "dollars", "meters", etc.)
- No explanations
- No additional text
- Just the number or expression itself"""

    # Removed _normalize_math_answer - now using the central utility in extraction.processors

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

        # Extract answers using the extraction system
        candidates = extract_answer(model_output, context)

        # Get the best answer (highest confidence)
        extracted_answer = candidates[0].text if candidates else ""

        # Compare answers and determine correctness
        correct = compare_answers(extracted_answer, example.answer, domain="math500")

        # Create detailed metadata
        metadata: dict[str, object] = {
            "extracted_answer": extracted_answer,
            "expected_answer": example.answer,
            "category": example.category,
            "difficulty": example.difficulty,
        }

        # Add extraction details if available
        if candidates:
            best_candidate = candidates[0]
            metadata.update({
                "extraction_method": best_candidate.pattern_name,
                "method": best_candidate.pattern_name,  # Alternative key for backward compatibility
                "extraction_confidence": float(best_candidate.confidence),
                "confidence": float(best_candidate.confidence),  # Alternative key
                "pattern_type": best_candidate.metadata.get("pattern_type", "unknown")
            })

            # Add alternative candidates info if available
            if len(candidates) > 1:
                metadata["alternative_answers"] = [
                    {
                        "text": c.text,
                        "method": c.pattern_name,
                        "confidence": float(c.confidence)
                    }
                    for c in candidates[1:3]  # Just include top alternatives
                ]

        return TaskResult(
            question=example.question,
            raw_question=example.question,
            example_id=example.id,
            model_id="",  # Will be filled in by the evaluation engine
            model_output=model_output,
            correct=correct,
            metadata=metadata,
        )
