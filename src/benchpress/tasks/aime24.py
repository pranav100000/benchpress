"""AIME24 benchmark task."""

import os
from typing import List, Optional

from ..examples.aime24 import Aime24Example
from ..extraction import ExtractionContext, extract_answer
from ..utils.math_comparison import compare_answers
from .base import BaseTask, TaskResult
from .registry import register_task


@register_task
class Aime24Task(BaseTask[Aime24Example]):
    """AIME24 benchmark task implementation."""

    def __init__(self, data_path: Optional[str] = None, limit: Optional[int] = None):
        """Initialize the AIME24 task.

        Args:
            data_path: Path to the AIME24 data directory
            limit: Optional limit on the number of examples to load
        """
        self._data_path = data_path or os.environ.get("BENCHPRESS_DATA_PATH", "./data")
        self._limit = limit
        self._dataset = None

    @property
    def name(self) -> str:
        """Return the task name."""
        return "aime24"

    @property
    def description(self) -> str:
        """Return the task description."""
        return "A benchmark based on the American Invitational Mathematics Examination"

    @property
    def prompt_template(self) -> str:
        """Return the prompt template for the task."""
        return """You are a mathematical problem solver working on problems from the American Invitational Mathematics Examination (AIME).

Problem:
{question}

Solve this problem step by step, showing your work clearly. AIME problems typically have an integer answer that is a 3-digit number or less.

After solving the problem, please clearly indicate your final answer using the format:
\boxed{your_answer_here}

Remember, AIME problems require exact answers, not decimal approximations."""

    async def get_dataset(self):
        """Get the dataset for this task.

        Returns:
            The dataset instance
        """
        # Import here to avoid circular imports
        from ..datasets import dataset_registry

        try:
            # Try to get the HF dataset first
            dataset_class = dataset_registry.get("aime24_hf")
            # Create an instance
            dataset = dataset_class(data_path=self._data_path)
            return dataset
        except (ImportError, ValueError, KeyError):
            # If HF dataset is not available, fall back to the sample dataset
            return None

    async def load_examples(self) -> List[Aime24Example]:
        """Load AIME24 examples from Hugging Face dataset.

        Returns:
            A list of AIME24 examples

        Raises:
            RuntimeError: If the dataset cannot be loaded
        """
        # Load from the Hugging Face dataset
        dataset = await self.get_dataset()

        if not dataset:
            raise RuntimeError(
                "AIME24 dataset could not be initialized. Make sure the 'datasets' library is installed and you have internet access."
            )

        try:
            examples = await dataset.load()
            # Apply limit if specified
            if self._limit is not None and examples:
                examples = examples[: self._limit]

            if not examples:
                raise RuntimeError("No examples were loaded from the AIME24 dataset.")

            return examples
        except Exception as e:
            # Provide a more detailed error message
            raise RuntimeError(
                f"Error loading AIME24 examples from Hugging Face dataset: {e}"
            )

    async def evaluate_example(
        self, example: Aime24Example, model_output: str
    ) -> TaskResult:
        """Evaluate a model's output on an AIME24 example.

        Args:
            example: The AIME24 example
            model_output: The model's generated output

        Returns:
            A task result containing the evaluation
        """
        # Create extraction context
        context = ExtractionContext(
            domain="aime24",
            task_name="aime24",
            expected_format="integer",
            question_type="math",
            metadata={
                "year": example.year,
                "problem_number": example.problem_number,
            }
        )

        # Extract answers using the simplified extraction system
        candidates = extract_answer(model_output, context)

        # Get the best answer (highest confidence)
        extracted_answer = ""
        if candidates:
            extracted_answer = candidates[0].text

        # Use comprehensive answer comparison
        correct = compare_answers(extracted_answer, example.answer, domain="aime24")

        # Build metadata dictionary
        metadata = {
            "extracted_answer": extracted_answer,
            "expected_answer": example.answer,
            "year": example.year,
            "problem_number": example.problem_number,
        }

        # Add extraction details if available
        if candidates:
            best_candidate = candidates[0]
            metadata["extraction_method"] = best_candidate.pattern_name
            metadata["method"] = best_candidate.pattern_name  # Alternative key for backward compatibility
            metadata["extraction_confidence"] = float(best_candidate.confidence)
            metadata["confidence"] = float(best_candidate.confidence)  # Alternative key

            if best_candidate.metadata:
                for key, value in best_candidate.metadata.items():
                    metadata[key] = value

        return TaskResult(
            example_id=example.id,
            model_id="",  # Will be filled in by the evaluation engine
            model_output=model_output,
            correct=correct,
            metadata=metadata,
        )
