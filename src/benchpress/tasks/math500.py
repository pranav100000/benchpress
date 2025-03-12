"""MATH-500 benchmark task."""

import os
import re
from typing import Any, Dict, List, Optional

from .base import BaseTask, Example, TaskResult
from .registry import register_task


class Math500Example(Example):
    """An example from the MATH-500 dataset."""

    def __init__(
        self,
        id: str,
        question: str,
        answer: str,
        category: str,
        difficulty: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a MATH-500 example."""
        if metadata is None:
            metadata = {}
        metadata.update({"category": category, "difficulty": difficulty})
        super().__init__(id=id, question=question, answer=answer, metadata=metadata)
        self.category = category
        self.difficulty = difficulty


@register_task
class Math500Task(BaseTask[Math500Example]):
    """MATH-500 benchmark task implementation."""

    def __init__(self, data_path: Optional[str] = None):
        """Initialize the MATH-500 task.

        Args:
            data_path: Path to the MATH-500 data directory
        """
        self._data_path = data_path or os.environ.get("BENCHPRESS_DATA_PATH", "./data")

    @property
    def name(self) -> str:
        """Return the task name."""
        return "math500"

    @property
    def description(self) -> str:
        """Return the task description."""
        return "A benchmark of 500 challenging math problems"

    async def load_examples(self) -> List[Math500Example]:
        """Load MATH-500 examples.

        Returns:
            A list of MATH-500 examples
        """
        # In a real implementation, this would load from a file
        # For the simplified version, we'll return a few sample problems
        examples = [
            Math500Example(
                id="math500_1",
                question=(
                    "If $x^2 + y^2 = 25$ and $x + y = 7$, "
                    "find the value of $x^2 + 2xy + y^2$."
                ),
                answer="49",
                category="algebra",
                difficulty="easy",
                metadata={"source": "sample"},
            ),
            Math500Example(
                id="math500_2",
                question=(
                    "Find the sum of all positive integers $n$ such that "
                    "$n^2 + 3n + 5$ is a perfect square."
                ),
                answer="4",
                category="number_theory",
                difficulty="medium",
                metadata={"source": "sample"},
            ),
            Math500Example(
                id="math500_3",
                question=(
                    "In triangle $ABC$, we have $AB = 13$, $BC = 14$, and $AC = 15$. "
                    "Find the area of the triangle."
                ),
                answer="84",
                category="geometry",
                difficulty="easy",
                metadata={"source": "sample"},
            ),
        ]
        return examples

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
        # Extract the final answer from the model output
        # This is a simplified implementation - a real one would be more robust
        answer_pattern = r"(?:answer|result|solution):\s*(.+?)(?:\.|$)"
        match = re.search(answer_pattern, model_output.lower(), re.DOTALL)

        extracted_answer = None
        if match:
            extracted_answer = match.group(1).strip()
        else:
            # If no explicit answer format, try to extract the last number
            numbers = re.findall(r"\d+", model_output)
            extracted_answer = numbers[-1] if numbers else ""

        # Simple exact match evaluation
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
