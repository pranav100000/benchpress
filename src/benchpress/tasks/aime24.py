"""AIME24 benchmark task."""

import os
import re
from typing import Any, Dict, List, Optional

from .base import BaseTask, Example, TaskResult
from .registry import register_task


class Aime24Example(Example):
    """An example from the AIME24 dataset."""

    def __init__(
        self,
        id: str,
        question: str,
        answer: str,
        year: int,
        problem_number: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an AIME24 example."""
        if metadata is None:
            metadata = {}
        metadata.update({"year": year, "problem_number": problem_number})
        super().__init__(id=id, question=question, answer=answer, metadata=metadata)
        self.year = year
        self.problem_number = problem_number


@register_task
class Aime24Task(BaseTask[Aime24Example]):
    """AIME24 benchmark task implementation."""

    def __init__(self, data_path: Optional[str] = None):
        """Initialize the AIME24 task.

        Args:
            data_path: Path to the AIME24 data directory
        """
        self._data_path = data_path or os.environ.get("BENCHPRESS_DATA_PATH", "./data")

    @property
    def name(self) -> str:
        """Return the task name."""
        return "aime24"

    @property
    def description(self) -> str:
        """Return the task description."""
        return "A benchmark based on the American Invitational Mathematics Examination"

    async def load_examples(self) -> List[Aime24Example]:
        """Load AIME24 examples.

        Returns:
            A list of AIME24 examples
        """
        # In a real implementation, this would load from a file
        # For the simplified version, we'll return a few sample problems
        examples = [
            Aime24Example(
                id="aime24_2020_1",
                question=(
                    "Let S be the sum of all positive integers n such that "
                    "n^2 + 3n + 2 divides n^5 - n^3 + n + 3. Find the remainder when "
                    "S is divided by 1000."
                ),
                answer="168",
                year=2020,
                problem_number=1,
                metadata={"source": "sample"},
            ),
            Aime24Example(
                id="aime24_2021_5",
                question=(
                    "Triangle ABC has AB = 34, BC = 17, and CA = 29. Let D be a point "
                    "on AC such that BD bisects angle ABC. Let E be a point on AB such "
                    "that CE bisects angle BCA. Find the perimeter of triangle BDE."
                ),
                answer="50",
                year=2021,
                problem_number=5,
                metadata={"source": "sample"},
            ),
            Aime24Example(
                id="aime24_2022_8",
                question=(
                    "Find the number of ordered pairs (m,n) of positive integers such "
                    "that m ≤ 1000, n ≤ 1000, and (m^2 + n^2) / (mn + 1) is an integer."
                ),
                answer="824",
                year=2022,
                problem_number=8,
                metadata={"source": "sample"},
            ),
        ]
        return examples

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
        # AIME problems typically have integer answers, often 3 digits
        # Extract the final answer from the model output
        answer_pattern = r"(?:answer|result|solution):\s*(\d+)"
        match = re.search(answer_pattern, model_output.lower(), re.DOTALL)

        extracted_answer = None
        if match:
            extracted_answer = match.group(1).strip()
        else:
            # Try to find the last occurrence of a number with 1-3 digits
            # which is common for AIME problems
            number_matches = re.findall(r"\b(\d{1,3})\b", model_output)
            extracted_answer = number_matches[-1] if number_matches else ""

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
                "year": example.year,
                "problem_number": example.problem_number,
            },
        )
