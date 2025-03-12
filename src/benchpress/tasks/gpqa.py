"""GPQA Diamond benchmark task."""

import os
import re
from typing import Any, Dict, List, Optional

from .base import BaseTask, Example, TaskResult
from .registry import register_task


class GpqaExample(Example):
    """An example from the GPQA Diamond dataset."""

    def __init__(
        self,
        id: str,
        question: str,
        answer: str,
        subject: str,
        difficulty: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a GPQA Diamond example.

        Args:
            id: Unique identifier for the example
            question: The question text
            answer: The correct answer
            subject: The academic subject (e.g., physics, biology, etc.)
            difficulty: The difficulty level (e.g., easy, medium, hard)
            metadata: Additional metadata about the example
        """
        if metadata is None:
            metadata = {}
        metadata.update({"subject": subject, "difficulty": difficulty})
        super().__init__(id=id, question=question, answer=answer, metadata=metadata)
        self.subject = subject
        self.difficulty = difficulty


@register_task
class GpqaTask(BaseTask[GpqaExample]):
    """GPQA Diamond benchmark task implementation."""

    def __init__(self, data_path: Optional[str] = None):
        """Initialize the GPQA Diamond task.

        Args:
            data_path: Path to the GPQA Diamond data directory
        """
        self._data_path = data_path or os.environ.get("BENCHPRESS_DATA_PATH", "./data")

    @property
    def name(self) -> str:
        """Return the task name."""
        return "gpqa"

    @property
    def description(self) -> str:
        """Return the task description."""
        return "Diamond: Graduate-level Problem-solving Questions and Answers benchmark"

    async def load_examples(self) -> List[GpqaExample]:
        """Load GPQA Diamond examples.

        Returns:
            A list of GPQA Diamond examples
        """
        # In a real implementation, this would load from a file
        # For the simplified version, we'll return a few sample problems
        examples = [
            GpqaExample(
                id="gpqa_physics_1",
                question=(
                    "A uniform magnetic field B = 1.5 T is directed along the positive z-axis. "
                    "A particle with charge q = 3.2 × 10^-19 C and mass m = 6.64 × 10^-27 kg "
                    "is moving in this field with a velocity v = (2 × 10^5 î + 3 × 10^5 ĵ + 4 × 10^5 k̂) m/s. "
                    "Calculate the radius of the circular component of the particle's motion."
                ),
                answer="2.5 × 10^-4",
                subject="physics",
                difficulty="hard",
                metadata={"topic": "electromagnetism", "source": "sample"},
            ),
            GpqaExample(
                id="gpqa_biology_1",
                question=(
                    "In a population genetic study, a researcher found a gene with two alleles, A and a. "
                    "The frequency of the A allele is 0.7 and the frequency of the a allele is 0.3. "
                    "Assuming Hardy-Weinberg equilibrium, what is the expected frequency of heterozygotes (Aa) "
                    "in the population?"
                ),
                answer="0.42",
                subject="biology",
                difficulty="medium",
                metadata={"topic": "population genetics", "source": "sample"},
            ),
            GpqaExample(
                id="gpqa_chemistry_1",
                question=(
                    "Calculate the pH of a buffer solution prepared by mixing 0.15 mol of acetic acid "
                    "(CH3COOH) with 0.20 mol of sodium acetate (CH3COONa) in water to make 500 mL of solution. "
                    "The Ka of acetic acid is 1.8 × 10^-5."
                ),
                answer="4.88",
                subject="chemistry",
                difficulty="medium",
                metadata={"topic": "acid-base equilibria", "source": "sample"},
            ),
            GpqaExample(
                id="gpqa_cs_1",
                question=(
                    "Analyze the time complexity of the following algorithm to find the maximum subarray sum: "
                    "\n\n```python\n"
                    "def max_subarray_sum(arr):\n"
                    "    n = len(arr)\n"
                    "    max_so_far = float('-inf')\n"
                    "    max_ending_here = 0\n"
                    "    \n"
                    "    for i in range(n):\n"
                    "        max_ending_here = max(arr[i], max_ending_here + arr[i])\n"
                    "        max_so_far = max(max_so_far, max_ending_here)\n"
                    "    \n"
                    "    return max_so_far\n"
                    "```\n\n"
                    "What is the time complexity in big O notation?"
                ),
                answer="O(n)",
                subject="computer science",
                difficulty="easy",
                metadata={"topic": "algorithms", "source": "sample"},
            ),
            GpqaExample(
                id="gpqa_economics_1",
                question=(
                    "A monopolist faces the demand curve P = 100 - Q and has a cost function "
                    "C(Q) = 20 + 10Q + Q^2. What is the profit-maximizing quantity and price, "
                    "and what is the resulting profit?"
                ),
                answer="Q = 22.5, P = 77.5, Profit = 992.5",
                subject="economics",
                difficulty="medium",
                metadata={"topic": "microeconomics", "source": "sample"},
            ),
        ]
        return examples

    async def evaluate_example(
        self, example: GpqaExample, model_output: str
    ) -> TaskResult:
        """Evaluate a model's output on a GPQA Diamond example.

        Args:
            example: The GPQA Diamond example
            model_output: The model's generated output

        Returns:
            A task result containing the evaluation
        """
        # Extract the final answer from the model output
        answer_pattern = r"(?:answer|result|solution):\s*(.+?)(?:$|\n)"
        match = re.search(answer_pattern, model_output.lower(), re.DOTALL)

        extracted_answer = None
        if match:
            extracted_answer = match.group(1).strip()
        else:
            # If no explicit answer format, try to extract the last sentence or expression
            sentences = re.split(r'(?<=[.!?])\s+', model_output)
            extracted_answer = sentences[-1].strip() if sentences else ""

        # For GPQA, we need a more sophisticated answer matching approach
        # For now, we'll use a simple substring check which is less strict
        # In a real implementation, we would normalize answers and use semantic matching
        correct = example.answer.lower() in extracted_answer.lower()

        return TaskResult(
            example_id=example.id,
            model_id="",  # Will be filled in by the evaluation engine
            model_output=model_output,
            correct=correct,
            metadata={
                "extracted_answer": extracted_answer,
                "expected_answer": example.answer,
                "subject": example.subject,
                "difficulty": example.difficulty,
            },
        )