"""GPQA Diamond benchmark task."""

import os
import re
from typing import Any, Dict, List, Optional

from ..datasets.v2.gpqa_dataset import GpqaDataset
from .base import BaseTask, TaskResult
from .gpqa_example import GpqaExample
from .registry import register_task


@register_task
class GpqaTask(BaseTask[GpqaExample]):
    """GPQA Diamond benchmark task implementation."""

    def __init__(self, data_path: Optional[str] = None):
        """Initialize the GPQA Diamond task.

        Args:
            data_path: Path to the GPQA Diamond data directory
        """
        self._data_path = data_path

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
        # Use the new dataset system
        dataset = GpqaDataset(data_path=self._data_path)
        return await dataset.load()

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