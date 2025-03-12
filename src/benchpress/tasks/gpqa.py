"""GPQA Diamond benchmark task."""

import os
import re
import csv
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
        self._data_path = data_path or os.environ.get(
            "BENCHPRESS_DATA_PATH", 
            "/Users/pranavsharan/Developer/benchpress/datasets/gpqa_dataset"
        )

    @property
    def name(self) -> str:
        """Return the task name."""
        return "gpqa"

    @property
    def description(self) -> str:
        """Return the task description."""
        return "Diamond: Graduate-level Problem-solving Questions and Answers benchmark"

    async def load_examples(self) -> List[GpqaExample]:
        """Load GPQA Diamond examples from CSV file.

        Returns:
            A list of GPQA Diamond examples
        """
        examples = []
        csv_path = os.path.join(self._data_path, "gpqa_diamond.csv")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Skip header row if present
                if i == 0 and "Question" not in row:
                    continue
                    
                # Extract difficulty from writer's estimate, defaulting to "graduate"
                difficulty = row.get("Writer's Difficulty Estimate", "graduate")
                
                # Create example with data from CSV
                example = GpqaExample(
                    id=f"gpqa_diamond_{row.get('Record ID', i)}",
                    question=row.get("Question", ""),
                    answer=row.get("Correct Answer", ""),
                    subject=row.get("High-level domain", ""),
                    difficulty=difficulty,
                    metadata={
                        "subdomain": row.get("Subdomain", ""),
                        "incorrect_answers": [
                            row.get("Incorrect Answer 1", ""),
                            row.get("Incorrect Answer 2", ""),
                            row.get("Incorrect Answer 3", "")
                        ],
                        "explanation": row.get("Explanation", ""),
                        "record_id": row.get("Record ID", "")
                    }
                )
                examples.append(example)
                
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