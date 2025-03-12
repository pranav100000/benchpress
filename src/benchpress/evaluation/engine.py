"""Evaluation engine for benchpress."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..models.base import BaseModel
from ..tasks.base import BaseTask


@dataclass
class EvaluationSummary:
    """Summary of an evaluation run."""

    task_name: str
    model_id: str
    total_examples: int
    correct: int
    accuracy: float
    metadata: Optional[Dict[str, Any]] = None


class EvaluationEngine:
    """Evaluation engine for running benchmarks."""

    def __init__(
        self,
        model: BaseModel,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the evaluation engine.

        Args:
            model: The model to evaluate
            output_dir: Directory to save evaluation results (optional)
        """
        self.model = model
        self.output_dir = Path(output_dir) if output_dir else None

    async def evaluate_task(
        self, task: BaseTask, limit: Optional[int] = None
    ) -> EvaluationSummary:
        """Evaluate a model on a task.

        Args:
            task: The task to evaluate
            limit: Maximum number of examples to evaluate (optional)

        Returns:
            An evaluation summary
        """
        examples = await task.load_examples()
        if limit is not None:
            examples = examples[:limit]

        results = []
        for example in examples:
            model_output = await self.model.generate(example.question)
            result = await task.evaluate_example(example, model_output)
            result.model_id = self.model.model_id
            results.append(result)

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Sanitize the model ID to create a valid filename
            sanitized_model_id = self.model.model_id.replace("/", "_").replace(":", "_")
            output_file = (
                self.output_dir / f"{task.name}_{sanitized_model_id}_results.json"
            )
            with open(output_file, "w") as f:
                json.dump([asdict(result) for result in results], f, indent=2)

        # Compute summary
        correct = sum(1 for result in results if result.correct)
        accuracy = correct / len(results) if results else 0.0

        return EvaluationSummary(
            task_name=task.name,
            model_id=self.model.model_id,
            total_examples=len(results),
            correct=correct,
            accuracy=accuracy,
            metadata={
                "model_metadata": self.model.get_response_metadata(),
            },
        )
