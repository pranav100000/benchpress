"""GPQA Diamond benchmark task."""

import re
from typing import List, Literal, Optional

from ..datasets.gpqa_dataset import GpqaDataset
from ..datasets.gpqa_hf_dataset import GpqaHfDataset
from ..examples.gpqa import GpqaExample
from ..extraction import ExtractionContext, extract_answer, ExtractedAnswer
from ..utils import get_hf_token
from ..utils.math_comparison import compare_answers
from .base import BaseTask, TaskResult
from .registry import register_task


@register_task
class GpqaTask(BaseTask[GpqaExample]):
    """GPQA Diamond benchmark task implementation."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        dataset_source: Literal["csv", "huggingface"] = "csv",
        hf_dataset_name: str = "Idavidrein/gpqa",
        hf_config_name: str = "gpqa_diamond",
        hf_split: str = "train",
        hf_token: Optional[str] = None,
    ):
        """Initialize the GPQA Diamond task.

        Args:
            data_path: Path to the GPQA Diamond data directory
            dataset_source: Source of the dataset ("csv" or "huggingface")
            hf_dataset_name: Hugging Face dataset name (if using huggingface)
            hf_config_name: Hugging Face dataset config (if using huggingface)
            hf_split: Hugging Face dataset split to use
            hf_token: Hugging Face API token (if None, will try to get from env)
        """
        self._data_path = data_path
        self._dataset_source = dataset_source
        self._hf_dataset_name = hf_dataset_name
        self._hf_config_name = hf_config_name
        self._hf_split = hf_split
        self._hf_token = hf_token

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
        # Use the appropriate dataset implementation based on the source
        if self._dataset_source == "huggingface":
            # Get token from environment if not provided
            token = self._hf_token or get_hf_token()

            dataset = GpqaHfDataset(
                data_path=self._data_path,
                dataset_name=self._hf_dataset_name,
                config_name=self._hf_config_name,
                split=self._hf_split,
                token=token,
            )
        else:
            # Default to CSV dataset
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
        # Use the extraction framework to extract answers with GPQA-specific context
        extraction_context = ExtractionContext(
            domain="gpqa",
            task_name="gpqa",
            expected_format=example.metadata.get("answer_format", "free-text"),
            metadata={"subject": example.subject}
        )

        # Extract all candidate answers
        candidate_answers = self._extractor.extract(model_output, extraction_context)

        # Use the highest confidence answer if available
        if candidate_answers and candidate_answers[0].confidence >= 0.3:
            extracted_answer = candidate_answers[0]
        else:
            # Basic fallback extraction for GPQA
            answer_pattern = r"(?:answer|result|solution):\s*(.+?)(?:$|\n)"
            match = re.search(answer_pattern, model_output.lower(), re.DOTALL)

            if match:
                extracted_text = match.group(1).strip()
                extracted_answer = ExtractedAnswer(
                    text=extracted_text,
                    pattern_name="fallback_regex",
                    confidence=0.5,
                    metadata={"pattern_type": "fallback"}
                )
            else:
                # If no explicit answer format, try to extract the last sentence
                sentences = re.split(r"(?<=[.!?])\s+", model_output)
                extracted_text = sentences[-1].strip() if sentences else ""
                extracted_answer = ExtractedAnswer(
                    text=extracted_text,
                    pattern_name="fallback_last_sentence",
                    confidence=0.2,  # Low confidence for this method
                    metadata={"pattern_type": "fallback"}
                )

        # Use our comprehensive comparison approach for more consistent results
        # This will handle various formatting differences
        correct = compare_answers(extracted_answer.text, example.answer, domain="gpqa")
        
        # For GPQA, we also want to do a fallback substring check if the above fails
        # This is less strict but catches cases where the answer is embedded in text
        if not correct and hasattr(extracted_answer, "text"):
            correct = example.answer.lower() in extracted_answer.text.lower()

        # Prepare the metadata with extraction information
        metadata = {
            "extracted_answer": extracted_answer.text,
            "extraction_confidence": extracted_answer.confidence,
            "extraction_method": extracted_answer.pattern_name,
            "expected_answer": example.answer,
            "subject": example.subject,
            "difficulty": example.difficulty,
        }

        # Include alternatives if available (other candidates)
        alternative_answers = candidate_answers[1:] if len(candidate_answers) > 1 else []
        if alternative_answers:
            metadata["alternative_answers"] = [
                {"text": alt.text, "confidence": alt.confidence}
                for alt in alternative_answers
            ]

        return TaskResult(
            example_id=example.id,
            model_id="",  # Will be filled in by the evaluation engine
            model_output=model_output,
            correct=correct,
            metadata=metadata,
        )
