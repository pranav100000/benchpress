"""Evaluation engine for benchpress."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from rich.console import Console

from ..models.base import BaseModel
from ..tasks.base import BaseTask
# from ..utils.latex import format_latex_for_terminal
from ..utils.latex_to_unicode import latex_to_unicode


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
        silent: bool = False,
        debug: bool = False,
        console: Optional[Console] = None,
    ):
        """Initialize the evaluation engine.

        Args:
            model: The model to evaluate
            output_dir: Directory to save evaluation results (optional)
            silent: Whether to suppress real-time output (optional)
            debug: Whether to show detailed debug information (optional)
            console: Rich console for output formatting (optional)
        """
        self.model = model
        self.output_dir = Path(output_dir) if output_dir else None
        self.silent = silent
        self.debug = debug
        self.console = console

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
        # Check if the task already supports limiting through constructor

        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

        if not self.silent and self.console:
            self.console.print(f"[bold]Loading examples for task:[/bold] {task.name}")

        # We handle limit via task's internal mechanism or externally

        # Check if task handles limit internally (like Math500Task)
        if hasattr(task, "_limit") and limit is not None:
            # Task might already have a limit set in constructor
            # For safety, use the smaller of the two limits
            if task._limit is None or limit < task._limit:
                task._limit = limit

            # Load examples with task's internal limiting
            examples = await task.load_examples()
        else:
            # Load examples and apply limit afterwards
            examples = await task.load_examples()

            # Apply limit if specified
            if limit is not None:
                examples = examples[:limit]

        if not self.silent and self.console:
            self.console.print(f"[bold]Loaded[/bold] {len(examples)} examples")

        results = []

        # Create progress bar if not in silent mode and console is available
        if not self.silent and self.console:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            )
            task_id = progress.add_task(f"Evaluating {task.name}", total=len(examples))

            with progress:
                for i, example in enumerate(examples):
                    # Show the question
                    self.console.print(f"\n[bold cyan]Example {i+1}/{len(examples)}")
                    # Format the question for display using the new unicodeit-based formatter
                    formatted_question = latex_to_unicode(example.question)
                    self.console.print(
                        Panel(
                            formatted_question,
                            title="Question",
                            border_style="blue",
                            width=88  # Limit width to prevent wrapping issues
                        )
                    )

                    # Generate the model output
                    # Use a clear prompt that works well with our extraction patterns
                    prompt = f"""Solve this problem: {example.question}

Provide your step-by-step solution, then end your response with:
ANSWER: [your final answer]

IMPORTANT: The answer must be ONLY the numeric or algebraic result with:
- No units (don't write "dollars", "meters", etc.)
- No explanations
- No additional text
- Just the number or expression itself"""

                    # Generate response using the prompt
                    model_output = await self.model.generate(prompt)
                    # Format the model output for display using the new unicodeit-based formatter
                    formatted_output = latex_to_unicode(model_output)
                    self.console.print(
                        Panel(
                            formatted_output,
                            title="Model Response",
                            border_style="yellow",
                            width=88  # Limit width to prevent wrapping issues
                        )
                    )

                    # Evaluate the example
                    result = await task.evaluate_example(example, model_output)
                    result.model_id = self.model.model_id

                    # Display extracted answer from model output if available
                    if result.metadata and "extracted_answer" in result.metadata:
                        extracted_answer = result.metadata["extracted_answer"]
                        formatted_answer = latex_to_unicode(extracted_answer)

                        # Show both versions if they differ significantly
                        if (formatted_answer != extracted_answer and
                            len(extracted_answer) > 5):
                            self.console.print(
                                f"[bold yellow]Extracted Answer:[/bold yellow] "
                                f"{formatted_answer}"
                            )
                            self.console.print(
                                f"[dim]Raw: {extracted_answer}[/dim]"
                            )
                        else:
                            self.console.print(
                                f"[bold yellow]Extracted Answer:[/bold yellow] "
                                f"{extracted_answer}"
                            )

                    # Show the result
                    correct_style = "green" if result.correct else "red"
                    correct_text = "✓ CORRECT" if result.correct else "✗ INCORRECT"
                    self.console.print(
                        f"[bold {correct_style}]{correct_text}[/bold {correct_style}]"
                    )

                    # If the example has a reference answer, show it
                    if hasattr(example, "answer") and example.answer:
                        raw_answer = str(example.answer)
                        # Format the answer using the new unicodeit-based formatter
                        formatted_answer = latex_to_unicode(raw_answer)
                        self.console.print(
                            Panel(
                                formatted_answer,
                                title="Reference Answer",
                                border_style="green",
                                width=88  # Limit width to prevent wrapping issues
                            )
                        )

                        # If formatting changed it significantly, also show the raw form
                        if formatted_answer != raw_answer and len(raw_answer) > 5:
                            self.console.print(
                                f"[dim]Raw reference: {raw_answer}[/dim]"
                            )

                        # Show additional info for debugging incorrect answers
                        if not result.correct and result.metadata:
                            raw_expected = str(example.answer)
                            raw_extracted = result.metadata.get(
                                "extracted_answer", "N/A")

                            # Format both for better display
                            # formatted_expected = latex_to_unicode(raw_expected)
                            # formatted_extracted = latex_to_unicode(
                            #     raw_extracted)

                            # Use the same approach for method and confidence as our debug panel
                            method = result.metadata.get('extraction_method', 
                                    result.metadata.get('method', 'N/A'))
                            
                            confidence_val = result.metadata.get('extraction_confidence', 
                                             result.metadata.get('confidence', 'N/A'))
                            # Format confidence as float if it's a number
                            confidence = f"{float(confidence_val):.2f}" if isinstance(confidence_val, (int, float)) else confidence_val

                            self.console.print(
                                f"[dim]Debug: Expected='{raw_expected}' "
                                f"vs Extracted='{raw_extracted}' "
                                f"(method={method}, confidence={confidence})[/dim]"
                            )

                        # Show detailed debug information if debug mode is enabled
                        if self.debug:
                            from rich.panel import Panel

                            # Show unformatted question
                            self.console.print("[bold]Debug Information:[/bold]")
                            self.console.print(
                                Panel(
                                    example.question,
                                    title="Raw Question",
                                    border_style="blue",
                                    width=88
                                )
                            )

                            # Show unformatted model output
                            self.console.print(
                                Panel(
                                    model_output,
                                    title="Raw Model Response",
                                    border_style="yellow",
                                    width=88
                                )
                            )

                            # Show extraction details
                            metadata = result.metadata or {}
                            
                            # Uncomment for debugging metadata keys
                            # print(f"DEBUG - metadata keys: {metadata.keys()}")
                            
                            # Get extraction details with safeguards
                            extracted = metadata.get('extracted_answer', 'N/A')
                            
                            # Check for method and confidence - handle both string and numeric types
                            method = metadata.get('extraction_method', metadata.get('method', 'N/A'))
                            
                            # Handle confidence which could be a string, float, or missing
                            confidence_val = metadata.get('extraction_confidence', metadata.get('confidence', 'N/A'))
                            # Format confidence as float if it's a number
                            confidence = f"{float(confidence_val):.2f}" if isinstance(confidence_val, (int, float)) else confidence_val
                            
                            extraction_details = [
                                f"Extracted Answer: {extracted}",
                                f"Method: {method}",
                                f"Confidence: {confidence}",
                            ]
                            
                            # Add additional extraction details if available
                            if 'pattern_type' in metadata:
                                extraction_details.append(f"Pattern Type: {metadata['pattern_type']}")
                                
                            if 'extractor' in metadata:
                                extraction_details.append(f"Extractor: {metadata['extractor']}")
                            
                            if (result.metadata and 
                                "alternative_answers" in result.metadata):
                                alt_candidates = result.metadata["alternative_answers"]
                                if alt_candidates:
                                    extraction_details.append("")
                                    extraction_details.append("Alternative candidates:")
                                    for c in alt_candidates:
                                        alt_text = (f"- {c['text']} (method={c['method']}, "
                                                   f"confidence={c['confidence']})")
                                        extraction_details.append(alt_text)

                            self.console.print(
                                Panel(
                                    "\n".join(extraction_details),
                                    title="Extraction Details",
                                    border_style="green",
                                    width=88
                                )
                            )

                    # Add to results
                    results.append(result)

                    # Update progress
                    progress.update(task_id, advance=1)
        else:
            # Silent mode: just process everything without output
            for example in examples:
                # Use the same properly formatted prompt as in verbose mode
                prompt = f"""Solve this problem: {example.question}

Provide your step-by-step solution, then end your response with:
ANSWER: [your final answer]

IMPORTANT: The answer must be ONLY the numeric or algebraic result with:
- No units (don't write "dollars", "meters", etc.)
- No explanations
- No additional text
- Just the number or expression itself"""

                model_output = await self.model.generate(prompt)
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
