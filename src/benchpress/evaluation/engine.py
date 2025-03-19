"""Evaluation engine for benchpress."""

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from rich.console import Console, Group, Text
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from benchpress.utils.constants import (
    CORRECT_ANSWER_BORDER_PANEL_COLOR,
    DEBUG_BORDER_PANEL_COLOR,
    INCORRECT_ANSWER_BORDER_PANEL_COLOR,
    MAX_REFRESH_RATE,
    MODEL_RESPONSE_BORDER_PANEL_COLOR,
    PANEL_WIDTH,
    QUESTION_BORDER_PANEL_COLOR,
)

from ..models.base import BaseModel
from ..tasks.base import BaseTask, TaskResult
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
        streaming: bool = False,
        max_tokens: Optional[int] = None,
        sequential: bool = False,
    ):
        """Initialize the evaluation engine.

        Args:
            model: The model to evaluate
            output_dir: Directory to save evaluation results (optional)
            silent: Whether to suppress real-time output (optional)
            debug: Whether to show detailed debug information (optional)
            console: Rich console for output formatting (optional)
            streaming: Whether to use streaming API for model generation (optional)
            max_tokens: Maximum number of tokens to generate (optional)
            sequential: Whether to process examples sequentially (optional, defaults to False)
        """
        self.model = model
        self.output_dir = Path(output_dir) if output_dir else None
        self.silent = silent
        self.debug = debug
        self.console = console
        self.streaming = streaming
        self.max_tokens = max_tokens
        self.sequential = sequential

    def get_prompt(self, question: str) -> str:
        """Get the prompt for a question.

        Args:
            question: The question to create a prompt for

        Returns:
            Formatted prompt for the model
        """
        return f"""Solve this problem: {question}

Provide your step-by-step solution, then end your response with:
ANSWER: [your final answer]

IMPORTANT: The answer must be ONLY the numeric or algebraic result with:
- No units (don't write "dollars", "meters", etc.)
- No explanations
- No additional text
- Just the number or expression itself"""

    async def process_example(self, task: BaseTask, example: Any, example_index: int, total_examples: int) -> TaskResult:
        """Process a single example.

        Args:
            task: The task being evaluated
            example: The example to process
            example_index: The index of the example in the list
            total_examples: The total number of examples being evaluated

        Returns:
            The evaluation result for this example
        """
        # Format the prompt
        prompt = self.get_prompt(example.question)

        # Generate model output
        if self.streaming:
            model_output = ""
            async for chunk in self.model.stream_generate(
                prompt,
                max_tokens=self.max_tokens
            ):
                model_output += chunk
        else:
            model_output = await self.model.generate(
                prompt,
                max_tokens=self.max_tokens
            )

        # Evaluate the example
        result = await task.evaluate_example(example, model_output)
        result.model_id = self.model.model_id

        # Store additional information for display
        result.raw_output = model_output
        result.example_index = example_index
        result.total_examples = total_examples

        return result

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

        # Set up progress display if not in silent mode
        if not self.silent and self.console:
            # Add a live accuracy column to the progress bar with raw count
            accuracy_column = TextColumn(
                "[bold green]Accuracy: {task.fields[accuracy]:.1%} ({task.fields[correct]}/{task.completed})[/bold green]"
            )

            # Create progress bar component
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                accuracy_column,
                console=self.console
            )

            # Initialize progress bar with 0% accuracy
            task_id = progress.add_task(
                f"Evaluating {task.name}",
                total=len(examples),
                accuracy=0.0,
                correct=0
            )

            # Start the progress bar
            progress.start()

            try:
                # Process examples (either sequentially or in parallel)
                if self.sequential:
                    # Process examples sequentially (original behavior)
                    results = []
                    correct_count = 0

                    for i, example in enumerate(examples):
                        # Update progress description
                        progress.update(task_id, description=f"Evaluating {task.name} (Example {i+1}/{len(examples)})")

                        # Display question
                        formatted_question = latex_to_unicode(example.question)
                        self.console.print("")  # Add spacing
                        self.console.print(
                            Panel(
                                formatted_question,
                                title=f"Question {i+1}/{len(examples)}",
                                border_style=QUESTION_BORDER_PANEL_COLOR,
                                width=PANEL_WIDTH
                            )
                        )

                        # Process example with detailed output
                        if self.streaming:
                            # Stop the progress bar before streaming
                            progress.stop()

                            # Initialize empty output
                            model_output = ""
                            streaming_content = ""

                            # Create panel for streaming content
                            streaming_panel = Panel(
                                "",
                                title="Model Response (Streaming...)",
                                border_style=MODEL_RESPONSE_BORDER_PANEL_COLOR,
                                width=PANEL_WIDTH,
                                height=None  # Allow panel to grow as needed
                            )

                            # Stream the response with live updates
                            with Live(streaming_panel, console=self.console, refresh_per_second=MAX_REFRESH_RATE,
                                     transient=False, auto_refresh=True, vertical_overflow="visible") as live:
                                prompt = self.get_prompt(example.question)
                                async for chunk in self.model.stream_generate(
                                    prompt,
                                    max_tokens=self.max_tokens
                                ):
                                    model_output += chunk
                                    streaming_content += chunk

                                    # Format and update panel
                                    formatted_streaming = latex_to_unicode(streaming_content)
                                    streaming_panel.renderable = formatted_streaming

                                # Update panel title when complete
                                streaming_panel.title = "Model Response (Complete)"

                            # Restart progress bar
                            progress.start()
                        else:
                            # Standard non-streaming approach
                            self.console.print("Generating response...", style=MODEL_RESPONSE_BORDER_PANEL_COLOR)

                            prompt = self.get_prompt(example.question)
                            model_output = await self.model.generate(
                                prompt,
                                max_tokens=self.max_tokens
                            )

                            # Display formatted response
                            formatted_output = latex_to_unicode(model_output)
                            self.console.print(
                                Panel(
                                    formatted_output,
                                    title="Model Response",
                                    border_style=MODEL_RESPONSE_BORDER_PANEL_COLOR,
                                    width=PANEL_WIDTH
                                )
                            )

                        # Evaluate the example
                        result = await task.evaluate_example(example, model_output)
                        result.model_id = self.model.model_id

                        # Display evaluation results
                        result_items = []

                        # Add extracted answer if available
                        if result.metadata and "extracted_answer" in result.metadata:
                            extracted_answer = result.metadata["extracted_answer"]
                            formatted_answer = latex_to_unicode(extracted_answer)

                            # Create formatted answer text
                            answer_text = Text("Extracted Answer: ")
                            answer_text.append(formatted_answer, style="yellow bold")
                            result_items.append(answer_text)

                            # Show raw if it differs significantly
                            if formatted_answer != extracted_answer and len(extracted_answer) > 5:
                                raw_text = Text("Raw: ")
                                raw_text.append(extracted_answer, style="dim")
                                result_items.append(raw_text)

                        # Add correct/incorrect indicator
                        correct_style = "green" if result.correct else "red"
                        correct_text = "✓ CORRECT" if result.correct else "✗ INCORRECT"
                        result_items.append(Text(correct_text, style=f"bold {correct_style}"))

                        # Display the results panel
                        self.console.print(
                            Panel(
                                Group(*result_items),
                                title="Evaluation Results",
                                border_style=CORRECT_ANSWER_BORDER_PANEL_COLOR if result.correct else INCORRECT_ANSWER_BORDER_PANEL_COLOR,
                                width=PANEL_WIDTH
                            )
                        )

                        # Display reference answer if available
                        if hasattr(example, "answer") and example.answer:
                            raw_answer = str(example.answer)
                            formatted_answer = latex_to_unicode(raw_answer)

                            reference_texts = []
                            ref_text = Text("Reference Answer: ")
                            ref_text.append(formatted_answer, style="green bold")
                            reference_texts.append(ref_text)

                            # Show raw if it differs significantly
                            if formatted_answer != raw_answer and len(raw_answer) > 5:
                                raw_ref_text = Text("Raw reference: ")
                                raw_ref_text.append(raw_answer, style="dim")
                                reference_texts.append(raw_ref_text)

                            # Display the reference answer
                            self.console.print(
                                Panel(
                                    Group(*reference_texts),
                                    title="Reference Answer",
                                    border_style=CORRECT_ANSWER_BORDER_PANEL_COLOR if result.correct else INCORRECT_ANSWER_BORDER_PANEL_COLOR,
                                    width=PANEL_WIDTH
                                )
                            )

                            # Show debug info for incorrect answers
                            if not result.correct and result.metadata:
                                raw_expected = str(example.answer)
                                raw_extracted = result.metadata.get("extracted_answer", "N/A")

                                method = result.metadata.get('extraction_method',
                                        result.metadata.get('method', 'N/A'))

                                confidence_val = result.metadata.get('extraction_confidence',
                                                 result.metadata.get('confidence', 'N/A'))
                                # Format confidence as float if it's a number
                                confidence = f"{float(confidence_val):.2f}" if isinstance(confidence_val, (int, float)) else confidence_val

                                # Create debug panel
                                debug_items = []
                                debug_text = Text(f"Expected='{raw_expected}' vs Extracted='{raw_extracted}'", style="dim")
                                debug_items.append(debug_text)
                                debug_items.append(Text(f"Method: {method}, Confidence: {confidence}", style="dim"))

                                self.console.print(
                                    Panel(
                                        Group(*debug_items),
                                        title="Debug Information",
                                        border_style=DEBUG_BORDER_PANEL_COLOR,
                                        width=PANEL_WIDTH
                                    )
                                )

                            # Detailed debug information if enabled
                            if self.debug:
                                self.console.print("\n[bold]Detailed Debug Information:[/bold]")

                                # Raw question
                                self.console.print(
                                    Panel(
                                        example.question,
                                        title="Raw Question",
                                        border_style=QUESTION_BORDER_PANEL_COLOR,
                                        width=PANEL_WIDTH
                                    )
                                )

                                # Raw model output
                                self.console.print(
                                    Panel(
                                        model_output,
                                        title="Raw Model Response",
                                        border_style=MODEL_RESPONSE_BORDER_PANEL_COLOR,
                                        width=PANEL_WIDTH
                                    )
                                )

                                # Extraction details
                                metadata = result.metadata or {}
                                extracted = metadata.get('extracted_answer', 'N/A')
                                method = metadata.get('extraction_method', metadata.get('method', 'N/A'))

                                confidence_val = metadata.get('extraction_confidence', metadata.get('confidence', 'N/A'))
                                confidence = f"{float(confidence_val):.2f}" if isinstance(confidence_val, (int, float)) else confidence_val

                                extraction_details = [
                                    f"Extracted Answer: {extracted}",
                                    f"Method: {method}",
                                    f"Confidence: {confidence}",
                                ]

                                if 'pattern_type' in metadata:
                                    extraction_details.append(f"Pattern Type: {metadata['pattern_type']}")

                                if 'extractor' in metadata:
                                    extraction_details.append(f"Extractor: {metadata['extractor']}")

                                if (result.metadata and "alternative_answers" in result.metadata):
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
                                        border_style=CORRECT_ANSWER_BORDER_PANEL_COLOR if result.correct else INCORRECT_ANSWER_BORDER_PANEL_COLOR,
                                        width=PANEL_WIDTH
                                    )
                                )

                        # Add to results
                        results.append(result)

                        # Update correct count and progress
                        if result.correct:
                            correct_count += 1

                        current_accuracy = correct_count / (i + 1)

                        progress.update(
                            task_id,
                            advance=1,
                            accuracy=current_accuracy,
                            correct=correct_count
                        )
                else:
                    # Process examples in parallel
                    # Create tasks for parallel processing
                    tasks = [
                        self.process_example(task, example, i, len(examples))
                        for i, example in enumerate(examples)
                    ]

                    # Track results as they complete
                    processed_count = 0
                    correct_count = 0
                    results = []

                    # Set up for collecting results
                    pending = set(asyncio.create_task(t) for t in tasks)

                    # Process results as they come in
                    while pending:
                        # Wait for the next result
                        done, pending = await asyncio.wait(
                            pending,
                            return_when=asyncio.FIRST_COMPLETED
                        )

                        # Process completed results
                        for future in done:
                            try:
                                result = future.result()
                                processed_count += 1

                                # Add to results
                                results.append(result)

                                # Update correct count if the answer is correct
                                if result.correct:
                                    correct_count += 1

                                # Calculate current accuracy
                                current_accuracy = correct_count / processed_count if processed_count > 0 else 0

                                # Update progress
                                progress.update(
                                    task_id,
                                    completed=processed_count,
                                    accuracy=current_accuracy,
                                    correct=correct_count
                                )
                            except Exception as e:
                                if self.console:
                                    self.console.print(f"[red]Error processing example:[/red] {str(e)}")
            finally:
                # Make sure to stop the progress bar
                progress.stop()
        else:
            # Silent mode - just process everything without UI output
            if self.sequential:
                # Process sequentially in silent mode
                results = []
                for example in examples:
                    prompt = self.get_prompt(example.question)

                    # Handle streaming in silent mode
                    if self.streaming:
                        model_output = ""
                        async for chunk in self.model.stream_generate(
                            prompt,
                            max_tokens=self.max_tokens
                        ):
                            model_output += chunk
                    else:
                        model_output = await self.model.generate(
                            prompt,
                            max_tokens=self.max_tokens
                        )

                    result = await task.evaluate_example(example, model_output)
                    result.model_id = self.model.model_id
                    results.append(result)
            else:
                # Process in parallel in silent mode
                tasks = [
                    self.process_example(task, example, i, len(examples))
                    for i, example in enumerate(examples)
                ]

                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks)

        # Save results to file if output directory is specified
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
