"""Evaluation engine for benchpress."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from rich.console import Console

from ..models.base import BaseModel
from ..tasks.base import BaseTask
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
    ):
        """Initialize the evaluation engine.

        Args:
            model: The model to evaluate
            output_dir: Directory to save evaluation results (optional)
            silent: Whether to suppress real-time output (optional)
            debug: Whether to show detailed debug information (optional)
            console: Rich console for output formatting (optional)
            streaming: Whether to use streaming API for model generation (optional)
        """
        self.model = model
        self.output_dir = Path(output_dir) if output_dir else None
        self.silent = silent
        self.debug = debug
        self.console = console
        self.streaming = streaming

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

        # Set up progress display if not in silent mode
        if not self.silent and self.console:
            from rich.panel import Panel
            from rich.console import Group
            from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn
            
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
            
            # Process examples
            try:
                correct_count = 0
                for i, example in enumerate(examples):
                    # Update example counter in progress description
                    progress.update(task_id, description=f"Evaluating {task.name} (Example {i+1}/{len(examples)})")
                    
                    # Format the question for display using the unicodeit-based formatter
                    formatted_question = latex_to_unicode(example.question)
                    
                    # Display the question in a panel directly (not inside the live display)
                    self.console.print("")  # Add spacing
                    self.console.print(
                        Panel(
                            formatted_question,
                            title=f"Question {i+1}/{len(examples)}",
                            border_style="blue",
                            width=88
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

                    # Use streaming if enabled
                    if self.streaming:
                        # Stop the progress bar before streaming
                        progress.stop()
                        
                        # Initialize empty output
                        model_output = ""
                        
                        from rich.live import Live
                        from rich.panel import Panel
                        
                        # Initialize streaming content
                        streaming_content = ""
                        model_output = ""
                        
                        # Create a panel that we'll update with streaming content
                        streaming_panel = Panel(
                            "",
                            title="Model Response (Streaming...)",
                            border_style="yellow",
                            width=88,
                            height=None  # Allow panel to grow as needed
                        )
                        
                        # Use a Live display for updating the panel
                        # Set refresh_per_second higher for more responsive updates
                        # Set auto_refresh=False to ensure we control when to refresh
                        # Set vertical_overflow="visible" to allow content to be scrollable
                        with Live(streaming_panel, console=self.console, refresh_per_second=10, 
                                 transient=False, auto_refresh=False, vertical_overflow="visible") as live:
                            async for chunk in self.model.stream_generate(prompt):
                                model_output += chunk
                                streaming_content += chunk
                                
                                # Apply LaTeX formatting to the entire accumulated content
                                formatted_streaming = latex_to_unicode(streaming_content)
                                
                                # Update the panel content with formatted text
                                streaming_panel.renderable = formatted_streaming
                                # Force a refresh to update the display
                                live.refresh()
                            
                            # After streaming completes, update the panel title
                            streaming_panel.title = "Model Response (Complete)"
                            # No need to reformat as we've been formatting all along
                            live.refresh()
                        
                        # Restart the progress bar
                        progress.start()
                    else:
                        # Standard non-streaming approach
                        # Show a "Generating..." message
                        self.console.print("Generating response...", style="yellow")
                        
                        # Generate the response
                        model_output = await self.model.generate(prompt)
                        
                        # Format and display the complete response
                        formatted_output = latex_to_unicode(model_output)
                        self.console.print(
                            Panel(
                                formatted_output,
                                title="Model Response",
                                border_style="yellow",
                                width=88
                            )
                        )

                    # Evaluate the example
                    result = await task.evaluate_example(example, model_output)
                    result.model_id = self.model.model_id

                    # Display the evaluation results directly in the console
                    from rich.text import Text
                    from rich.console import Group
                    from rich.panel import Panel
                    
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
                            border_style="green" if result.correct else "red",
                            width=88
                        )
                    )

                    # If the example has a reference answer, display it
                    if hasattr(example, "answer") and example.answer:
                        raw_answer = str(example.answer)
                        # Format the answer using the unicodeit-based formatter
                        formatted_answer = latex_to_unicode(raw_answer)
                        
                        # Create a panel for the reference answer
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
                                border_style="green",
                                width=88
                            )
                        )

                        # Show additional info for debugging incorrect answers
                        if not result.correct and result.metadata:
                            raw_expected = str(example.answer)
                            raw_extracted = result.metadata.get(
                                "extracted_answer", "N/A")

                            # Get method and confidence
                            method = result.metadata.get('extraction_method', 
                                    result.metadata.get('method', 'N/A'))
                            
                            confidence_val = result.metadata.get('extraction_confidence', 
                                             result.metadata.get('confidence', 'N/A'))
                            # Format confidence as float if it's a number
                            confidence = f"{float(confidence_val):.2f}" if isinstance(confidence_val, (int, float)) else confidence_val
                            
                            # Create a panel for the debug info
                            debug_items = []
                            
                            # Add debug info
                            debug_text = Text(f"Expected='{raw_expected}' vs Extracted='{raw_extracted}'", style="dim")
                            debug_items.append(debug_text)
                            debug_items.append(Text(f"Method: {method}, Confidence: {confidence}", style="dim"))
                            
                            # Display the debug info
                            self.console.print(
                                Panel(
                                    Group(*debug_items),
                                    title="Debug Information",
                                    border_style="red",
                                    width=88
                                )
                            )

                        # Show detailed debug information if debug mode is enabled
                        if self.debug:
                            # Display a header for the debug section
                            self.console.print("\n[bold]Detailed Debug Information:[/bold]")
                            
                            # Display raw question
                            self.console.print(
                                Panel(
                                    example.question,
                                    title="Raw Question",
                                    border_style="blue",
                                    width=88
                                )
                            )
                            
                            # Display raw model output
                            self.console.print(
                                Panel(
                                    model_output,
                                    title="Raw Model Response",
                                    border_style="yellow",
                                    width=88
                                )
                            )
                            
                            # Get extraction details with safeguards
                            metadata = result.metadata or {}
                            extracted = metadata.get('extracted_answer', 'N/A')
                            method = metadata.get('extraction_method', metadata.get('method', 'N/A'))
                            
                            # Handle confidence which could be a string, float, or missing
                            confidence_val = metadata.get('extraction_confidence', metadata.get('confidence', 'N/A'))
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
                            
                            # Display extraction details
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
                    
                    # Update correct count if the answer is correct
                    if result.correct:
                        correct_count += 1
                    
                    # Calculate current accuracy
                    current_accuracy = correct_count / (i + 1)
                    
                    # Update progress with new accuracy
                    progress.update(
                        task_id, 
                        advance=1, 
                        accuracy=current_accuracy,
                        correct=correct_count
                    )
            
            finally:
                # Make sure to stop the progress bar
                progress.stop()
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

                # Handle streaming in silent mode too
                if self.streaming:
                    model_output = ""
                    async for chunk in self.model.stream_generate(prompt):
                        model_output += chunk
                else:
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
