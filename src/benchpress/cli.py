"""Command-line interface for benchpress."""

import asyncio
import sys
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from benchpress.evaluation import EvaluationEngine
from benchpress.models import GlhfModel, OpenAICompatibleModel
from benchpress.tasks import task_registry

app = typer.Typer(
    name="benchpress",
    help="Benchpress: LLM evaluation framework for standardized benchmarks",
)
console = Console()


@app.command("list-tasks")
def list_tasks() -> None:
    """List all available benchmark tasks."""
    table = Table(title="Available Benchmark Tasks")
    table.add_column("Task Name", style="cyan")
    table.add_column("Description")

    for _name, task_class in task_registry.items():
        task = task_class()
        table.add_row(task.name, task.description)

    console.print(table)


@app.command("evaluate")
def evaluate(
    task: List[str] = typer.Option(  # noqa: B008
        ...,
        "--task",
        "-t",
        help="Name of the task(s) to evaluate. Can be specified multiple times.",
    ),
    model: str = typer.Option(  # noqa: B008
        ...,
        "--model",
        "-m",
        help="Model identifier (e.g., openai:gpt-4, glhf:mistralai/Mistral-7B-Instruct-v0.3)",  # noqa: E501
    ),
    output_dir: Optional[str] = typer.Option(  # noqa: B008
        None, "--output-dir", "-o", help="Directory to save results"
    ),
    limit: Optional[int] = typer.Option(  # noqa: B008
        None, "--limit", "-l", help="Maximum number of examples to evaluate"
    ),
    api_key: Optional[str] = typer.Option(  # noqa: B008
        None, "--api-key", "-k", help="API key for the model"
    ),
    api_base: Optional[str] = typer.Option(  # noqa: B008
        None, "--api-base", "-b", help="Base URL for the API"
    ),
    system_prompt: Optional[str] = typer.Option(  # noqa: B008
        None, "--system-prompt", "-s", help="System prompt to use for the model"
    ),
) -> None:
    """Evaluate a model on one or more benchmark tasks."""
    # Validate tasks
    invalid_tasks = [t for t in task if t not in task_registry]
    if invalid_tasks:
        console.print(
            f"[red]Error:[/red] Task(s) not found: {', '.join(invalid_tasks)}"
        )
        console.print("Use 'benchpress list-tasks' to see available tasks")
        sys.exit(1)

    # Parse model identifier
    if ":" not in model:
        console.print(
            f"[red]Error:[/red] Invalid model identifier: '{model}'. "
            "Format should be 'provider:model_name'"
        )
        sys.exit(1)

    provider, model_name = model.split(":", 1)

    # Initialize model
    try:
        if provider == "openai" or provider == "compatible":
            model_instance = OpenAICompatibleModel(
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
                system_prompt=system_prompt,
            )
        elif provider == "glhf":
            model_instance = GlhfModel(
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
                system_prompt=system_prompt,
            )
        else:
            console.print(f"[red]Error:[/red] Unsupported model provider: '{provider}'")
            console.print("Supported providers: 'openai', 'compatible', 'glhf'")
            sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)

    # Initialize evaluation engine
    engine = EvaluationEngine(
        model=model_instance,
        output_dir=output_dir,
    )

    # Prepare results table
    results_table = Table(title=f"Evaluation Results for {model}")
    results_table.add_column("Task", style="cyan")
    results_table.add_column("Examples")
    results_table.add_column("Correct")
    results_table.add_column("Accuracy")

    all_summaries = []
    errors = []

    # Evaluate each task
    for task_name in task:
        console.print(
            f"Evaluating [cyan]{model}[/cyan] on task [cyan]{task_name}[/cyan]..."
        )

        # Initialize task
        task_instance = task_registry[task_name]()

        # Run evaluation
        try:
            summary = asyncio.run(engine.evaluate_task(task_instance, limit=limit))
            all_summaries.append(summary)

            # Add to results table
            results_table.add_row(
                summary.task_name,
                str(summary.total_examples),
                str(summary.correct),
                f"{summary.accuracy:.2%}",
            )

        except Exception as e:
            error_msg = str(e)
            console.print(f"[red]Error evaluating {task_name}:[/red] {error_msg}")
            errors.append((task_name, error_msg))

    # Display results
    if all_summaries:
        console.print("\n[green]Evaluation complete![/green]\n")
        console.print(results_table)

        if len(all_summaries) > 1:
            # Calculate overall accuracy across all tasks
            total_examples = sum(s.total_examples for s in all_summaries)
            total_correct = sum(s.correct for s in all_summaries)
            overall_accuracy = (
                total_correct / total_examples if total_examples > 0 else 0
            )

            console.print(f"\n[bold]Overall accuracy:[/bold] {overall_accuracy:.2%}")

        if output_dir:
            console.print(f"\nResults saved to: {output_dir}")

    # Report any errors
    if errors:
        console.print("\n[red]Some tasks failed:[/red]")
        for task_name, error_msg in errors:
            console.print(f"  - {task_name}: {error_msg}")

        if not all_summaries:
            sys.exit(1)


if __name__ == "__main__":
    app()
