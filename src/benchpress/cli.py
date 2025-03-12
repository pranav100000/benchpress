"""Command-line interface for benchpress."""

import asyncio
import sys
from typing import Optional

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
    task: str = typer.Option(..., "--task", "-t", help="Name of the task to evaluate"),
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model identifier (e.g., openai:gpt-4, glhf:mistralai/Mistral-7B-Instruct-v0.3)",
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Directory to save results"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Maximum number of examples to evaluate"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key for the model"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", "-b", help="Base URL for the API"
    ),
    system_prompt: Optional[str] = typer.Option(
        None, "--system-prompt", "-s", help="System prompt to use for the model"
    ),
) -> None:
    """Evaluate a model on a benchmark task."""
    # Check if task exists
    if task not in task_registry:
        console.print(f"[red]Error:[/red] Task '{task}' not found")
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

    # Initialize task
    task_instance = task_registry[task]()

    # Initialize evaluation engine
    engine = EvaluationEngine(
        model=model_instance,
        output_dir=output_dir,
    )

    console.print(f"Evaluating [cyan]{model}[/cyan] on task [cyan]{task}[/cyan]...")

    # Run evaluation
    try:
        summary = asyncio.run(engine.evaluate_task(task_instance, limit=limit))

        # Display results
        console.print("\n[green]Evaluation complete![/green]\n")
        console.print(f"Task: {summary.task_name}")
        console.print(f"Model: {summary.model_id}")
        console.print(f"Examples: {summary.total_examples}")
        console.print(f"Correct: {summary.correct}")
        console.print(f"Accuracy: {summary.accuracy:.2%}")

        if output_dir:
            console.print(f"\nResults saved to: {output_dir}")

    except Exception as e:
        console.print(f"[red]Error during evaluation:[/red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    app()
