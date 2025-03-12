# Benchpress: LLM Evaluation Framework

Benchpress is a modern Python framework for evaluating Large Language Models (LLMs) using standardized benchmarks through OpenAI-compatible APIs.

## Features

- Run standardized benchmarks against any LLM with an OpenAI-compatible API
- Strongly typed codebase with full type annotations
- Extendable architecture for adding new benchmarks and model providers
- Command-line interface for easy usage
- Support for multiple benchmarks:
  - MATH-500: A benchmark of 500 challenging math problems
  - AIME24: A benchmark based on the American Invitational Mathematics Examination
  - GPQA Diamond: A benchmark of graduate-level problems across various academic disciplines
- Support for multiple model providers:
  - OpenAI API (GPT models)
  - GLHF.chat (access to Hugging Face models)
  - Any OpenAI-compatible API endpoints
- Clean, consistent API for benchmark execution and result analysis

## Installation

Benchpress uses [uv](https://github.com/astral-sh/uv) as its package manager for fast, reliable dependency management.

```bash
# Clone the repository
git clone https://github.com/yourusername/benchpress.git
cd benchpress

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

## Quick Start

### Environment Setup

Benchpress uses environment variables for configuration. You can set them up by:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your API keys and configuration
# Set at least one of OPENAI_API_KEY or GLHF_API_KEY
nano .env  # or use your preferred editor
```

### List available benchmark tasks

```bash
benchpress list-tasks
```

### Evaluate a model on a benchmark

```bash
# With environment variables set in .env
# Evaluate on the MATH-500 benchmark
benchpress evaluate --task math500 --model openai:gpt-4

# Evaluate on the AIME24 benchmark
benchpress evaluate --task aime24 --model openai:gpt-4

# Evaluate on the GPQA Diamond benchmark
benchpress evaluate --task gpqa --model openai:gpt-4

# Evaluate on multiple benchmarks simultaneously
benchpress evaluate --task math500 --task aime24 --task gpqa --model openai:gpt-4

# Or provide the API key directly
benchpress evaluate --task aime24 --model openai:gpt-4 --api-key "your-api-key" --limit 1
```

### Using other OpenAI-compatible APIs

```bash
# Example using an Anthropic API through an OpenAI-compatible endpoint
benchpress evaluate --task math500 --model compatible:claude-3-opus-20240229 --api-base "https://your-compatible-api-endpoint" --api-key "your-api-key"

# Or evaluate AIME24 with a custom endpoint
benchpress evaluate --task aime24 --model compatible:llama-3-70b-instruct --api-base "https://your-compatible-api-endpoint" --api-key "your-api-key"

# Using GLHF.chat to access Hugging Face models (requires GLHF credits)
benchpress evaluate --task math500 --model glhf:mistralai/Mistral-7B-Instruct-v0.3 --api-key "your-glhf-api-key"
benchpress evaluate --task aime24 --model glhf:meta-llama/Meta-Llama-3.1-8B-Instruct --system-prompt "You are a math tutor specializing in competition math."

# Run multiple benchmarks against a GLHF model
benchpress evaluate --task math500 --task aime24 --task gpqa --model glhf:meta-llama/Meta-Llama-3.1-8B-Instruct --limit 5

# Save results to a specific directory
benchpress evaluate --task math500 --task gpqa --model glhf:meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir ./my_results

# Note: GLHF.chat is a pay-per-token service - you'll need to add credits at https://glhf.chat/billing
```

## Multi-Task Evaluation

Benchpress supports evaluating models on multiple tasks in a single command:

```bash
# Run all available benchmarks
benchpress evaluate --task math500 --task aime24 --task gpqa --model openai:gpt-4 --output-dir results

# Run multiple benchmarks with a limit
benchpress evaluate --task math500 --task gpqa --model openai:gpt-4 --limit 10

# Compare different models on the same tasks
benchpress evaluate --task math500 --task aime24 --model openai:gpt-4 --output-dir results/gpt4
benchpress evaluate --task math500 --task aime24 --model glhf:meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir results/llama
```

This produces:
- Individual results files for each task-model combination
- A combined accuracy report showing performance across all tasks
- An overall accuracy metric that aggregates results from all evaluated examples

## Adding New Tasks

1. Create a new file in `src/benchpress/tasks/` for your task
2. Define your task class extending `BaseTask`
3. Implement the required methods (`name`, `description`, `load_examples`, `evaluate_example`)
4. Register your task with the `@register_task` decorator

Example:

```python
from benchpress.tasks import BaseTask, Example, register_task

@register_task
class MyNewTask(BaseTask):
    @property
    def name(self) -> str:
        return "my_new_task"
    
    @property
    def description(self) -> str:
        return "Description of my new task"
    
    # Implement other required methods
```

## Development

Benchpress uses several tools to ensure code quality:

- **black**: Code formatting
- **ruff**: Linting
- **mypy**: Static type checking
- **pytest**: Testing

Run the quality checks:

```bash
# Format code
black src/

# Lint code
ruff check src/

# Type check
mypy src/

# Run tests
pytest
```

## License

MIT License

## Acknowledgments

- The MATH-500 benchmark is based on the [MATH dataset](https://github.com/hendrycks/math) by Hendrycks et al.
- The AIME24 benchmark is based on the American Invitational Mathematics Examination (AIME) administered by the Mathematical Association of America.
- The GPQA Diamond benchmark is inspired by the [GPQA dataset](https://github.com/openai/GPQA) that evaluates models on graduate-level problems across various academic disciplines.