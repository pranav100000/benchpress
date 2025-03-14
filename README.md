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
- Sophisticated answer extraction system:
  - Pattern-based extraction for multiple answer formats
  - Domain-specific extractors for mathematical expressions
  - Answer normalization for consistent comparison
  - Extraction metadata tracking (method, confidence)
- Debug mode for detailed extraction information
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

## Dataset Download

datasets/gpqa_dataset.zip contains the data files, and is password-proteced with this password: `deserted-untie-orchid`.

Alternatively, the dataset is available on Hugging Face: https://huggingface.co/datasets/idavidrein/gpqa

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

# Run with debug mode to see detailed extraction information
benchpress evaluate --task math500 --model openai:gpt-4 --debug

# Run evaluation for a specific example ID
benchpress evaluate --task math500 --model openai:gpt-4 --id "example_id"

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

## Answer Extraction System

Benchpress includes a sophisticated extraction system to parse model outputs and extract standardized answers.

### Key Components

- **Pattern Registry**: Central registry of extraction patterns in `extraction/registry.py`
- **Pattern Definitions**: Common patterns in `extraction/patterns.py`, math-specific patterns in `extraction/math.py`
- **Normalizers**: Functions to standardize extracted answers in `extraction/processors.py`
- **Base Extractor**: Core extraction logic in `extraction/base.py`

### Adding New Extraction Patterns

1. Define a new pattern with a regular expression that captures the answer in a named group
2. Register the pattern with priority, preprocessor, and normalizer functions
3. Add the pattern to the registry

Example:

```python
from benchpress.extraction.registry import register_pattern
from benchpress.extraction.processors import normalize_decimal

# Define and register a new extraction pattern
register_pattern(
    name="custom_answer_format",
    pattern=r"My answer is: (?P<answer>[\d\.]+)",
    priority=50,  # Higher priority patterns are tried first
    preprocessor=None,  # Optional function to preprocess the text
    normalizer=normalize_decimal  # Optional function to normalize the extracted answer
)
```

### Debugging Extraction

Use the `--debug` flag to see detailed information about the extraction process:

```bash
benchpress evaluate --task math500 --model openai:gpt-4 --debug --limit 1
```

The debug output includes:
- Raw model input and output
- Extraction pattern matched
- Pre and post-normalization values
- Extraction metadata (method, confidence)

## Development

Benchpress uses several tools to ensure code quality:

- **black**: Code formatting
- **ruff**: Linting
- **mypy**: Static type checking
- **pytest**: Testing

Run the quality checks:

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/

# Run all tests
pytest

# Run specific test files
pytest tests/test_extraction.py
pytest tests/test_math500.py

# Run test with verbose output
pytest tests/test_extraction.py -v
```

### Testing Extraction Patterns

When adding new extraction patterns, write tests to verify their behavior:

```python
# Example test for a new extraction pattern
def test_custom_extraction_pattern():
    from benchpress.extraction.base import extract_answer
    
    # Test with a sample response
    response = "My analysis is complete. My answer is: 42.5"
    result = extract_answer(response)
    
    assert result is not None
    assert result.value == "42.5"
    assert result.normalized == "42.5"
    assert result.metadata["method"] == "custom_answer_format"
```

## License

MIT License

## Acknowledgments

- The MATH-500 benchmark is based on the [MATH dataset](https://github.com/hendrycks/math) by Hendrycks et al.
- The AIME24 benchmark is based on the American Invitational Mathematics Examination (AIME) administered by the Mathematical Association of America.
- The GPQA Diamond benchmark is inspired by the [GPQA dataset](https://github.com/openai/GPQA) that evaluates models on graduate-level problems across various academic disciplines.