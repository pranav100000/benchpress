# Benchpress Development Guidelines

IMPORTANT GUIDELINES:
1. ALWAYS think hard for every prompt you have. There is no concern of wasting API tokens, use as many as needed to get the best result possible.
2. DO NOT ever put imports anywhere except the top of the file.
3. DO NOT ever manually fix lint issues unless you have to. Always try to fix lint issues by running the linter first.
4. DO NOT ever make code changes without being sure you have the best design and approach. Before you make a code change, compare and contrast potential approaches and pick the best one. Then when you pick the best one, determine how sure you are that it is the best approach as a % number. EXPLICITLY STATE what this % number of how sure you are this is the best approach/design before you write any code.
5. If you are unsure, keep thinking. 
6. DO NOT try to use an API key in any CLI commands. it is already set up

## Development Environment

### Setup
- **Clone Repository**: `git clone <repo-url> && cd benchpress`
- **Environment Variables**: `cp .env.example .env` then edit .env with your API keys
  - Required API keys: OpenAI, Anthropic, Hugging Face (for dataset access)
  - Optional: GLHF_API_KEY for GLHF model access
- **Python Environment**: 
  - Recommended: Python 3.10+ (tested on 3.10 and 3.11)
  - Virtual environment: `uv venv && source .venv/bin/activate`
  - Installation: `uv pip install -e ".[dev]"`

### Common Commands
- **Code Formatting**: `black src/ tests/`
- **Linting**: `ruff check src/ tests/`
- **Type Checking**: `mypy src/`
- **Testing**:
  - Run all tests: `pytest`
  - Run specific test file: `pytest tests/test_math500.py`
  - Run specific test: `pytest tests/test_math500.py::test_math500_load_examples -v`
  - Run with coverage: `pytest --cov=src/benchpress`
- **CLI Commands**:
  - List available tasks: `benchpress list-tasks`
  - Evaluate model on task: `benchpress evaluate --task math500 --model openai:gpt-4`
  - Evaluate with sample limit: `benchpress evaluate --task math500 --model openai:gpt-4 --limit 10`
  - Specify output file: `benchpress evaluate --task math500 --model openai:gpt-4 --output results/my_results.json`
  - Debug mode with detailed extraction info: `benchpress evaluate --task math500 --model openai:gpt-4 --debug`
  - Run evaluation with specific example ID: `benchpress evaluate --task math500 --model openai:gpt-4 --id "example_id"`

## Code Organization

### Project Structure
- `src/benchpress/` - Main package
  - `cli.py` - Command-line interface
  - `evaluation/` - Evaluation engine and metrics
  - `models/` - Model adapters and interfaces
  - `tasks/` - Task definitions and example classes
  - `datasets/` - Dataset management system
  - `extraction/` - Answer extraction system

### Core Components
- **Tasks**: Define benchmarks (GPQA, MATH500, AIME24)
- **Models**: Adapters for LLM APIs (OpenAI, Anthropic, HuggingFace, etc.)
- **Datasets**: Management system for loading and processing data
- **Evaluation**: Engine for running evaluations and computing metrics
- **Extraction**: System for extracting answers from model responses

## Extraction System

### Architecture
- **Base Classes**: `BaseExtractor`, `extract_answer` in `extraction/base.py`
- **Extraction Patterns**: 
  - Pattern definitions in `extraction/patterns.py`
  - Domain-specific patterns in `extraction/math.py`
  - Custom extractors in task-specific files (e.g., `tasks/aime24.py`)
- **Processing Components**:
  - Answer normalization in `extraction/processors.py`
  - Response cleaning in `extraction/general.py`
- **Registry**: Pattern registration in `extraction/registry.py`

### Extraction Pattern System
- **Pattern Definition**: Named regex patterns with optional preprocessors and normalizers
- **Pattern Registration**: Register patterns with `register_pattern(name, pattern, priority, preprocessor, normalizer)`
- **Pattern Matching**: Apply patterns in priority order, returning first match
- **Metadata Tracking**: Each extraction includes method name and confidence score
- **Normalization**: Standardize formats (e.g., fractions, decimals) for consistent comparison

### Debugging Features
- **Debug Mode**: Enable with `--debug` flag in CLI commands
- **Debug Output**:
  - Raw model input/output
  - Exact extraction pattern matched
  - Pre and post-normalization values
  - Extraction metadata (method, confidence)
- **Troubleshooting**: Debug mode helps identify extraction issues for complex responses

### Best Practices
- **Add New Patterns**: When model output formats aren't captured by existing patterns
- **Add Normalizers**: To handle domain-specific answer formatting
- **Test New Patterns**: Use `test_extraction.py` to verify pattern behavior
- **Hybrid Approach**: Balance between general extraction framework and task-specific customization

## Dataset Management System (v2)

### Architecture
- **Base Classes**: `Dataset`, `DatasetRegistry` in `datasets/v2/base.py`
- **Format Adapters**:
  - `CSVDataset` in `datasets/csv_dataset.py`
  - `JSONDataset` in `datasets/json_dataset.py`
  - `HuggingFaceDataset` in `datasets/huggingface_dataset.py`
- **Task-Specific Implementations**:
  - `GPQADataset` in `datasets/gpqa_dataset.py`
  - `MATH500HfDataset` in `datasets/math500_hf_dataset.py`

### Usage Patterns
- **Registry**: Register datasets with `DatasetRegistry.register("name", dataset_class)`
- **Loading**: Load datasets with `DatasetRegistry.get("name")`
- **Filtering**: Use `dataset.filter(lambda example: ...)` for subset selection
- **Sampling**: Use `dataset.sample(n)` for random sampling
- **Integration**: Task classes use datasets via `get_dataset()` method

### Hugging Face Datasets Features

#### Loading Options
- **Hub Datasets**: Load directly from hub with `load_dataset("namespace/dataset_name")`
- **Revision Control**: Specify version with `revision="main"` (tag, branch, commit)
- **Split Mapping**: Map files to splits with `data_files={"train": "train.csv", "test": "test.csv"}`
- **Subsetting**: Load subset of files with `data_files="pattern*.json"` or `data_dir="specific_dir"`

#### Performance Optimization
- **Multiprocessing**: Speed up dataset loading with `num_proc=8` for parallel processing
- **Memory Mapping**: Use `Dataset.from_file("data.arrow")` for Arrow files to save disk space
- **Streaming**: Enable with `streaming=True` for large datasets that don't fit in memory

#### Advanced Split Handling
- **Concatenation**: Combine splits with `split="train+test"`
- **Indexed Slicing**: Select specific rows with `split="train[10:20]"`
- **Percentage Slicing**: Select by percentage with `split="train[:10%]"`
- **Cross-Validation**: Create CV splits with percentages - `[f"train[{k}%:{k+10}%]" for k in range(0, 100, 10)]`
- **Rounding Control**: Specify rounding behavior with `rounding="pct1_dropremainder"` for equal-sized splits

#### Format Support
- **Multiple Formats**: Support for CSV, JSON, Parquet, Arrow, SQL, WebDataset
- **Custom Features**: Define schema with `features=Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})`

## Style Guidelines

### Code Style
- **Formatting**: 
  - Black with 88 character line length
  - Follows PEP 8 conventions
  - Use trailing commas in multi-line collections
- **Imports**: Organized in groups (separated by blank line):
  1. Standard library imports (sorted alphabetically)
  2. Third-party library imports (sorted alphabetically)
  3. Local application imports (sorted alphabetically)
- **Types**: 
  - Full type annotations required on all functions and methods
  - Type annotations on class attributes and variables
  - Use `typing` module for complex types (Optional, Union, etc.)
  - Prefer typed collections (List[str], Dict[str, int], etc.)
- **Naming**: 
  - `snake_case` for variables, functions, methods, modules
  - `PascalCase` for classes and exceptions
  - `UPPER_CASE` for constants
  - Prefix private attributes/methods with underscore (_private_method)

### Documentation
- **Doc Style**: Google-style docstrings with explicit parameter/return annotations
  ```python
  def function(param1: str, param2: int) -> bool:
      """Short description of function.
      
      More detailed description if needed.
      
      Args:
          param1: Description of param1
          param2: Description of param2
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: Description of when this error is raised
      """
  ```
- **Module Docs**: Each module should have a docstring describing its purpose
- **Class Docs**: Each class should have a docstring describing its purpose, behavior, and usage

### Error Handling
- **Exception Types**: Prefer specific exceptions over generic ones
- **Context**: Add context when re-raising exceptions
- **Graceful Failures**: Implement fallbacks when appropriate (especially for dataset loading)
- **Logging**: Use the `logging` module for diagnostic information
- **Validation**: Validate inputs early with descriptive error messages

### Architecture Patterns
- **Inheritance**: Follow established abstractions with proper inheritance
- **Composition**: Prefer composition over inheritance when appropriate
- **Registry Pattern**: Use for plugin-like components (tasks, datasets, models)
- **Factory Pattern**: Use for creating instances (example creation)
- **Adapter Pattern**: Use for interfacing with external systems
- **Dependency Injection**: Inject dependencies rather than creating them internally

## Testing Guidelines

### Test Organization
- Tests mirror the package structure
- Test files named `test_*.py`
- Test functions named `test_*`

### Testing Patterns
- Use pytest fixtures for setup and teardown
- Mock external dependencies (API calls, filesystem)
- Test both success and failure paths
- Use parameterized tests for multiple input scenarios
- Test extraction patterns with a variety of input formats

### Extraction Testing
- **Pattern Testing**: Use `test_extraction.py` to verify pattern matching behavior
- **Normalizer Testing**: Use `test_normalizers.py` to verify normalization functions
- **End-to-End Testing**: Test extraction within the evaluation flow
- **Edge Cases**: Test with complex mathematical expressions, LaTeX, special characters
- **Regression Testing**: Add test cases for previously fixed bugs

### Test Coverage
- Aim for 80%+ coverage on core logic
- Test public interfaces thoroughly
- Integration tests for task-model interactions
- Test all extraction patterns with representative examples

## Performance Considerations

### Dataset Loading
- Use lazy loading when possible
- Implement caching for expensive operations
- Consider batch processing for large datasets

### Evaluation
- Use async/await for concurrent API calls
- Implement proper rate limiting for API calls
- Monitor memory usage with large datasets

## Dependency Management

### Primary Dependencies
- **pydantic**: Data validation and settings management
- **typer**: CLI interface
- **tqdm**: Progress bars
- **openai**: OpenAI API client
- **anthropic**: Anthropic API client
- **datasets**: Hugging Face datasets library

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting
- **ruff**: Linting
- **mypy**: Type checking
- **pytest-cov**: Test coverage