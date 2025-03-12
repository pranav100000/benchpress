# Benchpress Development Guidelines

## Commands
- **Environment**: `cp .env.example .env` then edit .env with your API keys
- **Install**: `uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"`
- **Format**: `black src/ tests/`
- **Lint**: `ruff check src/ tests/`
- **Type check**: `mypy src/`
- **Test all**: `pytest`
- **Test single**: `pytest tests/test_math500.py::test_math500_load_examples -v`
- **Run CLI**: `benchpress list-tasks` or `benchpress evaluate --task math500 --model openai:gpt-4`

## Style Guidelines
- **Imports**: Organized in groups - stdlib, third-party, local (sorted alphabetically)
- **Formatting**: Black (88 char line length), follows PEP 8
- **Types**: Full type annotations required on all functions, methods and variables
- **Naming**: Use `snake_case` for variables/functions, `PascalCase` for classes
- **Error handling**: Prefer specific exceptions, add context when re-raising
- **Doc style**: Google-style docstrings with explicit param/return annotations
- **Architecture**: Follow established abstractions with proper inheritance