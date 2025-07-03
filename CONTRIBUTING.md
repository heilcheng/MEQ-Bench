# Contributing to MEQ-Bench

Thank you for your interest in contributing to MEQ-Bench! This document provides guidelines and instructions for contributing to the project. We welcome contributions from researchers, developers, and healthcare professionals who want to help improve medical AI evaluation.

## Table of Contents

- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Conventions](#coding-conventions)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)
- [Getting Help](#getting-help)

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

#### 🐛 Bug Reports
If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs. actual behavior
- Your environment details (OS, Python version, etc.)
- Relevant code snippets or error messages

#### ✨ Feature Requests
For new features:
- Describe the feature and its use case
- Explain why it would be valuable for medical AI evaluation
- Provide examples of how it would be used
- Consider backward compatibility implications

#### 📝 Documentation Improvements
Help improve our documentation by:
- Fixing typos or unclear explanations
- Adding missing documentation
- Improving code examples
- Translating documentation

#### 🔬 Research Contributions
Contribute to the scientific validity of MEQ-Bench:
- New evaluation metrics or methodologies
- Validation studies with human experts
- Dataset improvements or additions
- Performance analysis and benchmarking results

#### 🚀 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Code refactoring
- Additional model integrations

### Before You Start

1. **Check existing issues** to see if your contribution is already being discussed
2. **Open an issue** for major changes to discuss the approach before implementation
3. **Fork the repository** and create a new branch for your contribution
4. **Read this contributing guide** thoroughly

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation Steps

1. **Fork and clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/MEQ-Bench.git
cd MEQ-Bench
```

2. **Create a virtual environment:**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n meq-bench python=3.9
conda activate meq-bench
```

3. **Install development dependencies:**
```bash
pip install -e .[dev]
```

This installs the package in editable mode with all development dependencies including:
- `pytest` for testing
- `black` for code formatting
- `flake8` for linting
- `isort` for import sorting
- `mypy` for type checking
- `bandit` for security scanning
- `pre-commit` for git hooks

4. **Install pre-commit hooks:**
```bash
pre-commit install
```

5. **Create required directories:**
```bash
mkdir -p logs data results
```

6. **Verify installation:**
```bash
python -c "import src.benchmark; print('Installation successful!')"
pytest tests/ -v
```

### Development Dependencies

The development environment includes several dependency groups:

- **Core (`.[dev]`)**: Essential development tools
- **Testing (`.[test]`)**: Testing framework and utilities
- **Documentation (`.[docs]`)**: Sphinx and documentation tools
- **Machine Learning (`.[ml]`)**: ML libraries for full functionality
- **LLM APIs (`.[llm]`)**: API clients for external LLM services
- **Apple Silicon (`.[apple]`)**: MLX framework for Apple M-series optimization
- **Analysis (`.[analysis]`)**: Jupyter, plotting, and analysis tools

For full development environment:
```bash
pip install -e .[dev-full]
```

## Coding Conventions

We follow strict coding conventions to maintain code quality and consistency.

### Code Style

**Formatter: Black**
- Line length: 88 characters
- String quotes: Double quotes preferred
- Run: `black src/ tests/ examples/`

**Import Sorting: isort**
- Compatible with Black
- Group imports: standard library, third-party, local
- Run: `isort src/ tests/ examples/`

**Linting: flake8**
- Max line length: 127 characters
- Max complexity: 10
- Run: `flake8 src/ tests/ examples/`

### Code Quality Standards

**Type Hints**
- Use type hints for all functions and methods
- Import types from `typing` module
- Use `Optional[T]` for nullable types
- Example:
```python
from typing import Dict, List, Optional

def process_data(items: List[str], config: Optional[Dict[str, Any]] = None) -> bool:
    """Process a list of items with optional configuration."""
    pass
```

**Documentation**
- Use Google-style docstrings
- Document all public functions, classes, and methods
- Include type information, parameter descriptions, and examples
- Example:
```python
def evaluate_model(self, model_func: Callable[[str], str]) -> Dict[str, float]:
    """Evaluate a model's performance on the benchmark.
    
    Args:
        model_func: Function that takes a prompt and returns model response.
            Must be callable with signature (str) -> str.
    
    Returns:
        Dictionary mapping metric names to scores between 0.0 and 1.0.
        
    Raises:
        ValueError: If model_func is not callable or returns invalid response.
        
    Example:
        ```python
        def my_model(prompt: str) -> str:
            return "Generated response"
            
        scores = evaluator.evaluate_model(my_model)
        print(f"Overall score: {scores['overall']}")
        ```
    """
```

**Error Handling**
- Use specific exception types
- Provide meaningful error messages
- Include context in error messages
- Handle errors gracefully with fallbacks where appropriate

**Logging**
- Use the `logging` module, not `print()`
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Include contextual information in log messages

### Pre-commit Hooks

We use pre-commit hooks to automatically check code quality:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]
```

### File Organization

- **Source code**: `src/` directory
- **Tests**: `tests/` directory (mirror structure of `src/`)
- **Examples**: `examples/` directory
- **Documentation**: `docs/` directory
- **Configuration**: Root directory (`config.yaml`, `setup.py`, etc.)

## Testing

We use `pytest` for our testing framework with comprehensive test coverage.

### Running Tests

**All tests:**
```bash
pytest
```

**With coverage:**
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

**Specific test file:**
```bash
pytest tests/test_benchmark.py -v
```

**Integration tests:**
```bash
pytest tests/integration/ -v
```

### Test Structure

```
tests/
├── unit/                   # Unit tests
│   ├── test_benchmark.py
│   ├── test_evaluator.py
│   └── test_strategies.py
├── integration/            # Integration tests
│   ├── test_full_pipeline.py
│   └── test_model_integration.py
├── fixtures/               # Test data and fixtures
│   ├── sample_data.json
│   └── mock_responses.py
└── conftest.py            # Pytest configuration and shared fixtures
```

### Writing Tests

**Test Function Naming:**
```python
def test_function_name_expected_behavior():
    """Test that function_name behaves correctly when condition."""
    pass
```

**Use Fixtures for Test Data:**
```python
@pytest.fixture
def sample_benchmark_item():
    """Provide a sample MEQBenchItem for testing."""
    return MEQBenchItem(
        id="test_001",
        medical_content="Sample medical content for testing",
        complexity_level="basic",
        source_dataset="test"
    )

def test_evaluate_explanation_valid_input(sample_benchmark_item):
    """Test that evaluate_explanation works with valid input."""
    # Test implementation
    pass
```

**Mock External Dependencies:**
```python
@pytest.mark.parametrize("model_response,expected_score", [
    ("Good response", 0.8),
    ("Poor response", 0.3),
])
def test_llm_judge_scoring(mock_llm_api, model_response, expected_score):
    """Test LLM judge scoring with different response qualities."""
    pass
```

### Test Categories

- **Unit Tests**: Test individual functions and methods in isolation
- **Integration Tests**: Test interactions between components
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test performance characteristics
- **Regression Tests**: Test that bugs don't reoccur

### Continuous Integration

Our CI pipeline runs:
- Linting (flake8, black, isort)
- Type checking (mypy)
- Security scanning (bandit)
- Tests on Python 3.8, 3.9, 3.10, 3.11
- Coverage reporting
- Documentation building

## Documentation

### Building Documentation

We use Sphinx for documentation generation.

**Install documentation dependencies:**
```bash
pip install .[docs]
```

**Build documentation:**
```bash
cd docs
make html
```

**Live preview during development:**
```bash
sphinx-autobuild docs docs/_build/html
```

### Documentation Types

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: Step-by-step tutorials
3. **Developer Guides**: Technical implementation details
4. **Examples**: Practical usage examples

### Writing Documentation

- Use clear, concise language
- Include code examples
- Provide context and rationale
- Update documentation when changing code
- Use proper Sphinx directives and cross-references

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass:**
```bash
pytest
flake8 src/ tests/
black --check src/ tests/
isort --check-only src/ tests/
mypy src/
```

2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Update CHANGELOG.md** if applicable

### Pull Request Guidelines

**Title Format:**
- Use clear, descriptive titles
- Start with action verb (Add, Fix, Update, etc.)
- Example: "Add support for Gemini models in LLMJudge"

**Description Template:**
```markdown
## Summary
Brief description of changes

## Changes Made
- Specific change 1
- Specific change 2

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Code is documented with docstrings
- [ ] README updated if needed
- [ ] Examples updated if needed

## Checklist
- [ ] Code follows project conventions
- [ ] Tests pass locally
- [ ] No breaking changes (or properly documented)
- [ ] Commit messages are clear and descriptive
```

### Review Process

1. **Automated Checks**: CI pipeline must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Reviewers may request additional tests
4. **Documentation**: Ensure documentation is adequate
5. **Merge**: Squash merge preferred for clean history

### Merge Requirements

- ✅ All CI checks pass
- ✅ At least one approval from maintainer
- ✅ No requested changes pending
- ✅ Branch is up to date with main
- ✅ No merge conflicts

## Code of Conduct

### Our Standards

We are committed to providing a welcoming and inclusive environment for all contributors. We expect:

- **Respectful Communication**: Be kind and professional
- **Constructive Feedback**: Focus on code and ideas, not people
- **Collaborative Spirit**: Help others learn and grow
- **Scientific Rigor**: Support claims with evidence
- **Ethical Responsibility**: Consider societal impact of medical AI

### Unacceptable Behavior

- Harassment, discrimination, or offensive language
- Personal attacks or trolling
- Publishing private information without permission
- Unethical use of medical data or AI systems
- Deliberate spreading of misinformation

### Reporting Issues

Report unacceptable behavior to [contact@meq-bench.org]. All reports will be reviewed confidentially.

## Getting Help

### Resources

- **Documentation**: Check the [docs](docs/) directory
- **Examples**: See [examples](examples/) directory
- **Issues**: Search [existing issues](https://github.com/heilcheng/MEQ-Bench/issues)
- **Discussions**: Use [GitHub Discussions](https://github.com/heilcheng/MEQ-Bench/discussions)

### Contact

- **General Questions**: Create a GitHub Discussion
- **Bug Reports**: Create a GitHub Issue
- **Security Issues**: Email [security@meq-bench.org]
- **Collaboration**: Email [contact@meq-bench.org]

### Community

- **Weekly Office Hours**: [Details TBD]
- **Research Meetings**: [Details TBD]
- **Slack Channel**: [Link TBD]

---

## Development Commands Quick Reference

```bash
# Setup
pip install -e .[dev]
pre-commit install

# Code Quality
black src/ tests/ examples/
isort src/ tests/ examples/
flake8 src/ tests/ examples/
mypy src/

# Testing
pytest                              # All tests
pytest --cov=src                   # With coverage
pytest tests/unit/ -v              # Unit tests only
pytest -k "test_evaluate" -v       # Specific test pattern

# Documentation
cd docs && make html                # Build docs
sphinx-autobuild docs docs/_build/html  # Live preview

# Security
bandit -r src/                      # Security scan

# Build
python -m build                     # Build package
twine check dist/*                  # Check package
```

Thank you for contributing to MEQ-Bench! Your contributions help advance the field of medical AI evaluation and make healthcare AI more reliable and accessible.