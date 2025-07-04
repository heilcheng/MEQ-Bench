"""MEQ-Bench: A Resource-Efficient Benchmark for Evaluating Medical LLM Explanation Quality.

MEQ-Bench is the first benchmark specifically designed to assess an LLM's ability to generate
audience-adaptive medical explanations for four key stakeholders: physicians, nurses, patients,
and caregivers. This package provides a comprehensive evaluation framework that combines
automated metrics with LLM-as-a-judge evaluation.

Key Features:
    - Novel evaluation framework for audience-adaptive medical explanations
    - Resource-efficient methodology using existing validated medical datasets
    - Multi-dimensional automated evaluation with LLM-as-a-judge paradigm
    - Optimized for open-weight models on consumer hardware

Typical usage example:
    ```python
    from meq_bench import MEQBench, MEQBenchEvaluator

    # Initialize benchmark
    bench = MEQBench()
    evaluator = MEQBenchEvaluator()

    # Generate and evaluate explanations
    explanations = bench.generate_explanations(medical_content, model_func)
    scores = evaluator.evaluate_all_audiences(medical_content, explanations)
    ```
"""

__version__ = "1.0.0"
__author__ = "MEQ-Bench Team"
__email__ = "contact@meq-bench.org"

# Initialize configuration and logging
from .config import config
from .benchmark import MEQBench
from .evaluator import MEQBenchEvaluator
from .prompt_templates import AudienceAdaptivePrompt
from .strategies import StrategyFactory

config.setup_logging()

__all__ = ["MEQBench", "MEQBenchEvaluator", "AudienceAdaptivePrompt", "StrategyFactory", "config"]
