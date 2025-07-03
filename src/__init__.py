"""
MEQ-Bench: A Resource-Efficient Benchmark for Evaluating 
Audience-Adaptive Explanation Quality in Medical Large Language Models
"""

__version__ = "1.0.0"
__author__ = "MEQ-Bench Team"
__email__ = "contact@meq-bench.org"

from .benchmark import MEQBench
from .evaluator import MEQBenchEvaluator
from .prompt_templates import AudienceAdaptivePrompt

__all__ = ["MEQBench", "MEQBenchEvaluator", "AudienceAdaptivePrompt"]