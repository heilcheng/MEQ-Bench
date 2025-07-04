"""
Pytest configuration and fixtures for MEQ-Bench tests
"""

import pytest
import os
import sys
import logging
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock environment variables for testing
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["ANTHROPIC_API_KEY"] = "test-key"

from src.benchmark import MEQBench, MEQBenchItem
from src.evaluator import MEQBenchEvaluator
from src.config import config

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def sample_medical_content():
    """Sample medical content for testing"""
    return "Hypertension is a condition where blood pressure is consistently elevated above normal levels."


@pytest.fixture
def sample_benchmark_item():
    """Sample benchmark item for testing"""
    return MEQBenchItem(
        id="test_001",
        medical_content="Diabetes is a metabolic disorder characterized by high blood sugar levels.",
        complexity_level="basic",
        source_dataset="test",
    )


@pytest.fixture
def sample_explanations():
    """Sample audience-adaptive explanations for testing"""
    return {
        "physician": "Essential hypertension with systolic BP >140 mmHg or diastolic >90 mmHg, requiring antihypertensive therapy and cardiovascular risk stratification.",
        "nurse": "Patient has high blood pressure requiring medication monitoring, lifestyle education, and regular BP checks. Watch for medication side effects.",
        "patient": "You have high blood pressure, which means your heart is working harder than it should. We'll give you medicine to help lower it.",
        "caregiver": "Their blood pressure is too high. Make sure they take their medicine daily and watch for dizziness or headaches.",
    }


@pytest.fixture
def benchmark_instance():
    """MEQBench instance for testing"""
    return MEQBench()


@pytest.fixture
def evaluator_instance():
    """MEQBenchEvaluator instance for testing"""
    return MEQBenchEvaluator()


@pytest.fixture
def dummy_model_function():
    """Dummy model function for testing"""

    def model_func(prompt):
        return """
        For a Physician: Technical medical explanation with proper terminology.
        For a Nurse: Practical care-focused explanation with monitoring points.
        For a Patient: Simple, clear explanation without medical jargon.
        For a Caregiver: Concrete instructions and warning signs to watch for.
        """

    return model_func


@pytest.fixture
def test_data_dir():
    """Test data directory path"""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs"""
    return tmp_path / "test_outputs"
