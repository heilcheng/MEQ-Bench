"""
Basic usage example for MEQ-Bench
"""

import os
import sys
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.benchmark import MEQBench
from src.evaluator import MEQBenchEvaluator
from src.config import config

# Set up logging
logger = logging.getLogger('meq_bench.examples')


def dummy_model_function(prompt: str) -> str:
    """
    Dummy model function for demonstration
    In practice, this would call your actual LLM
    """
    return """
    For a Physician: The patient presents with essential hypertension, likely multifactorial etiology including genetic predisposition and lifestyle factors. Recommend ACE inhibitor initiation with monitoring of renal function and electrolytes. Consider cardiovascular risk stratification.

    For a Nurse: Monitor blood pressure readings twice daily, document trends. Educate patient on medication compliance, dietary sodium restriction, and importance of regular follow-up. Watch for signs of medication side effects.

    For a Patient: Your blood pressure is higher than normal, which means your heart is working harder than it should. We'll start you on medication to help lower it. It's important to take your medicine every day and eat less salt.

    For a Caregiver: Help ensure they take their blood pressure medication at the same time each day. Monitor for dizziness or fatigue. Encourage a low-salt diet and regular gentle exercise. Call the doctor if blood pressure readings are very high.
    """


def main():
    """Main example function"""
    print("MEQ-Bench Basic Usage Example")
    print("=" * 40)
    
    # Initialize benchmark
    bench = MEQBench()
    
    # Create sample dataset
    print("Creating sample dataset...")
    sample_items = bench.create_sample_dataset()
    
    # Add items to benchmark
    for item in sample_items:
        bench.add_benchmark_item(item)
    
    # Get benchmark statistics
    stats = bench.get_benchmark_stats()
    print(f"\nBenchmark Statistics:")
    print(f"Total items: {stats['total_items']}")
    print(f"Complexity distribution: {stats['complexity_distribution']}")
    print(f"Source distribution: {stats['source_distribution']}")
    
    # Test single explanation generation
    print("\nGenerating explanations for sample content...")
    medical_content = "Diabetes is a condition where blood sugar levels are too high. It requires careful management through diet, exercise, and sometimes medication."
    
    explanations = bench.generate_explanations(medical_content, dummy_model_function)
    
    print("\nGenerated Explanations:")
    for audience, explanation in explanations.items():
        print(f"\n{audience.upper()}:")
        print(explanation[:200] + "..." if len(explanation) > 200 else explanation)
    
    # Evaluate explanations
    print("\nEvaluating explanations...")
    evaluator = MEQBenchEvaluator()
    results = evaluator.evaluate_all_audiences(medical_content, explanations)
    
    print("\nEvaluation Results:")
    for audience, score in results.items():
        print(f"\n{audience.upper()}:")
        print(f"  Readability: {score.readability:.3f}")
        print(f"  Terminology: {score.terminology:.3f}")
        print(f"  Safety: {score.safety:.3f}")
        print(f"  Coverage: {score.coverage:.3f}")
        print(f"  Quality: {score.quality:.3f}")
        print(f"  Overall: {score.overall:.3f}")
    
    # Run full benchmark evaluation (on sample data)
    print("\nRunning full benchmark evaluation...")
    full_results = bench.evaluate_model(dummy_model_function, max_items=2)
    
    print("\nFull Benchmark Results Summary:")
    summary = full_results['summary']
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.3f}")
    
    # Save results
    output_path = "sample_results.json"
    bench.save_results(full_results, output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()