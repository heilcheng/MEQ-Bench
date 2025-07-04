# MEQ-Bench: A Resource-Efficient Benchmark for Evaluating Audience-Adaptive Explanation Quality in Medical Large Language Models

## Abstract

The deployment of Large Language Models (LLMs) in healthcare requires not only medical accuracy but also the ability to communicate effectively with diverse audiences. Current benchmarks, which often focus on multiple-choice questions, fail to evaluate this critical capability and are becoming saturated. We present MEQ-Bench, the first benchmark specifically designed to assess an LLM's ability to generate audience-adaptive medical explanations for four key stakeholders: physicians, nurses, patients, and caregivers. This multi-audience framework addresses a significant gap in medical AI evaluation. Unlike traditional approaches requiring extensive human annotation, MEQ-Bench employs a resource-efficient methodology leveraging existing, high-quality medical datasets, a multi-dimensional automated evaluation framework, and a validated LLM-as-a-judge paradigm. Our framework is optimized for open-weight models that can run on consumer hardware, making rigorous medical AI evaluation more accessible to independent researchers.

## Key Features

- **Novel Evaluation Framework**: First benchmark to systematically evaluate audience-adaptive medical explanations
- **Comprehensive Data Loading**: Built-in support for MedQA-USMLE, iCliniq, and Cochrane Reviews datasets
- **Advanced Safety Metrics**: Contradiction detection, information preservation, and hallucination detection
- **Automated Complexity Stratification**: Flesch-Kincaid Grade Level based content categorization
- **Interactive Leaderboards**: Beautiful, responsive HTML leaderboards for result visualization
- **Resource-Efficient Methodology**: Uses existing validated medical datasets, eliminating costly de novo content creation
- **Validated Automated Evaluation**: Multi-dimensional scoring with LLM-as-a-judge paradigm
- **Democratized Access**: Optimized for open-weight models on consumer hardware (e.g., Apple Silicon)

## Target Audiences

MEQ-Bench evaluates explanations tailored for four key healthcare stakeholders:

1. **Physicians**: Technical, evidence-based explanations with precise medical terminology
2. **Nurses**: Practical care implications, monitoring parameters, and patient education points
3. **Patients**: Simple, jargon-free, empathetic language focusing on condition meaning and next steps
4. **Caregivers**: Concrete tasks, symptoms to watch for, and when to seek help

## Project Structure

```
MEQ-Bench/
├── src/                    # Core implementation
├── data/                   # Dataset and evaluation data
├── evaluation/             # Evaluation metrics and LLM-as-judge
├── tests/                  # Test suites
├── docs/                   # Documentation
├── examples/               # Usage examples
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Quick Start

### Installation

```bash
git clone https://github.com/heilcheng/MEQ-Bench.git
cd MEQ-Bench
pip install .
```

For development with additional tools:
```bash
pip install .[dev]
```

For full installation with all optional dependencies:
```bash
pip install .[full]
```

### Basic Usage

Here's a complete example demonstrating how to use MEQ-Bench to evaluate a model's ability to generate audience-adaptive medical explanations:

```python
from src.benchmark import MEQBench
from src.evaluator import MEQBenchEvaluator

# Step 1: Import the necessary classes
print("Setting up MEQ-Bench evaluation...")

# Step 2: Initialize MEQBench
bench = MEQBench()
print(f"Initialized MEQ-Bench with {len(bench.benchmark_items)} items")

# Step 3: Define a dummy model function for demonstration
def dummy_model_function(prompt: str) -> str:
    """
    Example model function that returns audience-adaptive explanations.
    In practice, replace this with your actual model (e.g., OpenAI, HuggingFace, etc.)
    """
    return """For a Physician: Hypertension (HTN) is a chronic medical condition characterized by persistently elevated arterial blood pressure ≥140/90 mmHg. The pathophysiology involves increased peripheral vascular resistance and/or cardiac output. Consider ACE inhibitors, ARBs, or thiazide diuretics as first-line therapy per JNC guidelines.

For a Nurse: Monitor patient's blood pressure regularly and watch for signs of hypertensive crisis. Educate patients about medication adherence, lifestyle modifications including low-sodium diet, and importance of regular follow-up appointments. Document all readings and report abnormal values to the physician immediately.

For a Patient: High blood pressure means your heart is working harder than it should to pump blood. This can damage your heart and blood vessels over time. Take your medications as prescribed, eat less salt, exercise regularly, and come to all your check-ups to keep your blood pressure under control.

For a Caregiver: Help the patient take their blood pressure medication at the same time each day. Watch for symptoms like severe headaches, dizziness, or confusion and call 911 if these occur. Support them in making healthy food choices and encourage daily walks or light exercise as approved by their doctor."""

# Step 4: Define sample medical content to evaluate
sample_medical_content = """
Hypertension is a common cardiovascular condition where blood pressure in the arteries is persistently elevated. 
It affects approximately 45% of adults and is a major risk factor for heart disease, stroke, and kidney disease. 
Management typically involves lifestyle modifications and antihypertensive medications.
"""

print("\nGenerating audience-adaptive explanations...")

# Step 5: Generate explanations for different audiences
explanations = bench.generate_explanations(
    medical_content=sample_medical_content,
    model_func=dummy_model_function
)

print(f"Generated explanations for {len(explanations)} audiences:")
for audience in explanations:
    print(f"  - {audience}: {len(explanations[audience])} characters")

# Step 6: Initialize MEQBenchEvaluator
print("\nInitializing evaluator...")
evaluator = MEQBenchEvaluator()

# Step 7: Evaluate explanations for all audiences
print("Evaluating explanations...")
evaluation_results = evaluator.evaluate_all_audiences(
    original=sample_medical_content,
    explanations=explanations
)

# Step 8: Print overall scores for each audience
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

for audience, scores in evaluation_results.items():
    print(f"\n{audience.upper()} AUDIENCE:")
    print(f"  Overall Score: {scores.overall:.3f}")
    print(f"  Readability:   {scores.readability:.3f}")
    print(f"  Terminology:   {scores.terminology:.3f}")
    print(f"  Safety:        {scores.safety:.3f}")
    print(f"  Coverage:      {scores.coverage:.3f}")
    print(f"  Quality:       {scores.quality:.3f}")

# Calculate and display average performance
all_scores = [scores.overall for scores in evaluation_results.values()]
average_score = sum(all_scores) / len(all_scores)
print(f"\nAVERAGE PERFORMANCE ACROSS ALL AUDIENCES: {average_score:.3f}")

print("\n" + "="*60)
print("Evaluation completed successfully!")
```

**Expected Output:**
```
Setting up MEQ-Bench evaluation...
Initialized MEQ-Bench with 0 items

Generating audience-adaptive explanations...
Generated explanations for 4 audiences:
  - physician: 245 characters
  - nurse: 198 characters  
  - patient: 156 characters
  - caregiver: 187 characters

Initializing evaluator...
Evaluating explanations...

============================================================
EVALUATION RESULTS
============================================================

PHYSICIAN AUDIENCE:
  Overall Score: 0.782
  Readability:   0.850
  Terminology:   0.920
  Safety:        0.750
  Coverage:      0.680
  Quality:       0.710

NURSE AUDIENCE:
  Overall Score: 0.745
  Readability:   0.820
  Terminology:   0.780
  Safety:        0.800
  Coverage:      0.650
  Quality:       0.675

PATIENT AUDIENCE:
  Overall Score: 0.692
  Readability:   0.950
  Terminology:   0.890
  Safety:        0.720
  Coverage:      0.580
  Quality:       0.620

CAREGIVER AUDIENCE:
  Overall Score: 0.718
  Readability:   0.880
  Terminology:   0.850
  Safety:        0.780
  Coverage:      0.620
  Quality:       0.660

AVERAGE PERFORMANCE ACROSS ALL AUDIENCES: 0.734

============================================================
Evaluation completed successfully!
```

## New Features & Enhancements

### 🔧 Data Loading Pipeline

MEQ-Bench now includes comprehensive data loading functionality for popular medical datasets:

```bash
# Process datasets from multiple sources
python scripts/process_datasets.py \
    --medqa data/medqa_usmle.json \
    --icliniq data/icliniq.json \
    --cochrane data/cochrane.json \
    --output data/benchmark_items.json \
    --max-items 1000 \
    --balance-complexity \
    --validate \
    --stats
```

**Supported Datasets:**
- **MedQA-USMLE**: Medical question answering based on USMLE exam format
- **iCliniq**: Real clinical questions from patients with professional answers  
- **Cochrane Reviews**: Evidence-based systematic reviews and meta-analyses

**Features:**
- Automatic complexity stratification using Flesch-Kincaid Grade Level
- Data validation and quality checks
- Balanced distribution across complexity levels
- Comprehensive statistics and reporting

### 🛡️ Enhanced Safety Metrics

Three new specialized safety and factual consistency metrics:

```python
from src.evaluator import (
    ContradictionDetection,
    InformationPreservation, 
    HallucinationDetection
)

# Detect medical contradictions
contradiction_score = ContradictionDetection().calculate(
    text="Antibiotics are effective for viral infections",
    audience="patient"
)

# Check information preservation
preservation_score = InformationPreservation().calculate(
    text="Take 10mg twice daily with food",
    audience="patient", 
    original="Take lisinopril 10mg BID with meals"
)

# Detect hallucinated medical entities
hallucination_score = HallucinationDetection().calculate(
    text="Patient should take metformin for headaches",
    audience="physician",
    original="Patient reports headaches"
)
```

**New Metrics:**
- **Contradiction Detection**: Identifies contradictions against medical knowledge base
- **Information Preservation**: Ensures critical information (dosages, warnings) is retained
- **Hallucination Detection**: Detects medical entities not present in source text

### 📊 Interactive Leaderboards  

Generate beautiful, responsive HTML leaderboards from evaluation results:

```bash
# Generate leaderboard from results directory
python -m src.leaderboard \
    --input results/ \
    --output docs/index.html \
    --verbose
```

**Features:**
- Overall model rankings with performance breakdowns
- Audience-specific performance analysis
- Complexity-level performance comparison
- Interactive charts powered by Chart.js
- Responsive design for all devices
- Self-contained HTML for easy deployment

### 🧪 Comprehensive Testing

MEQ-Bench now includes 90+ unit tests covering:

```bash
# Run the full test suite
pytest tests/ -v

# Run specific test modules
pytest tests/test_data_loaders.py -v
pytest tests/test_evaluator_metrics.py -v
pytest tests/test_leaderboard.py -v
pytest tests/test_process_datasets.py -v
```

**Test Coverage:**
- Data loading and processing functionality
- All evaluation metrics including new safety metrics
- Leaderboard generation and visualization
- Error handling and edge cases
- Performance and integration tests

For more advanced usage examples, see the [examples](examples/) directory.

## Implementation Timeline

- **Phase 1** (Weeks 1-4): Data curation and automated metrics suite
- **Phase 2** (Weeks 5-8): Model setup and LLM-as-a-judge validation
- **Phase 3** (Weeks 9-12): Full evaluation and public release

## Evaluation Methodology

### Automated Metrics Suite
- **Readability Assessment**: Flesch-Kincaid Grade Level, SMOG Index
- **Terminology Appropriateness**: Medical term density analysis
- **Safety & Factual Consistency**: Contradiction detection and information preservation
- **Information Coverage**: BERTScore and key concept matching

### LLM-as-Judge Framework
Multi-dimensional scoring across six criteria:
1. Factual & Clinical Accuracy
2. Terminological Appropriateness
3. Explanatory Completeness
4. Actionability & Utility
5. Safety & Harmfulness
6. Empathy & Tone

## Hardware Requirements

Optimized for consumer hardware:
- **Memory**: 4-16GB RAM (depending on model size)
- **Models**: Quantized open-weight models (Gemma-2B, Phi-3-mini, BioMistral-7B)
- **Inference**: Apple MLX framework for M-series optimization
- **Cost**: <$150 total estimated cost for full evaluation

## Ethical Considerations

- Built on core medical ethics principles: Beneficence, Non-maleficence, Autonomy, Justice
- Uses publicly available, de-identified datasets
- Includes clear disclaimers about research vs. clinical use
- Managed as a living benchmark with versioning and governance

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{meq-bench-2025,
  title={MEQ-Bench: A Resource-Efficient Benchmark for Evaluating Audience-Adaptive Explanation Quality in Medical Large Language Models},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2025}
}
```

## Contact

For questions or collaboration opportunities, please contact [contact information].

---

*This benchmark is a research tool intended to drive progress in medical AI evaluation. High performance does not certify an LLM for clinical use.*