# MEQ-Bench: A Resource-Efficient Benchmark for Evaluating Audience-Adaptive Explanation Quality in Medical Large Language Models

## Abstract

The deployment of Large Language Models (LLMs) in healthcare requires not only medical accuracy but also the ability to communicate effectively with diverse audiences. Current benchmarks, which often focus on multiple-choice questions, fail to evaluate this critical capability and are becoming saturated. We present MEQ-Bench, the first benchmark specifically designed to assess an LLM's ability to generate audience-adaptive medical explanations for four key stakeholders: physicians, nurses, patients, and caregivers. This multi-audience framework addresses a significant gap in medical AI evaluation. Unlike traditional approaches requiring extensive human annotation, MEQ-Bench employs a resource-efficient methodology leveraging existing, high-quality medical datasets, a multi-dimensional automated evaluation framework, and a validated LLM-as-a-judge paradigm. Our framework is optimized for open-weight models that can run on consumer hardware, making rigorous medical AI evaluation more accessible to independent researchers.

## Key Features

- **Novel Evaluation Framework**: First benchmark to systematically evaluate audience-adaptive medical explanations
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
git clone https://github.com/[username]/MEQ-Bench.git
cd MEQ-Bench
pip install -r requirements.txt
```

### Basic Usage

```python
from src.benchmark import MEQBench
from src.evaluator import MEQBenchEvaluator

# Initialize benchmark
bench = MEQBench()

# Generate audience-adaptive explanations
explanations = bench.generate_explanations(medical_content, model)

# Evaluate explanations
evaluator = MEQBenchEvaluator()
scores = evaluator.evaluate_all_audiences(explanations)
```

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