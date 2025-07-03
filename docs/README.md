# MEQ-Bench Documentation

This directory contains comprehensive documentation for the MEQ-Bench framework.

## Documentation Structure

```
docs/
├── api/                   # API documentation
├── evaluation/           # Evaluation methodology details
├── examples/             # Usage examples and tutorials
├── paper/               # Research paper and citations
└── development/         # Development guidelines
```

## Quick Links

- [Installation Guide](installation.md)
- [API Reference](api/README.md)
- [Evaluation Framework](evaluation/README.md)
- [Contributing Guidelines](development/contributing.md)
- [Research Paper](paper/MEQ-Bench-Paper.pdf)

## Key Concepts

### Audience-Adaptive Explanations

MEQ-Bench evaluates how well models can tailor medical explanations for:

1. **Physicians** - Technical, evidence-based explanations
2. **Nurses** - Practical care implications and monitoring
3. **Patients** - Simple, empathetic, jargon-free language
4. **Caregivers** - Concrete tasks and warning signs

### Evaluation Dimensions

1. **Factual & Clinical Accuracy**
2. **Terminological Appropriateness**
3. **Explanatory Completeness**
4. **Actionability & Utility**
5. **Safety & Harmfulness**
6. **Empathy & Tone**

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run basic example: `python examples/basic_usage.py`
3. See [API documentation](api/README.md) for detailed usage

## Citation

```bibtex
@article{meq-bench-2025,
  title={MEQ-Bench: A Resource-Efficient Benchmark for Evaluating Audience-Adaptive Explanation Quality in Medical Large Language Models},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2025}
}
```