MEQ-Bench Documentation
=======================

Welcome to MEQ-Bench, a resource-efficient benchmark for evaluating audience-adaptive explanation quality in medical Large Language Models.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Core Functionality:

   data_loading
   evaluation_metrics
   leaderboard

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics:

   evaluation
   examples
   contributing

Overview
--------

MEQ-Bench addresses a critical gap in medical AI evaluation by providing the first benchmark specifically designed to assess an LLM's ability to generate audience-adaptive medical explanations for four key stakeholders:

* **Physicians** - Technical, evidence-based explanations
* **Nurses** - Practical care implications and monitoring
* **Patients** - Simple, empathetic, jargon-free language  
* **Caregivers** - Concrete tasks and warning signs

Key Features
------------

* **Novel Evaluation Framework**: First benchmark to systematically evaluate audience-adaptive medical explanations
* **Comprehensive Data Loading**: Support for MedQA-USMLE, iCliniq, and Cochrane Reviews datasets
* **Advanced Safety Metrics**: Contradiction detection, information preservation, and hallucination detection
* **Automated Complexity Stratification**: Flesch-Kincaid Grade Level based content categorization
* **Interactive Leaderboards**: Beautiful, responsive HTML leaderboards for result visualization
* **Resource-Efficient Methodology**: Uses existing validated medical datasets
* **Validated Automated Evaluation**: Multi-dimensional scoring with LLM-as-a-judge paradigm
* **Democratized Access**: Optimized for open-weight models on consumer hardware

Quick Start
-----------

.. code-block:: bash

   pip install -r requirements.txt

.. code-block:: python

   from src.benchmark import MEQBench
   from src.evaluator import MEQBenchEvaluator

   # Initialize benchmark
   bench = MEQBench()
   
   # Generate audience-adaptive explanations
   explanations = bench.generate_explanations(medical_content, model)
   
   # Evaluate explanations
   evaluator = MEQBenchEvaluator()
   scores = evaluator.evaluate_all_audiences(explanations)

Architecture
------------

MEQ-Bench is built with SOLID principles and uses:

* **Strategy Pattern** for audience-specific scoring
* **Dependency Injection** for flexible component management
* **Configuration-Driven** design with YAML configuration
* **Comprehensive Logging** for debugging and monitoring

Getting Help
------------

We provide comprehensive support channels to help you successfully use MEQ-Bench. Choose the most appropriate channel based on your needs:

📚 **Documentation and Self-Help**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before reaching out, please check these resources:

* **Primary Documentation**: This comprehensive guide covers installation, usage, and advanced topics
* **API Reference**: Detailed function and class documentation with examples
* **Quickstart Guide**: :doc:`quickstart` - Get up and running in minutes  
* **Installation Guide**: :doc:`installation` - Step-by-step setup instructions
* **Evaluation Guide**: :doc:`evaluation_metrics` - Understanding MEQ-Bench metrics
* **Data Loading Guide**: :doc:`data_loading` - Working with datasets

**Code Examples and Tutorials**

* **Basic Usage**: `examples/basic_usage.py <https://github.com/heilcheng/MEQ-Bench/blob/main/examples/basic_usage.py>`_ - Simple getting started example
* **Model Integration**: Examples for OpenAI, Anthropic, Google Gemini, and MLX backends
* **Custom Datasets**: How to load and process your own medical datasets
* **Evaluation Examples**: Custom scoring and validation scenarios

**Quick Validation Commands**

.. code-block:: bash

   # Verify installation
   python -c "import src; print('✅ MEQ-Bench is working')"
   
   # Run basic test
   python run_benchmark.py --model_name dummy --max_items 2
   
   # Validate environment
   python scripts/validate_release.py

🆘 **Support Channels by Issue Type**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**🐛 Bug Reports and Technical Issues**

* **Where**: `GitHub Issues <https://github.com/heilcheng/MEQ-Bench/issues>`_
* **When**: Errors, crashes, unexpected behavior, or performance problems
* **Include**:
  
  * Clear problem description and steps to reproduce
  * Environment details (OS, Python version, package versions)
  * Complete error messages and stack traces
  * Minimal code example that demonstrates the issue

.. code-block:: bash

   # Gather environment info for bug reports
   python --version
   pip list | grep -E "(torch|transformers|openai|anthropic)"
   python -c "import platform; print(platform.platform())"

**💡 Feature Requests and Ideas**

* **Where**: `GitHub Issues <https://github.com/heilcheng/MEQ-Bench/issues>`_ (use "enhancement" label)
* **When**: You have ideas for new features, metrics, or improvements
* **Include**:
  
  * Clear use case description and benefits
  * Proposed implementation approach if known
  * Examples from other tools or research
  * Considerations for medical AI safety and ethics

**❓ Usage Questions and Best Practices**

* **Where**: `GitHub Discussions <https://github.com/heilcheng/MEQ-Bench/discussions>`_
* **When**: Questions about how to use MEQ-Bench effectively
* **Categories**:
  
  * **Q&A**: General usage and troubleshooting questions
  * **Ideas**: Feature discussions and feedback
  * **Show and Tell**: Share your research and applications
  * **Research**: Scientific methodology and validation discussions

**🔬 Research and Academic Support**

* **Where**: Email to `research@meq-bench.org <mailto:research@meq-bench.org>`_
* **When**: Academic collaborations, methodology questions, or validation studies
* **Topics**: Metric interpretation, benchmark design, evaluation methodologies

**🚨 Security and Safety Issues**

* **Where**: Email to `security@meq-bench.org <mailto:security@meq-bench.org>`_  
* **When**: Security vulnerabilities, medical safety concerns, or ethical issues
* **Note**: Please do not report security issues in public forums

🔧 **Troubleshooting Common Issues**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Installation Problems**

.. code-block:: bash

   # Clean reinstall
   pip uninstall meq-bench
   pip install --no-cache-dir -e .
   
   # Install with specific dependency groups
   pip install -e .[dev]  # Development tools
   pip install -e .[ml]   # Machine learning libraries
   pip install -e .[llm]  # LLM API clients

**Import Errors**

.. code-block:: bash

   # Ensure correct directory
   cd /path/to/MEQ-Bench
   python -c "import src"
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"

**Model Integration Issues**

.. code-block:: bash

   # Test with dummy model first
   python run_benchmark.py --model_name dummy --max_items 2
   
   # Verify API credentials
   echo $OPENAI_API_KEY
   echo $ANTHROPIC_API_KEY  
   echo $GOOGLE_API_KEY

**Performance and Memory Issues**

* Use smaller models or reduce ``max_items`` for testing
* Check available GPU memory: ``nvidia-smi`` (if using CUDA)
* Consider MLX backend for Apple Silicon: ``--model_name mlx:model_id``
* Enable logging for debugging: ``--verbose``

📞 **Response Times and Expectations**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **GitHub Issues**: 48 hours response time for bugs and urgent issues
* **GitHub Discussions**: Community-driven, responses vary by topic  
* **Email Support**: 3-5 business days for general inquiries
* **Research Inquiries**: 1 week for academic collaboration requests
* **Security Issues**: 24 hours acknowledgment, 1 week for full assessment

🎯 **Getting Effective Help**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To receive the best possible assistance:

1. **Search First**: Check existing issues, discussions, and documentation
2. **Be Specific**: Include exact error messages and reproduction steps  
3. **Provide Context**: Explain your goal and what you've already tried
4. **Share Code**: Include minimal, reproducible examples
5. **Follow Templates**: Use issue templates when available
6. **Stay Engaged**: Respond to follow-up questions promptly

🤝 **Community and Contribution**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Contributing Back**

If you receive help, consider contributing to the community:

* Answer questions in GitHub Discussions
* Improve documentation based on your experience  
* Submit bug fixes or feature enhancements
* Share usage examples and tutorials
* Participate in validation studies

**Development and Research Collaboration**

We welcome:

* **Code Contributions**: See our `Contributing Guidelines <https://github.com/heilcheng/MEQ-Bench/blob/main/CONTRIBUTING.md>`_
* **Research Partnerships**: Academic collaborations and validation studies
* **Dataset Contributions**: New medical datasets and evaluation benchmarks
* **Methodology Improvements**: Enhanced metrics and evaluation frameworks

**Community Standards**

Our community values:

* **Respectful Communication**: Professional and courteous interactions
* **Scientific Rigor**: Evidence-based discussions and peer review
* **Open Collaboration**: Sharing knowledge and helping others succeed
* **Medical Ethics**: Responsible development of medical AI systems
* **Inclusivity**: Welcoming contributors from diverse backgrounds

**Quick Reference for Contributors**

.. code-block:: bash

   # Development setup
   git clone https://github.com/YOUR_USERNAME/MEQ-Bench.git
   cd MEQ-Bench
   pip install -e .[dev]
   pre-commit install
   
   # Run tests
   pytest tests/ -v
   
   # Code quality checks  
   black src/ tests/
   flake8 src/ tests/
   mypy src/

📧 **Direct Contact Information**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **General Support**: `contact@meq-bench.org <mailto:contact@meq-bench.org>`_
* **Research Collaboration**: `research@meq-bench.org <mailto:research@meq-bench.org>`_  
* **Security Issues**: `security@meq-bench.org <mailto:security@meq-bench.org>`_
* **Media and Press**: `press@meq-bench.org <mailto:press@meq-bench.org>`_

For urgent medical AI safety concerns, include "URGENT" in your email subject line.

Citation
--------

.. code-block:: bibtex

   @article{meq-bench-2025,
     title={MEQ-Bench: A Resource-Efficient Benchmark for Evaluating Audience-Adaptive Explanation Quality in Medical Large Language Models},
     author={[Author Names]},
     journal={[Journal Name]},
     year={2025}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`