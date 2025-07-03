MEQ-Bench Documentation
=======================

Welcome to MEQ-Bench, a resource-efficient benchmark for evaluating audience-adaptive explanation quality in medical Large Language Models.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
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

If you need assistance with MEQ-Bench, here are the best ways to get help:

**Check Existing Resources**

* **Documentation**: Start by reviewing this documentation for answers to common questions and detailed usage examples.
* **GitHub Issues**: Search our `GitHub Issues page <https://github.com/heilcheng/MEQ-Bench/issues>`_ to see if your question has already been addressed.
* **Examples**: Check the `examples/ <https://github.com/heilcheng/MEQ-Bench/tree/main/examples>`_ directory for practical usage patterns.

**Report Issues**

If you encounter a bug or have a feature request:

1. **Search existing issues** first to avoid duplicates
2. **Open a new issue** with:
   
   * Clear description of the problem or request
   * Steps to reproduce (for bugs)
   * Your environment details (OS, Python version, etc.)
   * Relevant code snippets or error messages

**Ask Questions**

For general questions about usage, best practices, or research applications:

* **GitHub Discussions**: Use our `GitHub Discussions <https://github.com/heilcheng/MEQ-Bench/discussions>`_ for community support
* **Email Support**: For direct assistance, contact our development team at: contact@meq-bench.org

**Contribute**

We welcome contributions! See our `Contributing Guidelines <https://github.com/heilcheng/MEQ-Bench/blob/main/CONTRIBUTING.md>`_ for:

* Development setup instructions
* Coding standards and best practices
* Pull request process
* How to add new features or datasets

**Community Guidelines**

When seeking help:

* Be respectful and professional
* Provide sufficient context and details
* Include code examples when relevant
* Follow up if issues are resolved
* Help others when you can

For urgent issues related to medical AI safety or security concerns, please email us directly at contact@meq-bench.org with "URGENT" in the subject line.

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