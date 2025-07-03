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