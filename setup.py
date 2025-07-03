"""
Setup script for MEQ-Bench
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core production dependencies
requirements = [
    "numpy>=1.21.0",
    "pandas>=1.5.0", 
    "PyYAML>=6.0.0",
    "requests>=2.28.0",
    "textstat>=0.7.0",
]

setup(
    name="meq-bench",
    version="1.0.0",
    author="MEQ-Bench Team",
    author_email="contact@meq-bench.org",
    description="A Resource-Efficient Benchmark for Evaluating Audience-Adaptive Explanation Quality in Medical Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/MEQ-Bench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
            "bandit>=1.7.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "datasets>=2.10.0",
            "scikit-learn>=1.3.0",
            "sentence-transformers>=2.2.0",
            "bert-score>=0.3.13",
            "spacy>=3.6.0",
            "scispacy>=0.5.0",
            "nltk>=3.8.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.3.0",
        ],
        "apple": [
            "mlx>=0.5.0",
            "mlx-lm>=0.5.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=0.18.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "sphinx-autobuild>=2021.3.14",
        ],
        "full": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "datasets>=2.10.0",
            "scikit-learn>=1.3.0",
            "sentence-transformers>=2.2.0",
            "bert-score>=0.3.13",
            "spacy>=3.6.0",
            "scispacy>=0.5.0",
            "nltk>=3.8.0",
            "openai>=1.0.0",
            "anthropic>=0.3.0",
            "mlx>=0.5.0",
            "mlx-lm>=0.5.0",
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "tqdm>=4.65.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "meq-bench=src.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "meq-bench": ["data/*.json", "docs/*.md"],
    },
)