"""
Setup script for MEQ-Bench
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="meq-bench",
    version="1.0.0",
    author="MEQ-Bench Team",
    author_email="contact@meq-bench.org",
    description="A Resource-Efficient Benchmark for Evaluating Audience-Adaptive Explanation Quality in Medical Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/heilcheng/MEQ-Bench",
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
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
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