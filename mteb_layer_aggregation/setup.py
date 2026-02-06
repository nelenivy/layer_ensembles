"""Setup script for MTEB Layer Aggregation package."""
from setuptools import setup, find_packages

setup(
    name="mteb-layer-aggregation",
    version="1.0.0",
    description="Layer aggregation experiments for NLP models on MTEB benchmark",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "mteb>=1.0.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "cvxpy>=1.3.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
)
