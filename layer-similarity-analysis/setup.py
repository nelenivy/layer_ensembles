"""Setup script for layer-similarity-analysis."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Layer Similarity Analysis Framework"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'datasets>=2.14.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'mteb>=1.0.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
    ]

setup(
    name="layer-similarity-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Framework for analyzing layer-wise representational similarity in transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/layer-similarity-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'layer-sim-pipeline=scripts.run_pipeline:main',
            'layer-sim-grid=scripts.run_grid_search:main',
            'layer-sim-batch=scripts.run_batch_experiments:main',
        ],
    },
)
