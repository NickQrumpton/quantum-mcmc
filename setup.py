"""Setup configuration for quantum-mcmc package.

Research-grade quantum Markov Chain Monte Carlo implementation
for reproducible scientific computing.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Package metadata
NAME = "quantum-mcmc"
VERSION = "0.1.0"
AUTHOR = "Nicholas Zhao"
AUTHOR_EMAIL = "nz422@ic.ac.uk"
DESCRIPTION = "Quantum Markov Chain Monte Carlo (MCMC) simulation framework for research"
URL = "https://github.com/nicholaszhao/quantum-mcmc"  # Replace with actual URL

# Long description from README
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    long_description = DESCRIPTION

# Python version requirement
PYTHON_REQUIRES = ">=3.9"

# Core dependencies with version constraints for reproducibility
INSTALL_REQUIRES = [
    "numpy>=1.21.0,<2.0.0",
    "scipy>=1.7.0,<2.0.0",
    "pandas>=1.3.0,<3.0.0",
    "matplotlib>=3.4.0,<4.0.0",
    "seaborn>=0.11.0,<1.0.0",
    "qiskit>=0.45.0,<1.0.0",
    # "qiskit-aer>=0.13.0,<1.0.0",  # Optional for now due to compilation issues
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0,<8.0.0",
        "pytest-cov>=4.0.0,<5.0.0",
        "pytest-xdist>=3.0.0,<4.0.0",
        "black>=22.0.0,<24.0.0",
        "flake8>=5.0.0,<7.0.0",
        "mypy>=1.0.0,<2.0.0",
        "isort>=5.10.0,<6.0.0",
        "pre-commit>=3.0.0,<4.0.0",
        "sphinx>=5.0.0,<8.0.0",
        "sphinx-rtd-theme>=1.0.0,<2.0.0",
        "nbsphinx>=0.8.0,<1.0.0",
    ],
    "notebooks": [
        "jupyter>=1.0.0,<2.0.0",
        "jupyterlab>=3.0.0,<5.0.0",
        "notebook>=6.4.0,<8.0.0",
        "ipywidgets>=8.0.0,<9.0.0",
        "tqdm>=4.65.0,<5.0.0",
    ],
    "viz": [
        "networkx>=2.8.0,<4.0.0",
        "plotly>=5.0.0,<6.0.0",
        "graphviz>=0.20.0,<1.0.0",
    ],
    "all": [],  # Will be populated below
}

# Combine all optional dependencies
EXTRAS_REQUIRE["all"] = list(set(
    dep for deps in EXTRAS_REQUIRE.values() if isinstance(deps, list)
    for dep in deps
))

# Package classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Typing :: Typed",
]

# Keywords for discoverability
KEYWORDS = [
    "quantum computing",
    "quantum algorithms",
    "mcmc",
    "markov chain monte carlo",
    "quantum walks",
    "phase estimation",
    "sampling",
    "simulation",
    "qiskit",
    "research",
    "reproducible science",
]

# Entry points (if any command-line tools are provided)
ENTRY_POINTS = {
    "console_scripts": [
        # Example: "quantum-mcmc=quantum_mcmc.cli:main",
    ],
}

# Package data to include
PACKAGE_DATA = {
    "quantum_mcmc": [
        "py.typed",  # PEP 561 type information
        "data/*.json",  # Any data files
        "data/*.yaml",
    ],
}

# Additional files to include in source distribution
MANIFEST_IN = """
include README.md
include LICENSE
include CHANGELOG.md
include requirements*.txt
recursive-include docs *.md *.rst *.txt
recursive-include examples *.py *.ipynb
recursive-include tests *.py
recursive-exclude * __pycache__
recursive-exclude * *.py[co]
"""

# Write MANIFEST.in if it doesn't exist
manifest_path = Path(__file__).parent / "MANIFEST.in"
if not manifest_path.exists():
    manifest_path.write_text(MANIFEST_IN.strip())

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={
        "Bug Tracker": f"{URL}/issues",
        "Documentation": f"{URL}#documentation",
        "Source Code": URL,
        "Citation": f"{URL}#citation",
    },
    license="MIT",
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data=PACKAGE_DATA,
    include_package_data=True,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points=ENTRY_POINTS,
    zip_safe=False,  # Required for mypy to find the py.typed file
    platforms=["any"],
)

# Development setup check
if __name__ == "__main__":
    # Verify Python version
    if sys.version_info < (3, 9):
        print(f"ERROR: Python {sys.version_info.major}.{sys.version_info.minor} is not supported.")
        print("Please use Python 3.9 or higher.")
        sys.exit(1)
    
    # Check if running in development mode
    if any(arg in sys.argv for arg in ["develop", "dev", "editable"]):
        print("\nSetting up quantum-mcmc in development mode...")
        print("Consider installing development dependencies:")
        print("  pip install -e '.[dev]'")
        print("\nFor notebook support:")
        print("  pip install -e '.[notebooks]'")
        print("\nFor all features:")
        print("  pip install -e '.[all]'")