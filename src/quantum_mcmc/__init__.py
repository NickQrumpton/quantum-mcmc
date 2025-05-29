"""Quantum Markov Chain Monte Carlo (MCMC) Simulation Framework.

A research-grade implementation of quantum-enhanced MCMC sampling algorithms,
providing tools for classical Markov chain analysis, Szegedy quantum walk
construction, quantum phase estimation, reflection operators, and comprehensive
benchmarking and visualization capabilities.

This package implements state-of-the-art quantum algorithms for accelerated
sampling from complex probability distributions, with applications in statistical
physics, machine learning, and optimization.

Author: Nicholas Zhao  
Affiliation: Imperial College London  
Contact: nz422@ic.ac.uk
Version: 0.1.0

Modules:
    classical: Classical Markov chain construction and analysis
    core: Quantum algorithm implementations (walks, QPE, reflection)
    utils: Utilities for state preparation, analysis, and visualization

Example:
    >>> from quantum_mcmc import build_two_state_chain, prepare_walk_operator
    >>> P = build_two_state_chain(0.3, 0.7)
    >>> W = prepare_walk_operator(P)
"""

__version__ = "0.1.0"
__author__ = "Nicholas Zhao"
__affiliation__ = "Imperial College London"
__email__ = "nz422@ic.ac.uk"

# Classical Markov chain components
from quantum_mcmc.classical import (
    build_two_state_chain,
    build_metropolis_chain,
    is_stochastic,
    stationary_distribution,
    is_reversible,
    sample_random_reversible_chain,
    discriminant_matrix,
    singular_values,
    spectral_gap,
    phase_gap,
)

# Core quantum components
from quantum_mcmc.core import (
    prepare_walk_operator,
    walk_eigenvalues,
    quantum_phase_estimation,
    analyze_qpe_results,
    approximate_reflection_operator,
    apply_reflection_operator,
)

# Utility functions
from quantum_mcmc.utils import (
    prepare_stationary_state,
    prepare_basis_state,
    prepare_uniform_superposition,
    compute_overlap,
    total_variation_distance,
    effective_sample_size,
    plot_phase_histogram,
    plot_singular_values,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__affiliation__",
    
    # Classical Markov chains
    "build_two_state_chain",
    "build_metropolis_chain",
    "is_stochastic",
    "stationary_distribution",
    "is_reversible",
    "sample_random_reversible_chain",
    
    # Discriminant matrix analysis
    "discriminant_matrix",
    "singular_values",
    "spectral_gap",
    "phase_gap",
    
    # Quantum walk operators
    "prepare_walk_operator",
    "walk_eigenvalues",
    
    # Quantum phase estimation
    "quantum_phase_estimation",
    "analyze_qpe_results",
    
    # Reflection operators
    "approximate_reflection_operator",
    "apply_reflection_operator",
    
    # State preparation
    "prepare_stationary_state",
    "prepare_basis_state",
    "prepare_uniform_superposition",
    
    # Analysis utilities
    "compute_overlap",
    "total_variation_distance",
    "effective_sample_size",
    
    # Visualization
    "plot_phase_histogram",
    "plot_singular_values",
]