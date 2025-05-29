"""Classical Markov Chain Components for Quantum MCMC.

This subpackage provides tools for constructing and analyzing classical
reversible Markov chains, which serve as the foundation for quantum walk
operators in quantum-enhanced MCMC algorithms.

Key components:
- Markov chain construction (two-state, Metropolis-Hastings, random reversible)
- Stochasticity and reversibility verification
- Stationary distribution computation
- Discriminant matrix analysis for quantum walk construction
- Spectral gap and mixing time analysis

All chains are designed to satisfy detailed balance for compatibility
with Szegedy's quantum walk framework.
"""

from quantum_mcmc.classical.markov_chain import (
    build_two_state_chain,
    build_metropolis_chain,
    is_stochastic,
    stationary_distribution,
    is_reversible,
    sample_random_reversible_chain,
    build_imhk_chain,
)

from quantum_mcmc.classical.discriminant import (
    discriminant_matrix,
    singular_values,
    spectral_gap,
    phase_gap,
    mixing_time_bound,
    validate_discriminant,
    effective_dimension,
    condition_number,
    spectral_analysis,
)

__all__ = [
    # Markov chain construction
    "build_two_state_chain",
    "build_metropolis_chain",
    "build_imhk_chain",
    "sample_random_reversible_chain",
    
    # Markov chain properties
    "is_stochastic",
    "is_reversible",
    "stationary_distribution",
    
    # Discriminant matrix
    "discriminant_matrix",
    "singular_values",
    "spectral_gap",
    "phase_gap",
    "mixing_time_bound",
    "validate_discriminant",
    "effective_dimension",
    "condition_number",
    "spectral_analysis",
]