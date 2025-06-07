"""Core Quantum Algorithm Components for Quantum MCMC.

This subpackage implements the quantum algorithmic primitives required
for quantum-enhanced Markov Chain Monte Carlo sampling, following the
theoretical framework of Szegedy quantum walks and amplitude amplification.

Key components:
- Szegedy quantum walk operator construction
- Quantum phase estimation (QPE) for eigenvalue extraction
- Approximate reflection operators for amplitude amplification
- Quantum state analysis and validation

These implementations are designed for both theoretical analysis and
practical quantum circuit construction using Qiskit.
"""

from quantum_mcmc.core.quantum_walk import (
    prepare_walk_operator,
    is_unitary,
    walk_eigenvalues,
    validate_walk_operator,
    prepare_initial_state,
    measure_walk_mixing,
    quantum_mixing_time,
)

from quantum_mcmc.core.phase_estimation import (
    quantum_phase_estimation,
    analyze_qpe_results,
    plot_qpe_histogram,
    estimate_required_precision,
    adaptive_qpe,
    qpe_for_quantum_walk,
)

from quantum_mcmc.core.reflection_operator import (
    approximate_reflection_operator,
    apply_reflection_operator,
    analyze_reflection_quality,
    optimize_reflection_parameters,
    plot_reflection_analysis,
)

__all__ = [
    # Quantum walk operators
    "prepare_walk_operator",
    "is_unitary",
    "walk_eigenvalues",
    "validate_walk_operator",
    "prepare_initial_state",
    "measure_walk_mixing",
    "quantum_mixing_time",
    
    # Phase estimation
    "quantum_phase_estimation",
    "analyze_qpe_results",
    "plot_qpe_histogram",
    "estimate_required_precision",
    "adaptive_qpe",
    "qpe_for_quantum_walk",
    
    # Reflection operators
    "approximate_reflection_operator",
    "apply_reflection_operator",
    "analyze_reflection_quality",
    "optimize_reflection_parameters",
    "plot_reflection_analysis",
]