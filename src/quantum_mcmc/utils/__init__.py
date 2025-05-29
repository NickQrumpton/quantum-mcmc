"""Utilities for Quantum MCMC Implementation.

This subpackage provides supporting tools for quantum MCMC algorithms,
including quantum state preparation, performance analysis, visualization,
and diagnostics for both classical and quantum sampling algorithms.

Key components:
- Quantum state preparation (stationary, basis, superposition states)
- Performance analysis and convergence diagnostics
- Publication-quality visualization tools
- Sampling quality metrics and comparisons

These utilities are essential for practical implementation, debugging,
and research analysis of quantum MCMC algorithms.
"""

from quantum_mcmc.utils.state_preparation import (
    prepare_stationary_state,
    prepare_basis_state,
    prepare_uniform_superposition,
    prepare_gaussian_state,
    prepare_binomial_state,
    prepare_thermal_state,
    prepare_w_state,
    prepare_dicke_state,
    prepare_edge_superposition,
    validate_state_preparation,
    optimize_state_preparation,
)

from quantum_mcmc.utils.analysis import (
    compute_spectral_gap,
    total_variation_distance,
    compute_overlap,
    fidelity,
    trace_distance,
    mixing_time,
    effective_sample_size,
    quantum_process_fidelity,
    phase_estimation_accuracy,
    convergence_diagnostics,
    quantum_speedup_estimate,
    plot_convergence_diagnostics,
    entropy_analysis,
    sample_quality_metrics,
)

from quantum_mcmc.utils.visualization import (
    plot_phase_histogram,
    plot_singular_values,
    plot_total_variation,
    plot_eigenvalue_spectrum,
    plot_state_fidelity,
    plot_markov_chain_graph,
    create_multi_panel_figure,
    save_figure,
)

__all__ = [
    # State preparation
    "prepare_stationary_state",
    "prepare_basis_state",
    "prepare_uniform_superposition",
    "prepare_gaussian_state",
    "prepare_binomial_state",
    "prepare_thermal_state",
    "prepare_w_state",
    "prepare_dicke_state",
    "prepare_edge_superposition",
    "validate_state_preparation",
    "optimize_state_preparation",
    
    # Analysis functions
    "compute_spectral_gap",
    "total_variation_distance",
    "compute_overlap",
    "fidelity",
    "trace_distance",
    "mixing_time",
    "effective_sample_size",
    "quantum_process_fidelity",
    "phase_estimation_accuracy",
    "convergence_diagnostics",
    "quantum_speedup_estimate",
    "plot_convergence_diagnostics",
    "entropy_analysis",
    "sample_quality_metrics",
    
    # Visualization
    "plot_phase_histogram",
    "plot_singular_values",
    "plot_total_variation",
    "plot_eigenvalue_spectrum",
    "plot_state_fidelity",
    "plot_markov_chain_graph",
    "create_multi_panel_figure",
    "save_figure",
]