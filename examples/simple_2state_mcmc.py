#!/usr/bin/env python3
"""Simple 2-State Quantum MCMC Demonstration.

This example demonstrates the complete quantum MCMC sampling pipeline
on a minimal two-state reversible Markov chain. The script illustrates:

1. Classical Markov chain construction and analysis
2. Quantum walk operator construction using Szegedy's framework
3. Quantum state preparation for stationary and computational basis states
4. Quantum phase estimation to extract eigenphases
5. Approximate reflection operator construction and application
6. Analysis of spectral properties and sampling performance

This serves as a pedagogical example and validation of the quantum MCMC
framework on the simplest non-trivial case.

References:
    Szegedy, M. (2004). Quantum speed-up of Markov chain based algorithms.
    FOCS 2004: 32-41.
    
    Lemieux, J., et al. (2019). Efficient quantum walk circuits for 
    Metropolis-Hastings algorithm. Quantum, 4, 287.

Author: [Your Name]
Date: 2025-01-23
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

# Classical Markov chain components
from quantum_mcmc.classical.markov_chain import (
    build_two_state_chain,
    stationary_distribution,
    is_reversible,
    is_stochastic
)
from quantum_mcmc.classical.discriminant import (
    discriminant_matrix,
    phase_gap,
    spectral_gap
)

# Quantum core components
from quantum_mcmc.core.quantum_walk import (
    prepare_walk_operator,
    walk_eigenvalues,
    is_unitary,
    validate_walk_operator
)
from quantum_mcmc.core.phase_estimation import (
    quantum_phase_estimation,
    analyze_qpe_results,
    plot_qpe_histogram
)
from quantum_mcmc.core.reflection_operator import (
    approximate_reflection_operator,
    apply_reflection_operator,
    analyze_reflection_quality
)

# Utility components
from quantum_mcmc.utils.state_preparation import (
    prepare_stationary_state,
    prepare_computational_basis_state
)
from quantum_mcmc.utils.analysis import (
    total_variation_distance,
    mixing_time_bound
)


def main():
    """Run the complete 2-state quantum MCMC demonstration."""
    
    print("=" * 60)
    print("QUANTUM MCMC SAMPLING: 2-STATE CHAIN DEMONSTRATION")
    print("=" * 60)
    print()
    
    # ================================================================
    # STEP 1: Build Classical Markov Chain
    # ================================================================
    print("Step 1: Building Classical Markov Chain")
    print("-" * 40)
    
    # Create asymmetric but reversible 2-state chain
    # P[0,1] = ± (probability 0 í 1)
    # P[1,0] = ≤ (probability 1 í 0)
    alpha, beta = 0.3, 0.7
    
    print(f"Transition probabilities: ± = {alpha}, ≤ = {beta}")
    
    # Build transition matrix
    P = build_two_state_chain(alpha, beta)
    print(f"Transition matrix P =")
    print(P)
    
    # Compute and validate stationary distribution
    pi = stationary_distribution(P)
    print(f"Stationary distribution ¿ = {pi}")
    
    # Validate classical properties
    assert is_stochastic(P), "Transition matrix must be stochastic"
    assert is_reversible(P, pi), "Chain must be reversible"
    print(" Chain is stochastic and reversible")
    
    # Compute classical spectral properties
    D = discriminant_matrix(P, pi)
    classical_gap = spectral_gap(P)
    phase_gap_value = phase_gap(D)
    
    print(f"Classical spectral gap: {classical_gap:.4f}")
    print(f"Phase gap: {phase_gap_value:.4f}")
    print()
    
    # ================================================================
    # STEP 2: Construct Quantum Walk Operator
    # ================================================================
    print("Step 2: Constructing Szegedy Quantum Walk Operator")
    print("-" * 50)
    
    # Build quantum walk operator in both representations
    W_circuit = prepare_walk_operator(P, pi=pi, backend="qiskit")
    W_matrix = prepare_walk_operator(P, pi=pi, backend="matrix")
    
    print(f"Walk operator dimensions: {W_matrix.shape}")
    print(f"Walk operator circuit depth: {W_circuit.depth()}")
    print(f"Walk operator gate count: {W_circuit.size()}")
    
    # Validate quantum walk properties
    assert is_unitary(W_matrix), "Walk operator must be unitary"
    assert validate_walk_operator(W_circuit, P, pi), "Walk operator validation failed"
    print(" Walk operator is unitary and valid")
    
    # Compute theoretical eigenvalues
    walk_eigenvals = walk_eigenvalues(P, pi)
    print(f"Theoretical eigenvalues: {walk_eigenvals}")
    
    # Check eigenvalue magnitudes (should all be 1 for unitary)
    eigenval_magnitudes = np.abs(walk_eigenvals)
    print(f"Eigenvalue magnitudes: {eigenval_magnitudes}")
    assert np.allclose(eigenval_magnitudes, 1.0), "All eigenvalues should have magnitude 1"
    print(" All eigenvalues have unit magnitude")
    print()
    
    # ================================================================
    # STEP 3: Prepare Quantum States
    # ================================================================
    print("Step 3: Preparing Quantum States")
    print("-" * 35)
    
    # Prepare stationary state approximation
    print("Preparing stationary state...")
    try:
        stationary_circuit = prepare_stationary_state(P, pi, method="amplitude_encoding")
        print(f" Stationary state prepared using amplitude encoding")
    except:
        # Fallback: manual preparation for 2-state case
        stationary_circuit = QuantumCircuit(2)
        # Prepare |»_¿È = ¿[0]|00È + ¿[1]|11È (simplified)
        theta = 2 * np.arccos(np.sqrt(pi[0]))
        stationary_circuit.ry(theta, 0)
        stationary_circuit.cx(0, 1)  # Create correlation between registers
        print(f" Stationary state prepared manually")
    
    print(f"Stationary circuit depth: {stationary_circuit.depth()}")
    
    # Prepare computational basis state |01È (edge 0í1)
    print("Preparing computational basis state |01È...")
    basis_circuit = prepare_computational_basis_state(n_qubits=2, state_index=1)
    print(f" Basis state |01È prepared")
    print()
    
    # ================================================================
    # STEP 4: Quantum Phase Estimation
    # ================================================================
    print("Step 4: Quantum Phase Estimation")
    print("-" * 35)
    
    # QPE parameters
    num_ancilla_qpe = 8  # Good precision for demonstration
    print(f"Using {num_ancilla_qpe} ancilla qubits for QPE")
    
    # QPE on stationary state
    print("Running QPE on stationary state...")
    qpe_stationary = quantum_phase_estimation(
        W_circuit,
        num_ancilla=num_ancilla_qpe,
        initial_state=stationary_circuit,
        backend="statevector"
    )
    
    # QPE on basis state
    print("Running QPE on computational basis state...")
    qpe_basis = quantum_phase_estimation(
        W_circuit,
        num_ancilla=num_ancilla_qpe,
        initial_state=basis_circuit,
        backend="statevector"
    )
    
    # Analyze QPE results
    analysis_stationary = analyze_qpe_results(qpe_stationary)
    analysis_basis = analyze_qpe_results(qpe_basis)
    
    print("QPE Results Summary:")
    print(f"Stationary state - Dominant phases: {analysis_stationary['dominant_phases'][:3]}")
    print(f"Basis state - Dominant phases: {analysis_basis['dominant_phases'][:3]}")
    
    # Check for stationary eigenvalue (phase H 0)
    stationary_phases = [p for p in analysis_stationary['dominant_phases'] if abs(p) < 0.1]
    print(f"Stationary eigenvalue found: {len(stationary_phases) > 0}")
    
    if len(stationary_phases) > 0:
        print(f"Best stationary phase estimate: {stationary_phases[0]:.6f}")
    
    # Plot QPE histograms
    print("Plotting QPE results...")
    fig1 = plot_qpe_histogram(qpe_stationary, analysis_stationary, 
                             figsize=(10, 6), show_phases=True)
    plt.title("QPE Results: Stationary State")
    plt.tight_layout()
    plt.savefig("qpe_stationary_state.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    fig2 = plot_qpe_histogram(qpe_basis, analysis_basis, 
                             figsize=(10, 6), show_phases=True)
    plt.title("QPE Results: Computational Basis State |01È")
    plt.tight_layout()
    plt.savefig("qpe_basis_state.png", dpi=150, bbox_inches='tight')
    plt.show()
    print()
    
    # ================================================================
    # STEP 5: Approximate Reflection Operator
    # ================================================================
    print("Step 5: Constructing Approximate Reflection Operator")
    print("-" * 50)
    
    # Reflection operator parameters
    num_ancilla_refl = 6
    phase_threshold = 0.05  # Conservative threshold
    
    print(f"Using {num_ancilla_refl} ancilla qubits for reflection")
    print(f"Phase threshold: {phase_threshold}")
    
    # Build reflection operator
    reflection_circuit = approximate_reflection_operator(
        W_circuit,
        num_ancilla=num_ancilla_refl,
        phase_threshold=phase_threshold
    )
    
    print(f"Reflection circuit depth: {reflection_circuit.depth()}")
    print(f"Reflection circuit gate count: {reflection_circuit.size()}")
    
    # Analyze reflection operator quality
    print("Analyzing reflection operator...")
    reflection_analysis = analyze_reflection_quality(
        reflection_circuit,
        target_state=stationary_circuit,
        num_samples=10
    )
    
    print("Reflection Analysis Results:")
    print(f"Average reflection fidelity: {reflection_analysis['average_reflection_fidelity']:.4f}")
    if 'target_preservation_fidelity' in reflection_analysis:
        print(f"Stationary state preservation: {reflection_analysis['target_preservation_fidelity']:.4f}")
    
    eigen_analysis = reflection_analysis['eigenvalue_analysis']
    print(f"Eigenvalues near +1: {eigen_analysis['num_near_plus_one']}")
    print(f"Eigenvalues near -1: {eigen_analysis['num_near_minus_one']}")
    print(f"Total eigenvalues: {eigen_analysis['num_eigenvalues']}")
    print()
    
    # ================================================================
    # STEP 6: Apply Reflection Operator
    # ================================================================
    print("Step 6: Demonstrating Reflection Operator Action")
    print("-" * 45)
    
    # Apply reflection to stationary state
    print("Applying reflection to stationary state...")
    stationary_reflected = apply_reflection_operator(
        stationary_circuit,
        reflection_circuit,
        backend="statevector",
        return_statevector=True
    )
    
    # Apply reflection to basis state
    print("Applying reflection to basis state...")
    basis_reflected = apply_reflection_operator(
        basis_circuit,
        reflection_circuit,
        backend="statevector",
        return_statevector=True
    )
    
    # Compute fidelities
    if stationary_reflected['statevector'] is not None:
        original_stationary = Statevector(stationary_circuit)
        reflected_stationary = stationary_reflected['statevector']
        stationary_fidelity = state_fidelity(original_stationary, reflected_stationary)
        print(f"Stationary state fidelity after reflection: {stationary_fidelity:.4f}")
    else:
        print("Stationary state fidelity: Using purity as proxy")
        print(f"Reflected state purity: {stationary_reflected.get('purity', 'N/A')}")
    
    if basis_reflected['statevector'] is not None:
        original_basis = Statevector(basis_circuit)
        reflected_basis = basis_reflected['statevector']
        basis_fidelity = state_fidelity(original_basis, reflected_basis)
        print(f"Basis state fidelity after reflection: {basis_fidelity:.4f}")
    else:
        print("Basis state fidelity: Using purity as proxy")
        print(f"Reflected state purity: {basis_reflected.get('purity', 'N/A')}")
    
    print()
    
    # ================================================================
    # STEP 7: Performance Analysis and Diagnostics
    # ================================================================
    print("Step 7: Performance Analysis and Diagnostics")
    print("-" * 45)
    
    # Classical vs quantum mixing time estimates
    classical_mixing = mixing_time_bound(P, epsilon=0.01)
    print(f"Classical mixing time bound (µ=0.01): {classical_mixing}")
    
    # Quantum advantage estimate
    if phase_gap_value > 0:
        quantum_mixing_estimate = int(np.ceil(1.0 / phase_gap_value))
        advantage_ratio = classical_mixing / quantum_mixing_estimate if quantum_mixing_estimate > 0 else float('inf')
        print(f"Quantum mixing time estimate: {quantum_mixing_estimate}")
        print(f"Potential quantum advantage: {advantage_ratio:.2f}x")
    else:
        print("Phase gap too small for reliable quantum mixing estimate")
    
    # Total variation distance analysis
    # Compare distributions after different numbers of classical steps
    print("\nTotal Variation Distance Analysis:")
    uniform_dist = np.array([0.5, 0.5])
    
    for t in [1, 5, 10, 20]:
        # Classical evolution
        P_t = np.linalg.matrix_power(P, t)
        evolved_dist = uniform_dist @ P_t
        tv_distance = total_variation_distance(evolved_dist, pi)
        print(f"Classical TV distance after {t:2d} steps: {tv_distance:.6f}")
    
    # Summary of quantum circuit resources
    print(f"\nQuantum Resource Summary:")
    print(f"Walk operator qubits: {W_circuit.num_qubits}")
    print(f"QPE total qubits: {qpe_stationary['circuit'].num_qubits}")
    print(f"Reflection total qubits: {reflection_circuit.num_qubits}")
    print(f"QPE circuit depth: {qpe_stationary['circuit'].depth()}")
    print(f"Reflection circuit depth: {reflection_circuit.depth()}")
    
    # ================================================================
    # STEP 8: Visualization and Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY AND CONCLUSIONS")
    print("=" * 60)
    
    print(f" Successfully demonstrated quantum MCMC pipeline on 2-state chain")
    print(f" Chain parameters: ±={alpha}, ≤={beta}")
    print(f" Stationary distribution: ¿ = {pi}")
    print(f" Classical spectral gap: {classical_gap:.4f}")
    print(f" Quantum phase gap: {phase_gap_value:.4f}")
    
    if len(stationary_phases) > 0:
        print(f" QPE successfully identified stationary eigenvalue")
        print(f"  Best phase estimate: {stationary_phases[0]:.6f}")
    else:
        print("† QPE did not clearly identify stationary eigenvalue")
    
    print(f" Reflection operator constructed and analyzed")
    print(f"  Average fidelity: {reflection_analysis['average_reflection_fidelity']:.4f}")
    
    # Theoretical vs practical comparison
    print(f"\nTheoretical vs Practical Results:")
    print(f"Expected eigenvalues: {walk_eigenvals}")
    print(f"QPE dominant phases: {analysis_stationary['dominant_phases'][:3]}")
    
    # Save results plot
    create_summary_plot(P, pi, qpe_stationary, qpe_basis, reflection_analysis)
    
    print(f"\n Results saved to qpe_stationary_state.png, qpe_basis_state.png, and summary_plot.png")
    print(f" Quantum MCMC demonstration completed successfully!")


def create_summary_plot(P: np.ndarray, pi: np.ndarray, 
                       qpe_stationary: Dict, qpe_basis: Dict,
                       reflection_analysis: Dict) -> None:
    """Create a comprehensive summary plot of the quantum MCMC results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Markov chain visualization
    ax = axes[0, 0]
    states = ['State 0', 'State 1']
    ax.bar(states, pi, color=['skyblue', 'lightcoral'], alpha=0.7)
    ax.set_title('Stationary Distribution ¿')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    for i, prob in enumerate(pi):
        ax.text(i, prob + 0.02, f'{prob:.3f}', ha='center', fontweight='bold')
    
    # Add transition matrix as text
    ax.text(0.02, 0.98, f'Transition Matrix P:\n[{P[0,0]:.2f} {P[0,1]:.2f}]\n[{P[1,0]:.2f} {P[1,1]:.2f}]',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: QPE phases comparison
    ax = axes[0, 1]
    phases_stat = qpe_stationary['phases'][:5]  # Top 5 phases
    phases_basis = qpe_basis['phases'][:5]
    probs_stat = qpe_stationary['probabilities'][:5]
    probs_basis = qpe_basis['probabilities'][:5]
    
    x = np.arange(len(phases_stat))
    width = 0.35
    
    ax.bar(x - width/2, probs_stat, width, label='Stationary State', 
           color='darkblue', alpha=0.7)
    ax.bar(x + width/2, probs_basis, width, label='Basis State |01È', 
           color='darkred', alpha=0.7)
    
    ax.set_title('QPE Phase Probabilities')
    ax.set_xlabel('Phase Index')
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p:.3f}' for p in phases_stat], rotation=45)
    ax.legend()
    
    # Plot 3: Reflection operator eigenvalue analysis
    ax = axes[1, 0]
    eigen_data = reflection_analysis['eigenvalue_analysis']
    categories = ['Near +1', 'Near -1', 'Other']
    counts = [
        eigen_data['num_near_plus_one'],
        eigen_data['num_near_minus_one'],
        eigen_data['num_eigenvalues'] - eigen_data['num_near_plus_one'] - eigen_data['num_near_minus_one']
    ]
    colors = ['green', 'red', 'gray']
    
    wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors, 
                                     autopct='%1.1f%%', startangle=90)
    ax.set_title('Reflection Operator Eigenvalues')
    
    # Plot 4: Performance metrics
    ax = axes[1, 1]
    metrics = ['Reflection\nFidelity', 'Phase\nResolution', 'Spectral\nGap']
    values = [
        reflection_analysis['average_reflection_fidelity'],
        1.0 / (2 ** qpe_stationary['num_ancilla']),  # QPE resolution
        phase_gap(discriminant_matrix(P, pi))  # Normalized for display
    ]
    
    bars = ax.bar(metrics, values, color=['purple', 'orange', 'green'], alpha=0.7)
    ax.set_title('Performance Metrics')
    ax.set_ylabel('Value')
    ax.set_ylim(0, max(values) * 1.2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("summary_plot.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()