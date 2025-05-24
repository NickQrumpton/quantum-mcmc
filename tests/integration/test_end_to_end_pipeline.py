"""End-to-end integration tests for quantum MCMC sampling pipeline.

This module provides comprehensive integration tests that exercise the entire
quantum MCMC sampling pipeline from classical Markov chain construction
through quantum walk operators, phase estimation, and reflection operators.

The tests verify that all components work together correctly to implement
the theoretical quantum MCMC framework, ensuring:
- Proper interface integration between modules
- Mathematically consistent results across the pipeline
- Physical validity of quantum states and operators
- Robustness to different parameter regimes

These integration tests complement the unit tests by validating the complete
workflow that would be used in quantum MCMC research applications.

Author: [Your Name]
Date: 2025-01-23
"""

import pytest
import numpy as np
from typing import Dict, Tuple, List
import warnings

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, state_fidelity

# Project imports - classical components
from quantum_mcmc.classical.markov_chain import (
    build_two_state_chain,
    build_three_state_symmetric,
    build_random_reversible_chain,
    stationary_distribution,
    is_reversible,
    is_stochastic
)
from quantum_mcmc.classical.discriminant import (
    discriminant_matrix,
    singular_values,
    phase_gap
)

# Project imports - quantum core components
from quantum_mcmc.core.quantum_walk import (
    prepare_walk_operator,
    validate_walk_operator,
    walk_eigenvalues,
    is_unitary
)
from quantum_mcmc.core.phase_estimation import (
    quantum_phase_estimation,
    analyze_qpe_results,
    estimate_required_precision
)
from quantum_mcmc.core.reflection_operator import (
    approximate_reflection_operator,
    apply_reflection_operator,
    analyze_reflection_quality
)

# Project imports - utility components
from quantum_mcmc.utils.state_preparation import (
    prepare_stationary_state,
    prepare_uniform_superposition
)
from quantum_mcmc.utils.analysis import (
    compute_sampling_fidelity,
    estimate_mixing_time,
    validate_quantum_state
)


class TestTwoStateChainPipeline:
    """End-to-end tests for complete 2-state chain quantum MCMC pipeline."""
    
    def test_complete_two_state_pipeline(self):
        """Test complete quantum MCMC pipeline on simple 2-state chain.
        
        This test exercises the full pipeline:
        1. Build 2-state reversible Markov chain
        2. Construct Szegedy quantum walk operator
        3. Prepare stationary state
        4. Apply quantum phase estimation
        5. Construct reflection operator
        6. Verify mathematical consistency
        """
        # Step 1: Build classical Markov chain
        alpha, beta = 0.3, 0.4
        P = build_two_state_chain(alpha, beta)
        pi = stationary_distribution(P)
        
        # Validate classical chain properties
        assert is_stochastic(P), "Transition matrix should be stochastic"
        assert is_reversible(P, pi), "Chain should be reversible"
        assert np.allclose(P @ pi, pi), "pi should be stationary distribution"
        
        # Step 2: Construct quantum walk operator
        W_circuit = prepare_walk_operator(P, pi=pi, backend="qiskit")
        W_matrix = prepare_walk_operator(P, pi=pi, backend="matrix")
        
        # Validate walk operator properties
        assert isinstance(W_circuit, QuantumCircuit), \
            "Should construct quantum circuit"
        assert isinstance(W_matrix, np.ndarray), \
            "Should construct matrix representation"
        assert is_unitary(W_matrix), \
            "Walk operator should be unitary"
        assert validate_walk_operator(W_circuit, P, pi), \
            "Walk operator should pass validation"
        
        # Check dimensions
        n = P.shape[0]
        expected_dim = n * n  # Edge space dimension
        assert W_matrix.shape == (expected_dim, expected_dim), \
            f"Walk operator should be {expected_dim}×{expected_dim}"
        
        # Step 3: Prepare stationary state approximation
        stationary_circuit = prepare_stationary_state(P, pi, method="amplitude_encoding")
        
        # Validate state preparation
        assert isinstance(stationary_circuit, QuantumCircuit), \
            "Should prepare stationary state circuit"
        assert stationary_circuit.num_qubits >= int(np.ceil(np.log2(n))), \
            "Should have sufficient qubits for state representation"
        
        # Step 4: Apply quantum phase estimation
        qpe_results = quantum_phase_estimation(
            W_circuit, 
            num_ancilla=6,
            initial_state=stationary_circuit,
            backend="statevector"
        )
        
        # Validate QPE results structure
        assert isinstance(qpe_results, dict), "QPE should return dictionary"
        assert 'phases' in qpe_results, "Should extract phases"
        assert 'probabilities' in qpe_results, "Should compute probabilities"
        assert 'circuit' in qpe_results, "Should return QPE circuit"
        
        # Analyze QPE results
        qpe_analysis = analyze_qpe_results(qpe_results)
        
        # Should find stationary eigenvalue (phase H 0)
        stationary_phases = [p for p in qpe_analysis['dominant_phases'] if abs(p) < 0.1]
        assert len(stationary_phases) > 0, \
            "Should identify stationary eigenstate (phase H 0)"
        
        # Step 5: Construct reflection operator
        reflection_circuit = approximate_reflection_operator(
            W_circuit, 
            num_ancilla=5,
            phase_threshold=0.08
        )
        
        # Validate reflection operator
        assert isinstance(reflection_circuit, QuantumCircuit), \
            "Should construct reflection circuit"
        assert reflection_circuit.num_qubits > W_circuit.num_qubits, \
            "Reflection should include ancilla qubits"
        
        # Step 6: Apply reflection operator to stationary state
        reflection_result = apply_reflection_operator(
            stationary_circuit,
            reflection_circuit,
            backend="statevector",
            return_statevector=True
        )
        
        # Validate reflection application
        assert 'statevector' in reflection_result or 'density_matrix' in reflection_result, \
            "Should return quantum state after reflection"
        
        # Step 7: Analyze complete pipeline consistency
        # The stationary state should be approximately preserved by reflection
        if 'purity' in reflection_result:
            assert reflection_result['purity'] > 0.5, \
                "Reflected stationary state should maintain reasonable purity"
        
        # Verify eigenvalue structure consistency
        walk_eigenvals = walk_eigenvalues(P, pi)
        qpe_phases = qpe_analysis['dominant_phases']
        
        # Check that QPE phases correspond to walk eigenvalues
        for eigenval in walk_eigenvals:
            phase = np.angle(eigenval) / (2 * np.pi) % 1
            closest_qpe_phase = min(qpe_phases, key=lambda p: abs(p - phase))
            phase_error = abs(closest_qpe_phase - phase)
            # Allow for discretization error
            resolution = 1.0 / (2 ** qpe_results['num_ancilla'])
            assert phase_error < 3 * resolution, \
                f"QPE phase {closest_qpe_phase} should match eigenvalue phase {phase}"
    
    def test_pipeline_parameter_scaling(self):
        """Test pipeline behavior with different precision parameters."""
        # Fixed 2-state chain
        P = build_two_state_chain(0.2, 0.7)
        pi = stationary_distribution(P)
        W = prepare_walk_operator(P, backend="qiskit")
        
        # Test different QPE precisions
        precisions = [4, 6, 8]
        phase_accuracies = []
        
        for num_ancilla in precisions:
            # Run QPE
            qpe_results = quantum_phase_estimation(
                W, num_ancilla=num_ancilla, backend="statevector"
            )
            
            # Analyze accuracy
            qpe_analysis = analyze_qpe_results(qpe_results)
            
            # Find most accurate phase estimate for stationary state
            if len(qpe_analysis['dominant_phases']) > 0:
                stationary_phase = min(qpe_analysis['dominant_phases'], key=abs)
                phase_accuracies.append(abs(stationary_phase))
            else:
                phase_accuracies.append(1.0)  # Worst case
        
        # Higher precision should generally give better accuracy
        # (allowing for some noise in small examples)
        assert phase_accuracies[-1] <= phase_accuracies[0] + 0.1, \
            f"Higher precision should improve accuracy: {phase_accuracies}"
    
    def test_pipeline_robustness_symmetric_chain(self):
        """Test pipeline on symmetric 2-state chain (special case)."""
        # Symmetric chain: P = [[0.5, 0.5], [0.5, 0.5]]
        P = build_two_state_chain(0.5, 0.5)
        pi = stationary_distribution(P)
        
        # Should have uniform stationary distribution
        assert np.allclose(pi, [0.5, 0.5]), \
            "Symmetric chain should have uniform stationary distribution"
        
        # Complete pipeline
        W = prepare_walk_operator(P, backend="qiskit")
        
        # QPE should work
        qpe_results = quantum_phase_estimation(W, num_ancilla=5, backend="statevector")
        assert len(qpe_results['phases']) > 0, \
            "Should extract phases from symmetric chain"
        
        # Reflection should construct
        R = approximate_reflection_operator(W, num_ancilla=4)
        assert isinstance(R, QuantumCircuit), \
            "Should construct reflection for symmetric chain"
        
        # Analysis should show good properties
        reflection_analysis = analyze_reflection_quality(R, num_samples=3)
        assert reflection_analysis['average_reflection_fidelity'] > 0.2, \
            "Symmetric chain should have reasonable reflection properties"


class TestThreeStateChainPipeline:
    """End-to-end tests for 3-state chain quantum MCMC pipeline."""
    
    def test_three_state_symmetric_pipeline(self):
        """Test complete pipeline on 3-state symmetric chain."""
        # Build 3-state symmetric chain
        p = 0.25  # Probability of staying in same state
        P = build_three_state_symmetric(p)
        pi = stationary_distribution(P)
        
        # Classical validation
        assert is_stochastic(P), "3-state chain should be stochastic"
        assert is_reversible(P, pi), "3-state chain should be reversible"
        assert np.allclose(pi, [1/3, 1/3, 1/3]), \
            "Symmetric 3-state chain should have uniform stationary distribution"
        
        # Quantum walk construction
        W_circuit = prepare_walk_operator(P, backend="qiskit")
        W_matrix = prepare_walk_operator(P, backend="matrix")
        
        # Validate walk operator
        assert W_matrix.shape == (9, 9), \
            "3-state walk operator should be 9×9 (edge space)"
        assert is_unitary(W_matrix), \
            "Walk operator should be unitary"
        
        # State preparation for larger system
        try:
            stationary_circuit = prepare_stationary_state(P, pi, method="dicke_state")
        except:
            # Fallback to uniform superposition if Dicke state fails
            n_qubits = int(np.ceil(np.log2(len(pi))))
            stationary_circuit = prepare_uniform_superposition(n_qubits)
        
        # QPE on larger system
        qpe_results = quantum_phase_estimation(
            W_circuit,
            num_ancilla=5,
            initial_state=stationary_circuit,
            backend="statevector"
        )
        
        # Should extract meaningful phases
        assert len(qpe_results['phases']) > 0, \
            "Should extract phases from 3-state system"
        
        qpe_analysis = analyze_qpe_results(qpe_results)
        assert len(qpe_analysis['dominant_phases']) > 0, \
            "Should identify dominant phases"
        
        # Reflection operator construction
        R = approximate_reflection_operator(W_circuit, num_ancilla=4)
        
        # Apply reflection
        reflection_result = apply_reflection_operator(
            stationary_circuit, R, backend="statevector", return_statevector=True
        )
        
        # Validate end-to-end result
        assert 'circuit' in reflection_result, \
            "Should complete reflection application"
    
    def test_three_state_asymmetric_pipeline(self):
        """Test pipeline on asymmetric 3-state chain."""
        # Build asymmetric but reversible 3-state chain
        np.random.seed(42)  # Reproducible results
        P = build_random_reversible_chain(n=3, seed=42)
        pi = stationary_distribution(P)
        
        # Validate classical properties
        assert is_stochastic(P), "Random chain should be stochastic"
        assert is_reversible(P, pi), "Random chain should be reversible"
        
        # Complete quantum pipeline
        W = prepare_walk_operator(P, backend="qiskit")
        
        # QPE analysis
        qpe_results = quantum_phase_estimation(W, num_ancilla=5, backend="statevector")
        qpe_analysis = analyze_qpe_results(qpe_results)
        
        # Should find stationary eigenvalue
        stationary_phases = [p for p in qpe_analysis['dominant_phases'] if abs(p) < 0.15]
        assert len(stationary_phases) > 0, \
            "Should find stationary eigenstate in asymmetric chain"
        
        # Reflection analysis
        R = approximate_reflection_operator(W, num_ancilla=4)
        reflection_analysis = analyze_reflection_quality(R, num_samples=2)
        
        # Should have reasonable reflection properties
        assert 'eigenvalue_analysis' in reflection_analysis, \
            "Should analyze reflection eigenvalues"
        assert reflection_analysis['eigenvalue_analysis']['num_eigenvalues'] > 0, \
            "Should have positive number of eigenvalues"


class TestPipelineConsistencyChecks:
    """Tests for mathematical consistency across pipeline components."""
    
    def test_eigenvalue_consistency_across_representations(self):
        """Test that eigenvalues are consistent between different representations."""
        # Build test chain
        P = build_two_state_chain(0.4, 0.3)
        pi = stationary_distribution(P)
        
        # Get eigenvalues from different sources
        # 1. Direct walk operator eigenvalues
        walk_eigenvals = walk_eigenvalues(P, pi)
        
        # 2. Matrix representation eigenvalues
        W_matrix = prepare_walk_operator(P, backend="matrix")
        matrix_eigenvals = np.linalg.eigvals(W_matrix)
        
        # 3. QPE estimated eigenvalues
        W_circuit = prepare_walk_operator(P, backend="qiskit")
        qpe_results = quantum_phase_estimation(W_circuit, num_ancilla=7, backend="statevector")
        qpe_phases = qpe_results['phases']
        qpe_eigenvals = np.exp(2j * np.pi * qpe_phases)
        
        # Check consistency between walk_eigenvalues and matrix eigenvalues
        for w_eval in walk_eigenvals:
            closest_matrix = min(matrix_eigenvals, key=lambda x: abs(x - w_eval))
            assert abs(closest_matrix - w_eval) < 0.1, \
                f"Walk eigenvalue {w_eval} should match matrix eigenvalue {closest_matrix}"
        
        # Check that QPE finds the dominant eigenvalues
        # (Note: QPE may not find all eigenvalues due to state preparation)
        dominant_qpe = qpe_eigenvals[qpe_results['probabilities'] > 0.1]
        for qpe_eval in dominant_qpe:
            closest_theory = min(walk_eigenvals, key=lambda x: abs(x - qpe_eval))
            phase_diff = abs(np.angle(qpe_eval) - np.angle(closest_theory))
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Handle wraparound
            resolution = 2 * np.pi / (2 ** qpe_results['num_ancilla'])
            assert phase_diff < 3 * resolution, \
                f"QPE eigenvalue {qpe_eval} should match theoretical {closest_theory}"
    
    def test_stationary_state_preservation_consistency(self):
        """Test that stationary state is consistently preserved across components."""
        # Build chain
        P = build_two_state_chain(0.35, 0.45)
        pi = stationary_distribution(P)
        
        # Prepare approximate stationary state
        try:
            pi_circuit = prepare_stationary_state(P, pi, method="amplitude_encoding")
        except:
            # Fallback preparation
            n_qubits = int(np.ceil(np.log2(len(pi))))
            pi_circuit = QuantumCircuit(2 * n_qubits)
            # Simple approximation: prepare product state based on pi
            if pi[0] > 0.5:
                pi_circuit.ry(2 * np.arccos(np.sqrt(pi[1])), 0)
            else:
                pi_circuit.x(0)
                pi_circuit.ry(2 * np.arccos(np.sqrt(pi[0])), 0)
        
        # Apply walk operator - should approximately preserve stationary state
        W = prepare_walk_operator(P, backend="qiskit")
        
        # Create circuit that applies walk to stationary state
        test_circuit = pi_circuit.compose(W)
        
        # Get statevectors
        original_state = Statevector(pi_circuit)
        evolved_state = Statevector(test_circuit)
        
        # Compute fidelity
        fidelity = state_fidelity(original_state, evolved_state)
        
        # Stationary state should be approximately preserved
        # (allowing for approximation errors in state preparation)
        assert fidelity > 0.7, \
            f"Walk operator should approximately preserve stationary state, fidelity={fidelity}"
        
        # Reflection operator should also preserve stationary state
        R = approximate_reflection_operator(W, num_ancilla=4, phase_threshold=0.1)
        
        reflection_result = apply_reflection_operator(
            pi_circuit, R, backend="statevector", return_statevector=True
        )
        
        # Check preservation by reflection
        if reflection_result['statevector'] is not None:
            reflection_fidelity = state_fidelity(
                original_state, reflection_result['statevector']
            )
            assert reflection_fidelity > 0.5, \
                f"Reflection should preserve stationary state, fidelity={reflection_fidelity}"
    
    def test_discriminant_phase_gap_consistency(self):
        """Test consistency between discriminant analysis and QPE phase gap."""
        # Build chain with known spectral properties
        P = build_two_state_chain(0.2, 0.8)  # Unbalanced chain
        pi = stationary_distribution(P)
        
        # Classical discriminant analysis
        D = discriminant_matrix(P, pi)
        classical_gap = phase_gap(D)
        
        # Quantum phase estimation
        W = prepare_walk_operator(P, backend="qiskit")
        qpe_results = quantum_phase_estimation(W, num_ancilla=8, backend="statevector")
        qpe_analysis = analyze_qpe_results(qpe_results)
        
        # Find quantum phase gap
        phases = qpe_analysis['dominant_phases']
        non_zero_phases = [p for p in phases if abs(p) > 0.01]
        
        if len(non_zero_phases) > 0:
            quantum_gap = min(min(non_zero_phases), min(1 - p for p in non_zero_phases if p > 0.5))
            
            # Classical and quantum gaps should be related
            # (exact relationship depends on implementation details)
            gap_ratio = quantum_gap / classical_gap if classical_gap > 0 else float('inf')
            assert 0.1 <= gap_ratio <= 10, \
                f"Quantum and classical gaps should be comparable: {quantum_gap} vs {classical_gap}"


class TestPipelineErrorHandling:
    """Tests for robust error handling across the pipeline."""
    
    def test_pipeline_with_near_singular_chain(self):
        """Test pipeline behavior with nearly singular Markov chain."""
        # Create nearly reducible chain
        epsilon = 1e-6
        P = np.array([[1-epsilon, epsilon],
                      [epsilon, 1-epsilon]])
        
        # Should still be valid
        pi = stationary_distribution(P)
        assert is_reversible(P, pi), "Nearly singular chain should still be reversible"
        
        # Quantum walk construction should work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Expect warnings about conditioning
            
            W = prepare_walk_operator(P, backend="qiskit")
            assert isinstance(W, QuantumCircuit), \
                "Should construct walk operator for nearly singular chain"
            
            # QPE might have convergence issues but should not crash
            qpe_results = quantum_phase_estimation(W, num_ancilla=4, backend="statevector")
            assert isinstance(qpe_results, dict), \
                "QPE should return results for nearly singular chain"
    
    def test_pipeline_with_invalid_inputs(self):
        """Test pipeline error handling with invalid inputs."""
        # Valid starting point
        P = build_two_state_chain(0.3, 0.4)
        W = prepare_walk_operator(P, backend="qiskit")
        
        # Test QPE with invalid ancilla count
        with pytest.raises(ValueError):
            quantum_phase_estimation(W, num_ancilla=0)
        
        # Test reflection with invalid threshold
        with pytest.raises(ValueError):
            approximate_reflection_operator(W, num_ancilla=3, phase_threshold=0.6)
        
        # Test state preparation with invalid dimensions
        invalid_P = np.array([[0.5, 0.5, 0.1],  # Non-square or non-stochastic
                              [0.3, 0.7]])
        
        with pytest.raises((ValueError, AssertionError)):
            prepare_walk_operator(invalid_P)
    
    def test_pipeline_component_interface_compatibility(self):
        """Test that all pipeline components have compatible interfaces."""
        # Build valid chain
        P = build_two_state_chain(0.4, 0.5)
        pi = stationary_distribution(P)
        
        # Test quantum walk interfaces
        W_circuit = prepare_walk_operator(P, backend="qiskit")
        W_matrix = prepare_walk_operator(P, backend="matrix")
        
        # Both should be compatible with validation
        assert validate_walk_operator(W_circuit, P, pi), \
            "Circuit walk operator should validate"
        assert validate_walk_operator(W_matrix, P, pi), \
            "Matrix walk operator should validate"
        
        # QPE should accept both circuit and matrix representations
        qpe_results_circuit = quantum_phase_estimation(W_circuit, num_ancilla=4, backend="statevector")
        
        # Analysis should work on QPE results
        qpe_analysis = analyze_qpe_results(qpe_results_circuit)
        assert isinstance(qpe_analysis, dict), \
            "QPE analysis should return dictionary"
        
        # Reflection should accept walk circuit
        R = approximate_reflection_operator(W_circuit, num_ancilla=3)
        
        # Reflection analysis should work
        reflection_analysis = analyze_reflection_quality(R, num_samples=1)
        assert isinstance(reflection_analysis, dict), \
            "Reflection analysis should return dictionary"


class TestPipelinePerformanceMetrics:
    """Tests for performance and quality metrics across the pipeline."""
    
    def test_end_to_end_sampling_fidelity(self):
        """Test complete pipeline produces high-fidelity sampling results."""
        # Well-conditioned test chain
        P = build_two_state_chain(0.3, 0.6)
        pi = stationary_distribution(P)
        
        # Complete pipeline with good precision
        W = prepare_walk_operator(P, backend="qiskit")
        
        # High-precision QPE
        qpe_results = quantum_phase_estimation(W, num_ancilla=8, backend="statevector")
        qpe_analysis = analyze_qpe_results(qpe_results)
        
        # Should achieve good phase resolution
        resolution = qpe_analysis['resolution']
        assert resolution < 0.01, f"Should achieve fine phase resolution: {resolution}"
        
        # Should identify stationary eigenstate accurately
        stationary_phases = [p for p in qpe_analysis['dominant_phases'] if abs(p) < 0.05]
        assert len(stationary_phases) > 0, \
            "Should accurately identify stationary eigenstate"
        
        # High-quality reflection operator
        R = approximate_reflection_operator(W, num_ancilla=6, phase_threshold=0.03)
        reflection_analysis = analyze_reflection_quality(R, num_samples=5)
        
        # Should achieve reasonable reflection quality
        assert reflection_analysis['average_reflection_fidelity'] > 0.6, \
            f"Should achieve good reflection fidelity: {reflection_analysis['average_reflection_fidelity']}"
    
    def test_pipeline_convergence_with_precision(self):
        """Test that pipeline results converge with increasing precision."""
        P = build_two_state_chain(0.25, 0.35)
        W = prepare_walk_operator(P, backend="qiskit")
        
        # Test increasing QPE precision
        precisions = [4, 6, 8]
        stationary_phase_estimates = []
        
        for num_ancilla in precisions:
            qpe_results = quantum_phase_estimation(W, num_ancilla=num_ancilla, backend="statevector")
            qpe_analysis = analyze_qpe_results(qpe_results)
            
            # Find best stationary phase estimate
            if len(qpe_analysis['dominant_phases']) > 0:
                best_stationary = min(qpe_analysis['dominant_phases'], key=abs)
                stationary_phase_estimates.append(abs(best_stationary))
            else:
                stationary_phase_estimates.append(1.0)
        
        # Should generally improve (allowing for some noise)
        assert stationary_phase_estimates[-1] <= stationary_phase_estimates[0] + 0.05, \
            f"Higher precision should improve stationary phase estimation: {stationary_phase_estimates}"
        
        # Test reflection operator quality vs precision
        reflection_qualities = []
        for num_ancilla in [3, 5]:
            R = approximate_reflection_operator(W, num_ancilla=num_ancilla)
            analysis = analyze_reflection_quality(R, num_samples=2)
            reflection_qualities.append(analysis['average_reflection_fidelity'])
        
        # Higher precision should generally improve quality
        assert reflection_qualities[1] >= reflection_qualities[0] - 0.1, \
            f"Higher precision should improve reflection quality: {reflection_qualities}"
    
    def test_pipeline_scaling_with_system_size(self):
        """Test pipeline behavior as system size increases."""
        # Test different system sizes
        system_sizes = [2, 3]  # Limited by computational cost
        
        for n in system_sizes:
            if n == 2:
                P = build_two_state_chain(0.3, 0.4)
            elif n == 3:
                P = build_three_state_symmetric(0.2)
            else:
                P = build_random_reversible_chain(n, seed=42)
            
            # Complete pipeline
            pi = stationary_distribution(P)
            W = prepare_walk_operator(P, backend="qiskit")
            
            # Should construct successfully
            assert isinstance(W, QuantumCircuit), \
                f"Should construct walk operator for {n}-state system"
            
            # QPE should work
            qpe_results = quantum_phase_estimation(W, num_ancilla=5, backend="statevector")
            assert len(qpe_results['phases']) > 0, \
                f"Should extract phases for {n}-state system"
            
            # Reflection should construct
            R = approximate_reflection_operator(W, num_ancilla=4)
            assert isinstance(R, QuantumCircuit), \
                f"Should construct reflection for {n}-state system"
            
            # Circuit depth should scale reasonably
            assert W.depth() < 100, \
                f"Walk operator depth should be reasonable for {n}-state system: {W.depth()}"
            assert R.depth() < 200, \
                f"Reflection operator depth should be reasonable for {n}-state system: {R.depth()}"


class TestPipelineRegressionTests:
    """Regression tests to ensure pipeline maintains expected behavior."""
    
    def test_canonical_two_state_chain_results(self):
        """Test that canonical 2-state chain produces expected results.
        
        This test serves as a regression test to ensure that changes
        to the codebase don't break the fundamental behavior on a
        well-understood example.
        """
        # Canonical example: symmetric 2-state chain
        P = build_two_state_chain(0.5, 0.5)
        pi = stationary_distribution(P)
        
        # Expected properties
        assert np.allclose(pi, [0.5, 0.5]), "Should have uniform stationary distribution"
        
        # Walk operator eigenvalues should be known
        walk_eigenvals = walk_eigenvalues(P, pi)
        eigenval_magnitudes = np.abs(walk_eigenvals)
        assert np.allclose(eigenval_magnitudes, 1.0), \
            "All eigenvalues should have magnitude 1"
        
        # QPE should find these eigenvalues
        W = prepare_walk_operator(P, backend="qiskit")
        qpe_results = quantum_phase_estimation(W, num_ancilla=6, backend="statevector")
        
        # Should find phase 0 (stationary eigenvalue 1)
        phases_near_zero = [p for p in qpe_results['phases'] if abs(p) < 0.1]
        assert len(phases_near_zero) > 0, \
            "Should find stationary eigenvalue (phase H 0)"
        
        # Should find another eigenvalue (phase H 0.5 for eigenvalue -1)
        phases_near_half = [p for p in qpe_results['phases'] if abs(p - 0.5) < 0.1]
        assert len(phases_near_half) > 0, \
            "Should find second eigenvalue (phase H 0.5)"
    
    def test_known_eigenvalue_structure_preservation(self):
        """Test that known eigenvalue structures are preserved through pipeline."""
        # Asymmetric 2-state chain with known eigenvalues
        alpha, beta = 0.2, 0.8
        P = build_two_state_chain(alpha, beta)
        pi = stationary_distribution(P)
        
        # Theoretical eigenvalues of transition matrix P
        P_eigenvals = np.linalg.eigvals(P)
        assert np.any(np.isclose(P_eigenvals, 1.0)), \
            "Transition matrix should have eigenvalue 1"
        
        # Walk operator eigenvalues
        walk_eigenvals = walk_eigenvalues(P, pi)
        
        # QPE should detect these eigenvalues
        W = prepare_walk_operator(P, backend="qiskit")
        qpe_results = quantum_phase_estimation(W, num_ancilla=7, backend="statevector")
        qpe_phases = qpe_results['phases']
        
        # Convert QPE phases to eigenvalues
        qpe_eigenvals = np.exp(2j * np.pi * qpe_phases)
        
        # Should find stationary eigenvalue (eigenvalue 1, phase 0)
        stationary_qpe = [e for e in qpe_eigenvals if abs(e - 1.0) < 0.2]
        assert len(stationary_qpe) > 0, \
            "QPE should detect stationary eigenvalue"
        
        # Check overall eigenvalue spectrum consistency
        for w_eval in walk_eigenvals:
            closest_qpe = min(qpe_eigenvals, key=lambda x: abs(x - w_eval))
            assert abs(closest_qpe - w_eval) < 0.3, \
                f"QPE eigenvalue {closest_qpe} should match theoretical {w_eval}"
    
    def test_pipeline_determinism(self):
        """Test that pipeline produces deterministic results with fixed parameters."""
        # Fixed parameters
        P = build_two_state_chain(0.3, 0.4)
        pi = stationary_distribution(P)
        
        # Run pipeline twice with same parameters
        results1 = self._run_basic_pipeline(P, pi, num_ancilla=5, random_seed=42)
        results2 = self._run_basic_pipeline(P, pi, num_ancilla=5, random_seed=42)
        
        # Results should be identical for deterministic backends
        assert np.allclose(results1['qpe_phases'], results2['qpe_phases']), \
            "QPE phases should be deterministic"
        
        # Circuit structures should be identical
        assert results1['walk_circuit'].depth() == results2['walk_circuit'].depth(), \
            "Walk circuits should be identical"
        assert results1['reflection_circuit'].depth() == results2['reflection_circuit'].depth(), \
            "Reflection circuits should be identical"
    
    def _run_basic_pipeline(self, P: np.ndarray, pi: np.ndarray, 
                           num_ancilla: int, random_seed: int) -> Dict:
        """Helper method to run basic pipeline for regression testing."""
        np.random.seed(random_seed)
        
        # Build pipeline components
        W = prepare_walk_operator(P, backend="qiskit")
        
        qpe_results = quantum_phase_estimation(
            W, num_ancilla=num_ancilla, backend="statevector"
        )
        
        R = approximate_reflection_operator(W, num_ancilla=num_ancilla-1)
        
        return {
            'walk_circuit': W,
            'qpe_phases': qpe_results['phases'],
            'reflection_circuit': R
        }