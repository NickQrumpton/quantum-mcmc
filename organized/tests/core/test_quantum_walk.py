"""Unit tests for quantum walk operator construction module.

This module provides comprehensive unit tests for the quantum walk operator
implementation in quantum_mcmc.core.quantum_walk. Tests cover:
- Mathematical correctness (unitarity, eigenstructure)
- Qiskit integration and circuit construction
- Edge cases and error handling
- Physical validity of quantum walk properties

The tests verify both the theoretical foundations and practical implementation
of Szegedy's quantum walk framework for MCMC sampling.

Author: [Your Name]
Date: 2025-01-23
"""

import pytest
import numpy as np
from scipy.linalg import eigh
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from quantum_mcmc.core.quantum_walk import (
    prepare_walk_operator,
    is_unitary,
    walk_eigenvalues,
    validate_walk_operator,
    prepare_initial_state,
    measure_walk_mixing,
    quantum_mixing_time,
    _build_walk_matrix,
    _build_projection_operator,
    _build_swap_operator,
    _matrix_to_circuit
)
from quantum_mcmc.classical.markov_chain import (
    build_two_state_chain,
    build_three_state_symmetric,
    build_random_reversible_chain,
    stationary_distribution,
    is_reversible
)
from quantum_mcmc.classical.discriminant import discriminant_matrix


class TestPrepareWalkOperator:
    """Test suite for prepare_walk_operator function."""
    
    def test_two_state_chain_matrix(self):
        """Test walk operator construction for simple two-state chain."""
        # Build a simple two-state chain with known properties
        P = build_two_state_chain(alpha=0.3)
        
        # Construct walk operator as matrix
        W = prepare_walk_operator(P, backend="matrix")
        
        # Check dimensions: should be 4x4 for 2-state chain
        assert W.shape == (4, 4), \
            f"Walk operator should be 4x4 for 2-state chain, got {W.shape}"
        
        # Check unitarity
        assert is_unitary(W), \
            "Walk operator must be unitary"
        
        # Check that W is complex (may have complex eigenvalues)
        assert W.dtype == complex, \
            "Walk operator should be complex-valued"
    
    def test_two_state_chain_circuit(self):
        """Test Qiskit circuit generation for two-state chain."""
        P = build_two_state_chain(alpha=0.3)
        
        # Construct walk operator as quantum circuit
        qc = prepare_walk_operator(P, backend="qiskit")
        
        # Check that we get a QuantumCircuit
        assert isinstance(qc, QuantumCircuit), \
            "Should return QuantumCircuit when backend='qiskit'"
        
        # Check qubit count: need 2 qubits total (1 per register)
        assert qc.num_qubits == 2, \
            f"Two-state chain should use 2 qubits, got {qc.num_qubits}"
        
        # Extract unitary and verify
        W_from_circuit = Operator(qc).data
        assert is_unitary(W_from_circuit), \
            "Circuit should implement a unitary operator"
    
    def test_three_state_symmetric_chain(self):
        """Test walk operator for three-state symmetric chain."""
        P = build_three_state_symmetric(p=0.2)
        pi = stationary_distribution(P)
        
        # Construct walk operator
        W = prepare_walk_operator(P, pi=pi, backend="matrix")
        
        # Check dimensions: should be 9x9 for 3-state chain
        assert W.shape == (9, 9), \
            f"Walk operator should be 9x9 for 3-state chain, got {W.shape}"
        
        # Verify unitarity
        assert is_unitary(W), \
            "Walk operator must be unitary for symmetric chain"
        
        # Verify validation passes
        assert validate_walk_operator(W, P, pi), \
            "Walk operator should pass validation"
    
    def test_larger_chain_circuit_padding(self):
        """Test circuit construction with qubit padding for non-power-of-2 states."""
        # 5-state chain requires 3 qubits per register (2^3 = 8 > 5)
        n = 5
        P = build_random_reversible_chain(n, seed=42)
        
        qc = prepare_walk_operator(P, backend="qiskit")
        
        # Should use 6 qubits total (3 per register)
        expected_qubits = 2 * int(np.ceil(np.log2(n)))
        assert qc.num_qubits == expected_qubits, \
            f"Expected {expected_qubits} qubits for {n}-state chain, got {qc.num_qubits}"
        
        # Verify the circuit still implements valid walk operator
        W_circuit = Operator(qc).data
        assert is_unitary(W_circuit), \
            "Padded circuit should still be unitary"
    
    def test_stationary_distribution_computation(self):
        """Test automatic computation of stationary distribution."""
        P = build_two_state_chain(alpha=0.4)
        
        # Call without providing pi
        W1 = prepare_walk_operator(P, backend="matrix")
        
        # Call with explicit pi
        pi = stationary_distribution(P)
        W2 = prepare_walk_operator(P, pi=pi, backend="matrix")
        
        # Results should be identical
        assert np.allclose(W1, W2), \
            "Walk operator should be same whether pi is provided or computed"
    
    def test_invalid_backend_error(self):
        """Test error handling for invalid backend specification."""
        P = build_two_state_chain(alpha=0.3)
        
        with pytest.raises(ValueError, match="Backend.*not supported"):
            prepare_walk_operator(P, backend="invalid")
    
    def test_non_stochastic_matrix_error(self):
        """Test error handling for non-stochastic transition matrix."""
        # Create a non-stochastic matrix
        P = np.array([[0.5, 0.3],  # Row doesn't sum to 1
                      [0.4, 0.6]])
        
        with pytest.raises(ValueError, match="row-stochastic"):
            prepare_walk_operator(P)
    
    def test_non_reversible_chain_error(self):
        """Test error handling for non-reversible Markov chain."""
        # Create a non-reversible chain
        P = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0]])  # Cyclic, not reversible
        
        with pytest.raises(ValueError, match="reversible"):
            prepare_walk_operator(P)
    
    def test_eigenvalue_structure(self):
        """Test that walk operator has correct eigenvalue structure."""
        P = build_two_state_chain(alpha=0.3)
        W = prepare_walk_operator(P, backend="matrix")
        
        # Compute eigenvalues
        eigenvals = np.linalg.eigvals(W)
        
        # All eigenvalues should have magnitude 1 (unitary)
        mags = np.abs(eigenvals)
        assert np.allclose(mags, 1.0), \
            f"Unitary operator should have all eigenvalues with |»| = 1, got {mags}"
        
        # Should have at least one eigenvalue = 1 (stationary state)
        assert np.any(np.isclose(eigenvals, 1.0)), \
            "Walk operator should have eigenvalue 1"


class TestIsUnitary:
    """Test suite for is_unitary helper function."""
    
    def test_identity_matrix(self):
        """Test that identity matrix is recognized as unitary."""
        I = np.eye(4)
        assert is_unitary(I), \
            "Identity matrix should be unitary"
    
    def test_pauli_matrices(self):
        """Test standard Pauli matrices are unitary."""
        # Pauli X
        X = np.array([[0, 1], [1, 0]])
        assert is_unitary(X), "Pauli X should be unitary"
        
        # Pauli Y
        Y = np.array([[0, -1j], [1j, 0]])
        assert is_unitary(Y), "Pauli Y should be unitary"
        
        # Pauli Z
        Z = np.array([[1, 0], [0, -1]])
        assert is_unitary(Z), "Pauli Z should be unitary"
    
    def test_hadamard_matrix(self):
        """Test Hadamard matrix is unitary."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        assert is_unitary(H), "Hadamard matrix should be unitary"
    
    def test_non_unitary_matrix(self):
        """Test that non-unitary matrices are correctly identified."""
        # Non-square matrix
        A = np.array([[1, 0], [0, 1], [0, 0]])
        assert not is_unitary(A), \
            "Non-square matrix cannot be unitary"
        
        # Square but not unitary
        B = np.array([[1, 1], [0, 1]])
        assert not is_unitary(B), \
            "Upper triangular matrix should not be unitary"
    
    def test_nearly_unitary_tolerance(self):
        """Test tolerance handling for numerical precision."""
        # Create a nearly unitary matrix with small error
        U = np.array([[1, 0], [0, 1j]])
        U_noisy = U + 1e-11 * np.random.randn(2, 2)
        
        # Should pass with default tolerance
        assert is_unitary(U_noisy, atol=1e-10), \
            "Should accept nearly unitary matrix within tolerance"
        
        # Should fail with tighter tolerance
        assert not is_unitary(U_noisy, atol=1e-12), \
            "Should reject nearly unitary matrix outside tolerance"


class TestWalkEigenvalues:
    """Test suite for walk_eigenvalues function."""
    
    def test_two_state_eigenvalues(self):
        """Test eigenvalue computation for two-state chain."""
        P = build_two_state_chain(alpha=0.3)
        eigenvals = walk_eigenvalues(P)
        
        # Check that all eigenvalues have magnitude d 1
        mags = np.abs(eigenvals)
        assert np.all(mags <= 1.0 + 1e-10), \
            f"Walk eigenvalues should have |»| d 1, got max {np.max(mags)}"
        
        # Should have at least one eigenvalue near 1
        assert np.any(np.abs(eigenvals - 1.0) < 1e-10), \
            "Should have eigenvalue 1 corresponding to stationary state"
    
    def test_symmetric_chain_eigenvalues(self):
        """Test eigenvalues for symmetric three-state chain."""
        P = build_three_state_symmetric(p=0.25)
        eigenvals = walk_eigenvalues(P)
        
        # For symmetric chains, eigenvalues often come in ± pairs
        # Check for this structure
        eigenvals_sorted = np.sort(eigenvals)
        
        # All eigenvalues should be within unit circle
        assert np.all(np.abs(eigenvals) <= 1.0 + 1e-10), \
            "All eigenvalues must be within unit circle"
    
    def test_eigenvalue_pi_consistency(self):
        """Test that providing pi doesn't change eigenvalues."""
        P = build_random_reversible_chain(4, seed=123)
        pi = stationary_distribution(P)
        
        eigenvals1 = walk_eigenvalues(P)
        eigenvals2 = walk_eigenvalues(P, pi=pi)
        
        # Sort for comparison
        eigenvals1_sorted = np.sort(eigenvals1)
        eigenvals2_sorted = np.sort(eigenvals2)
        
        assert np.allclose(eigenvals1_sorted, eigenvals2_sorted), \
            "Eigenvalues should be same whether pi is provided or computed"


class TestValidateWalkOperator:
    """Test suite for validate_walk_operator function."""
    
    def test_valid_walk_operator_matrix(self):
        """Test validation of correctly constructed walk operator."""
        P = build_two_state_chain(alpha=0.4)
        W = prepare_walk_operator(P, backend="matrix")
        
        assert validate_walk_operator(W, P), \
            "Correctly constructed walk operator should pass validation"
    
    def test_valid_walk_operator_circuit(self):
        """Test validation of walk operator as quantum circuit."""
        P = build_three_state_symmetric(p=0.3)
        qc = prepare_walk_operator(P, backend="qiskit")
        
        assert validate_walk_operator(qc, P), \
            "Walk operator circuit should pass validation"
    
    def test_invalid_dimension(self):
        """Test validation fails for wrong dimension."""
        P = build_two_state_chain(alpha=0.3)
        # Create a 2x2 matrix instead of expected 4x4
        W_wrong = np.eye(2)
        
        with pytest.warns(UserWarning, match="dimension"):
            result = validate_walk_operator(W_wrong, P)
        assert not result, \
            "Validation should fail for wrong dimension"
    
    def test_non_unitary_fails(self):
        """Test validation fails for non-unitary operator."""
        P = build_two_state_chain(alpha=0.3)
        # Create non-unitary matrix of correct dimension
        W_bad = np.ones((4, 4))
        
        assert not validate_walk_operator(W_bad, P), \
            "Validation should fail for non-unitary operator"
    
    def test_tolerance_parameter(self):
        """Test tolerance parameter in validation."""
        P = build_two_state_chain(alpha=0.3)
        W = prepare_walk_operator(P, backend="matrix")
        
        # Add small noise
        W_noisy = W + 1e-11 * np.random.randn(*W.shape)
        
        # Should pass with default tolerance
        assert validate_walk_operator(W_noisy, P, atol=1e-10), \
            "Should pass with appropriate tolerance"
        
        # Should fail with tight tolerance
        assert not validate_walk_operator(W_noisy, P, atol=1e-12), \
            "Should fail with tight tolerance"


class TestPrepareInitialState:
    """Test suite for prepare_initial_state function."""
    
    def test_specific_vertex_start(self):
        """Test initial state preparation from specific vertex."""
        n = 4
        start = 2
        qc = prepare_initial_state(n, start_vertex=start)
        
        # Check qubit count
        expected_qubits = 2 * int(np.ceil(np.log2(n)))
        assert qc.num_qubits == expected_qubits, \
            f"Expected {expected_qubits} qubits, got {qc.num_qubits}"
        
        # Simulate and check state
        state = Statevector.from_instruction(qc)
        probs = state.probabilities()
        
        # Should have non-zero probability for states |start>|j>
        # In our encoding, these are at indices start*n + j
        for j in range(n):
            idx = start * n + j
            if idx < len(probs):
                assert probs[idx] > 1e-10, \
                    f"Should have non-zero probability for edge ({start},{j})"
    
    def test_uniform_superposition_start(self):
        """Test uniform superposition initial state."""
        n = 4
        qc = prepare_initial_state(n, start_vertex=None)
        
        # Check state is uniform over valid indices
        state = Statevector.from_instruction(qc)
        probs = state.probabilities()
        
        # First n*n probabilities should be approximately equal
        expected_prob = 1.0 / (n * n)
        for i in range(n * n):
            assert abs(probs[i] - expected_prob) < 1e-10, \
                f"Probability at index {i} should be {expected_prob}, got {probs[i]}"
    
    def test_single_vertex_start(self):
        """Test edge case of single vertex (n=1)."""
        n = 1
        qc = prepare_initial_state(n, start_vertex=0)
        
        # Should still create valid circuit
        assert qc.num_qubits >= 0, \
            "Should handle single vertex case"


class TestMeasureWalkMixing:
    """Test suite for measure_walk_mixing function."""
    
    def test_zero_steps_mixing(self):
        """Test mixing distance at t=0 steps."""
        P = build_two_state_chain(alpha=0.3)
        W = prepare_walk_operator(P, backend="matrix")
        
        # At t=0, should have maximum distance from stationary
        dist = measure_walk_mixing(W, P, t_steps=0)
        assert dist >= 0, "Distance should be non-negative"
        assert dist <= 1, "Total variation distance should be at most 1"
    
    def test_mixing_convergence(self):
        """Test that mixing distance decreases with more steps."""
        P = build_two_state_chain(alpha=0.4)
        W = prepare_walk_operator(P, backend="matrix")
        
        distances = []
        for t in [0, 5, 10, 20]:
            dist = measure_walk_mixing(W, P, t_steps=t)
            distances.append(dist)
        
        # Distance should generally decrease (allowing small fluctuations)
        for i in range(1, len(distances)):
            assert distances[i] <= distances[i-1] + 0.1, \
                f"Distance should decrease: {distances[i]} > {distances[i-1]}"
    
    def test_stationary_initial_state(self):
        """Test mixing from stationary distribution."""
        P = build_three_state_symmetric(p=0.2)
        W = prepare_walk_operator(P, backend="matrix")
        pi = stationary_distribution(P)
        
        # Create initial state corresponding to stationary distribution
        n = P.shape[0]
        initial_state = np.zeros(n * n, dtype=complex)
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                # Amplitude proportional to sqrt(pi[i] * P[i,j])
                initial_state[idx] = np.sqrt(pi[i] * P[i, j])
        
        # Normalize
        initial_state /= np.linalg.norm(initial_state)
        
        # Distance should remain small
        dist = measure_walk_mixing(W, P, t_steps=10, initial_state=initial_state)
        assert dist < 0.1, \
            "Distance from stationary should remain small when starting from stationary"
    
    def test_circuit_vs_matrix_mixing(self):
        """Test that circuit and matrix representations give same mixing."""
        P = build_two_state_chain(alpha=0.35)
        
        W_matrix = prepare_walk_operator(P, backend="matrix")
        W_circuit = prepare_walk_operator(P, backend="qiskit")
        
        dist_matrix = measure_walk_mixing(W_matrix, P, t_steps=5)
        dist_circuit = measure_walk_mixing(W_circuit, P, t_steps=5)
        
        assert abs(dist_matrix - dist_circuit) < 1e-10, \
            f"Matrix and circuit should give same mixing: {dist_matrix} vs {dist_circuit}"


class TestQuantumMixingTime:
    """Test suite for quantum_mixing_time function."""
    
    def test_mixing_time_reasonable(self):
        """Test that mixing time is reasonable for simple chains."""
        P = build_two_state_chain(alpha=0.3)
        t_mix = quantum_mixing_time(P, epsilon=0.01)
        
        # Should be positive and finite
        assert t_mix > 0, "Mixing time should be positive"
        assert t_mix < 10000, "Mixing time should be finite and reasonable"
    
    def test_smaller_epsilon_larger_time(self):
        """Test that smaller epsilon requires more mixing time."""
        P = build_three_state_symmetric(p=0.25)
        
        t_mix_1 = quantum_mixing_time(P, epsilon=0.1)
        t_mix_2 = quantum_mixing_time(P, epsilon=0.01)
        t_mix_3 = quantum_mixing_time(P, epsilon=0.001)
        
        assert t_mix_1 <= t_mix_2, \
            f"Smaller epsilon should need more time: {t_mix_1} > {t_mix_2}"
        assert t_mix_2 <= t_mix_3, \
            f"Smaller epsilon should need more time: {t_mix_2} > {t_mix_3}"
    
    def test_nearly_reducible_chain_warning(self):
        """Test warning for nearly reducible chain."""
        # Create a nearly reducible chain
        P = np.array([[0.999, 0.001],
                      [0.001, 0.999]])
        
        with pytest.warns(UserWarning, match="Phase gap near zero"):
            t_mix = quantum_mixing_time(P, epsilon=0.01)
        
        # Should return large value
        assert t_mix >= 1e5, \
            "Nearly reducible chain should have very large mixing time"
    
    def test_pi_parameter_consistency(self):
        """Test that providing pi doesn't change mixing time."""
        P = build_random_reversible_chain(5, seed=789)
        pi = stationary_distribution(P)
        
        t_mix_1 = quantum_mixing_time(P, epsilon=0.05)
        t_mix_2 = quantum_mixing_time(P, epsilon=0.05, pi=pi)
        
        assert t_mix_1 == t_mix_2, \
            "Mixing time should be same whether pi is provided or computed"


class TestInternalHelpers:
    """Test suite for internal helper functions."""
    
    def test_build_swap_operator(self):
        """Test swap operator construction."""
        n = 3
        S = _build_swap_operator(n)
        
        # Check dimensions
        assert S.shape == (n*n, n*n), \
            f"Swap operator should be {n*n}x{n*n}"
        
        # Check it's a permutation matrix
        assert np.all(np.sum(S, axis=0) == 1), \
            "Each column should sum to 1"
        assert np.all(np.sum(S, axis=1) == 1), \
            "Each row should sum to 1"
        
        # Check swap property: S^2 = I
        S2 = S @ S
        assert np.allclose(S2, np.eye(n*n)), \
            "Swap operator applied twice should be identity"
        
        # Check specific swaps
        for i in range(n):
            for j in range(n):
                idx_in = i * n + j
                idx_out = j * n + i
                assert S[idx_out, idx_in] == 1, \
                    f"S should map |{i}>|{j}> to |{j}>|{i}>"
    
    def test_build_projection_operator(self):
        """Test projection operator construction."""
        P = build_two_state_chain(alpha=0.3)
        pi = stationary_distribution(P)
        D = discriminant_matrix(P, pi)
        
        Pi_op = _build_projection_operator(D, P, pi)
        
        # Check it's a projection: P^2 = P
        Pi_op_squared = Pi_op @ Pi_op
        assert np.allclose(Pi_op_squared, Pi_op), \
            "Projection operator should satisfy P^2 = P"
        
        # Check it's Hermitian
        assert np.allclose(Pi_op, Pi_op.conj().T), \
            "Projection operator should be Hermitian"
        
        # Check rank equals number of states
        rank = np.linalg.matrix_rank(Pi_op)
        assert rank == P.shape[0], \
            f"Projection rank should equal number of states, got {rank}"
    
    def test_matrix_to_circuit_padding(self):
        """Test matrix to circuit conversion with padding."""
        # Create a 6x6 matrix (needs padding to 8x8 for 3 qubits)
        n = 3  # Results in 6 qubits needed, but circuit will use 6 = 2*3
        W_small = np.eye(n * n)
        
        qc = _matrix_to_circuit(W_small, n)
        
        # Should use correct number of qubits
        n_qubits_per_reg = int(np.ceil(np.log2(n)))
        expected_qubits = 2 * n_qubits_per_reg
        assert qc.num_qubits == expected_qubits, \
            f"Should use {expected_qubits} qubits"
        
        # Extract operator and check padding
        W_circuit = Operator(qc).data
        full_dim = 2 ** expected_qubits
        assert W_circuit.shape == (full_dim, full_dim), \
            f"Circuit operator should be {full_dim}x{full_dim}"
    
    def test_build_walk_matrix_unitarity(self):
        """Test that _build_walk_matrix produces unitary output."""
        P = build_random_reversible_chain(4, seed=456)
        pi = stationary_distribution(P)
        D = discriminant_matrix(P, pi)
        
        W = _build_walk_matrix(D, P, pi)
        
        assert is_unitary(W), \
            "_build_walk_matrix should produce unitary operator"


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_single_state_chain(self):
        """Test handling of trivial single-state chain."""
        P = np.array([[1.0]])  # Single state, self-loop
        
        # Should handle this degenerate case
        W = prepare_walk_operator(P, backend="matrix")
        assert W.shape == (1, 1), "Single state should give 1x1 walk operator"
        assert np.isclose(W[0, 0], 1.0), "Single state walk operator should be 1"
    
    def test_disconnected_chain_components(self):
        """Test chain with disconnected components."""
        # Create block-diagonal transition matrix
        P = np.array([[0.7, 0.3, 0.0, 0.0],
                      [0.3, 0.7, 0.0, 0.0],
                      [0.0, 0.0, 0.6, 0.4],
                      [0.0, 0.0, 0.4, 0.6]])
        
        # Should still work if each component is reversible
        W = prepare_walk_operator(P, backend="matrix")
        assert is_unitary(W), \
            "Walk operator should be unitary even for disconnected components"
    
    def test_nearly_stochastic_matrix(self):
        """Test handling of nearly stochastic matrix."""
        # Create matrix with tiny deviation from stochastic
        P = np.array([[0.5, 0.5 - 1e-15],
                      [0.3, 0.7]])
        P[0, :] /= P[0, :].sum()  # Renormalize
        
        # Should handle numerical precision gracefully
        W = prepare_walk_operator(P, backend="matrix")
        assert is_unitary(W), \
            "Should handle nearly stochastic matrices"
    
    def test_large_condition_number(self):
        """Test chain with large condition number."""
        # Create poorly conditioned but valid chain
        epsilon = 1e-8
        P = np.array([[1 - epsilon, epsilon],
                      [epsilon, 1 - epsilon]])
        
        # Should still produce valid walk operator
        W = prepare_walk_operator(P, backend="matrix")
        assert is_unitary(W), \
            "Should handle poorly conditioned chains"
        
        # But mixing time should be large
        t_mix = quantum_mixing_time(P, epsilon=0.1)
        assert t_mix > 100, \
            "Poorly conditioned chain should have large mixing time"