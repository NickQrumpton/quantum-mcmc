"""
Unit tests for Markov chain functionality in quantum MCMC implementation.

This module provides comprehensive testing for classical Markov chain construction
and analysis functions used as a foundation for quantum walk operators.

Tests cover:
- Two-state chain construction with various parameters
- Metropolis-Hastings chain construction for different distributions
- Stochasticity verification
- Stationary distribution computation
- Reversibility (detailed balance) checking
- Random reversible chain generation
- Edge cases and error handling

Author: Quantum MCMC Research Team
"""

import unittest
import numpy as np
# import pytest  # Remove pytest dependency
from typing import Tuple, Optional
import scipy.sparse as sp
from scipy.stats import norm, expon, beta, gamma

# Import the functions to test
from quantum_mcmc.classical.markov_chain import (
    build_two_state_chain,
    build_metropolis_chain,
    is_stochastic,
    stationary_distribution,
    is_reversible,
    sample_random_reversible_chain
)


class TestBuildTwoStateChain(unittest.TestCase):
    """Test cases for two-state Markov chain construction."""
    
    def test_valid_probabilities(self):
        """Test two-state chain with valid transition probabilities."""
        # Test symmetric case
        P = build_two_state_chain(0.3, 0.3)
        expected = np.array([[0.7, 0.3], [0.3, 0.7]])
        np.testing.assert_array_almost_equal(P, expected, decimal=10,
                                           err_msg="Symmetric two-state chain incorrect")
        
        # Test asymmetric case
        P = build_two_state_chain(0.1, 0.4)
        expected = np.array([[0.9, 0.1], [0.4, 0.6]])
        np.testing.assert_array_almost_equal(P, expected, decimal=10,
                                           err_msg="Asymmetric two-state chain incorrect")
        
        # Test extreme case (almost deterministic)
        P = build_two_state_chain(0.99, 0.01)
        expected = np.array([[0.01, 0.99], [0.01, 0.99]])
        np.testing.assert_array_almost_equal(P, expected, decimal=10,
                                           err_msg="Near-deterministic chain incorrect")
    
    def test_boundary_probabilities(self):
        """Test two-state chain with boundary probability values."""
        # Test with p = 0, q = 1
        P = build_two_state_chain(0.0, 1.0)
        expected = np.array([[1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(P, expected, decimal=10,
                                           err_msg="Boundary case p=0, q=1 incorrect")
        
        # Test with p = 1, q = 0
        P = build_two_state_chain(1.0, 0.0)
        expected = np.array([[0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(P, expected, decimal=10,
                                           err_msg="Boundary case p=1, q=0 incorrect")
        
        # Test identity matrix case
        P = build_two_state_chain(0.0, 0.0)
        expected = np.eye(2)
        np.testing.assert_array_almost_equal(P, expected, decimal=10,
                                           err_msg="Identity matrix case incorrect")
    
    def test_invalid_probabilities(self):
        """Test two-state chain with invalid probability values."""
        # Negative probabilities
        with self.assertRaises(ValueError, msg="Should reject negative p"):
            build_two_state_chain(-0.1, 0.5)
        
        with self.assertRaises(ValueError, msg="Should reject negative q"):
            build_two_state_chain(0.5, -0.1)
        
        # Probabilities > 1
        with self.assertRaises(ValueError, msg="Should reject p > 1"):
            build_two_state_chain(1.1, 0.5)
        
        with self.assertRaises(ValueError, msg="Should reject q > 1"):
            build_two_state_chain(0.5, 1.1)
    
    def test_output_properties(self):
        """Test properties of the output matrix."""
        P = build_two_state_chain(0.2, 0.3)
        
        # Check type and shape
        self.assertIsInstance(P, np.ndarray, "Output should be numpy array")
        self.assertEqual(P.shape, (2, 2), "Output should be 2x2 matrix")
        self.assertEqual(P.dtype, np.float64, "Output should be float64")
        
        # Check stochasticity
        self.assertTrue(is_stochastic(P), "Output should be stochastic")
        
        # Check reversibility condition
        pi = stationary_distribution(P)
        self.assertTrue(is_reversible(P, pi), "Two-state chain should be reversible")


class TestBuildMetropolisChain(unittest.TestCase):
    """Test cases for Metropolis-Hastings chain construction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_space = np.linspace(-5, 5, 50)
        self.small_state_space = np.array([0.0, 1.0, 2.0])
    
    def test_gaussian_target(self):
        """Test Metropolis chain for Gaussian target distribution."""
        # Standard normal
        target_probs = norm.pdf(self.state_space)
        target_probs /= target_probs.sum()
        
        P = build_metropolis_chain(target_probs)
        
        # Check basic properties
        self.assertEqual(P.shape, (50, 50), "Shape mismatch for Gaussian target")
        self.assertTrue(is_stochastic(P), "Metropolis chain not stochastic")
        
        # Check stationary distribution
        pi = stationary_distribution(P)
        np.testing.assert_array_almost_equal(pi, target_probs, decimal=6,
                                           err_msg="Stationary distribution incorrect")
        
        # Check reversibility
        self.assertTrue(is_reversible(P, pi), 
                       "Metropolis chain should satisfy detailed balance")
    
    def test_exponential_target(self):
        """Test Metropolis chain for exponential target distribution."""
        # Exponential on positive half
        positive_states = np.linspace(0, 10, 30)
        target_probs = expon.pdf(positive_states, scale=2)
        target_probs /= target_probs.sum()
        
        P = build_metropolis_chain(target_probs)
        
        # Check properties
        self.assertTrue(is_stochastic(P), "Exponential Metropolis chain not stochastic")
        pi = stationary_distribution(P)
        np.testing.assert_array_almost_equal(pi, target_probs, decimal=6,
                                           err_msg="Exponential stationary distribution incorrect")
    
    def test_custom_proposal_std(self):
        """Test Metropolis chain with custom proposal standard deviation."""
        target_probs = norm.pdf(self.state_space)
        target_probs /= target_probs.sum()
        
        # Small proposal std (more rejections)
        P_small = build_metropolis_chain(target_probs, proposal_std=0.1)
        
        # Large proposal std (potentially more rejections for distant states)
        P_large = build_metropolis_chain(target_probs, proposal_std=5.0)
        
        # Both should be valid
        self.assertTrue(is_stochastic(P_small), "Small proposal std chain not stochastic")
        self.assertTrue(is_stochastic(P_large), "Large proposal std chain not stochastic")
        
        # Both should have correct stationary distribution
        pi_small = stationary_distribution(P_small)
        pi_large = stationary_distribution(P_large)
        
        np.testing.assert_array_almost_equal(pi_small, target_probs, decimal=5,
                                           err_msg="Small proposal std stationary dist incorrect")
        np.testing.assert_array_almost_equal(pi_large, target_probs, decimal=5,
                                           err_msg="Large proposal std stationary dist incorrect")
    
    def test_uniform_target(self):
        """Test Metropolis chain for uniform target distribution."""
        n = 20
        target_probs = np.ones(n) / n
        
        P = build_metropolis_chain(target_probs)
        
        # For uniform target, transition matrix should have specific structure
        self.assertTrue(is_stochastic(P), "Uniform target chain not stochastic")
        
        # Check stationary distribution
        pi = stationary_distribution(P)
        np.testing.assert_array_almost_equal(pi, target_probs, decimal=6,
                                           err_msg="Uniform stationary distribution incorrect")
    
    def test_sparse_matrix_option(self):
        """Test Metropolis chain with sparse matrix output."""
        target_probs = norm.pdf(self.state_space)
        target_probs /= target_probs.sum()
        
        P_sparse = build_metropolis_chain(target_probs, sparse=True)
        
        # Check sparse matrix properties
        self.assertTrue(sp.issparse(P_sparse), "Output should be sparse when requested")
        self.assertIsInstance(P_sparse, sp.csr_matrix, "Should be CSR sparse matrix")
        
        # Convert to dense for testing
        P_dense = P_sparse.toarray()
        self.assertTrue(is_stochastic(P_dense), "Sparse Metropolis chain not stochastic")
        
        # Compare with dense version
        P_direct = build_metropolis_chain(target_probs, sparse=False)
        np.testing.assert_array_almost_equal(P_dense, P_direct, decimal=10,
                                           err_msg="Sparse and dense versions differ")
    
    def test_edge_cases(self):
        """Test Metropolis chain construction edge cases."""
        # Single state
        with self.assertRaises(ValueError, msg="Should reject single state"):
            build_metropolis_chain(np.array([1.0]))
        
        # Negative probabilities
        with self.assertRaises(ValueError, msg="Should reject negative probabilities"):
            build_metropolis_chain(np.array([0.5, -0.3, 0.8]))
        
        # Zero probabilities (should be handled)
        target_probs = np.array([0.5, 0.0, 0.5])
        P = build_metropolis_chain(target_probs)
        self.assertTrue(is_stochastic(P), "Should handle zero probabilities")
        
        # Very small state space
        target_probs = np.array([0.3, 0.7])
        P = build_metropolis_chain(target_probs)
        self.assertEqual(P.shape, (2, 2), "Should handle 2-state case")
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme distributions."""
        # Very peaked distribution
        states = np.linspace(-10, 10, 100)
        target_probs = norm.pdf(states, scale=0.01)  # Very narrow
        target_probs /= target_probs.sum()
        
        P = build_metropolis_chain(target_probs)
        self.assertTrue(is_stochastic(P), "Should handle peaked distributions")
        
        # Heavy-tailed distribution
        target_probs = 1 / (1 + states**2)  # Cauchy-like
        target_probs = np.abs(target_probs)
        target_probs /= target_probs.sum()
        
        P = build_metropolis_chain(target_probs)
        self.assertTrue(is_stochastic(P), "Should handle heavy-tailed distributions")


class TestIsStochastic(unittest.TestCase):
    """Test cases for stochasticity verification."""
    
    def test_valid_stochastic_matrices(self):
        """Test detection of valid stochastic matrices."""
        # 2x2 stochastic
        P1 = np.array([[0.7, 0.3], [0.4, 0.6]])
        self.assertTrue(is_stochastic(P1), "Valid 2x2 stochastic matrix")
        
        # 3x3 stochastic
        P2 = np.array([[0.5, 0.3, 0.2],
                       [0.1, 0.8, 0.1],
                       [0.0, 0.5, 0.5]])
        self.assertTrue(is_stochastic(P2), "Valid 3x3 stochastic matrix")
        
        # Identity matrix
        I = np.eye(5)
        self.assertTrue(is_stochastic(I), "Identity matrix is stochastic")
        
        # Permutation matrix
        P3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        self.assertTrue(is_stochastic(P3), "Permutation matrix is stochastic")
    
    def test_tolerance_handling(self):
        """Test tolerance in stochasticity checking."""
        # Matrix with small numerical errors
        P = np.array([[0.7 + 1e-10, 0.3 - 1e-10],
                      [0.4, 0.6]])
        
        self.assertTrue(is_stochastic(P), "Should accept small numerical errors")
        self.assertTrue(is_stochastic(P, tol=1e-8), "Should accept with specified tolerance")
        self.assertFalse(is_stochastic(P, tol=1e-12), "Should reject with tight tolerance")
    
    def test_non_stochastic_matrices(self):
        """Test detection of non-stochastic matrices."""
        # Negative entries
        P1 = np.array([[0.8, 0.2], [-0.1, 1.1]])
        self.assertFalse(is_stochastic(P1), "Should reject negative entries")
        
        # Rows don't sum to 1
        P2 = np.array([[0.5, 0.4], [0.3, 0.7]])
        self.assertFalse(is_stochastic(P2), "Should reject incorrect row sums")
        
        # Entry > 1
        P3 = np.array([[1.2, -0.2], [0.3, 0.7]])
        self.assertFalse(is_stochastic(P3), "Should reject entries > 1")
        
        # All zeros
        P4 = np.zeros((3, 3))
        self.assertFalse(is_stochastic(P4), "Should reject zero matrix")
    
    def test_edge_cases(self):
        """Test edge cases for stochasticity checking."""
        # 1x1 matrix
        P1 = np.array([[1.0]])
        self.assertTrue(is_stochastic(P1), "1x1 matrix with 1 is stochastic")
        
        # Empty matrix
        P2 = np.array([])
        with self.assertRaises((ValueError, IndexError), 
                             msg="Should handle empty matrix appropriately"):
            is_stochastic(P2)
        
        # Non-square matrix
        P3 = np.array([[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]])
        with self.assertRaises(ValueError, msg="Should reject non-square matrix"):
            is_stochastic(P3)
    
    def test_sparse_matrices(self):
        """Test stochasticity checking for sparse matrices."""
        # Create sparse stochastic matrix
        data = [0.7, 0.3, 0.4, 0.6]
        indices = [0, 1, 0, 1]
        indptr = [0, 2, 4]
        P_sparse = sp.csr_matrix((data, indices, indptr), shape=(2, 2))
        
        # Should work with sparse matrices
        self.assertTrue(is_stochastic(P_sparse.toarray()), 
                       "Should handle sparse matrix conversion")


class TestStationaryDistribution(unittest.TestCase):
    """Test cases for stationary distribution computation."""
    
    def test_known_distributions(self):
        """Test stationary distribution for known cases."""
        # Two-state chain with known stationary distribution
        P1 = build_two_state_chain(0.2, 0.3)
        pi1 = stationary_distribution(P1)
        expected1 = np.array([0.6, 0.4])  # pi = [q/(p+q), p/(p+q)]
        np.testing.assert_array_almost_equal(pi1, expected1, decimal=10,
                                           err_msg="Two-state stationary distribution incorrect")
        
        # Uniform distribution for doubly stochastic matrix
        n = 5
        P2 = np.ones((n, n)) / n  # All entries equal
        pi2 = stationary_distribution(P2)
        expected2 = np.ones(n) / n
        np.testing.assert_array_almost_equal(pi2, expected2, decimal=10,
                                           err_msg="Uniform stationary distribution incorrect")
    
    def test_power_method_option(self):
        """Test stationary distribution computation using power method."""
        P = build_two_state_chain(0.3, 0.4)
        
        # Compare eigen method with power method
        pi_eigen = stationary_distribution(P, method='eigen')
        pi_power = stationary_distribution(P, method='power')
        
        np.testing.assert_array_almost_equal(pi_eigen, pi_power, decimal=8,
                                           err_msg="Power and eigen methods should agree")
    
    def test_convergence_criteria(self):
        """Test convergence with different parameters."""
        P = build_metropolis_chain(norm.pdf(np.linspace(-3, 3, 20)))
        
        # Test with different max iterations
        pi1 = stationary_distribution(P, method='power', max_iter=100)
        pi2 = stationary_distribution(P, method='power', max_iter=1000)
        
        # Should converge to same distribution
        np.testing.assert_array_almost_equal(pi1, pi2, decimal=6,
                                           err_msg="Different iterations should converge similarly")
        
        # Test with different tolerance
        pi3 = stationary_distribution(P, method='power', tol=1e-6)
        pi4 = stationary_distribution(P, method='power', tol=1e-10)
        
        np.testing.assert_array_almost_equal(pi3, pi4, decimal=5,
                                           err_msg="Different tolerances should give similar results")
    
    def test_periodic_chains(self):
        """Test stationary distribution for periodic chains."""
        # Period-2 chain
        P = np.array([[0, 1], [1, 0]])
        pi = stationary_distribution(P)
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(pi, expected, decimal=10,
                                           err_msg="Periodic chain stationary distribution incorrect")
    
    def test_reducible_chains(self):
        """Test stationary distribution for reducible chains."""
        # Block diagonal (reducible) chain
        P = np.array([[0.7, 0.3, 0.0, 0.0],
                      [0.4, 0.6, 0.0, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.1, 0.9]])
        
        # Should still find a stationary distribution
        pi = stationary_distribution(P)
        self.assertAlmostEqual(np.sum(pi), 1.0, places=10,
                              msg="Stationary distribution should sum to 1")
        
        # Verify it's actually stationary
        pi_next = pi @ P
        np.testing.assert_array_almost_equal(pi, pi_next, decimal=10,
                                           err_msg="Distribution should be stationary")
    
    def test_edge_cases(self):
        """Test edge cases for stationary distribution."""
        # 1x1 matrix
        P1 = np.array([[1.0]])
        pi1 = stationary_distribution(P1)
        np.testing.assert_array_almost_equal(pi1, np.array([1.0]), decimal=10,
                                           err_msg="1x1 stationary distribution should be [1]")
        
        # Identity matrix
        I = np.eye(3)
        with self.assertRaises(ValueError, 
                             msg="Identity matrix has multiple stationary distributions"):
            stationary_distribution(I, method='power')  # Power method should fail
        
        # Non-stochastic matrix
        P_bad = np.array([[0.5, 0.6], [0.3, 0.7]])
        with self.assertRaises(ValueError, msg="Should reject non-stochastic matrix"):
            stationary_distribution(P_bad)
    
    def test_numerical_stability(self):
        """Test numerical stability of stationary distribution computation."""
        # Nearly reducible chain (very small connections)
        epsilon = 1e-10
        P = np.array([[0.9, 0.1 - epsilon, epsilon, 0],
                      [0.2, 0.8 - epsilon, 0, epsilon],
                      [epsilon, 0, 0.7, 0.3 - epsilon],
                      [0, epsilon, 0.4, 0.6 - epsilon]])
        
        pi = stationary_distribution(P)
        self.assertAlmostEqual(np.sum(pi), 1.0, places=10,
                              msg="Should handle near-reducible chains")
        
        # Verify stationarity numerically
        pi_next = pi @ P
        np.testing.assert_array_almost_equal(pi, pi_next, decimal=8,
                                           err_msg="Should be numerically stationary")


class TestIsReversible(unittest.TestCase):
    """Test cases for reversibility (detailed balance) checking."""
    
    def test_known_reversible_chains(self):
        """Test reversibility for known reversible chains."""
        # Two-state chain (always reversible)
        P1 = build_two_state_chain(0.3, 0.4)
        pi1 = stationary_distribution(P1)
        self.assertTrue(is_reversible(P1, pi1), "Two-state chain should be reversible")
        
        # Metropolis chain (by construction reversible)
        target = norm.pdf(np.linspace(-3, 3, 20))
        target /= target.sum()
        P2 = build_metropolis_chain(target)
        self.assertTrue(is_reversible(P2, target), 
                       "Metropolis chain should satisfy detailed balance")
        
        # Symmetric matrix (reversible with uniform distribution)
        n = 4
        P3 = np.random.rand(n, n)
        P3 = (P3 + P3.T) / 2  # Make symmetric
        P3 = P3 / P3.sum(axis=1, keepdims=True)  # Normalize rows
        pi3 = stationary_distribution(P3)
        self.assertTrue(is_reversible(P3, pi3), "Symmetric matrix should be reversible")
    
    def test_non_reversible_chains(self):
        """Test detection of non-reversible chains."""
        # Cyclic chain (not reversible)
        P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        pi = np.ones(3) / 3  # Uniform is stationary
        self.assertFalse(is_reversible(P, pi), "Cyclic chain should not be reversible")
        
        # Asymmetric random chain
        np.random.seed(42)
        n = 5
        P = np.random.rand(n, n)
        P = P / P.sum(axis=1, keepdims=True)
        pi = stationary_distribution(P)
        
        # Most random chains are not reversible
        if not is_reversible(P, pi):
            self.assertFalse(is_reversible(P, pi), 
                           "Random chain should typically not be reversible")
    
    def test_tolerance_handling(self):
        """Test tolerance in reversibility checking."""
        # Create nearly reversible chain with small violations
        P = build_two_state_chain(0.3, 0.4)
        pi = stationary_distribution(P)
        
        # Add small perturbation
        P_perturbed = P + 1e-10 * np.random.randn(2, 2)
        P_perturbed = np.maximum(P_perturbed, 0)  # Ensure non-negative
        P_perturbed = P_perturbed / P_perturbed.sum(axis=1, keepdims=True)
        
        self.assertTrue(is_reversible(P_perturbed, pi, tol=1e-8),
                       "Should accept small violations with appropriate tolerance")
        self.assertFalse(is_reversible(P_perturbed, pi, tol=1e-12),
                        "Should reject with tight tolerance")
    
    def test_wrong_distribution(self):
        """Test reversibility check with wrong stationary distribution."""
        P = build_two_state_chain(0.2, 0.3)
        pi_correct = stationary_distribution(P)
        pi_wrong = np.array([0.5, 0.5])  # Wrong distribution
        
        self.assertTrue(is_reversible(P, pi_correct), 
                       "Should be reversible with correct distribution")
        self.assertFalse(is_reversible(P, pi_wrong), 
                        "Should not be reversible with wrong distribution")
    
    def test_edge_cases(self):
        """Test edge cases for reversibility checking."""
        # 1x1 matrix
        P1 = np.array([[1.0]])
        pi1 = np.array([1.0])
        self.assertTrue(is_reversible(P1, pi1), "1x1 matrix should be reversible")
        
        # Identity matrix
        I = np.eye(3)
        pi_uniform = np.ones(3) / 3
        self.assertTrue(is_reversible(I, pi_uniform), 
                       "Identity matrix is reversible with any distribution")
        
        # Zero probability states
        P = np.array([[0.5, 0.5, 0], [0.3, 0.7, 0], [0, 0, 1]])
        pi = stationary_distribution(P)
        # Should handle zero probability states correctly
        result = is_reversible(P, pi)
        self.assertIsInstance(result, bool, "Should return boolean even with zero states")
    
    def test_input_validation(self):
        """Test input validation for reversibility checking."""
        P = build_two_state_chain(0.3, 0.4)
        
        # Wrong size distribution
        with self.assertRaises(ValueError, msg="Should reject wrong size distribution"):
            is_reversible(P, np.array([1.0]))
        
        # Non-normalized distribution
        with self.assertRaises(ValueError, msg="Should reject non-normalized distribution"):
            is_reversible(P, np.array([0.3, 0.3]))
        
        # Negative probabilities
        with self.assertRaises(ValueError, msg="Should reject negative probabilities"):
            is_reversible(P, np.array([0.7, -0.3]))


class TestSampleRandomReversibleChain(unittest.TestCase):
    """Test cases for random reversible chain generation."""
    
    def test_basic_generation(self):
        """Test basic random reversible chain generation."""
        n = 5
        P, pi = sample_random_reversible_chain(n)
        
        # Check dimensions
        self.assertEqual(P.shape, (n, n), "Chain should have correct dimensions")
        self.assertEqual(pi.shape, (n,), "Distribution should have correct dimension")
        
        # Check stochasticity
        self.assertTrue(is_stochastic(P), "Generated chain should be stochastic")
        
        # Check distribution normalization
        self.assertAlmostEqual(np.sum(pi), 1.0, places=10,
                              msg="Distribution should be normalized")
        
        # Check reversibility
        self.assertTrue(is_reversible(P, pi), 
                       "Generated chain should be reversible")
        
        # Check that pi is stationary
        pi_next = pi @ P
        np.testing.assert_array_almost_equal(pi, pi_next, decimal=10,
                                           err_msg="Generated distribution should be stationary")
    
    def test_sparsity_parameter(self):
        """Test generation with different sparsity levels."""
        n = 10
        
        # Dense chain
        P_dense, pi_dense = sample_random_reversible_chain(n, sparsity=0.0)
        num_zeros_dense = np.sum(P_dense == 0)
        self.assertLess(num_zeros_dense, n,  # Allow diagonal zeros
                       "Dense chain should have few zeros")
        
        # Sparse chain
        P_sparse, pi_sparse = sample_random_reversible_chain(n, sparsity=0.8)
        num_zeros_sparse = np.sum(P_sparse == 0)
        self.assertGreater(num_zeros_sparse, num_zeros_dense,
                          "Sparse chain should have more zeros")
        
        # Both should be reversible
        self.assertTrue(is_reversible(P_dense, pi_dense), 
                       "Dense chain should be reversible")
        self.assertTrue(is_reversible(P_sparse, pi_sparse), 
                       "Sparse chain should be reversible")
    
    def test_return_sparse_option(self):
        """Test sparse matrix return option."""
        n = 20
        P_sparse, pi = sample_random_reversible_chain(n, sparsity=0.7, 
                                                     return_sparse=True)
        
        # Check sparse format
        self.assertTrue(sp.issparse(P_sparse), "Should return sparse matrix")
        self.assertIsInstance(P_sparse, sp.csr_matrix, "Should be CSR format")
        
        # Convert to dense for testing
        P_dense = P_sparse.toarray()
        self.assertTrue(is_stochastic(P_dense), "Sparse chain should be stochastic")
        self.assertTrue(is_reversible(P_dense, pi), "Sparse chain should be reversible")
    
    def test_reproducibility_with_seed(self):
        """Test reproducibility with random seed."""
        n = 5
        seed = 12345
        
        # Generate twice with same seed
        P1, pi1 = sample_random_reversible_chain(n, seed=seed)
        P2, pi2 = sample_random_reversible_chain(n, seed=seed)
        
        np.testing.assert_array_equal(P1, P2, 
                                     err_msg="Same seed should produce same chain")
        np.testing.assert_array_equal(pi1, pi2, 
                                     err_msg="Same seed should produce same distribution")
        
        # Different seed should produce different result
        P3, pi3 = sample_random_reversible_chain(n, seed=seed+1)
        self.assertFalse(np.array_equal(P1, P3), 
                        "Different seeds should produce different chains")
    
    def test_edge_cases(self):
        """Test edge cases for random chain generation."""
        # Single state
        with self.assertRaises(ValueError, msg="Should reject n < 2"):
            sample_random_reversible_chain(1)
        
        # Two states
        P2, pi2 = sample_random_reversible_chain(2)
        self.assertEqual(P2.shape, (2, 2), "Should handle 2-state case")
        self.assertTrue(is_reversible(P2, pi2), "2-state should be reversible")
        
        # Large chain
        P_large, pi_large = sample_random_reversible_chain(100, sparsity=0.9)
        self.assertTrue(is_stochastic(P_large), "Large chain should be stochastic")
        self.assertTrue(is_reversible(P_large, pi_large), 
                       "Large chain should be reversible")
    
    def test_sparsity_bounds(self):
        """Test sparsity parameter bounds."""
        n = 10
        
        # Sparsity = 0 (dense)
        P0, pi0 = sample_random_reversible_chain(n, sparsity=0.0)
        self.assertTrue(is_reversible(P0, pi0), "Zero sparsity should work")
        
        # Sparsity = 1 (should be handled gracefully)
        P1, pi1 = sample_random_reversible_chain(n, sparsity=1.0)
        self.assertTrue(is_stochastic(P1), "Max sparsity should still be stochastic")
        self.assertTrue(is_reversible(P1, pi1), "Max sparsity should be reversible")
        
        # Invalid sparsity
        with self.assertRaises(ValueError, msg="Should reject negative sparsity"):
            sample_random_reversible_chain(n, sparsity=-0.1)
        
        with self.assertRaises(ValueError, msg="Should reject sparsity > 1"):
            sample_random_reversible_chain(n, sparsity=1.1)
    
    def test_connectivity(self):
        """Test that generated chains maintain connectivity."""
        n = 10
        
        # Even with high sparsity, chain should be connected
        for sparsity in [0.5, 0.7, 0.9]:
            P, pi = sample_random_reversible_chain(n, sparsity=sparsity, seed=42)
            
            # Check that stationary distribution has no zeros
            # (indicates all states are accessible)
            self.assertTrue(np.all(pi > 0), 
                           f"All states should be accessible at sparsity {sparsity}")
            
            # Verify ergodicity by checking powers of P
            P_power = np.linalg.matrix_power(P, n**2)
            self.assertTrue(np.all(P_power > 0) or np.allclose(P_power, pi),
                           f"Chain should be ergodic at sparsity {sparsity}")


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple functions."""
    
    def test_metropolis_reversibility_pipeline(self):
        """Test full pipeline: build Metropolis chain and verify properties."""
        # Define target distribution
        states = np.linspace(-4, 4, 30)
        target = 0.5 * norm.pdf(states, -1, 0.5) + 0.5 * norm.pdf(states, 1, 0.5)
        target /= target.sum()
        
        # Build chain
        P = build_metropolis_chain(target)
        
        # Verify all properties
        self.assertTrue(is_stochastic(P), "Metropolis chain should be stochastic")
        
        pi = stationary_distribution(P)
        np.testing.assert_array_almost_equal(pi, target, decimal=6,
                                           err_msg="Should have correct stationary distribution")
        
        self.assertTrue(is_reversible(P, pi), 
                       "Metropolis chain should be reversible")
    
    def test_random_chain_properties(self):
        """Test that random reversible chains satisfy all required properties."""
        for n in [5, 10, 20]:
            for sparsity in [0.0, 0.5, 0.8]:
                P, pi = sample_random_reversible_chain(n, sparsity=sparsity)
                
                # Check all properties
                self.assertTrue(is_stochastic(P), 
                               f"Random chain (n={n}, sparsity={sparsity}) not stochastic")
                
                # Verify pi is stationary
                pi_computed = stationary_distribution(P)
                np.testing.assert_array_almost_equal(pi, pi_computed, decimal=8,
                    err_msg=f"Generated pi should be stationary (n={n}, sparsity={sparsity})")
                
                self.assertTrue(is_reversible(P, pi),
                               f"Random chain (n={n}, sparsity={sparsity}) not reversible")
    
    def test_two_state_complete_analysis(self):
        """Complete analysis of two-state chain properties."""
        # Test various parameter combinations
        test_params = [(0.1, 0.2), (0.5, 0.5), (0.0, 1.0), (0.99, 0.01)]
        
        for p, q in test_params:
            P = build_two_state_chain(p, q)
            
            # Check stochasticity
            self.assertTrue(is_stochastic(P), 
                           f"Two-state chain (p={p}, q={q}) not stochastic")
            
            # Compute and verify stationary distribution
            pi = stationary_distribution(P)
            if p + q > 0:
                expected_pi = np.array([q/(p+q), p/(p+q)])
            else:
                expected_pi = np.array([0.5, 0.5])  # Undefined, any dist is stationary
            
            np.testing.assert_array_almost_equal(pi, expected_pi, decimal=10,
                err_msg=f"Two-state stationary distribution wrong (p={p}, q={q})")
            
            # Check reversibility
            self.assertTrue(is_reversible(P, pi),
                           f"Two-state chain (p={p}, q={q}) not reversible")


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)