"""
Unit tests for discriminant matrix functionality in quantum MCMC implementation.

This module provides comprehensive testing for discriminant matrix construction
and spectral analysis functions used in Szegedy quantum walk construction.

Tests cover:
- Discriminant matrix construction for reversible chains
- Singular value decomposition and spectral analysis
- Phase gap and spectral gap computation
- Validation and error handling
- Numerical stability and edge cases

Author: Quantum MCMC Research Team
"""

import unittest
import numpy as np
from typing import Optional
import warnings

# Import the functions to test
from quantum_mcmc.classical.markov_chain import (
    build_two_state_chain,
    build_metropolis_chain,
    stationary_distribution,
    is_reversible,
    sample_random_reversible_chain
)

from quantum_mcmc.classical.discriminant import (
    discriminant_matrix,
    singular_values,
    spectral_gap,
    phase_gap,
    validate_discriminant,
    classical_spectral_gap,
    mixing_time_bound,
    effective_dimension,
    condition_number,
    spectral_analysis
)


class TestDiscriminantMatrix(unittest.TestCase):
    """Test cases for discriminant matrix construction."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Simple two-state chain
        self.P_simple = build_two_state_chain(0.3, 0.4)
        self.pi_simple = stationary_distribution(self.P_simple)
        
        # Symmetric chain
        self.P_symmetric = build_two_state_chain(0.3, 0.3)
        self.pi_symmetric = stationary_distribution(self.P_symmetric)
        
        # Metropolis chain for normal distribution
        states = np.linspace(-3, 3, 20)
        target = np.exp(-0.5 * states**2)
        target /= target.sum()
        self.P_metropolis = build_metropolis_chain(target)
        self.pi_metropolis = target
    
    def test_basic_construction(self):
        """Test basic discriminant matrix construction."""
        D = discriminant_matrix(self.P_simple, self.pi_simple)
        
        # Check dimensions
        self.assertEqual(D.shape, self.P_simple.shape, "Discriminant matrix should have same shape as P")
        
        # Check symmetry
        np.testing.assert_allclose(D, D.T, rtol=1e-10, 
                                 err_msg="Discriminant matrix should be symmetric")
        
        # Check range [0, 1]
        self.assertTrue(np.all(D >= -1e-10), "All entries should be non-negative")
        self.assertTrue(np.all(D <= 1 + 1e-10), "All entries should be ≤ 1")
    
    def test_symmetric_chain(self):
        """Test discriminant matrix for symmetric chain."""
        D = discriminant_matrix(self.P_symmetric, self.pi_symmetric)
        
        # For symmetric chain with uniform stationary distribution,
        # D should equal P
        np.testing.assert_allclose(D, self.P_symmetric, rtol=1e-10,
                                 err_msg="For symmetric chain, D should equal P")
    
    def test_reversibility_preservation(self):
        """Test that discriminant matrix is symmetric (which implies reversibility structure)."""
        D = discriminant_matrix(self.P_simple, self.pi_simple)
        
        # The discriminant matrix should be symmetric
        # This is the key property for Szegedy walks
        np.testing.assert_allclose(D, D.T, rtol=1e-10,
                                 err_msg="Discriminant matrix should be symmetric")
        
        # Also verify the construction is correct: D[i,j] = sqrt(P[i,j] * P[j,i])
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if self.P_simple[i, j] > 0 and self.P_simple[j, i] > 0:
                    expected = np.sqrt(self.P_simple[i, j] * self.P_simple[j, i])
                    self.assertAlmostEqual(D[i, j], expected, places=10,
                                         msg=f"Discriminant formula incorrect at ({i},{j})")
                elif i == j:
                    expected = self.P_simple[i, j]
                    self.assertAlmostEqual(D[i, j], expected, places=10,
                                         msg=f"Diagonal entry incorrect at ({i},{j})")
    
    def test_spectral_properties(self):
        """Test spectral properties of discriminant matrix."""
        D = discriminant_matrix(self.P_simple, self.pi_simple)
        
        # Compute singular values
        sigma = singular_values(D)
        
        # Largest singular value should be 1
        self.assertAlmostEqual(sigma[0], 1.0, places=6,
                             msg="Largest singular value should be 1")
        
        # All singular values should be in [0, 1]
        self.assertTrue(np.all(sigma >= -1e-10), "All singular values should be non-negative")
        self.assertTrue(np.all(sigma <= 1 + 1e-10), "All singular values should be ≤ 1")
        
        # Should be sorted in descending order
        self.assertTrue(np.all(np.diff(sigma) <= 1e-10), 
                       "Singular values should be sorted in descending order")
    
    def test_input_validation(self):
        """Test input validation for discriminant matrix."""
        # Non-stochastic matrix
        P_bad = np.array([[0.5, 0.4], [0.3, 0.7]])
        with self.assertRaises(ValueError, msg="Should reject non-stochastic matrix"):
            discriminant_matrix(P_bad)
        
        # Non-reversible matrix
        P_irreversible = np.array([[0, 1], [1, 0]])  # Cyclic, reversible actually
        pi_irreversible = np.array([0.5, 0.5])
        # This should work since cyclic is reversible with uniform stationary
        D = discriminant_matrix(P_irreversible, pi_irreversible)
        self.assertEqual(D.shape, (2, 2), "Should handle reversible cyclic chain")
        
        # Wrong dimension stationary distribution
        with self.assertRaises(ValueError, msg="Should reject wrong dimension pi"):
            discriminant_matrix(self.P_simple, np.array([1.0]))
        
        # Non-normalized distribution
        with self.assertRaises(ValueError, msg="Should reject non-normalized pi"):
            discriminant_matrix(self.P_simple, np.array([0.3, 0.3]))


class TestSpectralAnalysis(unittest.TestCase):
    """Test cases for spectral analysis functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.P = build_two_state_chain(0.2, 0.3)
        self.pi = stationary_distribution(self.P)
        self.D = discriminant_matrix(self.P, self.pi)
    
    def test_spectral_gap(self):
        """Test spectral gap computation."""
        gap = spectral_gap(self.D)
        
        # Should be non-negative
        self.assertGreaterEqual(gap, 0, "Spectral gap should be non-negative")
        
        # Should be at most 1
        self.assertLessEqual(gap, 1, "Spectral gap should be at most 1")
        
        # For two-state chain, can compute analytically
        sigma = singular_values(self.D)
        expected_gap = sigma[0] - sigma[1]
        self.assertAlmostEqual(gap, expected_gap, places=10,
                             msg="Spectral gap should match manual calculation")
    
    def test_phase_gap(self):
        """Test phase gap computation."""
        delta = phase_gap(self.D)
        
        # Should be positive for ergodic chain
        self.assertGreater(delta, 0, "Phase gap should be positive")
        
        # Should be at most π
        self.assertLessEqual(delta, np.pi, "Phase gap should be at most π")
        
        # Relationship with classical gap
        classical_gap = classical_spectral_gap(self.P)
        theoretical_bound = 2 * np.sqrt(classical_gap)
        
        # Phase gap should be related to classical gap (approximate relationship)
        # For two-state chains, this relationship is typically tight
        self.assertGreater(delta, theoretical_bound * 0.5,
                          msg="Phase gap should be related to classical spectral gap")
    
    def test_mixing_time_bound(self):
        """Test quantum mixing time bound computation."""
        bound = mixing_time_bound(self.D, epsilon=0.01)
        
        # Should be positive and finite
        self.assertGreater(bound, 0, "Mixing time bound should be positive")
        self.assertLess(bound, np.inf, "Mixing time bound should be finite")
        
        # Should scale with log(1/epsilon)
        bound_small = mixing_time_bound(self.D, epsilon=0.001)
        self.assertGreater(bound_small, bound, 
                          "Smaller epsilon should give larger bound")
    
    def test_effective_dimension(self):
        """Test effective dimension computation."""
        d_eff = effective_dimension(self.D, threshold=0.01)
        
        # Should be a positive integer
        self.assertIsInstance(d_eff, int, "Effective dimension should be integer")
        self.assertGreater(d_eff, 0, "Effective dimension should be positive")
        
        # Should be at most the matrix dimension
        self.assertLessEqual(d_eff, self.D.shape[0], 
                           "Effective dimension should be ≤ matrix dimension")
        
        # Higher threshold should give smaller effective dimension
        d_eff_high = effective_dimension(self.D, threshold=0.1)
        self.assertLessEqual(d_eff_high, d_eff,
                           "Higher threshold should give smaller effective dimension")
    
    def test_condition_number(self):
        """Test condition number computation."""
        kappa = condition_number(self.D)
        
        # Should be ≥ 1
        self.assertGreaterEqual(kappa, 1.0, "Condition number should be ≥ 1")
        
        # Should be finite for full-rank matrix
        self.assertLess(kappa, np.inf, "Condition number should be finite")


class TestValidation(unittest.TestCase):
    """Test cases for discriminant matrix validation."""
    
    def test_validate_correct_discriminant(self):
        """Test validation of correct discriminant matrix."""
        P = build_two_state_chain(0.3, 0.4)
        pi = stationary_distribution(P)
        D = discriminant_matrix(P, pi)
        
        # Should validate successfully
        self.assertTrue(validate_discriminant(D, P, pi),
                       "Correct discriminant matrix should validate")
    
    def test_validate_incorrect_discriminant(self):
        """Test validation rejects incorrect discriminant matrix."""
        P = build_two_state_chain(0.3, 0.4)
        pi = stationary_distribution(P)
        
        # Wrong discriminant matrix
        D_wrong = np.array([[0.5, 0.2], [0.2, 0.5]])
        
        self.assertFalse(validate_discriminant(D_wrong, P, pi),
                        "Incorrect discriminant matrix should not validate")
    
    def test_validate_asymmetric_matrix(self):
        """Test validation rejects asymmetric matrix."""
        P = build_two_state_chain(0.3, 0.4)
        pi = stationary_distribution(P)
        
        # Asymmetric matrix
        D_asym = np.array([[0.7, 0.2], [0.3, 0.6]])
        
        self.assertFalse(validate_discriminant(D_asym, P, pi),
                        "Asymmetric matrix should not validate")


class TestSpectralAnalysisComprehensive(unittest.TestCase):
    """Test comprehensive spectral analysis function."""
    
    def test_spectral_analysis_complete(self):
        """Test complete spectral analysis function."""
        P = build_two_state_chain(0.25, 0.4)
        pi = stationary_distribution(P)
        D = discriminant_matrix(P, pi)
        
        analysis = spectral_analysis(D)
        
        # Check all required keys are present
        required_keys = [
            'singular_values', 'spectral_gap', 'phase_gap',
            'mixing_time', 'condition_number', 'effective_dimension',
            'largest_singular_value', 'dimension'
        ]
        
        for key in required_keys:
            self.assertIn(key, analysis, f"Analysis should contain key '{key}'")
        
        # Check types and ranges
        self.assertIsInstance(analysis['singular_values'], np.ndarray)
        self.assertIsInstance(analysis['spectral_gap'], float)
        self.assertIsInstance(analysis['phase_gap'], float)
        self.assertIsInstance(analysis['mixing_time'], float)
        self.assertIsInstance(analysis['condition_number'], float)
        self.assertIsInstance(analysis['effective_dimension'], int)
        self.assertIsInstance(analysis['largest_singular_value'], float)
        self.assertIsInstance(analysis['dimension'], int)
        
        # Check consistency
        self.assertAlmostEqual(analysis['largest_singular_value'], 
                             analysis['singular_values'][0], places=10)
        self.assertEqual(analysis['dimension'], D.shape[0])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and numerical stability."""
    
    def test_identity_matrix(self):
        """Test discriminant matrix for identity matrix."""
        I = np.eye(3)
        pi_uniform = np.ones(3) / 3
        
        D = discriminant_matrix(I, pi_uniform)
        
        # For identity matrix, D should equal I
        np.testing.assert_allclose(D, I, rtol=1e-10,
                                 err_msg="Discriminant of identity should be identity")
        
        # All singular values should be 1
        sigma = singular_values(D)
        np.testing.assert_allclose(sigma, np.ones(3), rtol=1e-10,
                                 err_msg="Identity matrix should have all singular values = 1")
    
    def test_nearly_singular_chain(self):
        """Test discriminant matrix for nearly singular chain."""
        # Chain with very small transition probabilities
        P = build_two_state_chain(1e-8, 1e-6)
        pi = stationary_distribution(P)
        
        D = discriminant_matrix(P, pi)
        
        # Should still be valid
        self.assertTrue(validate_discriminant(D, P, pi),
                       "Nearly singular chain should still validate")
        
        # Should have proper spectral structure
        sigma = singular_values(D)
        self.assertAlmostEqual(sigma[0], 1.0, places=6,
                             msg="Largest singular value should still be 1")
    
    def test_random_chains(self):
        """Test discriminant matrix for random reversible chains."""
        for n in [3, 5, 8]:
            for sparsity in [0.3, 0.7]:
                with self.subTest(n=n, sparsity=sparsity):
                    P, pi = sample_random_reversible_chain(n, sparsity=sparsity, seed=42)
                    
                    # Should be able to construct discriminant matrix
                    D = discriminant_matrix(P, pi)
                    
                    # Should validate
                    self.assertTrue(validate_discriminant(D, P, pi),
                                   f"Random chain (n={n}, sparsity={sparsity}) should validate")
                    
                    # Should have proper spectral properties
                    sigma = singular_values(D)
                    self.assertAlmostEqual(sigma[0], 1.0, places=6,
                                         msg=f"Random chain largest singular value should be 1")


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability and precision."""
    
    def test_very_small_probabilities(self):
        """Test with very small transition probabilities."""
        P = build_two_state_chain(1e-12, 1e-10)
        pi = stationary_distribution(P)
        
        # Should handle gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            D = discriminant_matrix(P, pi)
            
        # Should still have valid structure
        self.assertTrue(np.allclose(D, D.T), "Should remain symmetric")
        self.assertTrue(np.all(D >= -1e-10), "Should remain non-negative")
    
    def test_near_deterministic_chain(self):
        """Test with near-deterministic chain."""
        P = build_two_state_chain(0.9999, 0.0001)
        pi = stationary_distribution(P)
        D = discriminant_matrix(P, pi)
        
        # Should handle extreme case
        sigma = singular_values(D)
        self.assertAlmostEqual(sigma[0], 1.0, places=4,
                             msg="Should handle near-deterministic case")
        
        # Spectral gap should be very large
        gap = spectral_gap(D)
        self.assertGreater(gap, 0.9, "Spectral gap should be large for near-deterministic chain")


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)