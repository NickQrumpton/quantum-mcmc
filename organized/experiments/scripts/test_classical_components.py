#!/usr/bin/env python3
"""Test classical MCMC components independently."""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_markov_chain_basic():
    """Test basic Markov chain functionality."""
    print("Testing basic Markov chain functionality...")
    
    try:
        from quantum_mcmc.classical.markov_chain import (
            build_two_state_chain,
            is_stochastic,
            stationary_distribution,
            is_reversible,
            build_metropolis_chain
        )
        
        # Test 1: Two-state chain
        print("  Test 1: Two-state chain construction")
        P = build_two_state_chain(0.3, 0.4)
        print(f"    Transition matrix: \n{P}")
        
        # Check stochasticity
        assert is_stochastic(P), "Matrix should be stochastic"
        print("    ✓ Matrix is stochastic")
        
        # Check stationary distribution
        pi = stationary_distribution(P)
        print(f"    Stationary distribution: {pi}")
        expected_pi = np.array([0.4/(0.3+0.4), 0.3/(0.3+0.4)])
        np.testing.assert_allclose(pi, expected_pi, rtol=1e-10)
        print("    ✓ Stationary distribution correct")
        
        # Check reversibility
        assert is_reversible(P, pi), "Chain should be reversible"
        print("    ✓ Chain is reversible")
        
        # Test 2: Symmetric chain
        print("  Test 2: Symmetric chain")
        P_sym = build_two_state_chain(0.3)
        pi_sym = stationary_distribution(P_sym)
        expected_sym = np.array([0.5, 0.5])
        np.testing.assert_allclose(pi_sym, expected_sym, rtol=1e-10)
        print("    ✓ Symmetric chain has uniform stationary distribution")
        
        # Test 3: Metropolis chain
        print("  Test 3: Metropolis chain construction")
        target = np.array([0.3, 0.7])
        P_metro = build_metropolis_chain(target)
        pi_metro = stationary_distribution(P_metro)
        np.testing.assert_allclose(pi_metro, target, rtol=1e-6)
        assert is_reversible(P_metro, pi_metro), "Metropolis chain should be reversible"
        print("    ✓ Metropolis chain correct")
        
        print("✓ All Markov chain tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Markov chain test failed: {e}")
        return False

def test_discriminant_matrix():
    """Test discriminant matrix functionality."""
    print("Testing discriminant matrix functionality...")
    
    try:
        from quantum_mcmc.classical.markov_chain import build_two_state_chain, stationary_distribution
        from quantum_mcmc.classical.discriminant import (
            discriminant_matrix,
            singular_values,
            spectral_gap,
            phase_gap,
            validate_discriminant
        )
        
        # Test with known two-state chain
        P = build_two_state_chain(0.3, 0.4)
        pi = stationary_distribution(P)
        
        print("  Test 1: Discriminant matrix construction")
        D = discriminant_matrix(P, pi)
        print(f"    Discriminant matrix: \n{D}")
        
        # Check symmetry
        assert np.allclose(D, D.T), "Discriminant matrix should be symmetric"
        print("    ✓ Discriminant matrix is symmetric")
        
        # Check range [0,1]
        assert np.all(D >= -1e-10) and np.all(D <= 1 + 1e-10), "D should have entries in [0,1]"
        print("    ✓ Discriminant matrix entries in [0,1]")
        
        # Test singular values
        print("  Test 2: Singular value analysis")
        sigma = singular_values(D)
        print(f"    Singular values: {sigma}")
        
        # Largest singular value should be 1 (allow small numerical error)
        assert abs(sigma[0] - 1.0) < 1e-6, f"Largest singular value should be 1, got {sigma[0]}"
        print("    ✓ Largest singular value is 1")
        
        # Test spectral gap
        gap = spectral_gap(D)
        print(f"    Spectral gap: {gap}")
        assert gap >= -1e-10, "Spectral gap should be non-negative"
        print("    ✓ Spectral gap is non-negative")
        
        # Test phase gap
        delta = phase_gap(D)
        print(f"    Phase gap: {delta}")
        assert delta > 0, "Phase gap should be positive"
        print("    ✓ Phase gap is positive")
        
        # Test validation
        print("  Test 3: Discriminant matrix validation")
        assert validate_discriminant(D, P, pi), "Discriminant matrix validation should pass"
        print("    ✓ Discriminant matrix validation passed")
        
        print("✓ All discriminant matrix tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Discriminant matrix test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases and error handling...")
    
    try:
        from quantum_mcmc.classical.markov_chain import (
            build_two_state_chain,
            is_stochastic,
            stationary_distribution,
            build_metropolis_chain
        )
        
        print("  Test 1: Invalid probabilities")
        
        # Test negative probabilities
        try:
            build_two_state_chain(-0.1, 0.3)
            assert False, "Should raise ValueError for negative p"
        except ValueError:
            print("    ✓ Correctly rejects negative p")
        
        try:
            build_two_state_chain(0.3, -0.1)
            assert False, "Should raise ValueError for negative q"
        except ValueError:
            print("    ✓ Correctly rejects negative q")
        
        # Test probabilities > 1
        try:
            build_two_state_chain(1.1, 0.3)
            assert False, "Should raise ValueError for p > 1"
        except ValueError:
            print("    ✓ Correctly rejects p > 1")
        
        print("  Test 2: Boundary cases")
        
        # Test with p=0, q=1
        P_boundary = build_two_state_chain(0.0, 1.0)
        expected = np.array([[1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(P_boundary, expected)
        print("    ✓ Boundary case p=0, q=1 correct")
        
        # Test identity matrix case
        P_identity = build_two_state_chain(0.0, 0.0)
        np.testing.assert_allclose(P_identity, np.eye(2))
        print("    ✓ Identity matrix case correct")
        
        print("  Test 3: Non-stochastic matrix detection")
        
        # Non-stochastic matrix
        P_bad = np.array([[0.5, 0.4], [0.3, 0.7]])  # Rows don't sum to 1
        assert not is_stochastic(P_bad), "Should detect non-stochastic matrix"
        print("    ✓ Correctly detects non-stochastic matrix")
        
        # Matrix with negative entries
        P_neg = np.array([[0.8, 0.2], [-0.1, 1.1]])
        assert not is_stochastic(P_neg), "Should detect negative entries"
        print("    ✓ Correctly detects negative entries")
        
        print("✓ All edge case tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        return False

def test_numerical_stability():
    """Test numerical stability and precision."""
    print("Testing numerical stability...")
    
    try:
        from quantum_mcmc.classical.markov_chain import (
            build_two_state_chain,
            stationary_distribution,
            is_reversible,
            sample_random_reversible_chain
        )
        
        print("  Test 1: Near-singular matrices")
        
        # Very small transition probabilities
        P_small = build_two_state_chain(1e-10, 1e-8)
        pi_small = stationary_distribution(P_small)
        assert abs(np.sum(pi_small) - 1.0) < 1e-10, "Distribution should sum to 1"
        print("    ✓ Handles small probabilities correctly")
        
        # Near-deterministic matrix
        P_det = build_two_state_chain(0.999, 0.001)
        pi_det = stationary_distribution(P_det)
        assert is_reversible(P_det, pi_det), "Near-deterministic chain should be reversible"
        print("    ✓ Handles near-deterministic matrices")
        
        print("  Test 2: Random chain generation")
        
        # Generate random reversible chains
        for n in [5, 10]:
            P_rand, pi_rand = sample_random_reversible_chain(n, seed=42)
            assert abs(np.sum(pi_rand) - 1.0) < 1e-10, "Random distribution should sum to 1"
            assert is_reversible(P_rand, pi_rand), "Random chain should be reversible"
        print("    ✓ Random chain generation works correctly")
        
        print("  Test 3: Power method convergence")
        
        # Test both eigen and power methods give same result
        P = build_two_state_chain(0.2, 0.6)
        pi_eigen = stationary_distribution(P, method='eigen')
        pi_power = stationary_distribution(P, method='power')
        np.testing.assert_allclose(pi_eigen, pi_power, rtol=1e-6)
        print("    ✓ Eigen and power methods agree")
        
        print("✓ All numerical stability tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Numerical stability test failed: {e}")
        return False

def main():
    """Run all classical component tests."""
    print("=" * 60)
    print("CLASSICAL MCMC COMPONENT TESTING")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Run all test suites
    all_passed &= test_markov_chain_basic()
    all_passed &= test_discriminant_matrix()
    all_passed &= test_edge_cases()
    all_passed &= test_numerical_stability()
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Classical components are working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Classical components need refinement!")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)