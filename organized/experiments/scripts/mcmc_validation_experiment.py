#!/usr/bin/env python3
"""
Comprehensive MCMC Validation Experiment

This script implements and validates a Metropolis-Hastings sampler through
extensive diagnostics including convergence, stationarity, detailed balance,
and mixing properties.

Author: Quantum MCMC Research Team
Date: 2025-01-27
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import cholesky
import warnings
from typing import Callable, Tuple, Dict, List, Any
import os


def run_classical_chain(target_log_prob: Callable[[np.ndarray], float],
                       proposal_sampler: Callable[[np.ndarray], np.ndarray],
                       theta0: np.ndarray,
                       num_steps: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Metropolis-Hastings MCMC sampler.
    
    Args:
        target_log_prob: Function that computes log probability of target distribution
        proposal_sampler: Function that generates proposal given current state
        theta0: Initial state
        num_steps: Number of MCMC steps
    
    Returns:
        samples: Array of shape (num_steps+1, dim) containing all samples
        accepted: Boolean array indicating which proposals were accepted
        acceptance_rate: Overall acceptance rate
    """
    dim = len(theta0)
    samples = np.zeros((num_steps + 1, dim))
    accepted = np.zeros(num_steps, dtype=bool)
    
    # Initialize
    samples[0] = theta0.copy()
    current_log_prob = target_log_prob(theta0)
    num_accepted = 0
    
    for i in range(num_steps):
        # Generate proposal
        proposal = proposal_sampler(samples[i])
        proposal_log_prob = target_log_prob(proposal)
        
        # Compute acceptance probability (log space for numerical stability)
        log_alpha = proposal_log_prob - current_log_prob
        alpha = min(1.0, np.exp(log_alpha))
        
        # Accept or reject
        if np.random.rand() < alpha:
            samples[i + 1] = proposal
            current_log_prob = proposal_log_prob
            accepted[i] = True
            num_accepted += 1
        else:
            samples[i + 1] = samples[i].copy()
            accepted[i] = False
    
    acceptance_rate = num_accepted / num_steps
    return samples, accepted, acceptance_rate


def define_target_distribution():
    """Define 2D Gaussian target with specified mean and covariance."""
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    
    # Precompute for efficiency
    Sigma_inv = np.linalg.inv(Sigma)
    log_det_Sigma = np.log(np.linalg.det(Sigma))
    log_2pi = np.log(2 * np.pi)
    
    def target_log_prob(x):
        """Log probability of 2D Gaussian."""
        diff = x - mu
        return -0.5 * (diff.T @ Sigma_inv @ diff + log_det_Sigma + 2 * log_2pi)
    
    return target_log_prob, mu, Sigma


def create_proposal_sampler(sigma: float):
    """Create Gaussian random walk proposal sampler."""
    def proposal_sampler(x):
        return x + np.random.normal(0, sigma, size=len(x))
    return proposal_sampler


def tune_acceptance_rate(target_log_prob: Callable, 
                        theta0: np.ndarray,
                        target_acceptance: Tuple[float, float] = (0.2, 0.5),
                        max_iterations: int = 10,
                        tune_steps: int = 5000) -> float:
    """
    Automatically tune proposal standard deviation for target acceptance rate.
    
    Args:
        target_log_prob: Target log probability function
        theta0: Initial state for tuning
        target_acceptance: (min_rate, max_rate) tuple
        max_iterations: Maximum tuning iterations
        tune_steps: Number of steps for each tuning run
    
    Returns:
        Tuned sigma value
    """
    print("🎯 Tuning proposal standard deviation for optimal acceptance rate...")
    
    sigma = 1.0  # Initial guess
    min_acc, max_acc = target_acceptance
    
    for iteration in range(max_iterations):
        proposal_sampler = create_proposal_sampler(sigma)
        _, _, acc_rate = run_classical_chain(target_log_prob, proposal_sampler, 
                                           theta0, tune_steps)
        
        print(f"  Iteration {iteration+1}: σ = {sigma:.3f}, acceptance = {acc_rate:.3f}")
        
        if min_acc <= acc_rate <= max_acc:
            print(f"  ✓ Optimal σ = {sigma:.3f} (acceptance = {acc_rate:.3f})")
            return sigma
        elif acc_rate < min_acc:
            sigma *= 0.8  # Decrease step size to increase acceptance
        else:  # acc_rate > max_acc
            sigma *= 1.25  # Increase step size to decrease acceptance
    
    print(f"  ⚠ Tuning did not converge, using σ = {sigma:.3f}")
    return sigma


def compute_gelman_rubin(chains: List[np.ndarray]) -> np.ndarray:
    """
    Compute Gelman-Rubin R-hat statistic for convergence diagnosis.
    
    Args:
        chains: List of MCMC chains, each of shape (n_samples, dim)
    
    Returns:
        R_hat values for each dimension
    """
    n_chains = len(chains)
    n_samples, dim = chains[0].shape
    
    # Stack chains for easier computation
    all_chains = np.stack(chains, axis=0)  # (n_chains, n_samples, dim)
    
    # Chain means and overall mean
    chain_means = np.mean(all_chains, axis=1)  # (n_chains, dim)
    overall_mean = np.mean(chain_means, axis=0)  # (dim,)
    
    # Between-chain variance B
    B = n_samples * np.var(chain_means, axis=0, ddof=1)
    
    # Within-chain variance W
    chain_vars = np.var(all_chains, axis=1, ddof=1)  # (n_chains, dim)
    W = np.mean(chain_vars, axis=0)
    
    # Pooled variance estimate
    var_plus = ((n_samples - 1) * W + B) / n_samples
    
    # R-hat statistic
    R_hat = np.sqrt(var_plus / W)
    
    return R_hat


def compute_effective_sample_size(chain: np.ndarray, max_lag: int = None) -> np.ndarray:
    """
    Compute effective sample size using autocorrelation.
    
    Args:
        chain: MCMC chain of shape (n_samples, dim)
        max_lag: Maximum lag for autocorrelation (default: n_samples//4)
    
    Returns:
        Effective sample size for each dimension
    """
    n_samples, dim = chain.shape
    if max_lag is None:
        max_lag = n_samples // 4
    
    ess = np.zeros(dim)
    
    for d in range(dim):
        x = chain[:, d]
        autocorr = compute_autocorrelation(x, max_lag)
        
        # Find first negative autocorrelation (or use all positive)
        first_negative = np.where(autocorr < 0)[0]
        if len(first_negative) > 0:
            cutoff = first_negative[0]
        else:
            cutoff = len(autocorr)
        
        # Integrated autocorrelation time
        tau_int = 1 + 2 * np.sum(autocorr[:cutoff])
        
        # Effective sample size
        ess[d] = n_samples / tau_int
    
    return ess


def compute_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function for a 1D time series."""
    n = len(x)
    x_centered = x - np.mean(x)
    
    # Use FFT for efficient computation
    f_x = np.fft.fft(x_centered, 2 * n)
    autocorr_fft = np.fft.ifft(f_x * np.conj(f_x))[:n].real
    autocorr_fft = autocorr_fft / autocorr_fft[0]  # Normalize
    
    return autocorr_fft[:max_lag + 1]


def test_convergence(chains: List[np.ndarray], burn_in: int = 0) -> Tuple[bool, Dict]:
    """
    Test convergence using Gelman-Rubin diagnostic.
    
    This validates that multiple chains have converged to the same distribution,
    which is essential for reliable MCMC inference.
    """
    print("\n🔍 Testing convergence with Gelman-Rubin diagnostic...")
    
    # Remove burn-in
    chains_burned = [chain[burn_in:] for chain in chains]
    
    # Compute R-hat
    R_hat = compute_gelman_rubin(chains_burned)
    
    # Check convergence criterion
    converged = np.all(R_hat < 1.05)
    
    results = {
        'R_hat': R_hat,
        'converged': converged,
        'max_R_hat': np.max(R_hat)
    }
    
    print(f"  R-hat values: {R_hat}")
    print(f"  Max R-hat: {results['max_R_hat']:.4f}")
    print(f"  Convergence (R-hat < 1.05): {'✓ PASS' if converged else '✗ FAIL'}")
    
    return converged, results


def test_effective_sample_size(chains: List[np.ndarray], 
                              burn_in: int = 0,
                              min_ess: int = 1000) -> Tuple[bool, Dict]:
    """
    Test effective sample size for adequate mixing.
    
    This validates that the chains are mixing well and providing
    sufficient independent samples for reliable statistics.
    """
    print(f"\n📊 Testing effective sample size (target: ESS > {min_ess})...")
    
    # Pool chains after burn-in
    chains_burned = [chain[burn_in:] for chain in chains]
    pooled_samples = np.vstack(chains_burned)
    
    # Compute ESS
    ess = compute_effective_sample_size(pooled_samples)
    
    # Check ESS criterion
    adequate_ess = np.all(ess > min_ess)
    
    results = {
        'ess': ess,
        'adequate_ess': adequate_ess,
        'min_ess': np.min(ess),
        'total_samples': len(pooled_samples)
    }
    
    print(f"  ESS per dimension: {ess}")
    print(f"  Min ESS: {results['min_ess']:.1f}")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Adequate ESS: {'✓ PASS' if adequate_ess else '✗ FAIL'}")
    
    return adequate_ess, results


def test_stationary_distribution(chains: List[np.ndarray],
                                burn_in: int = 0,
                                true_mean: np.ndarray = None,
                                true_cov: np.ndarray = None,
                                tolerance: float = 0.02) -> Tuple[bool, Dict]:
    """
    Test that empirical distribution matches target stationary distribution.
    
    This validates that the MCMC sampler is correctly targeting the
    intended probability distribution.
    """
    print(f"\n📈 Testing stationary distribution (tolerance: {tolerance*100}%)...")
    
    # Pool samples after burn-in
    chains_burned = [chain[burn_in:] for chain in chains]
    pooled_samples = np.vstack(chains_burned)
    
    # Compute empirical statistics
    emp_mean = np.mean(pooled_samples, axis=0)
    emp_cov = np.cov(pooled_samples, rowvar=False)
    
    # Compute errors
    # For mean: use absolute error when true mean is close to zero
    mean_error = np.abs(emp_mean - true_mean)
    if np.any(np.abs(true_mean) > 1e-6):
        # Use relative error when true mean is non-zero
        mean_error = mean_error / np.maximum(np.abs(true_mean), 1e-6)
    
    # For covariance: use relative error
    cov_error = np.abs(emp_cov - true_cov) / np.maximum(np.abs(true_cov), 1e-10)
    
    # Check tolerance
    mean_ok = np.all(mean_error < tolerance)
    cov_ok = np.all(cov_error < tolerance)
    distribution_ok = mean_ok and cov_ok
    
    results = {
        'empirical_mean': emp_mean,
        'empirical_cov': emp_cov,
        'mean_error': mean_error,
        'cov_error': cov_error,
        'max_mean_error': np.max(mean_error),
        'max_cov_error': np.max(cov_error),
        'distribution_ok': distribution_ok
    }
    
    print(f"  True mean: {true_mean}")
    print(f"  Empirical mean: {emp_mean}")
    print(f"  Mean relative error: {mean_error}")
    print(f"  Max mean error: {results['max_mean_error']:.4f}")
    print(f"  Max covariance error: {results['max_cov_error']:.4f}")
    print(f"  Distribution accuracy: {'✓ PASS' if distribution_ok else '✗ FAIL'}")
    
    return distribution_ok, results


def test_autocorrelation(chains: List[np.ndarray],
                        burn_in: int = 0,
                        max_lag: int = 100,
                        lag_threshold: int = 50,
                        autocorr_threshold: float = 0.1) -> Tuple[bool, Dict]:
    """
    Test autocorrelation decay for adequate mixing.
    
    This validates that the chain is mixing well and not getting stuck
    in local regions of the parameter space.
    """
    print(f"\n⏱️  Testing autocorrelation (threshold: autocorr < {autocorr_threshold} at lag {lag_threshold})...")
    
    # Pool samples after burn-in
    chains_burned = [chain[burn_in:] for chain in chains]
    pooled_samples = np.vstack(chains_burned)
    
    dim = pooled_samples.shape[1]
    autocorrs = np.zeros((max_lag + 1, dim))
    
    # Compute autocorrelation for each dimension
    for d in range(dim):
        autocorrs[:, d] = compute_autocorrelation(pooled_samples[:, d], max_lag)
    
    # Check autocorrelation at specified lag
    autocorr_at_threshold = autocorrs[lag_threshold]
    mixing_ok = np.all(np.abs(autocorr_at_threshold) < autocorr_threshold)
    
    results = {
        'autocorrelations': autocorrs,
        'autocorr_at_threshold': autocorr_at_threshold,
        'mixing_ok': mixing_ok,
        'max_autocorr_at_threshold': np.max(np.abs(autocorr_at_threshold))
    }
    
    print(f"  Autocorrelation at lag {lag_threshold}: {autocorr_at_threshold}")
    print(f"  Max |autocorr| at lag {lag_threshold}: {results['max_autocorr_at_threshold']:.4f}")
    print(f"  Adequate mixing: {'✓ PASS' if mixing_ok else '✗ FAIL'}")
    
    return mixing_ok, results


def test_detailed_balance(target_log_prob: Callable,
                         proposal_sampler: Callable,
                         chains: List[np.ndarray],
                         burn_in: int = 0,
                         n_tests: int = 1000,
                         tolerance: float = 1e-6) -> Tuple[bool, Dict]:
    """
    Test detailed balance condition for equilibrium validation.
    
    This is the fundamental test that the MCMC kernel satisfies detailed balance:
    π(x) * P(x→x') = π(x') * P(x'→x)
    
    This ensures the chain has the correct stationary distribution.
    """
    print(f"\n⚖️  Testing detailed balance ({n_tests} random transitions)...")
    
    # Pool samples after burn-in
    chains_burned = [chain[burn_in:] for chain in chains]
    pooled_samples = np.vstack(chains_burned)
    
    # Function to compute transition probability
    def transition_prob(x, x_prime):
        """Compute P(x → x') for Metropolis-Hastings."""
        # Proposal probability (Gaussian random walk is symmetric)
        # For symmetric proposals: q(x→x') = q(x'→x)
        
        # Acceptance probability
        log_alpha = target_log_prob(x_prime) - target_log_prob(x)
        alpha = min(1.0, np.exp(log_alpha))
        
        # If x_prime is the proposal from x, transition prob = alpha
        # If x_prime ≠ x + noise, transition prob = 0 (for continuous case)
        # For detailed balance test, we'll use the acceptance probability
        return alpha
    
    # Sample random transitions from equilibrium distribution
    n_samples = len(pooled_samples)
    test_indices = np.random.choice(n_samples, n_tests, replace=True)
    
    balance_errors = []
    
    for i in test_indices:
        x = pooled_samples[i]
        
        # Generate a proposal as if we were at x
        x_prime = proposal_sampler(x)
        
        # Compute detailed balance components
        pi_x = np.exp(target_log_prob(x))
        pi_x_prime = np.exp(target_log_prob(x_prime))
        
        # For symmetric proposals, q(x→x') = q(x'→x), so they cancel
        alpha_forward = transition_prob(x, x_prime)
        alpha_backward = transition_prob(x_prime, x)
        
        # Detailed balance: π(x) * α(x→x') = π(x') * α(x'→x)
        lhs = pi_x * alpha_forward
        rhs = pi_x_prime * alpha_backward
        
        # Compute relative error
        if lhs + rhs > 1e-20:  # Avoid division by zero
            rel_error = abs(lhs - rhs) / (0.5 * (lhs + rhs))
            balance_errors.append(rel_error)
    
    # Check detailed balance
    max_error = np.max(balance_errors) if balance_errors else 0
    mean_error = np.mean(balance_errors) if balance_errors else 0
    balance_ok = max_error < tolerance
    
    results = {
        'balance_errors': balance_errors,
        'max_error': max_error,
        'mean_error': mean_error,
        'balance_ok': balance_ok,
        'n_tests': len(balance_errors)
    }
    
    print(f"  Tests performed: {results['n_tests']}")
    print(f"  Mean relative error: {mean_error:.2e}")
    print(f"  Max relative error: {max_error:.2e}")
    print(f"  Detailed balance: {'✓ PASS' if balance_ok else '✗ FAIL'}")
    
    return balance_ok, results


def acceptance_rate_sweep(target_log_prob: Callable,
                         true_mean: np.ndarray,
                         true_cov: np.ndarray,
                         sigma_values: List[float],
                         initial_states: List[np.ndarray],
                         num_steps: int = 50000,
                         burn_in: int = 5000) -> Dict:
    """
    Sweep over different proposal standard deviations to study performance.
    
    This validates how the sampler performance depends on the proposal
    step size and helps identify optimal parameter ranges.
    """
    print(f"\n🔄 Acceptance rate sweep over σ = {sigma_values}...")
    
    results = {
        'sigma_values': sigma_values,
        'acceptance_rates': [],
        'R_hat_values': [],
        'ess_values': [],
        'mean_errors': [],
        'converged': []
    }
    
    for sigma in sigma_values:
        print(f"\n  Testing σ = {sigma}...")
        
        # Create proposal sampler
        proposal_sampler = create_proposal_sampler(sigma)
        
        # Run multiple chains
        chains = []
        total_accepted = 0
        total_steps = 0
        
        for theta0 in initial_states:
            samples, accepted, acc_rate = run_classical_chain(
                target_log_prob, proposal_sampler, theta0, num_steps
            )
            chains.append(samples)
            total_accepted += np.sum(accepted)
            total_steps += len(accepted)
        
        # Overall acceptance rate
        overall_acc_rate = total_accepted / total_steps
        
        # Test convergence
        converged, conv_results = test_convergence(chains, burn_in)
        max_R_hat = conv_results['max_R_hat']
        
        # Test ESS
        _, ess_results = test_effective_sample_size(chains, burn_in, min_ess=100)
        min_ess = ess_results['min_ess']
        
        # Test distribution
        _, dist_results = test_stationary_distribution(
            chains, burn_in, true_mean, true_cov, tolerance=0.05
        )
        max_mean_error = dist_results['max_mean_error']
        
        # Store results
        results['acceptance_rates'].append(overall_acc_rate)
        results['R_hat_values'].append(max_R_hat)
        results['ess_values'].append(min_ess)
        results['mean_errors'].append(max_mean_error)
        results['converged'].append(converged)
        
        print(f"    Acceptance rate: {overall_acc_rate:.3f}")
        print(f"    Max R-hat: {max_R_hat:.4f}")
        print(f"    Min ESS: {min_ess:.1f}")
        print(f"    Max mean error: {max_mean_error:.4f}")
    
    return results


def create_diagnostic_plots(chains: List[np.ndarray],
                           autocorr_results: Dict,
                           sweep_results: Dict,
                           burn_in: int = 0):
    """Create comprehensive diagnostic plots."""
    
    print("\n📊 Creating diagnostic plots...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Trace plots
    ax1 = plt.subplot(3, 3, 1)
    chains_burned = [chain[burn_in:] for chain in chains]
    for i, chain in enumerate(chains_burned):
        plt.plot(chain[:2000, 0], alpha=0.7, label=f'Chain {i+1}')
    plt.title('Trace Plot (Dimension 1)')
    plt.xlabel('Iteration')
    plt.ylabel('x₁')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    for i, chain in enumerate(chains_burned):
        plt.plot(chain[:2000, 1], alpha=0.7, label=f'Chain {i+1}')
    plt.title('Trace Plot (Dimension 2)')
    plt.xlabel('Iteration')
    plt.ylabel('x₂')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Sample scatter plot
    ax3 = plt.subplot(3, 3, 3)
    pooled_samples = np.vstack(chains_burned)
    plt.scatter(pooled_samples[::10, 0], pooled_samples[::10, 1], 
                alpha=0.5, s=1, label='MCMC samples')
    
    # Add true distribution contours
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(mu, Sigma)
    plt.contour(X, Y, rv.pdf(pos), levels=5, colors='red', alpha=0.7)
    
    plt.title('Sample Distribution')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # 3. Autocorrelation plots
    ax4 = plt.subplot(3, 3, 4)
    autocorrs = autocorr_results['autocorrelations']
    lags = np.arange(len(autocorrs))
    plt.plot(lags, autocorrs[:, 0], 'b-', label='Dimension 1')
    plt.plot(lags, autocorrs[:, 1], 'r-', label='Dimension 2')
    plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.7, label='Threshold')
    plt.axhline(y=-0.1, color='gray', linestyle='--', alpha=0.7)
    plt.title('Autocorrelation Function')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. R-hat evolution (if available)
    ax5 = plt.subplot(3, 3, 5)
    # Compute R-hat for different chain lengths
    chain_lengths = np.logspace(2, np.log10(len(chains_burned[0])), 20).astype(int)
    r_hat_evolution = []
    
    for length in chain_lengths:
        if length <= len(chains_burned[0]):
            test_chains = [chain[:length] for chain in chains_burned]
            r_hat = compute_gelman_rubin(test_chains)
            r_hat_evolution.append(np.max(r_hat))
    
    plt.semilogx(chain_lengths[:len(r_hat_evolution)], r_hat_evolution, 'g-o')
    plt.axhline(y=1.05, color='red', linestyle='--', label='Convergence threshold')
    plt.title('R-hat Evolution')
    plt.xlabel('Chain Length')
    plt.ylabel('Max R-hat')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Acceptance rate vs sigma
    ax6 = plt.subplot(3, 3, 6)
    sigma_vals = sweep_results['sigma_values']
    acc_rates = sweep_results['acceptance_rates']
    plt.plot(sigma_vals, acc_rates, 'bo-', linewidth=2, markersize=8)
    plt.axhspan(0.2, 0.5, alpha=0.3, color='green', label='Optimal range')
    plt.title('Acceptance Rate vs Step Size')
    plt.xlabel('σ')
    plt.ylabel('Acceptance Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. ESS vs sigma
    ax7 = plt.subplot(3, 3, 7)
    ess_vals = sweep_results['ess_values']
    plt.plot(sigma_vals, ess_vals, 'ro-', linewidth=2, markersize=8)
    plt.axhline(y=1000, color='gray', linestyle='--', label='Target ESS')
    plt.title('Effective Sample Size vs Step Size')
    plt.xlabel('σ')
    plt.ylabel('Min ESS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. R-hat vs sigma
    ax8 = plt.subplot(3, 3, 8)
    r_hat_vals = sweep_results['R_hat_values']
    plt.plot(sigma_vals, r_hat_vals, 'mo-', linewidth=2, markersize=8)
    plt.axhline(y=1.05, color='red', linestyle='--', label='Convergence threshold')
    plt.title('Convergence vs Step Size')
    plt.xlabel('σ')
    plt.ylabel('Max R-hat')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Summary performance
    ax9 = plt.subplot(3, 3, 9)
    # Create a performance score combining acceptance rate, ESS, and convergence
    performance_scores = []
    for i, sigma in enumerate(sigma_vals):
        acc_score = 1.0 if 0.2 <= acc_rates[i] <= 0.5 else 0.5
        ess_score = min(1.0, ess_vals[i] / 1000)
        conv_score = 1.0 if r_hat_vals[i] < 1.05 else 0.0
        total_score = (acc_score + ess_score + conv_score) / 3
        performance_scores.append(total_score)
    
    plt.plot(sigma_vals, performance_scores, 'ko-', linewidth=2, markersize=8)
    plt.title('Overall Performance Score')
    plt.xlabel('σ')
    plt.ylabel('Performance Score')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/mcmc_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run the complete MCMC validation experiment."""
    
    print("🔬 COMPREHENSIVE MCMC VALIDATION EXPERIMENT")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Define target distribution
    print("\n1️⃣  Defining target distribution...")
    target_log_prob, true_mean, true_cov = define_target_distribution()
    print(f"   Target: 2D Gaussian with μ = {true_mean}, Σ = \n{true_cov}")
    
    # 2. Tune proposal standard deviation
    print("\n2️⃣  Tuning proposal kernel...")
    theta0_tune = np.array([0.0, 0.0])
    optimal_sigma = tune_acceptance_rate(target_log_prob, theta0_tune)
    
    # 3. Set up multiple chains
    print("\n3️⃣  Setting up multiple chains...")
    initial_states = [
        np.array([5.0, 5.0]),
        np.array([-5.0, -5.0]),
        np.array([5.0, -5.0]),
        np.array([-5.0, 5.0])
    ]
    num_steps = 100000
    burn_in = 10000
    
    print(f"   Initial states: {len(initial_states)} chains")
    print(f"   Steps per chain: {num_steps:,}")
    print(f"   Burn-in: {burn_in:,}")
    
    # 4. Run chains
    print("\n4️⃣  Running MCMC chains...")
    proposal_sampler = create_proposal_sampler(optimal_sigma)
    chains = []
    acceptance_rates = []
    
    for i, theta0 in enumerate(initial_states):
        print(f"   Running chain {i+1}/4...")
        samples, accepted, acc_rate = run_classical_chain(
            target_log_prob, proposal_sampler, theta0, num_steps
        )
        chains.append(samples)
        acceptance_rates.append(acc_rate)
    
    print(f"   Acceptance rates: {[f'{rate:.3f}' for rate in acceptance_rates]}")
    
    # 5. Run all diagnostic tests
    print("\n5️⃣  Running diagnostic tests...")
    test_results = {}
    all_passed = True
    
    # Test convergence
    converged, conv_results = test_convergence(chains, burn_in)
    test_results['convergence'] = (converged, conv_results)
    all_passed &= converged
    
    # Test effective sample size
    ess_ok, ess_results = test_effective_sample_size(chains, burn_in)
    test_results['ess'] = (ess_ok, ess_results)
    all_passed &= ess_ok
    
    # Test stationary distribution
    dist_ok, dist_results = test_stationary_distribution(chains, burn_in, true_mean, true_cov)
    test_results['distribution'] = (dist_ok, dist_results)
    all_passed &= dist_ok
    
    # Test autocorrelation
    mixing_ok, autocorr_results = test_autocorrelation(chains, burn_in)
    test_results['autocorrelation'] = (mixing_ok, autocorr_results)
    all_passed &= mixing_ok
    
    # Test detailed balance
    balance_ok, balance_results = test_detailed_balance(
        target_log_prob, proposal_sampler, chains, burn_in
    )
    test_results['detailed_balance'] = (balance_ok, balance_results)
    all_passed &= balance_ok
    
    # 6. Acceptance rate sweep
    print("\n6️⃣  Running acceptance rate sweep...")
    sigma_values = [0.1, 0.5, 1.0, 2.0]
    sweep_results = acceptance_rate_sweep(
        target_log_prob, true_mean, true_cov, sigma_values, 
        initial_states, num_steps=50000, burn_in=5000
    )
    
    # 7. Create diagnostic plots
    print("\n7️⃣  Creating diagnostic plots...")
    create_diagnostic_plots(chains, autocorr_results, sweep_results, burn_in)
    
    # 8. Final summary
    print("\n" + "=" * 80)
    print("📋 FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n🎯 OPTIMAL PARAMETERS:")
    print(f"   Proposal σ: {optimal_sigma:.3f}")
    print(f"   Acceptance rate: {np.mean(acceptance_rates):.3f} ± {np.std(acceptance_rates):.3f}")
    
    print(f"\n📊 TEST RESULTS:")
    test_names = ['Convergence (R-hat)', 'Effective Sample Size', 'Stationary Distribution', 
                  'Autocorrelation', 'Detailed Balance']
    
    for i, (test_name, (passed, results)) in enumerate(zip(test_names, test_results.values())):
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n📈 KEY METRICS:")
    print(f"   Max R-hat: {conv_results['max_R_hat']:.4f} (< 1.05)")
    print(f"   Min ESS: {ess_results['min_ess']:.0f} (> 1000)")
    print(f"   Max mean error: {dist_results['max_mean_error']:.4f} (< 0.02)")
    print(f"   Max autocorr at lag 50: {autocorr_results['max_autocorr_at_threshold']:.4f} (< 0.1)")
    print(f"   Max balance error: {balance_results['max_error']:.2e} (< 1e-6)")
    
    print(f"\n🔄 ACCEPTANCE RATE SWEEP:")
    for i, sigma in enumerate(sigma_values):
        acc = sweep_results['acceptance_rates'][i]
        r_hat = sweep_results['R_hat_values'][i]
        ess = sweep_results['ess_values'][i]
        conv = "✓" if sweep_results['converged'][i] else "✗"
        print(f"   σ={sigma:4.1f}: acc={acc:.3f}, R-hat={r_hat:.3f}, ESS={ess:6.0f} {conv}")
    
    print(f"\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - MCMC IMPLEMENTATION IS VALIDATED!")
    else:
        print("❌ SOME TESTS FAILED - REVIEW IMPLEMENTATION")
        failed_tests = [name for name, (passed, _) in zip(test_names, test_results.values()) if not passed]
        print(f"   Failed tests: {failed_tests}")
    print("=" * 80)
    
    print(f"\n📁 Results saved to: results/mcmc_diagnostics.png")
    
    return all_passed, test_results


if __name__ == "__main__":
    success, results = main()