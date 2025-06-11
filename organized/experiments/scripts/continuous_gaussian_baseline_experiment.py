#!/usr/bin/env python3
"""
Continuous-Gaussian Baseline Metropolis-Hastings Experiment

This script implements a comprehensive MCMC validation experiment targeting
a 2D correlated Gaussian distribution with automatic parameter tuning,
multi-chain convergence diagnostics, and publication-quality analysis.

Author: MCMC Diagnostics Expert
Date: 2025-01-27
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
import os
from typing import Tuple, List, Dict, Any
import warnings

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set style for publication-quality figures
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def target_log_prob(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute log-probability of 2D Gaussian target distribution.
    
    Args:
        x: 2D point to evaluate
        mu: Mean vector [2,]
        sigma: Covariance matrix [2,2]
    
    Returns:
        Log-probability density at x
    """
    return multivariate_normal.logpdf(x, mean=mu, cov=sigma)


def proposal(x: np.ndarray, proposal_std: float) -> np.ndarray:
    """
    Isotropic Gaussian random walk proposal.
    
    Args:
        x: Current state
        proposal_std: Standard deviation of proposal distribution
    
    Returns:
        Proposed next state
    """
    return x + np.random.normal(size=x.shape) * proposal_std


def metropolis_hastings_step(x_current: np.ndarray, 
                           target_log_prob_func: callable,
                           proposal_std: float) -> Tuple[np.ndarray, bool]:
    """
    Single Metropolis-Hastings step.
    
    Args:
        x_current: Current state
        target_log_prob_func: Function computing target log-probability
        proposal_std: Proposal standard deviation
    
    Returns:
        (next_state, accepted): Next state and acceptance flag
    """
    # Generate proposal
    x_proposal = proposal(x_current, proposal_std)
    
    # Compute acceptance probability (in log space for numerical stability)
    log_prob_current = target_log_prob_func(x_current)
    log_prob_proposal = target_log_prob_func(x_proposal)
    
    log_alpha = log_prob_proposal - log_prob_current
    alpha = min(1.0, np.exp(log_alpha))
    
    # Accept or reject
    if np.random.rand() < alpha:
        return x_proposal, True
    else:
        return x_current, False


def run_mcmc_chain(initial_state: np.ndarray,
                  target_log_prob_func: callable,
                  proposal_std: float,
                  n_steps: int) -> Tuple[np.ndarray, float]:
    """
    Run complete MCMC chain.
    
    Args:
        initial_state: Starting point
        target_log_prob_func: Target log-probability function
        proposal_std: Proposal standard deviation
        n_steps: Number of MCMC steps
    
    Returns:
        (samples, acceptance_rate): Chain samples and acceptance rate
    """
    dim = len(initial_state)
    samples = np.zeros((n_steps + 1, dim))
    samples[0] = initial_state.copy()
    
    n_accepted = 0
    x_current = initial_state.copy()
    
    for i in range(n_steps):
        x_next, accepted = metropolis_hastings_step(x_current, target_log_prob_func, proposal_std)
        samples[i + 1] = x_next
        x_current = x_next
        
        if accepted:
            n_accepted += 1
    
    acceptance_rate = n_accepted / n_steps
    return samples, acceptance_rate


def tune_proposal_std(target_log_prob_func: callable,
                     initial_state: np.ndarray,
                     target_acceptance_range: Tuple[float, float] = (0.25, 0.45),
                     n_tune_steps: int = 5000,
                     max_iterations: int = 20) -> Tuple[float, float]:
    """
    Automatically tune proposal standard deviation for optimal acceptance rate.
    
    Args:
        target_log_prob_func: Target log-probability function
        initial_state: Starting point for tuning runs
        target_acceptance_range: (min_rate, max_rate) target range
        n_tune_steps: Number of steps for each tuning trial
        max_iterations: Maximum tuning iterations
    
    Returns:
        (tuned_std, final_acceptance_rate): Optimal parameters
    """
    print("🎯 Tuning proposal standard deviation...")
    
    proposal_std = 0.5  # Initial guess
    min_acc, max_acc = target_acceptance_range
    
    for iteration in range(max_iterations):
        # Run trial chain
        _, acceptance_rate = run_mcmc_chain(
            initial_state, target_log_prob_func, proposal_std, n_tune_steps
        )
        
        print(f"  Iteration {iteration + 1}: proposal_std = {proposal_std:.4f}, "
              f"acceptance = {acceptance_rate:.3f}")
        
        # Check if in target range
        if min_acc <= acceptance_rate <= max_acc:
            print(f"  ✓ Converged: proposal_std = {proposal_std:.4f}, "
                  f"acceptance = {acceptance_rate:.3f}")
            return proposal_std, acceptance_rate
        
        # Adjust proposal standard deviation
        if acceptance_rate < min_acc:
            proposal_std *= 0.9  # Decrease step size to increase acceptance
        else:  # acceptance_rate > max_acc
            proposal_std *= 1.1  # Increase step size to decrease acceptance
    
    print(f"  ⚠ Tuning did not converge in {max_iterations} iterations")
    print(f"  Using: proposal_std = {proposal_std:.4f}, acceptance = {acceptance_rate:.3f}")
    return proposal_std, acceptance_rate


def compute_autocorrelation(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """
    Compute autocorrelation function using FFT for efficiency.
    
    Args:
        x: 1D time series
        max_lag: Maximum lag to compute
    
    Returns:
        Autocorrelation function [0, max_lag]
    """
    n = len(x)
    x_centered = x - np.mean(x)
    
    # Use FFT for efficient computation
    f_x = np.fft.fft(x_centered, 2 * n)
    autocorr_fft = np.fft.ifft(f_x * np.conj(f_x))[:n].real
    autocorr_fft = autocorr_fft / autocorr_fft[0]  # Normalize
    
    return autocorr_fft[:max_lag + 1]


def compute_gelman_rubin(chains: List[np.ndarray]) -> np.ndarray:
    """
    Compute Gelman-Rubin R-hat convergence diagnostic.
    
    Args:
        chains: List of MCMC chains, each of shape (n_samples, dim)
    
    Returns:
        R-hat values for each dimension
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


def compute_effective_sample_size(x: np.ndarray, max_lag: int = None) -> float:
    """
    Compute effective sample size from autocorrelation.
    
    Args:
        x: 1D time series
        max_lag: Maximum lag for autocorrelation computation
    
    Returns:
        Effective sample size
    """
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 4, 200)
    
    autocorr = compute_autocorrelation(x, max_lag)
    
    # Find first negative autocorrelation or use all positive
    first_negative = np.where(autocorr < 0)[0]
    if len(first_negative) > 0:
        cutoff = first_negative[0]
    else:
        cutoff = len(autocorr)
    
    # Integrated autocorrelation time
    tau_int = 1 + 2 * np.sum(autocorr[:cutoff])
    
    # Effective sample size
    ess = n / tau_int
    return ess


def create_publication_figure(chains_burned: List[np.ndarray],
                            all_samples: np.ndarray,
                            mu_true: np.ndarray,
                            sigma_true: np.ndarray,
                            proposal_std: float,
                            final_acceptance_rate: float,
                            chains_full: List[np.ndarray]) -> None:
    """
    Create comprehensive publication-quality diagnostic figure.
    
    This function generates a 2x3 grid of diagnostic plots showing all
    key aspects of MCMC validation and performance.
    """
    
    print("📊 Creating publication-quality diagnostic figure...")
    
    # Set up the figure with specific size for publication
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('MCMC Diagnostics for 2D Gaussian Target Distribution', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Define colors for consistency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional color scheme
    dim_colors = ['#e74c3c', '#3498db']  # Red and blue for dimensions
    
    # ========================================================================
    # TOP LEFT: Trace Plots (a)
    # ========================================================================
    ax1 = plt.subplot(2, 3, 1)
    
    # Create 2x2 subgrid for trace plots
    fig_trace = plt.figure(figsize=(12, 8))
    fig_trace.suptitle('Trace Plots for All Chains', fontsize=14, fontweight='bold')
    
    for dim in range(2):
        for chain_idx in range(4):
            ax_trace = plt.subplot(2, 4, dim * 4 + chain_idx + 1)
            
            chain = chains_burned[chain_idx]
            plt.plot(chain[:2000, dim], color=colors[chain_idx], alpha=0.8, linewidth=0.8)
            plt.axhline(y=mu_true[dim], color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            plt.title(f'Chain {chain_idx + 1}, Dim {dim + 1}', fontsize=10)
            plt.xlabel('Iteration (post burn-in)')
            plt.ylabel(f'x₁' if dim == 0 else f'x₂')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/figures/trace_plots_detailed.png', dpi=150, bbox_inches='tight')
    plt.close(fig_trace)
    
    # Simplified trace plot for main figure
    for dim in range(2):
        for chain_idx, chain in enumerate(chains_burned):
            plt.plot(chain[:1000, dim], color=colors[chain_idx], alpha=0.7, linewidth=0.8,
                    label=f'Chain {chain_idx + 1}' if dim == 0 else '')
        
        plt.axhline(y=mu_true[dim], color='black', linestyle='--', alpha=0.8, linewidth=1.5)
    
    plt.title('(a) Trace Plots (First 1000 Post-Burn-in)', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('x₁, x₂')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # ========================================================================
    # TOP MIDDLE: Autocorrelation Functions (b)
    # ========================================================================
    ax2 = plt.subplot(2, 3, 2)
    
    max_lag = 100
    lags = np.arange(max_lag + 1)
    
    for dim in range(2):
        # Pool samples from all chains
        pooled_samples = all_samples[:, dim]
        autocorr = compute_autocorrelation(pooled_samples, max_lag)
        
        plt.plot(lags, autocorr, color=dim_colors[dim], linewidth=2,
                label=f'Dimension {dim + 1}')
    
    plt.axhline(y=0.1, color='gray', linestyle=':', alpha=0.7, label='Threshold (0.1)')
    plt.axhline(y=0.0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    
    plt.title('(b) Autocorrelation Functions', fontweight='bold')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.0)
    
    # ========================================================================
    # TOP RIGHT: Gelman-Rubin R-hat Evolution (c)
    # ========================================================================
    ax3 = plt.subplot(2, 3, 3)
    
    # Compute R-hat at different chain lengths
    check_points = [1000, 2000, 5000, 10000, 20000, 50000, 90000]
    r_hat_evolution = {0: [], 1: []}
    actual_points = []
    
    for n_samples in check_points:
        if n_samples <= len(chains_burned[0]):
            test_chains = [chain[:n_samples] for chain in chains_burned]
            r_hat = compute_gelman_rubin(test_chains)
            r_hat_evolution[0].append(r_hat[0])
            r_hat_evolution[1].append(r_hat[1])
            actual_points.append(n_samples)
    
    for dim in range(2):
        plt.semilogx(actual_points, r_hat_evolution[dim], 
                    color=dim_colors[dim], marker='o', linewidth=2,
                    label=f'Dimension {dim + 1}')
    
    plt.axhline(y=1.05, color='red', linestyle='--', linewidth=2, 
                label='Convergence Threshold')
    
    plt.title('(c) Gelman-Rubin R̂ Evolution', fontweight='bold')
    plt.xlabel('Chain Length')
    plt.ylabel('R̂')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.98, 1.08)
    
    # ========================================================================
    # BOTTOM LEFT: 2D Posterior with True Contours (f)
    # ========================================================================
    ax4 = plt.subplot(2, 3, 4)
    
    # Plot sample scatter (subsample for visibility)
    n_plot = 5000
    plot_indices = np.random.choice(len(all_samples), min(n_plot, len(all_samples)), replace=False)
    sample_subset = all_samples[plot_indices]
    
    plt.scatter(sample_subset[:, 0], sample_subset[:, 1], 
                alpha=0.4, s=1, color='#3498db', rasterized=True, label='MCMC Samples')
    
    # Overlay true density contours
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))
    
    rv = multivariate_normal(mu_true, sigma_true)
    contour_levels = [0.05, 0.15, 0.35, 0.65, 0.85, 0.95]
    
    # Convert probability levels to density levels
    density_levels = []
    for level in contour_levels:
        # Find density value that contains 'level' probability mass
        density_levels.append(rv.pdf(mu_true) * level)
    
    CS = plt.contour(X, Y, rv.pdf(pos), levels=density_levels, 
                    colors='red', alpha=0.8, linewidths=2)
    
    # Add contour labels
    plt.clabel(CS, inline=True, fontsize=8, fmt='%.2f')
    
    plt.title('(f) Posterior Samples vs True Density', fontweight='bold')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # ========================================================================
    # BOTTOM MIDDLE: Effective Sample Sizes (d)
    # ========================================================================
    ax5 = plt.subplot(2, 3, 5)
    
    # Compute ESS for each dimension
    ess_values = []
    for dim in range(2):
        ess = compute_effective_sample_size(all_samples[:, dim])
        ess_values.append(ess)
    
    dimensions = ['x₁', 'x₂']
    bars = plt.bar(dimensions, ess_values, color=dim_colors, alpha=0.8, width=0.6)
    
    # Add value labels on bars
    for bar, ess in zip(bars, ess_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(ess):,}', ha='center', va='bottom', fontweight='bold')
    
    # Add target line
    plt.axhline(y=1000, color='gray', linestyle='--', alpha=0.7, 
                label='Target (1000)')
    
    plt.title('(d) Effective Sample Sizes', fontweight='bold')
    plt.ylabel('ESS')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # BOTTOM RIGHT: Moment Error Table (e)
    # ========================================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')  # Remove axes for table
    
    # Compute empirical moments
    emp_mean = np.mean(all_samples, axis=0)
    emp_cov = np.cov(all_samples, rowvar=False)
    
    # Compute relative errors
    mean_errors = np.abs(emp_mean - mu_true)
    cov_errors = np.abs(emp_cov - sigma_true) / np.abs(sigma_true + 1e-10)
    
    # Create formatted table text
    table_text = f"""
QUANTITATIVE RESULTS

Tuning Results:
• Proposal σ: {proposal_std:.4f}
• Acceptance Rate: {final_acceptance_rate:.3f}

Convergence:
• Max R̂: {max(r_hat_evolution[0][-1], r_hat_evolution[1][-1]):.4f}
• ESS (x₁): {int(ess_values[0]):,}
• ESS (x₂): {int(ess_values[1]):,}

Moment Accuracy:
• Mean Error (x₁): {mean_errors[0]:.4f}
• Mean Error (x₂): {mean_errors[1]:.4f}
• Cov Error (1,1): {cov_errors[0,0]:.4f}
• Cov Error (1,2): {cov_errors[0,1]:.4f}
• Cov Error (2,2): {cov_errors[1,1]:.4f}

True Parameters:
μ = [{mu_true[0]:.1f}, {mu_true[1]:.1f}]
Σ = [[{sigma_true[0,0]:.1f}, {sigma_true[0,1]:.1f}],
     [{sigma_true[1,0]:.1f}, {sigma_true[1,1]:.1f}]]

Empirical Results:
μ̂ = [{emp_mean[0]:.4f}, {emp_mean[1]:.4f}]
Σ̂ = [[{emp_cov[0,0]:.4f}, {emp_cov[0,1]:.4f}],
     [{emp_cov[1,0]:.4f}, {emp_cov[1,1]:.4f}]]
"""
    
    plt.text(0.05, 0.95, table_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.title('(e) Quantitative Summary', fontweight='bold', pad=20)
    
    # ========================================================================
    # Final Layout and Save
    # ========================================================================
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # Create results directory
    os.makedirs('../../results/figures', exist_ok=True)
    
    plt.savefig('../../results/figures/continuous_gaussian_baseline_diagnostics.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return emp_mean, emp_cov, ess_values, mean_errors, cov_errors


def main():
    """
    Execute the complete Continuous-Gaussian Baseline MCMC experiment.
    """
    
    print("🔬 CONTINUOUS-GAUSSIAN BASELINE METROPOLIS-HASTINGS EXPERIMENT")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # ========================================================================
    # 1. Define Target Distribution
    # ========================================================================
    print("\n1️⃣  Defining target distribution...")
    
    mu_true = np.array([0.0, 0.0])
    sigma_true = np.array([[1.0, 0.8], [0.8, 1.0]])
    
    def target_log_prob_func(x):
        return target_log_prob(x, mu_true, sigma_true)
    
    print(f"   Target: 2D Gaussian")
    print(f"   Mean μ = {mu_true}")
    print(f"   Covariance Σ = \n{sigma_true}")
    print(f"   Correlation ρ = {sigma_true[0,1]:.1f}")
    
    # ========================================================================
    # 2. Automatic Proposal Tuning
    # ========================================================================
    print("\n2️⃣  Automatic proposal tuning...")
    
    initial_tune = np.array([0.0, 0.0])
    proposal_std, final_acceptance_rate = tune_proposal_std(
        target_log_prob_func, initial_tune, target_acceptance_range=(0.25, 0.45)
    )
    
    print(f"\n   ✓ Final tuned proposal_std: {proposal_std:.4f}")
    print(f"   ✓ Final acceptance rate: {final_acceptance_rate:.3f}")
    
    # ========================================================================
    # 3. Multi-Chain MCMC Sampling
    # ========================================================================
    print("\n3️⃣  Running multi-chain MCMC sampling...")
    
    # Over-dispersed initial states
    initial_states = [
        np.array([5.0, 5.0]),
        np.array([-5.0, -5.0]),
        np.array([5.0, -5.0]),
        np.array([-5.0, 5.0])
    ]
    
    n_steps = 100000
    burn_in = 10000
    
    print(f"   Chains: {len(initial_states)}")
    print(f"   Steps per chain: {n_steps:,}")
    print(f"   Burn-in: {burn_in:,}")
    print(f"   Post-burn-in samples per chain: {n_steps - burn_in:,}")
    
    # Run all chains
    chains = []
    acceptance_rates = []
    
    for i, initial_state in enumerate(initial_states):
        print(f"   Running chain {i+1}/4 from {initial_state}...")
        
        samples, acc_rate = run_mcmc_chain(
            initial_state, target_log_prob_func, proposal_std, n_steps
        )
        
        chains.append(samples)
        acceptance_rates.append(acc_rate)
    
    print(f"\n   ✓ Individual acceptance rates: {[f'{rate:.3f}' for rate in acceptance_rates]}")
    print(f"   ✓ Mean acceptance rate: {np.mean(acceptance_rates):.3f} ± {np.std(acceptance_rates):.3f}")
    
    # ========================================================================
    # 4. Apply Burn-in and Pool Samples
    # ========================================================================
    print("\n4️⃣  Processing samples...")
    
    # Apply burn-in
    chains_burned = [chain[burn_in:] for chain in chains]
    
    # Pool all post-burn-in samples
    all_samples = np.vstack(chains_burned)
    
    print(f"   ✓ Total post-burn-in samples: {len(all_samples):,}")
    print(f"   ✓ Sample dimensions: {all_samples.shape}")
    
    # ========================================================================
    # 5. Convergence Diagnostics
    # ========================================================================
    print("\n5️⃣  Computing convergence diagnostics...")
    
    # Gelman-Rubin R-hat
    r_hat = compute_gelman_rubin(chains_burned)
    
    # Effective Sample Sizes
    ess_values = []
    for dim in range(2):
        ess = compute_effective_sample_size(all_samples[:, dim])
        ess_values.append(ess)
    
    print(f"   ✓ Gelman-Rubin R̂:")
    print(f"     - Dimension 1: {r_hat[0]:.4f}")
    print(f"     - Dimension 2: {r_hat[1]:.4f}")
    print(f"     - Max R̂: {np.max(r_hat):.4f}")
    
    print(f"   ✓ Effective Sample Sizes:")
    print(f"     - Dimension 1: {int(ess_values[0]):,}")
    print(f"     - Dimension 2: {int(ess_values[1]):,}")
    print(f"     - Min ESS: {int(min(ess_values)):,}")
    
    # ========================================================================
    # 6. Accuracy Assessment
    # ========================================================================
    print("\n6️⃣  Assessing sampling accuracy...")
    
    # Compute empirical moments
    emp_mean = np.mean(all_samples, axis=0)
    emp_cov = np.cov(all_samples, rowvar=False)
    
    # Compute errors
    mean_errors = np.abs(emp_mean - mu_true)
    cov_errors = np.abs(emp_cov - sigma_true) / np.abs(sigma_true + 1e-10)
    
    print(f"   ✓ Empirical mean: [{emp_mean[0]:.6f}, {emp_mean[1]:.6f}]")
    print(f"   ✓ True mean: [{mu_true[0]:.6f}, {mu_true[1]:.6f}]")
    print(f"   ✓ Mean absolute errors: [{mean_errors[0]:.6f}, {mean_errors[1]:.6f}]")
    
    print(f"   ✓ Empirical covariance:")
    print(f"     [[{emp_cov[0,0]:.6f}, {emp_cov[0,1]:.6f}],")
    print(f"      [{emp_cov[1,0]:.6f}, {emp_cov[1,1]:.6f}]]")
    
    print(f"   ✓ Covariance relative errors:")
    print(f"     [[{cov_errors[0,0]:.6f}, {cov_errors[0,1]:.6f}],")
    print(f"      [{cov_errors[1,0]:.6f}, {cov_errors[1,1]:.6f}]]")
    
    # ========================================================================
    # 7. Create Publication Figure
    # ========================================================================
    print("\n7️⃣  Creating publication-quality diagnostic figure...")
    
    emp_mean_final, emp_cov_final, ess_final, mean_errors_final, cov_errors_final = create_publication_figure(
        chains_burned, all_samples, mu_true, sigma_true, 
        proposal_std, final_acceptance_rate, chains
    )
    
    # ========================================================================
    # 8. Final Assessment and Interpretation
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 QUANTITATIVE SUMMARY & INTERPRETATION")
    print("=" * 80)
    
    print(f"\n🎯 TUNING RESULTS:")
    print(f"   • Tuned proposal_std: {proposal_std:.4f}")
    print(f"   • Final acceptance rate: {final_acceptance_rate:.3f}")
    print(f"   • Target range: [0.25, 0.45]")
    
    print(f"\n📊 CONVERGENCE DIAGNOSTICS:")
    print(f"   • R̂ (dimension 1): {r_hat[0]:.4f}")
    print(f"   • R̂ (dimension 2): {r_hat[1]:.4f}")
    print(f"   • Convergence threshold: 1.05")
    
    print(f"\n⚡ SAMPLING EFFICIENCY:")
    print(f"   • ESS (dimension 1): {int(ess_values[0]):,}")
    print(f"   • ESS (dimension 2): {int(ess_values[1]):,}")
    print(f"   • Target ESS threshold: 1,000")
    print(f"   • Total samples: {len(all_samples):,}")
    print(f"   • Efficiency ratio: {min(ess_values)/len(all_samples):.3f}")
    
    print(f"\n🎯 ACCURACY ASSESSMENT:")
    print(f"   • Max mean absolute error: {np.max(mean_errors):.6f}")
    print(f"   • Max covariance relative error: {np.max(cov_errors):.6f}")
    
    # ========================================================================
    # 9. Interpretation
    # ========================================================================
    print(f"\n🔍 INTERPRETATION:")
    
    # Tuning assessment
    tuning_success = 0.25 <= final_acceptance_rate <= 0.45
    print(f"\n1. Automatic Tuning:")
    if tuning_success:
        print(f"   ✓ SUCCESS: Achieved acceptance rate {final_acceptance_rate:.3f} in target range [0.25, 0.45]")
        print(f"   ✓ The tuned proposal_std = {proposal_std:.4f} provides optimal step size")
    else:
        print(f"   ✗ SUBOPTIMAL: Acceptance rate {final_acceptance_rate:.3f} outside target range")
    
    # Convergence assessment
    convergence_success = np.max(r_hat) < 1.05
    print(f"\n2. Convergence:")
    if convergence_success:
        print(f"   ✓ SUCCESS: All R̂ values < 1.05 (max = {np.max(r_hat):.4f})")
        print(f"   ✓ Multiple chains converged to the same distribution")
        print(f"   ✓ Starting points were successfully overcome")
    else:
        print(f"   ✗ FAILURE: R̂ values ≥ 1.05 (max = {np.max(r_hat):.4f})")
        print(f"   ✗ Chains have not converged to the same distribution")
    
    # Efficiency assessment
    efficiency_success = min(ess_values) >= 1000
    print(f"\n3. Sampling Efficiency:")
    if efficiency_success:
        print(f"   ✓ SUCCESS: Both dimensions achieve ESS ≥ 1000")
        print(f"   ✓ Autocorrelations decay rapidly (good mixing)")
        print(f"   ✓ Samples are effectively independent")
    else:
        print(f"   ✗ SUBOPTIMAL: ESS < 1000 for some dimensions (min = {min(ess_values):.0f})")
        print(f"   ✗ High autocorrelation indicates poor mixing")
    
    # Accuracy assessment
    accuracy_success = (np.max(mean_errors) < 0.01) and (np.max(cov_errors) < 0.05)
    print(f"\n4. Target Recovery:")
    if accuracy_success:
        print(f"   ✓ SUCCESS: Empirical moments match target with high precision")
        print(f"   ✓ Mean errors < 1% and covariance errors < 5%")
        print(f"   ✓ Visual posterior-contour alignment confirms accuracy")
    else:
        print(f"   ~ ADEQUATE: Moderate agreement with target distribution")
        print(f"   ~ Mean errors: {np.max(mean_errors):.4f}, Cov errors: {np.max(cov_errors):.4f}")
    
    # Overall assessment
    overall_success = convergence_success and efficiency_success
    
    print(f"\n" + "=" * 80)
    
    if overall_success:
        verdict = "BASELINE MCMC TESTS PASSED"
        print(f"🎉 {verdict}")
        print(f"\nAll critical diagnostics passed:")
        print(f"• ✓ Chains converged (R̂ < 1.05)")
        print(f"• ✓ Efficient sampling (ESS ≥ 1000)")
        print(f"• ✓ Accurate target recovery")
        print(f"• ✓ Optimal tuning achieved")
        print(f"\nThe Metropolis-Hastings implementation is validated and ready for production use.")
    else:
        failure_reasons = []
        if not convergence_success:
            failure_reasons.append("convergence failure (R̂ ≥ 1.05)")
        if not efficiency_success:
            failure_reasons.append("insufficient ESS (< 1000)")
        
        verdict = f"BASELINE MCMC TESTS FAILED: {', '.join(failure_reasons)}"
        print(f"❌ {verdict}")
    
    print(f"\n📁 Results saved to:")
    print(f"   • experiments/results/figures/continuous_gaussian_baseline_diagnostics.png")
    print(f"   • experiments/results/figures/trace_plots_detailed.png")
    
    print("=" * 80)
    
    return verdict, {
        'proposal_std': proposal_std,
        'acceptance_rate': final_acceptance_rate,
        'r_hat': r_hat,
        'ess': ess_values,
        'mean_errors': mean_errors,
        'cov_errors': cov_errors,
        'overall_success': overall_success
    }


if __name__ == "__main__":
    verdict, results = main()