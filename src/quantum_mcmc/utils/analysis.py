"""Analysis and diagnostic utilities for quantum MCMC algorithms.

This module provides comprehensive tools for analyzing and evaluating the
performance of quantum MCMC algorithms, including quantum walks, phase
estimation, and reflection operators. The utilities focus on spectral
analysis, distribution comparison, state fidelity, and convergence diagnostics.

These tools are essential for understanding the efficiency and accuracy of
quantum MCMC implementations, enabling rigorous performance evaluation and
algorithm optimization.

References:
    Montanaro, A. (2015). Quantum speedup of Monte Carlo methods.
    Proceedings of the Royal Society A, 471(2181), 20150301.
    
    Chakraborty, S., et al. (2016). The power of block-encoded matrix powers:
    improved regression techniques via faster Hamiltonian simulation.
    arXiv:1804.01973.

Author: [Your Name]
Date: 2025-01-23
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy import stats
from scipy.special import rel_entr
import warnings

# Qiskit imports for quantum state analysis
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit.quantum_info import partial_trace, entropy
from qiskit import QuantumCircuit
from qiskit.result import Result

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def compute_spectral_gap(singular_values: np.ndarray) -> float:
    """Calculate the spectral gap of a matrix from its singular values.
    
    The spectral gap is the difference between the largest and second-largest
    singular values (or eigenvalues for normal matrices). This quantity is
    crucial for understanding mixing times and convergence rates in quantum
    and classical MCMC algorithms.
    
    For quantum walks, the spectral gap determines:
    - Mixing time: O(1/gap)
    - Phase estimation precision requirements
    - Potential quantum speedup
    
    Args:
        singular_values: Array of singular values sorted in descending order
    
    Returns:
        Spectral gap (Ã - Ã‚)
    
    Raises:
        ValueError: If fewer than 2 singular values provided
    
    Example:
        >>> sv = np.array([1.0, 0.4, 0.2, 0.1])
        >>> gap = compute_spectral_gap(sv)
        >>> print(f"Spectral gap: {gap}")
        Spectral gap: 0.6
    """
    if len(singular_values) < 2:
        raise ValueError("Need at least 2 singular values to compute gap")
    
    # Ensure descending order
    sv_sorted = np.sort(singular_values)[::-1]
    
    # Compute gap between largest and second-largest
    gap = sv_sorted[0] - sv_sorted[1]
    
    # Validate result
    if gap < 0:
        warnings.warn("Negative spectral gap detected; check singular values")
        gap = 0.0
    
    return float(gap)


def total_variation_distance(dist1: np.ndarray, dist2: np.ndarray) -> float:
    """Compute total variation distance between probability distributions.
    
    The total variation distance is defined as:
        TV(P, Q) = (1/2) £b |P(i) - Q(i)|
    
    This metric ranges from 0 (identical distributions) to 1 (disjoint support).
    It's the standard measure for MCMC convergence and mixing.
    
    Args:
        dist1: First probability distribution
        dist2: Second probability distribution
    
    Returns:
        Total variation distance in [0, 1]
    
    Raises:
        ValueError: If distributions have different sizes or invalid probabilities
    
    Example:
        >>> p = np.array([0.3, 0.7])
        >>> q = np.array([0.4, 0.6])
        >>> tv = total_variation_distance(p, q)
        >>> print(f"TV distance: {tv}")
        TV distance: 0.1
    """
    # Validate inputs
    if len(dist1) != len(dist2):
        raise ValueError(f"Distributions must have same size: {len(dist1)} vs {len(dist2)}")
    
    # Normalize if needed (with tolerance for numerical errors)
    sum1, sum2 = np.sum(dist1), np.sum(dist2)
    if not np.allclose(sum1, 1.0, atol=1e-10):
        warnings.warn(f"dist1 sums to {sum1}, normalizing")
        dist1 = dist1 / sum1
    if not np.allclose(sum2, 1.0, atol=1e-10):
        warnings.warn(f"dist2 sums to {sum2}, normalizing")
        dist2 = dist2 / sum2
    
    # Check non-negativity
    if np.any(dist1 < -1e-10) or np.any(dist2 < -1e-10):
        raise ValueError("Distributions must have non-negative entries")
    
    # Compute TV distance
    tv_distance = 0.5 * np.sum(np.abs(dist1 - dist2))
    
    # Ensure result is in valid range
    tv_distance = np.clip(tv_distance, 0.0, 1.0)
    
    return float(tv_distance)


def compute_overlap(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute quantum state overlap (inner product squared).
    
    For quantum states |Èé and |Æé, computes |èÈ|Æé|².
    This measures the probability of finding one state in the other.
    
    Args:
        state1: First quantum state vector
        state2: Second quantum state vector
    
    Returns:
        Overlap |èÈ|Æé|² in [0, 1]
    
    Raises:
        ValueError: If states have different dimensions
    
    Example:
        >>> psi = np.array([1, 0]) / np.sqrt(1)  # |0é
        >>> phi = np.array([1, 1]) / np.sqrt(2)  # |+é
        >>> overlap = compute_overlap(psi, phi)
        >>> print(f"Overlap: {overlap}")
        Overlap: 0.5
    """
    if len(state1) != len(state2):
        raise ValueError(f"States must have same dimension: {len(state1)} vs {len(state2)}")
    
    # Normalize states
    norm1 = np.linalg.norm(state1)
    norm2 = np.linalg.norm(state2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        warnings.warn("Zero norm state detected")
        return 0.0
    
    state1_normalized = state1 / norm1
    state2_normalized = state2 / norm2
    
    # Compute inner product
    inner_product = np.vdot(state1_normalized, state2_normalized)
    
    # Return squared magnitude
    overlap = np.abs(inner_product) ** 2
    
    # Ensure valid range
    overlap = np.clip(overlap, 0.0, 1.0)
    
    return float(overlap)


def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Compute quantum state fidelity between density matrices.
    
    For density matrices Á and Ã, the fidelity is:
        F(Á, Ã) = Tr((Á Ã Á))²
    
    For pure states, this reduces to |èÈ|Æé|².
    
    Args:
        rho1: First density matrix
        rho2: Second density matrix
    
    Returns:
        Fidelity in [0, 1]
    
    Example:
        >>> rho = np.array([[1, 0], [0, 0]])  # |0éè0|
        >>> sigma = np.array([[0.5, 0.5], [0.5, 0.5]])  # |+éè+|
        >>> f = fidelity(rho, sigma)
        >>> print(f"Fidelity: {f}")
        Fidelity: 0.5
    """
    # Use Qiskit's implementation for numerical stability
    return float(state_fidelity(rho1, rho2))


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Compute trace distance between density matrices.
    
    The trace distance is:
        D(Á, Ã) = (1/2) Tr|Á - Ã|
    
    where |A| = (A A) is the matrix absolute value.
    
    Args:
        rho1: First density matrix
        rho2: Second density matrix
    
    Returns:
        Trace distance in [0, 1]
    
    Example:
        >>> rho = np.eye(2) / 2  # Maximally mixed state
        >>> sigma = np.array([[1, 0], [0, 0]])  # Pure |0é
        >>> d = trace_distance(rho, sigma)
        >>> print(f"Trace distance: {d}")
        Trace distance: 0.5
    """
    diff = rho1 - rho2
    
    # Compute eigenvalues of difference
    eigenvals = np.linalg.eigvalsh(diff)
    
    # Trace distance is half the sum of absolute eigenvalues
    distance = 0.5 * np.sum(np.abs(eigenvals))
    
    return float(np.clip(distance, 0.0, 1.0))


def mixing_time(
    transition_matrix: np.ndarray,
    epsilon: float = 0.01,
    initial_dist: Optional[np.ndarray] = None,
    max_steps: int = 10000
) -> int:
    """Estimate mixing time of a Markov chain.
    
    Finds the number of steps needed for the chain to reach within
    µ total variation distance of the stationary distribution.
    
    Args:
        transition_matrix: Row-stochastic transition matrix
        epsilon: Target accuracy (TV distance)
        initial_dist: Initial distribution (uniform if None)
        max_steps: Maximum steps to simulate
    
    Returns:
        Mixing time (number of steps)
    
    Example:
        >>> P = np.array([[0.9, 0.1], [0.2, 0.8]])
        >>> t_mix = mixing_time(P, epsilon=0.01)
        >>> print(f"Mixing time: {t_mix} steps")
    """
    n = transition_matrix.shape[0]
    
    # Find stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
    stationary_idx = np.argmax(np.abs(eigenvals - 1.0) < 1e-10)
    stationary = np.real(eigenvecs[:, stationary_idx])
    stationary = stationary / np.sum(stationary)
    
    # Initialize distribution
    if initial_dist is None:
        current_dist = np.ones(n) / n
    else:
        current_dist = initial_dist.copy()
    
    # Simulate until mixed
    for t in range(max_steps):
        tv_dist = total_variation_distance(current_dist, stationary)
        
        if tv_dist <= epsilon:
            return t
        
        # Update distribution
        current_dist = current_dist @ transition_matrix
    
    warnings.warn(f"Did not mix within {max_steps} steps")
    return max_steps


def effective_sample_size(
    samples: np.ndarray,
    method: str = "autocorrelation"
) -> float:
    """Compute effective sample size for MCMC samples.
    
    Estimates the number of effectively independent samples accounting
    for autocorrelation in the Markov chain.
    
    Args:
        samples: Array of samples (can be multidimensional)
        method: ESS estimation method - "autocorrelation" or "batch_means"
    
    Returns:
        Effective sample size
    
    Example:
        >>> samples = np.random.normal(0, 1, 1000)
        >>> ess = effective_sample_size(samples)
        >>> print(f"ESS: {ess:.1f} out of {len(samples)} samples")
    """
    n = len(samples)
    
    if method == "autocorrelation":
        # Compute autocorrelation function
        if samples.ndim == 1:
            acf = _autocorrelation_1d(samples)
        else:
            # For multidimensional, use first dimension
            acf = _autocorrelation_1d(samples[:, 0])
        
        # Find first negative autocorrelation
        first_negative = np.where(acf < 0)[0]
        if len(first_negative) > 0:
            cutoff = first_negative[0]
        else:
            cutoff = len(acf)
        
        # Estimate autocorrelation time
        tau = 1 + 2 * np.sum(acf[1:cutoff])
        
        # ESS = N / tau
        ess = n / tau
        
    elif method == "batch_means":
        # Batch means method
        batch_size = int(np.sqrt(n))
        n_batches = n // batch_size
        
        if samples.ndim == 1:
            batches = samples[:n_batches * batch_size].reshape(n_batches, batch_size)
            batch_means = np.mean(batches, axis=1)
            
            # Variance of batch means
            var_batch_means = np.var(batch_means, ddof=1)
            
            # Sample variance
            var_samples = np.var(samples, ddof=1)
            
            # ESS estimate
            if var_batch_means > 0:
                ess = batch_size * var_samples / var_batch_means
            else:
                ess = n
        else:
            # For multidimensional, compute for each dimension
            ess_list = []
            for d in range(samples.shape[1]):
                ess_d = effective_sample_size(samples[:, d], method=method)
                ess_list.append(ess_d)
            ess = np.mean(ess_list)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure reasonable bounds
    ess = np.clip(ess, 1, n)
    
    return float(ess)


def _autocorrelation_1d(x: np.ndarray) -> np.ndarray:
    """Compute normalized autocorrelation function."""
    n = len(x)
    x = x - np.mean(x)
    c0 = np.dot(x, x) / n
    
    acf = []
    for k in range(n):
        if k < n // 4:  # Only compute for first quarter
            ck = np.dot(x[:-k or None], x[k:]) / (n - k)
            acf.append(ck / c0)
        else:
            break
    
    return np.array(acf)


def quantum_process_fidelity(
    ideal_circuit: QuantumCircuit,
    noisy_circuit: QuantumCircuit,
    num_qubits: int,
    shots: int = 8192
) -> float:
    """Estimate process fidelity between ideal and noisy quantum circuits.
    
    Uses state tomography on maximally entangled state to estimate
    average gate fidelity.
    
    Args:
        ideal_circuit: Ideal quantum circuit
        noisy_circuit: Noisy implementation
        num_qubits: Number of qubits
        shots: Number of measurement shots
    
    Returns:
        Process fidelity estimate
    
    Example:
        >>> ideal = QuantumCircuit(2)
        >>> ideal.h(0)
        >>> ideal.cx(0, 1)
        >>> noisy = ideal.copy()
        >>> # Add noise model to noisy
        >>> f = quantum_process_fidelity(ideal, noisy, 2)
    """
    # This is a simplified implementation
    # Full process tomography would be more complex
    
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    
    # Prepare maximally entangled state
    prep_circuit = QuantumCircuit(2 * num_qubits)
    for i in range(num_qubits):
        prep_circuit.h(i)
        prep_circuit.cx(i, i + num_qubits)
    
    # Apply process to first register
    test_ideal = prep_circuit.copy()
    test_noisy = prep_circuit.copy()
    
    # Add circuits to first register only
    test_ideal.append(ideal_circuit, range(num_qubits))
    test_noisy.append(noisy_circuit, range(num_qubits))
    
    # Simulate
    simulator = AerSimulator(method='statevector')
    
    # Get ideal statevector
    ideal_result = simulator.run(transpile(test_ideal, simulator)).result()
    ideal_sv = ideal_result.get_statevector()
    
    # Get noisy statevector
    noisy_result = simulator.run(transpile(test_noisy, simulator)).result()
    noisy_sv = noisy_result.get_statevector()
    
    # Compute fidelity
    process_fid = np.abs(ideal_sv.inner(noisy_sv)) ** 2
    
    return float(process_fid)


def phase_estimation_accuracy(
    measured_phases: np.ndarray,
    true_phases: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Analyze accuracy of phase estimation results.
    
    Computes various error metrics for quantum phase estimation.
    
    Args:
        measured_phases: Estimated phases from QPE
        true_phases: True eigenphases
        weights: Optional weights for phases
    
    Returns:
        Dictionary with error metrics:
            - mean_absolute_error
            - root_mean_square_error
            - max_error
            - success_probability (fraction within 1 std dev)
    
    Example:
        >>> measured = np.array([0.249, 0.502, 0.748])
        >>> true = np.array([0.25, 0.5, 0.75])
        >>> metrics = phase_estimation_accuracy(measured, true)
    """
    if len(measured_phases) != len(true_phases):
        raise ValueError("Phase arrays must have same length")
    
    # Handle phase wraparound
    errors = np.minimum(
        np.abs(measured_phases - true_phases),
        1 - np.abs(measured_phases - true_phases)
    )
    
    if weights is None:
        weights = np.ones(len(measured_phases))
    weights = weights / np.sum(weights)
    
    # Compute metrics
    mae = np.sum(weights * errors)
    rmse = np.sqrt(np.sum(weights * errors**2))
    max_err = np.max(errors)
    
    # Success probability (within resolution)
    resolution = 1.0 / (2 ** 8)  # Assume 8-bit precision
    success_prob = np.sum(weights[errors < resolution])
    
    return {
        'mean_absolute_error': float(mae),
        'root_mean_square_error': float(rmse),
        'max_error': float(max_err),
        'success_probability': float(success_prob),
        'resolution': float(resolution)
    }


def convergence_diagnostics(
    chain_samples: List[np.ndarray],
    target_distribution: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Comprehensive convergence diagnostics for MCMC chains.
    
    Implements multiple convergence tests including Gelman-Rubin,
    Geweke, and effective sample size.
    
    Args:
        chain_samples: List of sample arrays from different chains
        target_distribution: Known target distribution (if available)
    
    Returns:
        Dictionary with convergence metrics:
            - gelman_rubin: R-hat statistic
            - geweke_scores: Z-scores for each chain
            - effective_sample_sizes: ESS for each chain
            - inter_chain_variance: Between-chain variance
    
    Example:
        >>> chains = [np.random.normal(0, 1, 1000) for _ in range(4)]
        >>> diag = convergence_diagnostics(chains)
        >>> print(f"R-hat: {diag['gelman_rubin']:.3f}")
    """
    n_chains = len(chain_samples)
    
    if n_chains < 2:
        raise ValueError("Need at least 2 chains for convergence diagnostics")
    
    # Ensure all chains have same length
    chain_lengths = [len(chain) for chain in chain_samples]
    if len(set(chain_lengths)) > 1:
        warnings.warn("Chains have different lengths, truncating to minimum")
        min_length = min(chain_lengths)
        chain_samples = [chain[:min_length] for chain in chain_samples]
    
    n = len(chain_samples[0])
    
    # Gelman-Rubin statistic
    chain_means = np.array([np.mean(chain) for chain in chain_samples])
    chain_vars = np.array([np.var(chain, ddof=1) for chain in chain_samples])
    
    # Between-chain variance
    B = n * np.var(chain_means, ddof=1)
    
    # Within-chain variance
    W = np.mean(chain_vars)
    
    # Pooled variance estimate
    var_plus = ((n - 1) * W + B) / n
    
    # R-hat
    r_hat = np.sqrt(var_plus / W) if W > 0 else np.inf
    
    # Geweke diagnostic
    geweke_scores = []
    for chain in chain_samples:
        # Compare first 10% and last 50%
        n1 = int(0.1 * n)
        n2 = int(0.5 * n)
        
        mean1 = np.mean(chain[:n1])
        mean2 = np.mean(chain[-n2:])
        var1 = np.var(chain[:n1], ddof=1)
        var2 = np.var(chain[-n2:], ddof=1)
        
        # Z-score
        se = np.sqrt(var1/n1 + var2/n2)
        z = (mean1 - mean2) / se if se > 0 else 0
        geweke_scores.append(z)
    
    # Effective sample sizes
    ess_values = [effective_sample_size(chain) for chain in chain_samples]
    
    # If target distribution provided, compute KL divergence
    kl_divergences = []
    if target_distribution is not None:
        for chain in chain_samples:
            # Estimate empirical distribution
            hist, _ = np.histogram(chain, bins=len(target_distribution), 
                                  range=(0, len(target_distribution)))
            empirical = hist / np.sum(hist)
            
            # KL divergence
            kl = np.sum(rel_entr(empirical, target_distribution))
            kl_divergences.append(kl)
    
    diagnostics = {
        'gelman_rubin': float(r_hat),
        'geweke_scores': geweke_scores,
        'effective_sample_sizes': ess_values,
        'inter_chain_variance': float(B),
        'within_chain_variance': float(W),
        'n_chains': n_chains,
        'chain_length': n
    }
    
    if kl_divergences:
        diagnostics['kl_divergences'] = kl_divergences
    
    return diagnostics


def quantum_speedup_estimate(
    classical_mixing_time: int,
    quantum_mixing_time: int,
    classical_cost_per_step: float = 1.0,
    quantum_cost_per_step: float = 1.0,
    phase_estimation_overhead: float = 100.0
) -> Dict[str, float]:
    """Estimate quantum speedup for MCMC algorithm.
    
    Computes various speedup metrics accounting for different cost models
    and overheads.
    
    Args:
        classical_mixing_time: Classical random walk mixing time
        quantum_mixing_time: Quantum walk mixing time
        classical_cost_per_step: Cost of one classical step
        quantum_cost_per_step: Cost of one quantum step
        phase_estimation_overhead: QPE circuit overhead factor
    
    Returns:
        Dictionary with speedup metrics:
            - mixing_time_speedup: Raw mixing time ratio
            - wall_time_speedup: Actual time speedup
            - query_speedup: Oracle query speedup
            - break_even_point: When quantum becomes faster
    
    Example:
        >>> speedup = quantum_speedup_estimate(
        ...     classical_mixing_time=10000,
        ...     quantum_mixing_time=100
        ... )
        >>> print(f"Quantum speedup: {speedup['mixing_time_speedup']:.1f}x")
    """
    # Basic mixing time speedup
    mixing_speedup = classical_mixing_time / quantum_mixing_time
    
    # Account for cost per step
    classical_total_cost = classical_mixing_time * classical_cost_per_step
    quantum_total_cost = (quantum_mixing_time * quantum_cost_per_step * 
                         phase_estimation_overhead)
    
    wall_time_speedup = classical_total_cost / quantum_total_cost
    
    # Query complexity speedup (oracle calls)
    query_speedup = np.sqrt(mixing_speedup)  # Typical for quantum walks
    
    # Break-even point
    break_even = phase_estimation_overhead * quantum_cost_per_step / classical_cost_per_step
    
    return {
        'mixing_time_speedup': float(mixing_speedup),
        'wall_time_speedup': float(wall_time_speedup),
        'query_speedup': float(query_speedup),
        'break_even_point': float(break_even),
        'quantum_advantage': wall_time_speedup > 1.0
    }


def plot_convergence_diagnostics(
    diagnostics: Dict[str, Any],
    figsize: Tuple[float, float] = (12, 8)
) -> Figure:
    """Visualize convergence diagnostics.
    
    Creates diagnostic plots for MCMC convergence analysis.
    
    Args:
        diagnostics: Output from convergence_diagnostics()
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # R-hat value
    ax = axes[0, 0]
    ax.axhline(y=1.0, color='green', linestyle='--', label='Target')
    ax.axhline(y=1.1, color='orange', linestyle='--', label='Warning')
    ax.bar(['R-hat'], [diagnostics['gelman_rubin']], color='blue', alpha=0.7)
    ax.set_ylabel('Gelman-Rubin Statistic')
    ax.set_title('Chain Convergence (R-hat)')
    ax.legend()
    ax.set_ylim([0.9, max(1.2, diagnostics['gelman_rubin'] * 1.1)])
    
    # Geweke scores
    ax = axes[0, 1]
    z_scores = diagnostics['geweke_scores']
    chains = range(1, len(z_scores) + 1)
    ax.bar(chains, z_scores, color='purple', alpha=0.7)
    ax.axhline(y=2, color='red', linestyle='--', label='±2Ã')
    ax.axhline(y=-2, color='red', linestyle='--')
    ax.set_xlabel('Chain')
    ax.set_ylabel('Z-score')
    ax.set_title('Geweke Diagnostic')
    ax.legend()
    
    # Effective sample sizes
    ax = axes[1, 0]
    ess = diagnostics['effective_sample_sizes']
    chain_length = diagnostics['chain_length']
    
    x = range(1, len(ess) + 1)
    ax.bar(x, ess, color='green', alpha=0.7, label='ESS')
    ax.axhline(y=chain_length, color='blue', linestyle='--', 
               label=f'Total samples ({chain_length})')
    ax.set_xlabel('Chain')
    ax.set_ylabel('Effective Sample Size')
    ax.set_title('ESS by Chain')
    ax.legend()
    
    # Variance components
    ax = axes[1, 1]
    B = diagnostics['inter_chain_variance']
    W = diagnostics['within_chain_variance']
    
    ax.bar(['Between-chain', 'Within-chain'], [B, W], 
           color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('Variance')
    ax.set_title('Variance Decomposition')
    
    plt.tight_layout()
    return fig


def entropy_analysis(
    distribution: np.ndarray,
    base: float = 2
) -> Dict[str, float]:
    """Compute various entropy measures for a probability distribution.
    
    Args:
        distribution: Probability distribution
        base: Logarithm base (2 for bits, e for nats)
    
    Returns:
        Dictionary with entropy measures:
            - shannon_entropy
            - min_entropy
            - max_entropy
            - relative_entropy (from uniform)
    
    Example:
        >>> p = np.array([0.5, 0.3, 0.2])
        >>> ent = entropy_analysis(p)
        >>> print(f"Shannon entropy: {ent['shannon_entropy']:.3f} bits")
    """
    # Normalize
    p = distribution / np.sum(distribution)
    
    # Remove zeros for log
    p_nonzero = p[p > 0]
    
    # Shannon entropy
    if base == 2:
        shannon = -np.sum(p_nonzero * np.log2(p_nonzero))
    else:
        shannon = -np.sum(p_nonzero * np.log(p_nonzero)) / np.log(base)
    
    # Min-entropy (Rényi entropy of order )
    min_entropy = -np.log(np.max(p)) / np.log(base)
    
    # Max entropy (uniform distribution)
    n = len(p)
    max_entropy = np.log(n) / np.log(base)
    
    # Relative entropy from uniform
    uniform = np.ones(n) / n
    if base == 2:
        kl_uniform = np.sum(p_nonzero * np.log2(p_nonzero * n))
    else:
        kl_uniform = np.sum(p_nonzero * np.log(p_nonzero * n)) / np.log(base)
    
    return {
        'shannon_entropy': float(shannon),
        'min_entropy': float(min_entropy),
        'max_entropy': float(max_entropy),
        'relative_entropy_from_uniform': float(kl_uniform),
        'normalized_entropy': float(shannon / max_entropy) if max_entropy > 0 else 0
    }


def sample_quality_metrics(
    samples: np.ndarray,
    target_distribution: Optional[np.ndarray] = None,
    n_bins: int = 50
) -> Dict[str, float]:
    """Comprehensive quality metrics for MCMC samples.
    
    Args:
        samples: Array of samples
        target_distribution: Target distribution (if known)
        n_bins: Number of bins for histogram
    
    Returns:
        Dictionary with quality metrics
    
    Example:
        >>> samples = np.random.exponential(2, 1000)
        >>> metrics = sample_quality_metrics(samples)
    """
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = float(np.mean(samples))
    metrics['std'] = float(np.std(samples, ddof=1))
    metrics['skewness'] = float(stats.skew(samples))
    metrics['kurtosis'] = float(stats.kurtosis(samples))
    
    # Effective sample size
    metrics['effective_sample_size'] = effective_sample_size(samples)
    metrics['efficiency'] = metrics['effective_sample_size'] / len(samples)
    
    # Autocorrelation time
    acf = _autocorrelation_1d(samples)
    first_negative = np.where(acf < 0)[0]
    if len(first_negative) > 0:
        tau = 1 + 2 * np.sum(acf[1:first_negative[0]])
    else:
        tau = 1 + 2 * np.sum(acf[1:])
    metrics['autocorrelation_time'] = float(tau)
    
    # If target distribution provided
    if target_distribution is not None:
        # Empirical distribution
        hist, bin_edges = np.histogram(samples, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Interpolate target to same bins
        if len(target_distribution) == n_bins:
            target_binned = target_distribution
        else:
            # Simple nearest neighbor interpolation
            indices = np.searchsorted(
                np.linspace(0, 1, len(target_distribution)), 
                bin_centers / np.max(bin_centers)
            )
            indices = np.clip(indices, 0, len(target_distribution) - 1)
            target_binned = target_distribution[indices]
        
        # Normalize
        empirical = hist * (bin_edges[1] - bin_edges[0])
        target_binned = target_binned / np.sum(target_binned)
        
        # TV distance
        metrics['tv_distance'] = total_variation_distance(empirical, target_binned)
        
        # KL divergence
        kl = np.sum(rel_entr(empirical[empirical > 0], 
                            target_binned[empirical > 0]))
        metrics['kl_divergence'] = float(kl)
    
    return metrics