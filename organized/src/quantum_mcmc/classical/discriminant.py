"""Discriminant matrix utilities for Szegedy-type quantum walks.

This module provides functions to construct and analyze the discriminant matrix
associated with classical Markov chains, which is fundamental to the construction
of quantum walks following Szegedy's framework. The discriminant matrix encodes
the transition amplitudes for the quantum walk operator and its spectral properties
determine the efficiency of quantum speedup.

Key functionalities:
- Construction of discriminant matrices from reversible Markov chains
- Spectral analysis including singular values and spectral gap
- Validation and consistency checks for quantum walk construction

References:
    Szegedy, M. (2004). Quantum speed-up of Markov chain based algorithms.
    FOCS 2004: 32-41.

Author: Nicholas Zhao  
Affiliation: Imperial College London  
Contact: nz422@ic.ac.uk
"""

from typing import Optional, Tuple
import numpy as np
from scipy.linalg import svd
import warnings


def discriminant_matrix(P: np.ndarray, pi: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the discriminant matrix D(P) for a reversible Markov chain.
    
    For a reversible Markov chain with transition matrix P and stationary
    distribution �, the discriminant matrix D is defined as:
        
        D[x,y] = (P[x,y] * P[y,x] * �[y] / �[x])
    
    This matrix arises naturally in Szegedy's quantum walk construction and
    encodes the transition amplitudes. The singular values of D determine
    the spectral properties of the quantum walk operator.
    
    Args:
        P: n�n reversible transition matrix (row-stochastic)
        pi: Stationary distribution. If None, it will be computed.
    
    Returns:
        D: n�n discriminant matrix with entries in [0,1]
    
    Raises:
        ValueError: If P is not stochastic, not reversible, or if the
                   stationary distribution cannot be computed
    
    Note:
        - The discriminant matrix is symmetric for reversible chains
        - D has the same sparsity pattern as P
        - The largest singular value of D is always 1
        - For doubly stochastic P, D = P
    
    Example:
        >>> from quantum_mcmc.classical.markov_chain import build_two_state_chain
        >>> P = build_two_state_chain(0.3)
        >>> D = discriminant_matrix(P)
        >>> np.allclose(D, D.T)  # Check symmetry
        True
    """
    # Import here to avoid circular dependency
    from .markov_chain import is_stochastic, is_reversible, stationary_distribution
    
    # Validate that P is stochastic
    if not is_stochastic(P):
        raise ValueError("Transition matrix P must be row-stochastic")
    
    n = P.shape[0]
    
    # Compute stationary distribution if not provided
    if pi is None:
        try:
            pi = stationary_distribution(P)
        except ValueError as e:
            raise ValueError(f"Cannot compute stationary distribution: {e}")
    
    # Validate pi
    if len(pi) != n:
        raise ValueError(f"Stationary distribution has wrong dimension: "
                        f"expected {n}, got {len(pi)}")
    
    if not np.allclose(np.sum(pi), 1.0):
        raise ValueError("Stationary distribution must sum to 1")
    
    if np.any(pi < 0):
        raise ValueError("Stationary distribution must have non-negative entries")
    
    # Handle zero probability states with a warning
    if np.any(pi == 0):
        import warnings
        warnings.warn("Stationary distribution has zero entries - results may be unreliable")
        # Replace zeros with small positive values for numerical stability
        pi = np.maximum(pi, 1e-12)
        pi = pi / np.sum(pi)  # Renormalize
    
    # Check reversibility
    if not is_reversible(P, pi):
        raise ValueError("Markov chain must be reversible with respect to �")
    
    # Construct discriminant matrix
    D = np.zeros((n, n), dtype=np.float64)
    
    # Compute entries using the correct discriminant formula
    # For reversible chains: D[i,j] = sqrt(P[i,j] * P[j,i])
    # This ensures D is symmetric and has the correct spectral properties
    for i in range(n):
        for j in range(n):
            if P[i, j] > 0 and P[j, i] > 0:
                # Correct discriminant formula for Szegedy walks
                D[i, j] = np.sqrt(P[i, j] * P[j, i])
            elif i == j:
                # Diagonal entries are just P[i,i]
                D[i, j] = P[i, j]
    
    # Ensure exact symmetry (handle numerical errors)
    D = (D + D.T) / 2
    
    return D


def singular_values(D: np.ndarray) -> np.ndarray:
    """Compute the sorted singular values of the discriminant matrix.
    
    The singular values of D determine the spectrum of the quantum walk operator.
    In particular:
    - The largest singular value is always 1 (for connected chains)
    - The second largest singular value determines the spectral gap
    - The singular values are related to the eigenvalues of the quantum walk
    
    Args:
        D: n�n discriminant matrix
    
    Returns:
        sigma: Array of singular values sorted in descending order
    
    Note:
        For symmetric matrices (which D is for reversible chains), the
        singular values equal the absolute values of the eigenvalues.
    
    Example:
        >>> D = discriminant_matrix(build_two_state_chain(0.3))
        >>> sigma = singular_values(D)
        >>> print(f"Largest singular value: {sigma[0]:.6f}")
        Largest singular value: 1.000000
    """
    # Compute SVD (we only need singular values)
    # For symmetric matrices, this is equivalent to eigenvalue decomposition
    _, sigma, _ = svd(D, full_matrices=False, compute_uv=True, overwrite_a=False)
    
    # Ensure descending order (svd should already return this, but be explicit)
    sigma = np.sort(sigma)[::-1]
    
    return sigma


def spectral_gap(D: np.ndarray) -> float:
    """Compute the spectral gap of the discriminant matrix.
    
    The spectral gap is defined as the difference between the largest and
    second largest singular values:
        
        gap = Á - Â
    
    This quantity determines the mixing time of the quantum walk and the
    potential quantum speedup. A larger spectral gap generally indicates
    faster mixing and better quantum advantage.
    
    Args:
        D: n�n discriminant matrix
    
    Returns:
        gap: Spectral gap (value between 0 and 1)
    
    Raises:
        ValueError: If D has fewer than 2 singular values
    
    Note:
        - For connected reversible chains, Á = 1
        - The spectral gap is related to the classical mixing time
        - Quantum speedup is approximately quadratic in the spectral gap
    
    Example:
        >>> D = discriminant_matrix(build_two_state_chain(0.3))
        >>> gap = spectral_gap(D)
        >>> print(f"Spectral gap: {gap:.6f}")
        Spectral gap: 0.600000
    """
    sigma = singular_values(D)
    
    if len(sigma) < 2:
        raise ValueError("Discriminant matrix must have at least 2 singular values")
    
    # The gap is between the largest and second largest
    gap = sigma[0] - sigma[1]
    
    # Validate the result
    if gap < -1e-10:
        warnings.warn(f"Negative spectral gap detected: {gap}. "
                     "This may indicate numerical issues.")
    
    # Ensure non-negative
    gap = max(0.0, gap)
    
    return gap


def phase_gap(D: np.ndarray) -> float:
    """Compute the phase gap of the quantum walk operator.
    
    The phase gap is defined as:
        Δ(P) = min{2θ | cos(θ) ∈ σ(D), θ ∈ (0,π/2)}
    
    where σ(D) are the singular values of the discriminant matrix.
    
    For a classical Markov chain with spectral gap δ, the quantum phase gap
    satisfies: Δ(P) ≥ 2√δ (with equality for 2-state chains).
    
    Args:
        D: n×n discriminant matrix
    
    Returns:
        delta: Phase gap in radians
    
    Note:
        The phase gap determines the quantum mixing time through:
        t_quantum = O(1/Δ × log(n/ε))
    
    Example:
        >>> D = discriminant_matrix(build_two_state_chain(0.3))
        >>> delta = phase_gap(D)
        >>> print(f"Phase gap: {delta:.6f} radians")
    """
    # Get singular values of D
    sigmas = singular_values(D)
    
    # Find minimum non-trivial phase
    min_phase = np.pi  # Initialize to maximum possible
    
    for sigma in sigmas:
        if sigma < 1e-14:  # Skip near-zero singular values
            continue
            
        if np.abs(sigma - 1.0) < 1e-14:  # Skip the stationary singular value
            continue
        
        # The phase θ is related to singular value σ by:
        # cos(θ) = σ for σ ∈ [0,1]
        # We want the minimum 2θ for θ ∈ (0, π/2)
        
        # Ensure sigma is in valid range [0,1]
        sigma_clipped = np.clip(sigma, 0.0, 1.0)
        
        # Compute θ = arccos(σ)
        theta = np.arccos(sigma_clipped)
        
        # We consider both θ and π-θ as phases
        # The phase gap is 2 times the minimum non-zero phase
        if 0 < theta < np.pi/2:
            min_phase = min(min_phase, theta)
        
        # Also consider π-θ if it's in the valid range
        alt_theta = np.pi - theta
        if 0 < alt_theta < np.pi/2:
            min_phase = min(min_phase, alt_theta)
    
    # The phase gap is 2 times the minimum phase
    phase_gap_value = 2 * min_phase
    
    # For verification, compute the theoretical lower bound
    # For classical gap δ, quantum phase gap should be ≥ 2√δ
    classical_gap = spectral_gap(D)
    theoretical_bound = 2 * np.sqrt(classical_gap)
    
    # In practice, for simple chains, we should have approximately
    # phase_gap ≈ theoretical_bound
    # But we use the exact calculation from singular values
    
    return phase_gap_value


def mixing_time_bound(D: np.ndarray, epsilon: float = 0.01) -> float:
    """Compute an upper bound on the quantum mixing time.
    
    For a quantum walk with discriminant matrix D, the mixing time to
    reach ε-distance from the stationary distribution is bounded by:
        
        T_quantum = O(1/phase_gap × log(n/ε))
    
    This provides a quadratic speedup over classical mixing when the
    spectral gap is small, since phase_gap ≈ 2√(classical_gap).
    
    Args:
        D: n×n discriminant matrix
        epsilon: Target distance from stationary distribution
    
    Returns:
        t_bound: Upper bound on quantum mixing time (in units of quantum steps)
    
    Example:
        >>> D = discriminant_matrix(build_two_state_chain(0.3))
        >>> t = mixing_time_bound(D, epsilon=0.01)
        >>> print(f"Quantum mixing time bound: {t:.1f} steps")
    """
    n = D.shape[0]
    delta = phase_gap(D)
    
    if delta < 1e-10:
        warnings.warn("Phase gap is near zero; mixing time may be infinite")
        return np.inf
    
    # Quantum mixing time bound: O(1/Δ × log(n/ε))
    # Using the standard constant factor
    t_bound = (1.0 / delta) * np.log(n / epsilon)
    
    return t_bound


def classical_spectral_gap(P: np.ndarray) -> float:
    """Compute the classical spectral gap of a transition matrix.
    
    The classical spectral gap is defined as:
        gap = 1 - |λ₂|
    where λ₂ is the second largest eigenvalue by magnitude.
    
    Args:
        P: n×n stochastic transition matrix
    
    Returns:
        gap: Classical spectral gap
    
    Example:
        >>> P = build_two_state_chain(0.3)
        >>> gap = classical_spectral_gap(P)
        >>> print(f"Classical gap: {gap:.4f}")
    """
    # Get eigenvalues of P
    eigenvals = np.linalg.eigvals(P)
    
    # Sort by magnitude in descending order
    eigenvals_sorted = sorted(eigenvals, key=lambda x: abs(x), reverse=True)
    
    # The largest eigenvalue should be 1 (stationary)
    # The gap is 1 - |λ₂|
    if len(eigenvals_sorted) < 2:
        return 1.0
    
    second_largest = abs(eigenvals_sorted[1])
    gap = 1.0 - second_largest
    
    # Ensure non-negative
    return max(0.0, gap)


def validate_discriminant(D: np.ndarray, P: np.ndarray, 
                         pi: Optional[np.ndarray] = None,
                         atol: float = 1e-10) -> bool:
    """Validate that D is a valid discriminant matrix for P.
    
    Checks that:
    1. D is symmetric
    2. D has entries in [0,1]
    3. D satisfies the discriminant relation with P and �
    4. D has the correct spectral properties
    
    Args:
        D: Candidate discriminant matrix
        P: Transition matrix
        pi: Stationary distribution (computed if not provided)
        atol: Absolute tolerance for numerical comparisons
    
    Returns:
        is_valid: True if D is a valid discriminant matrix for P
    
    Example:
        >>> P = build_two_state_chain(0.3)
        >>> D = discriminant_matrix(P)
        >>> validate_discriminant(D, P)
        True
    """
    from .markov_chain import stationary_distribution
    
    n = D.shape[0]
    
    # Check dimensions
    if D.shape != (n, n) or P.shape != (n, n):
        return False
    
    # Check symmetry
    if not np.allclose(D, D.T, atol=atol):
        return False
    
    # Check range [0, 1]
    if np.any(D < -atol) or np.any(D > 1 + atol):
        return False
    
    # Get stationary distribution
    if pi is None:
        try:
            pi = stationary_distribution(P)
        except:
            return False
    
    # Check discriminant relation
    D_reconstructed = discriminant_matrix(P, pi)
    if not np.allclose(D, D_reconstructed, atol=atol):
        return False
    
    # Check that largest singular value is 1
    sigma = singular_values(D)
    if not np.isclose(sigma[0], 1.0, atol=atol):
        return False
    
    return True


def effective_dimension(D: np.ndarray, threshold: float = 0.01) -> int:
    """Compute the effective dimension of the discriminant matrix.
    
    The effective dimension is the number of singular values above a
    threshold, indicating the number of "active" modes in the quantum walk.
    
    Args:
        D: n�n discriminant matrix
        threshold: Singular value threshold
    
    Returns:
        d_eff: Effective dimension
    
    Example:
        >>> D = discriminant_matrix(build_two_state_chain(0.3))
        >>> d_eff = effective_dimension(D)
        >>> print(f"Effective dimension: {d_eff}")
    """
    sigma = singular_values(D)
    d_eff = np.sum(sigma > threshold)
    return int(d_eff)


def condition_number(D: np.ndarray) -> float:
    """Compute the condition number of the discriminant matrix.
    
    The condition number �(D) = �_max / �_min indicates the numerical
    stability of quantum walk simulations. Large condition numbers may
    lead to numerical issues in quantum circuit implementations.
    
    Args:
        D: n�n discriminant matrix
    
    Returns:
        kappa: Condition number (e 1)
    
    Example:
        >>> D = discriminant_matrix(build_two_state_chain(0.3))
        >>> kappa = condition_number(D)
        >>> print(f"Condition number: {kappa:.2f}")
    """
    sigma = singular_values(D)
    
    # Find smallest non-zero singular value
    sigma_positive = sigma[sigma > 1e-14]
    
    if len(sigma_positive) == 0:
        return np.inf
    
    sigma_max = sigma_positive[0]
    sigma_min = sigma_positive[-1]
    
    if sigma_min == 0:
        return np.inf
    
    kappa = sigma_max / sigma_min
    return kappa


def spectral_analysis(D: np.ndarray) -> dict:
    """Perform comprehensive spectral analysis of the discriminant matrix.
    
    Computes various spectral properties that are relevant for understanding
    the quantum walk behavior and potential speedup.
    
    Args:
        D: n�n discriminant matrix
    
    Returns:
        analysis: Dictionary containing:
            - singular_values: Full spectrum
            - spectral_gap: Gap between largest two values
            - phase_gap: Corresponding phase gap
            - mixing_time: Quantum mixing time bound
            - condition_number: Numerical conditioning
            - effective_dimension: Number of significant modes
            - largest_singular_value: Should be 1 for valid D
    
    Example:
        >>> D = discriminant_matrix(build_two_state_chain(0.3))
        >>> analysis = spectral_analysis(D)
        >>> for key, value in analysis.items():
        ...     print(f"{key}: {value}")
    """
    sigma = singular_values(D)
    
    analysis = {
        'singular_values': sigma,
        'spectral_gap': spectral_gap(D),
        'phase_gap': phase_gap(D),
        'mixing_time': mixing_time_bound(D),
        'condition_number': condition_number(D),
        'effective_dimension': effective_dimension(D),
        'largest_singular_value': sigma[0] if len(sigma) > 0 else 0,
        'dimension': D.shape[0]
    }
    
    return analysis