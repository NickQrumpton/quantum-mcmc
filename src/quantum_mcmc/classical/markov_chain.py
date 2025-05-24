"""Classical Markov chain utilities for quantum MCMC sampling.

This module provides functions to construct and analyze classical Markov chains,
which serve as the foundation for quantum walk-based MCMC algorithms. The focus
is on reversible Markov chains that satisfy detailed balance, making them suitable
for quantum speedup via phase estimation.

Key functionalities:
- Construction of simple and complex Markov chains
- Verification of stochastic and reversibility properties
- Computation of stationary distributions
- Generation of random reversible chains for testing

Author: [Your Name]
Date: 2025-01-23
"""

from typing import Tuple, Optional
import numpy as np
from scipy.linalg import eig


def build_two_state_chain(p: float) -> np.ndarray:
    """Construct a 2x2 stochastic matrix for a two-state Markov chain.
    
    Creates a symmetric two-state Markov chain with transition probability p
    between states and self-loop probability 1-p.
    
    Args:
        p: Transition probability between states (0 <= p <= 1).
           p = 0 gives identity (no transitions)
           p = 1 gives perfect alternation
           p = 0.5 gives maximal mixing
    
    Returns:
        P: 2x2 row-stochastic transition matrix
    
    Raises:
        ValueError: If p is not in [0, 1]
    
    Example:
        >>> P = build_two_state_chain(0.3)
        >>> print(P)
        [[0.7 0.3]
         [0.3 0.7]]
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Transition probability p={p} must be in [0, 1]")
    
    P = np.array([[1 - p, p],
                  [p, 1 - p]], dtype=np.float64)
    
    return P


def build_metropolis_chain(target_dist: np.ndarray, 
                          proposal_matrix: np.ndarray) -> np.ndarray:
    """Construct a Metropolis-Hastings transition matrix.
    
    Given a target stationary distribution À and a proposal matrix Q,
    constructs the Metropolis-Hastings transition matrix P that satisfies
    detailed balance with respect to À.
    
    The acceptance probability from state i to j is:
        A(i,j) = min(1, À(j)Q(j,i) / À(i)Q(i,j))
    
    The transition probability is:
        P(i,j) = Q(i,j)A(i,j) for i ` j
        P(i,i) = 1 - £_{j`i} P(i,j)
    
    Args:
        target_dist: Target stationary distribution À (normalized probability vector)
        proposal_matrix: Row-stochastic proposal matrix Q
    
    Returns:
        P: Metropolis-Hastings transition matrix that leaves À invariant
    
    Raises:
        ValueError: If inputs have incompatible dimensions or invalid properties
    
    Example:
        >>> pi = np.array([0.3, 0.7])
        >>> Q = np.array([[0.5, 0.5], [0.5, 0.5]])  # Uniform proposal
        >>> P = build_metropolis_chain(pi, Q)
    """
    # Validate inputs
    n = len(target_dist)
    if proposal_matrix.shape != (n, n):
        raise ValueError(f"Proposal matrix shape {proposal_matrix.shape} incompatible "
                        f"with target distribution length {n}")
    
    if not np.allclose(np.sum(target_dist), 1.0):
        raise ValueError("Target distribution must be normalized (sum to 1)")
    
    if not is_stochastic(proposal_matrix):
        raise ValueError("Proposal matrix must be row-stochastic")
    
    if np.any(target_dist <= 0):
        raise ValueError("Target distribution must have positive entries")
    
    # Initialize transition matrix
    P = np.zeros((n, n), dtype=np.float64)
    
    # Compute off-diagonal entries using Metropolis acceptance ratio
    for i in range(n):
        for j in range(n):
            if i != j:
                # Avoid division by zero; if Q(i,j) = 0, then P(i,j) = 0
                if proposal_matrix[i, j] > 0:
                    # Metropolis acceptance ratio
                    acceptance_ratio = min(1.0, 
                                         (target_dist[j] * proposal_matrix[j, i]) / 
                                         (target_dist[i] * proposal_matrix[i, j]))
                    P[i, j] = proposal_matrix[i, j] * acceptance_ratio
    
    # Set diagonal entries to ensure row-stochasticity
    for i in range(n):
        P[i, i] = 1.0 - np.sum(P[i, :i]) - np.sum(P[i, i+1:])
    
    return P


def is_stochastic(P: np.ndarray, atol: float = 1e-10) -> bool:
    """Check if a matrix is row-stochastic.
    
    A matrix P is row-stochastic if:
    1. All entries are non-negative: P[i,j] >= 0
    2. Each row sums to 1: £_j P[i,j] = 1 for all i
    
    Args:
        P: Matrix to check
        atol: Absolute tolerance for numerical comparison
    
    Returns:
        True if P is row-stochastic, False otherwise
    
    Example:
        >>> P = np.array([[0.7, 0.3], [0.2, 0.8]])
        >>> is_stochastic(P)
        True
    """
    # Check non-negativity
    if np.any(P < -atol):
        return False
    
    # Check row sums
    row_sums = np.sum(P, axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        return False
    
    return True


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a stochastic matrix.
    
    Finds the normalized left eigenvector À such that ÀP = À,
    which corresponds to the stationary distribution of the Markov chain.
    
    For ergodic chains, this distribution is unique and represents the
    long-term fraction of time spent in each state.
    
    Args:
        P: Row-stochastic transition matrix
    
    Returns:
        pi: Stationary distribution (probability vector)
    
    Raises:
        ValueError: If P is not stochastic or no valid stationary distribution exists
    
    Note:
        This implementation assumes the chain is ergodic (irreducible and aperiodic).
        For reducible chains, the returned distribution may depend on numerical factors.
    
    Example:
        >>> P = build_two_state_chain(0.3)
        >>> pi = stationary_distribution(P)
        >>> print(pi)
        [0.5 0.5]
    """
    if not is_stochastic(P):
        raise ValueError("Input matrix must be row-stochastic")
    
    n = P.shape[0]
    
    # Find left eigenvectors by computing eigenvectors of P^T
    eigenvalues, eigenvectors = eig(P.T, left=False, right=True)
    
    # Find the eigenvector corresponding to eigenvalue 1
    # Due to numerical errors, we look for eigenvalues close to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    
    if not np.isclose(eigenvalues[idx], 1.0, atol=1e-9):
        raise ValueError("No eigenvalue close to 1 found; matrix may not be stochastic")
    
    # Extract the corresponding eigenvector
    pi = np.real(eigenvectors[:, idx])
    
    # Ensure all entries are non-negative (handle numerical errors)
    if np.any(pi < -1e-10):
        raise ValueError("Stationary distribution has negative entries")
    
    pi = np.maximum(pi, 0)
    
    # Normalize to get probability distribution
    pi = pi / np.sum(pi)
    
    return pi


def is_reversible(P: np.ndarray, pi: Optional[np.ndarray] = None, 
                  atol: float = 1e-10) -> bool:
    """Check if a Markov chain satisfies detailed balance (reversibility).
    
    A Markov chain with transition matrix P and stationary distribution À
    is reversible if it satisfies detailed balance:
        À[i] * P[i,j] = À[j] * P[j,i] for all i,j
    
    This property is crucial for many MCMC algorithms and enables
    quantum speedups through phase estimation.
    
    Args:
        P: Transition matrix
        pi: Stationary distribution. If None, it will be computed.
        atol: Absolute tolerance for numerical comparison
    
    Returns:
        True if the chain is reversible, False otherwise
    
    Example:
        >>> P = build_two_state_chain(0.3)
        >>> is_reversible(P)
        True
    """
    if not is_stochastic(P):
        return False
    
    # Compute stationary distribution if not provided
    if pi is None:
        try:
            pi = stationary_distribution(P)
        except ValueError:
            return False
    
    n = P.shape[0]
    
    # Check detailed balance for all pairs (i,j)
    for i in range(n):
        for j in range(i + 1, n):  # Only check upper triangle due to symmetry
            flow_ij = pi[i] * P[i, j]
            flow_ji = pi[j] * P[j, i]
            
            if not np.isclose(flow_ij, flow_ji, atol=atol):
                return False
    
    return True


def sample_random_reversible_chain(n: int, 
                                  sparsity: float = 0.3,
                                  seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random n-state reversible Markov chain.
    
    Constructs a random reversible Markov chain by:
    1. Generating a random stationary distribution
    2. Creating a sparse symmetric structure
    3. Applying the Metropolis filter to ensure reversibility
    
    This is useful for testing quantum MCMC algorithms on diverse chain structures.
    
    Args:
        n: Number of states
        sparsity: Fraction of zero entries in the transition matrix (0 to 1).
                 Lower values create denser, more connected chains.
        seed: Random seed for reproducibility
    
    Returns:
        P: n×n reversible transition matrix
        pi: Stationary distribution
    
    Raises:
        ValueError: If n < 2 or sparsity not in [0, 1]
    
    Example:
        >>> P, pi = sample_random_reversible_chain(5, sparsity=0.5, seed=42)
        >>> is_reversible(P, pi)
        True
    """
    if n < 2:
        raise ValueError(f"Number of states n={n} must be at least 2")
    
    if not 0 <= sparsity < 1:
        raise ValueError(f"Sparsity={sparsity} must be in [0, 1)")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random stationary distribution using Dirichlet
    # with uniform concentration parameters
    pi = np.random.dirichlet(np.ones(n))
    
    # Create a random symmetric proposal matrix with controlled sparsity
    Q = np.random.rand(n, n)
    Q = (Q + Q.T) / 2  # Make symmetric
    
    # Apply sparsity by zeroing out entries below threshold
    if sparsity > 0:
        threshold = np.percentile(Q.flatten(), sparsity * 100)
        Q[Q < threshold] = 0
    
    # Ensure at least a connected chain (add small diagonal if needed)
    np.fill_diagonal(Q, 0.1)
    
    # Normalize rows to make stochastic
    row_sums = np.sum(Q, axis=1, keepdims=True)
    Q = Q / row_sums
    
    # Apply Metropolis filter to ensure reversibility
    P = build_metropolis_chain(pi, Q)
    
    # Verify the result
    if not is_reversible(P, pi):
        raise RuntimeError("Failed to generate reversible chain")
    
    return P, pi


# Additional utility functions for lattice Gaussian sampling (optional extension)

def build_imhk_chain(dim: int, beta: float = 1.0) -> np.ndarray:
    """Build an IMHK (Iyer-Mandelshtam-Hore-Kais) type chain for lattice problems.
    
    Constructs a Markov chain suitable for sampling from distributions on
    integer lattices, commonly used in lattice Gaussian sampling and 
    discrete optimization problems.
    
    Args:
        dim: Dimension of the lattice
        beta: Inverse temperature parameter (higher beta = lower temperature)
    
    Returns:
        P: Transition matrix for IMHK-type chain
    
    Note:
        This is a simplified implementation. Full IMHK chains require
        problem-specific energy functions and proposal distributions.
    """
    # For a d-dimensional hypercube, we have 2^d states
    n_states = 2 ** dim
    
    # Energy function for Ising-like model (example)
    def energy(state_idx):
        # Convert index to binary representation
        state = [(state_idx >> i) & 1 for i in range(dim)]
        # Simple energy: number of 1s (can be generalized)
        return sum(state)
    
    # Compute target distribution using Boltzmann weights
    energies = np.array([energy(i) for i in range(n_states)])
    weights = np.exp(-beta * energies)
    pi = weights / np.sum(weights)
    
    # Build proposal matrix (nearest neighbor moves on hypercube)
    Q = np.zeros((n_states, n_states))
    
    for i in range(n_states):
        # Self-loop probability
        Q[i, i] = 0.5
        
        # Connect to neighbors (flip one bit)
        n_neighbors = 0
        for bit in range(dim):
            j = i ^ (1 << bit)  # Flip bit
            Q[i, j] = 1.0 / (2 * dim)
            n_neighbors += 1
        
        # Adjust self-loop to ensure stochasticity
        Q[i, i] = 1.0 - n_neighbors / (2 * dim)
    
    # Apply Metropolis filter
    P = build_metropolis_chain(pi, Q)
    
    return P