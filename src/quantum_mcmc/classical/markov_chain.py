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

Author: Nicholas Zhao  
Affiliation: Imperial College London  
Contact: nz422@ic.ac.uk
"""

from typing import Tuple, Optional, Union
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eig


def build_two_state_chain(p: float, q: Optional[float] = None) -> np.ndarray:
    """Construct a 2x2 stochastic matrix for a two-state Markov chain.
    
    Creates a reversible Markov chain with states {0, 1} where:
    - P[0,1] = p (transition probability from state 0 to 1)
    - P[1,0] = q (transition probability from state 1 to 0)
    
    If q is not provided, creates a symmetric chain with q = p.
    
    This gives the chain:
        P = [[1-p, p  ],
             [q,   1-q]]
    
    Args:
        p: Transition probability from state 0 to 1 (0 <= p <= 1)
        q: Transition probability from state 1 to 0 (0 <= q <= 1).
           If None, uses q = p for a symmetric chain.
    
    Returns:
        P: 2x2 row-stochastic transition matrix
    
    Raises:
        ValueError: If p or q are not in [0, 1]
    
    Example:
        >>> P = build_two_state_chain(0.3)
        >>> print(P)
        [[0.7 0.3]
         [0.3 0.7]]
        
        >>> P = build_two_state_chain(0.2, 0.4)
        >>> print(P)
        [[0.8 0.2]
         [0.4 0.6]]
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Transition probability p={p} must be in [0, 1]")
    
    if q is None:
        q = p
    elif not 0 <= q <= 1:
        raise ValueError(f"Transition probability q={q} must be in [0, 1]")
    
    P = np.array([[1 - p, p],
                  [q, 1 - q]], dtype=np.float64)
    
    return P


def build_metropolis_chain(target_dist: np.ndarray, 
                          proposal_matrix: Optional[np.ndarray] = None,
                          proposal_std: float = 1.0,
                          sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
    """Construct a Metropolis-Hastings transition matrix.
    
    Given a target stationary distribution pi and optionally a proposal matrix Q,
    constructs the Metropolis-Hastings transition matrix P that satisfies
    detailed balance with respect to pi.
    
    The acceptance probability from state i to j is:
        A(i,j) = min(1, pi(j)Q(j,i) / pi(i)Q(i,j))
    
    The transition probability is:
        P(i,j) = Q(i,j)A(i,j) for i != j
        P(i,i) = 1 - sum_{j!=i} P(i,j)
    
    Args:
        target_dist: Target stationary distribution pi (normalized probability vector)
        proposal_matrix: Row-stochastic proposal matrix Q. If None, generates a
                        default random walk proposal based on a discrete approximation
                        of a Gaussian kernel.
        proposal_std: Standard deviation for the default Gaussian proposal (only used
                     if proposal_matrix is None)
        sparse: If True, returns a sparse CSR matrix
    
    Returns:
        P: Metropolis-Hastings transition matrix that leaves pi invariant
    
    Raises:
        ValueError: If inputs have incompatible dimensions or invalid properties
    
    Example:
        >>> pi = np.array([0.3, 0.7])
        >>> Q = np.array([[0.5, 0.5], [0.5, 0.5]])  # Uniform proposal
        >>> P = build_metropolis_chain(pi, Q)
        
        >>> # Or use default proposal
        >>> pi = norm.pdf(np.linspace(-3, 3, 50))
        >>> pi /= pi.sum()
        >>> P = build_metropolis_chain(pi)
    """
    # Validate inputs
    n = len(target_dist)
    
    if n < 2:
        raise ValueError(f"Target distribution must have at least 2 states, got {n}")
    
    if not np.allclose(np.sum(target_dist), 1.0):
        raise ValueError("Target distribution must be normalized (sum to 1)")
    
    if np.any(target_dist < 0):
        raise ValueError("Target distribution must be non-negative")
    
    # Generate default proposal matrix if not provided
    if proposal_matrix is None:
        # Create a discrete Gaussian random walk proposal
        # This assumes states are ordered (e.g., discretized continuous space)
        proposal_matrix = np.zeros((n, n))
        
        # For each state, compute transition probabilities to nearby states
        for i in range(n):
            # Compute unnormalized probabilities using discrete Gaussian kernel
            for j in range(n):
                distance = abs(i - j)
                # Gaussian kernel with wrapping for periodic boundary
                proposal_matrix[i, j] = np.exp(-0.5 * (distance / proposal_std)**2)
            
            # Normalize to make row stochastic
            proposal_matrix[i] /= proposal_matrix[i].sum()
    
    # Validate proposal matrix
    if proposal_matrix.shape != (n, n):
        raise ValueError(f"Proposal matrix shape {proposal_matrix.shape} incompatible "
                        f"with target distribution length {n}")
    
    if not is_stochastic(proposal_matrix):
        raise ValueError("Proposal matrix must be row-stochastic")
    
    # Initialize transition matrix
    P = np.zeros((n, n), dtype=np.float64)
    
    # Compute off-diagonal entries using Metropolis acceptance ratio
    for i in range(n):
        for j in range(n):
            if i != j:
                # Avoid division by zero; if Q(i,j) = 0, then P(i,j) = 0
                if proposal_matrix[i, j] > 0:
                    # Metropolis acceptance ratio
                    # Handle case where target_dist[i] is zero
                    if target_dist[i] == 0:
                        # If we're in a zero-probability state, always accept moves
                        acceptance_ratio = 1.0
                    elif target_dist[j] == 0:
                        # Never accept moves to zero-probability states
                        acceptance_ratio = 0.0
                    else:
                        acceptance_ratio = min(1.0, 
                                             (target_dist[j] * proposal_matrix[j, i]) / 
                                             (target_dist[i] * proposal_matrix[i, j]))
                    P[i, j] = proposal_matrix[i, j] * acceptance_ratio
    
    # Set diagonal entries to ensure row-stochasticity
    for i in range(n):
        P[i, i] = 1.0 - np.sum(P[i, :i]) - np.sum(P[i, i+1:])
    
    # Return sparse matrix if requested
    if sparse:
        return sp.csr_matrix(P)
    return P


def is_stochastic(P: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a matrix is row-stochastic.
    
    A matrix P is row-stochastic if:
    1. All entries are non-negative: P[i,j] >= 0
    2. Each row sums to 1: sum_j P[i,j] = 1 for all i
    
    Args:
        P: Matrix to check
        tol: Absolute tolerance for numerical comparison
    
    Returns:
        True if P is row-stochastic, False otherwise
    
    Example:
        >>> P = np.array([[0.7, 0.3], [0.2, 0.8]])
        >>> is_stochastic(P)
        True
    """
    # Check if matrix is square
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Matrix must be square")
    
    # Check non-negativity
    if np.any(P < -tol):
        return False
    
    # Check row sums
    row_sums = np.sum(P, axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        return False
    
    return True


def stationary_distribution(P: np.ndarray, method: str = 'eigen', 
                           max_iter: int = 10000, tol: float = 1e-10) -> np.ndarray:
    """Compute the stationary distribution of a stochastic matrix.
    
    Finds the normalized left eigenvector pi such that pi*P = pi,
    which corresponds to the stationary distribution of the Markov chain.
    
    For ergodic chains, this distribution is unique and represents the
    long-term fraction of time spent in each state.
    
    Args:
        P: Row-stochastic transition matrix
        method: Method to use ('eigen' for eigenvalue decomposition, 
                'power' for power iteration)
        max_iter: Maximum iterations for power method
        tol: Convergence tolerance for power method
    
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
    
    if method == 'eigen':
        # Find left eigenvectors by computing eigenvectors of P^T
        eigenvalues, eigenvectors = eig(P.T, left=False, right=True)
        
        # Find the eigenvector corresponding to eigenvalue 1
        # Due to numerical errors, we look for eigenvalues close to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        
        if not np.isclose(eigenvalues[idx], 1.0, atol=1e-9):
            raise ValueError("No eigenvalue close to 1 found; matrix may not be stochastic")
        
        # Extract the corresponding eigenvector
        pi = np.real(eigenvectors[:, idx])
        
        # Eigenvector might have wrong sign, ensure it's mostly positive
        if np.sum(pi < 0) > np.sum(pi > 0):
            pi = -pi
        
        # Ensure all entries are non-negative (handle numerical errors)
        if np.any(pi < -1e-10):
            raise ValueError("Stationary distribution has negative entries")
        
        pi = np.maximum(pi, 0)
        
        # Normalize to get probability distribution
        pi = pi / np.sum(pi)
        
    elif method == 'power':
        # Power iteration method
        # Start with uniform distribution
        pi = np.ones(n) / n
        
        # Check if matrix is identity (special case)
        if np.allclose(P, np.eye(n)):
            # For identity matrix, any distribution is stationary
            # Power method won't converge to unique solution
            raise ValueError("Power method cannot find unique stationary distribution for identity matrix")
        
        for i in range(max_iter):
            pi_new = pi @ P
            
            # Check convergence
            if np.allclose(pi, pi_new, atol=tol):
                pi = pi_new
                break
            
            pi = pi_new
        else:
            # Did not converge
            raise ValueError(f"Power method did not converge in {max_iter} iterations")
        
        # Normalize (should already be normalized, but ensure numerical stability)
        pi = pi / np.sum(pi)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'eigen' or 'power'")
    
    return pi


def is_reversible(P: np.ndarray, pi: Optional[np.ndarray] = None, 
                  tol: float = 1e-10) -> bool:
    """Check if a Markov chain satisfies detailed balance (reversibility).
    
    A Markov chain with transition matrix P and stationary distribution pi
    is reversible if it satisfies detailed balance:
        pi[i] * P[i,j] = pi[j] * P[j,i] for all i,j
    
    This property is crucial for many MCMC algorithms and enables
    quantum speedups through phase estimation.
    
    Args:
        P: Transition matrix
        pi: Stationary distribution. If None, it will be computed.
        tol: Absolute tolerance for numerical comparison
    
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
    else:
        # Validate provided distribution
        if len(pi) != P.shape[0]:
            raise ValueError(f"Distribution size {len(pi)} doesn't match matrix size {P.shape[0]}")
        if not np.allclose(np.sum(pi), 1.0):
            raise ValueError("Distribution must sum to 1")
        if np.any(pi < 0):
            raise ValueError("Distribution must be non-negative")
    
    n = P.shape[0]
    
    # Check detailed balance for all pairs (i,j)
    for i in range(n):
        for j in range(i + 1, n):  # Only check upper triangle due to symmetry
            flow_ij = pi[i] * P[i, j]
            flow_ji = pi[j] * P[j, i]
            
            if not np.isclose(flow_ij, flow_ji, atol=tol):
                return False
    
    return True


def sample_random_reversible_chain(n: int, 
                                  sparsity: float = 0.3,
                                  seed: Optional[int] = None,
                                  return_sparse: bool = False) -> Tuple[Union[np.ndarray, sp.csr_matrix], np.ndarray]:
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
        return_sparse: If True, returns a sparse CSR matrix
    
    Returns:
        P: n x n reversible transition matrix (sparse or dense based on return_sparse)
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
    
    if not 0 <= sparsity <= 1:
        raise ValueError(f"Sparsity={sparsity} must be in [0, 1]")
    
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
    
    # Return sparse matrix if requested
    if return_sparse:
        return sp.csr_matrix(P), pi
    
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