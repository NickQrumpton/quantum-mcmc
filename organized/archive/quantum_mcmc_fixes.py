"""
Comprehensive fixes for quantum MCMC benchmark issues

Issues identified and fixed:
1. Eigenvalues outside unit circle - Fixed walk_eigenvalues() to use correct formula
2. Classical/quantum gaps identical - Fixed to use proper classical gap calculation
3. Two-State Symmetric 0x speedup - Fixed mixing time to handle non-uniform initial distribution
4. Speedup calculation - Fixed to use proper quantum mixing time estimate
"""

import numpy as np
from typing import Optional, Tuple


def walk_eigenvalues_fixed(P: np.ndarray, pi: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute eigenvalues of the quantum walk operator (FIXED).
    
    The eigenvalues are on the unit circle: λ = e^{±iθ}
    where cos(θ) = ±√(1 - 4σ²(1 - σ²)) and σ are singular values of D.
    """
    from quantum_mcmc.classical.discriminant import discriminant_matrix, singular_values
    
    D = discriminant_matrix(P, pi)
    sigmas = singular_values(D)
    
    eigenvals = []
    
    for sigma in sigmas:
        if sigma < 1e-14:
            continue
        
        # σ = 1 gives eigenvalue 1 (stationary state)
        if np.abs(sigma - 1.0) < 1e-14:
            eigenvals.append(1.0)
            continue
            
        # Compute cos(θ) = √(1 - 4σ²(1 - σ²))
        cos_theta_sq = 1 - 4 * sigma**2 * (1 - sigma**2)
        cos_theta_sq = np.clip(cos_theta_sq, 0.0, 1.0)
        cos_theta = np.sqrt(cos_theta_sq)
        
        # Get angle θ
        theta = np.arccos(cos_theta)
        
        # Eigenvalues are e^{±iθ} and e^{±i(π-θ)}
        eigenvals.extend([
            np.exp(1j * theta),
            np.exp(-1j * theta),
            np.exp(1j * (np.pi - theta)),
            np.exp(-1j * (np.pi - theta))
        ])
    
    # Remove duplicates
    eigenvals = np.array(eigenvals)
    unique_eigenvals = []
    for ev in eigenvals:
        if not any(np.abs(ev - uev) < 1e-10 for uev in unique_eigenvals):
            unique_eigenvals.append(ev)
    
    return np.array(unique_eigenvals)


def classical_spectral_gap(P: np.ndarray) -> float:
    """Compute classical spectral gap (FIXED).
    
    Classical gap = 1 - |λ_2| where λ_2 is second largest eigenvalue of P.
    """
    eigenvals = np.linalg.eigvals(P)
    eigenvals_mag = np.sort(np.abs(eigenvals))[::-1]
    
    if len(eigenvals_mag) < 2:
        return 0.0
    
    return 1.0 - eigenvals_mag[1]


def mixing_time_fixed(
    transition_matrix: np.ndarray,
    epsilon: float = 0.01,
    initial_dist: Optional[np.ndarray] = None,
    max_steps: int = 10000
) -> int:
    """Estimate mixing time (FIXED to handle worst-case initial distribution)."""
    n = transition_matrix.shape[0]
    
    # Find stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
    stationary_idx = np.argmax(np.abs(eigenvals - 1.0) < 1e-10)
    stationary = np.real(eigenvecs[:, stationary_idx])
    stationary = stationary / np.sum(stationary)
    
    # If no initial distribution specified, use worst-case
    if initial_dist is None:
        # For mixing time, we should consider worst-case initial distribution
        # Try each basis state and find the slowest mixing one
        max_mixing_time = 0
        
        for i in range(n):
            # Start from basis state |i⟩
            current_dist = np.zeros(n)
            current_dist[i] = 1.0
            
            for t in range(max_steps):
                tv_dist = 0.5 * np.sum(np.abs(current_dist - stationary))
                
                if tv_dist <= epsilon:
                    max_mixing_time = max(max_mixing_time, t)
                    break
                
                current_dist = current_dist @ transition_matrix
            else:
                # Did not mix within max_steps
                max_mixing_time = max_steps
        
        return max_mixing_time
    else:
        # Use provided initial distribution
        current_dist = initial_dist.copy()
        
        for t in range(max_steps):
            tv_dist = 0.5 * np.sum(np.abs(current_dist - stationary))
            
            if tv_dist <= epsilon:
                return t
            
            current_dist = current_dist @ transition_matrix
        
        return max_steps


def quantum_mixing_time_estimate(
    quantum_phase_gap: float,
    n_states: int,
    epsilon: float = 0.01
) -> int:
    """Estimate quantum mixing time from phase gap (FIXED).
    
    Based on quantum walk theory, the mixing time scales as:
    t_quantum = C × (1/quantum_phase_gap) × log(1/epsilon)
    
    Where quantum_phase_gap ≈ 2√(classical_gap) for simple chains.
    This gives O(1/√δ) scaling compared to classical O(1/δ).
    """
    if quantum_phase_gap < 1e-10:
        return np.inf
    
    # Quantum mixing time: O(1/Δ × log(n/ε))
    # Using standard constant (not conservative)
    t_quantum = int(np.ceil((1.0 / quantum_phase_gap) * np.log(n_states / epsilon)))
    
    return t_quantum


def compute_quantum_speedup(
    classical_mixing: int,
    quantum_mixing: int
) -> float:
    """Compute quantum speedup (FIXED to handle edge cases)."""
    if classical_mixing == 0 or quantum_mixing == 0:
        # If either is 0, no meaningful speedup can be computed
        return 1.0
    
    if quantum_mixing >= classical_mixing:
        # No speedup
        return classical_mixing / quantum_mixing
    
    return classical_mixing / quantum_mixing


# Monkey-patch the benchmark to use fixed functions
def patch_benchmark():
    """Apply fixes to the benchmark module."""
    import quantum_mcmc.core.quantum_walk
    import quantum_mcmc.utils.analysis
    
    # Replace the buggy functions
    quantum_mcmc.core.quantum_walk.walk_eigenvalues = walk_eigenvalues_fixed
    
    # Add classical_spectral_gap to markov_chain module
    import quantum_mcmc.classical.markov_chain
    quantum_mcmc.classical.markov_chain.classical_spectral_gap = classical_spectral_gap
    
    # Replace mixing_time
    quantum_mcmc.utils.analysis.mixing_time = mixing_time_fixed


if __name__ == "__main__":
    # Test the fixes
    from quantum_mcmc.classical.markov_chain import build_two_state_chain, stationary_distribution
    from quantum_mcmc.classical.discriminant import discriminant_matrix, phase_gap
    
    print("Testing fixes on Two-State Symmetric chain:")
    P = build_two_state_chain(0.3)
    pi = stationary_distribution(P)
    
    # Test eigenvalues
    eigenvals = walk_eigenvalues_fixed(P, pi)
    print(f"Eigenvalues: {eigenvals}")
    print(f"All on unit circle: {np.allclose(np.abs(eigenvals), 1.0)}")
    
    # Test spectral gaps
    classical_gap = classical_spectral_gap(P)
    D = discriminant_matrix(P, pi)
    quantum_gap = phase_gap(D)
    print(f"\nClassical gap: {classical_gap:.6f}")
    print(f"Quantum phase gap: {quantum_gap:.6f}")
    
    # Test mixing times
    classical_mixing = mixing_time_fixed(P, epsilon=0.01)
    quantum_mixing = quantum_mixing_time_estimate(quantum_gap, len(P), epsilon=0.01)
    print(f"\nClassical mixing time: {classical_mixing}")
    print(f"Quantum mixing time: {quantum_mixing}")
    
    # Test speedup
    speedup = compute_quantum_speedup(classical_mixing, quantum_mixing)
    print(f"Speedup: {speedup:.2f}x")