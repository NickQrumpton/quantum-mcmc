#!/usr/bin/env python3
"""
Debug the quantum walk construction step by step.
"""

import numpy as np
import warnings

def debug_n_cycle(N: int = 4):
    """Debug N-cycle construction step by step."""
    print(f"DEBUGGING {N}-CYCLE")
    print("=" * 40)
    
    # 1. Build transition matrix
    P = np.zeros((N, N), dtype=float)
    for x in range(N):
        P[x, (x + 1) % N] = 0.5
        P[x, (x - 1) % N] = 0.5
    
    print("1. Transition matrix P:")
    print(P)
    print(f"   Row sums: {np.sum(P, axis=1)}")
    print(f"   Col sums: {np.sum(P, axis=0)}")
    print()
    
    # 2. Classical eigenvalues (should be cos(2πk/N))
    classical_eigs = np.linalg.eigvals(P)
    classical_eigs_real = np.real(classical_eigs)
    classical_eigs_sorted = np.sort(classical_eigs_real)[::-1]
    
    print("2. Classical eigenvalues of P:")
    print(f"   Computed: {classical_eigs_sorted}")
    
    theoretical_eigs = [np.cos(2 * np.pi * k / N) for k in range(N)]
    theoretical_eigs.sort(reverse=True)
    print(f"   Theory:   {theoretical_eigs}")
    print()
    
    # 3. Build |p_x⟩ states
    print("3. |p_x⟩ states:")
    p_states = np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            p_states[x, y] = np.sqrt(P[x, y])
        print(f"   |p_{x}⟩ = {p_states[x]}")
    print()
    
    # 4. Check linear independence
    print("4. Linear independence check:")
    print(f"   Rank of p_states matrix: {np.linalg.matrix_rank(p_states)}")
    print(f"   Should be: {N}")
    print()
    
    # 5. Build projector Π_A carefully
    print("5. Building Π_A:")
    dim = N * N
    Pi_A = np.zeros((dim, dim))
    
    print(f"   Working in {dim}-dimensional space")
    
    for x in range(N):
        state_vector = np.zeros(dim)
        for y in range(N):
            idx = x * N + y
            state_vector[idx] = p_states[x, y]
        
        print(f"   |x={x}⟩⊗|p_{x}⟩: norm = {np.linalg.norm(state_vector):.6f}")
        
        # Check if state is normalized
        if abs(np.linalg.norm(state_vector) - 1.0) > 1e-10:
            print(f"   WARNING: State {x} not normalized!")
        
        Pi_A += np.outer(state_vector, state_vector)
    
    print(f"   Π_A rank: {np.linalg.matrix_rank(Pi_A)}")
    print(f"   Π_A trace: {np.trace(Pi_A):.6f}")
    print()
    
    # 6. Build projector Π_B  
    print("6. Building Π_B:")
    Pi_B = np.zeros((dim, dim))
    
    for y in range(N):
        state_vector = np.zeros(dim)
        for x in range(N):
            idx = x * N + y
            state_vector[idx] = p_states[y, x]  # Note: p_y[x] = sqrt(P[y,x])
        
        print(f"   |p_{y}*⟩⊗|y={y}⟩: norm = {np.linalg.norm(state_vector):.6f}")
        Pi_B += np.outer(state_vector, state_vector)
    
    print(f"   Π_B rank: {np.linalg.matrix_rank(Pi_B)}")
    print(f"   Π_B trace: {np.trace(Pi_B):.6f}")
    print()
    
    # 7. Check if Π_A and Π_B are proper projectors
    print("7. Projector properties:")
    Pi_A_squared = Pi_A @ Pi_A
    Pi_B_squared = Pi_B @ Pi_B
    
    print(f"   ||Π_A² - Π_A||_F = {np.linalg.norm(Pi_A_squared - Pi_A, 'fro'):.2e}")
    print(f"   ||Π_B² - Π_B||_F = {np.linalg.norm(Pi_B_squared - Pi_B, 'fro'):.2e}")
    
    # Check if they're different (should be for non-trivial cases)
    print(f"   ||Π_A - Π_B||_F = {np.linalg.norm(Pi_A - Pi_B, 'fro'):.6f}")
    print()
    
    # 8. Build W more carefully
    print("8. Building W(P):")
    I = np.eye(dim)
    
    # Check condition numbers
    cond_A = np.linalg.cond(2 * Pi_A - I)
    cond_B = np.linalg.cond(2 * Pi_B - I)
    print(f"   cond(2Π_A - I) = {cond_A:.2e}")
    print(f"   cond(2Π_B - I) = {cond_B:.2e}")
    
    if cond_A > 1e12 or cond_B > 1e12:
        print("   WARNING: Poor conditioning detected!")
    
    # Proceed with construction
    refl_A = 2 * Pi_A - I
    refl_B = 2 * Pi_B - I
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            W = refl_B @ refl_A
            print("   W constructed successfully")
        except RuntimeWarning as e:
            print(f"   ERROR in W construction: {e}")
            return None
    
    # Check unitarity
    WW_dag = W @ W.T
    unitarity_error = np.linalg.norm(WW_dag - I, 'fro')
    print(f"   Unitarity error: {unitarity_error:.2e}")
    
    if unitarity_error > 1e-10:
        print("   WARNING: W is not unitary!")
        return None
    
    print()
    
    # 9. Eigenvalue analysis
    print("9. Eigenvalue analysis of W:")
    try:
        eigenvals, eigenvecs = np.linalg.eigh(W)
        print(f"   Number of eigenvalues: {len(eigenvals)}")
        
        # Find stationary eigenvalue
        stationary_indices = np.where(np.abs(eigenvals - 1.0) < 1e-10)[0]
        print(f"   Stationary eigenvalues (≈1): {len(stationary_indices)}")
        
        # Find phases
        phases = np.angle(eigenvals.astype(complex))
        nonzero_phases = phases[np.abs(phases) > 1e-10]
        
        print(f"   Non-zero phases: {len(nonzero_phases)}")
        if len(nonzero_phases) > 0:
            min_phase = np.min(np.abs(nonzero_phases))
            phase_gap = 2 * min_phase
            
            print(f"   Smallest |phase|: {min_phase:.6f}")
            print(f"   Phase gap Δ(P) = 2θ: {phase_gap:.6f}")
            
            # Compare to theory
            theory_gap = 2 * np.pi / N
            print(f"   Theoretical Δ(P): {theory_gap:.6f}")
            print(f"   Ratio: {phase_gap / theory_gap:.6f}")
        else:
            print("   No non-zero phases found!")
            
    except Exception as e:
        print(f"   ERROR in eigenvalue analysis: {e}")
        return None
    
    print()
    print("Debug complete!")
    return W, eigenvals, eigenvecs

if __name__ == '__main__':
    for N in [3, 4]:
        result = debug_n_cycle(N)
        print("\n" + "="*60 + "\n")