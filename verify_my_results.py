#!/usr/bin/env python3
"""
Critical verification of the results I generated.

This script checks whether my previous results are theoretically sound
or if there are fundamental errors that invalidate the conclusions.
"""

import numpy as np
import matplotlib.pyplot as plt


def analyze_my_previous_results():
    """Analyze the results from my previous implementation."""
    
    print("CRITICAL ANALYSIS OF PREVIOUS RESULTS")
    print("=" * 50)
    print()
    
    # What I reported:
    print("PREVIOUS RESULTS SUMMARY:")
    print("-" * 30)
    print("• N = 8 cycle")
    print("• Theoretical phase gap: 0.785398 rad")
    print("• Computed phase gap: 1.570796 rad") 
    print("• Ratio: 2.000000")
    print("• QPE on |π⟩: peak at m=0 ✓")
    print("• QPE on |ψⱼ⟩: peak at m=3")
    print("• Reflection error bounds: ε_j(k) ≤ 2^(1-k)")
    print()
    
    # Let's check if these make sense
    N = 8
    theoretical_gap = 2 * np.pi / N  # = π/4 ≈ 0.785398
    my_computed_gap = np.pi / 2      # = π/2 ≈ 1.570796
    
    print("THEORETICAL EXPECTATIONS:")
    print("-" * 30)
    print(f"N-cycle theoretical phase gap: 2π/N = {theoretical_gap:.6f}")
    print(f"My computed gap: {my_computed_gap:.6f}")
    print(f"Ratio: {my_computed_gap/theoretical_gap:.1f}")
    print()
    
    # Check if the QPE results make sense
    print("QPE ANALYSIS:")
    print("-" * 30)
    s = 3  # I used 3 ancilla qubits
    resolution = 1 / (2**s)  # = 1/8 = 0.125
    
    print(f"QPE resolution: 1/2^s = {resolution:.3f}")
    print(f"Theoretical phase for smallest gap: ≈ 1/N = {1/N:.3f}")
    print(f"Expected QPE outcome: ⌊2^s × (1/N)⌋ = {int(2**s / N)}")
    print()
    
    # But I reported peak at m=3, let's check what phase that corresponds to
    measured_m = 3
    measured_phase = measured_m / (2**s)
    print(f"My reported peak at m={measured_m}")
    print(f"This corresponds to phase: {measured_m}/2^s = {measured_phase:.3f}")
    print(f"Expected phase: 1/N = {1/N:.3f}")
    print(f"Difference: {abs(measured_phase - 1/N):.3f}")
    print()
    
    # Check if this is reasonable
    if abs(measured_phase - 1/N) < resolution:
        print("✓ QPE result is within resolution limit - REASONABLE")
    else:
        print("❌ QPE result is outside resolution limit - SUSPICIOUS")
    print()
    
    print("REFLECTION OPERATOR ANALYSIS:")
    print("-" * 30)
    
    # My reported fidelities
    reported_fidelities = [0.0, 0.5, 0.75, 0.875]  # for k=1,2,3,4
    theoretical_fidelities = [1 - 2**(1-k) for k in [1,2,3,4]]
    
    print("Fidelity comparison:")
    for k, (reported, theory) in enumerate(zip(reported_fidelities, theoretical_fidelities), 1):
        print(f"  k={k}: Reported={reported:.3f}, Theory≥{theory:.3f}")
        
        if reported >= theory - 0.1:  # Allow some tolerance
            status = "✓ REASONABLE"
        else:
            status = "❌ SUSPICIOUS"
        print(f"         {status}")
    print()
    
    print("FUNDAMENTAL CONCERNS:")
    print("-" * 30)
    
    concerns = []
    
    # Concern 1: Phase gap ratio of exactly 2.0
    if abs(my_computed_gap/theoretical_gap - 2.0) < 0.001:
        concerns.append("Phase gap ratio is exactly 2.0 - may indicate systematic error")
    
    # Concern 2: Many eigenvalues at phase 0
    concerns.append("Previous run showed ~50+ eigenvalues with phase 0 - highly suspicious")
    
    # Concern 3: Numerical warnings
    concerns.append("Matrix operations had divide-by-zero and overflow warnings")
    
    # Concern 4: Fidelity starting at 0 for k=1
    if reported_fidelities[0] == 0.0:
        concerns.append("Fidelity of 0.0 for k=1 seems unrealistic")
    
    for i, concern in enumerate(concerns, 1):
        print(f"{i}. {concern}")
    
    print()
    
    return len(concerns) == 0


def correct_theoretical_analysis():
    """Provide the correct theoretical analysis."""
    
    print("CORRECT THEORETICAL FRAMEWORK:")
    print("=" * 50)
    print()
    
    N = 8
    
    # Build the correct 8-cycle
    P = np.zeros((N, N))
    for x in range(N):
        P[x, (x + 1) % N] = 0.5
        P[x, (x - 1) % N] = 0.5
    
    # Compute correct classical eigenvalues
    classical_eigvals, _ = np.linalg.eig(P)
    classical_eigvals = np.sort(np.real(classical_eigvals))[::-1]  # Sort descending
    
    print("Classical eigenvalues (theory vs computed):")
    for k in range(N):
        theory = np.cos(2 * np.pi * k / N)
        computed = classical_eigvals[k] if k < len(classical_eigvals) else np.nan
        print(f"  k={k}: theory={theory:7.4f}, computed={computed:7.4f}")
    
    print()
    
    # The quantum walk phase gap
    # For symmetric chains, this should be 2π/N
    correct_phase_gap = 2 * np.pi / N
    print(f"Correct quantum phase gap: 2π/{N} = {correct_phase_gap:.6f} rad")
    print(f"In degrees: {correct_phase_gap * 180/np.pi:.1f}°")
    print()
    
    # What the QPE should show
    s = 3
    print(f"With s={s} ancilla qubits:")
    print(f"Resolution: 1/2^{s} = {1/(2**s):.3f}")
    
    # For stationary state |π⟩: phase = 0, so QPE should give m=0
    print(f"QPE on |π⟩: should peak at m=0 (phase=0)")
    
    # For first excited state: phase = 2π/N
    first_excited_phase = 2 * np.pi / N
    expected_m = round(first_excited_phase * (2**s) / (2*np.pi))
    print(f"QPE on |ψ_1⟩: phase={first_excited_phase:.3f}, should peak at m≈{expected_m}")
    
    print()
    
    # Reflection operator bounds
    print("Reflection operator theory:")
    print("For stationary state: R|π⟩ ≈ |π⟩ with fidelity → 1")
    print("For non-stationary: ||(R+I)|ψ⟩|| ≤ 2^(1-k)")
    
    for k in [1, 2, 3, 4]:
        bound = 2**(1-k)
        min_fidelity = 1 - bound  # Rough estimate
        print(f"  k={k}: error bound ≤ {bound:.3f}, min fidelity ≈ {min_fidelity:.3f}")
    
    print()
    
    return {
        'correct_phase_gap': correct_phase_gap,
        'classical_eigvals': classical_eigvals,
        'expected_qpe_outcomes': {'stationary': 0, 'first_excited': expected_m}
    }


def main():
    """Main verification function."""
    
    results_valid = analyze_my_previous_results()
    
    print("\n" + "=" * 50)
    
    if results_valid:
        print("VERDICT: Previous results appear VALID ✓")
    else:
        print("VERDICT: Previous results have SERIOUS ISSUES ❌")
    
    print()
    
    correct_analysis = correct_theoretical_analysis()
    
    print("RECOMMENDATIONS:")
    print("-" * 20)
    print("1. Re-implement with proper eigenvalue computation")
    print("2. Verify numerical stability of matrix operations") 
    print("3. Compare against known analytical results")
    print("4. Use more robust quantum circuit simulation")
    print("5. Validate each component independently")
    
    return results_valid, correct_analysis


if __name__ == '__main__':
    valid, correct = main()