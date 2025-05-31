"""Direct test of Theorem 6 implementation without full module dependencies.

Author: Nicholas Zhao
Date: 5/31/2025
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import UnitaryGate, QFT
from qiskit.quantum_info import Statevector, Operator
import matplotlib.pyplot as plt
import pandas as pd


def create_simple_markov_chain():
    """Create a simple 2x2 Markov chain for testing."""
    # Simple 2-state chain
    p = 0.3
    P = np.array([[1-p, p], [p, 1-p]])
    
    # Stationary distribution is uniform for symmetric chain
    pi = np.array([0.5, 0.5])
    
    # Spectral gap is 2p for this chain
    delta = 2 * p
    
    return P, delta, pi


def build_simple_walk_operator(P: np.ndarray) -> QuantumCircuit:
    """Build a simple quantum walk operator for 2x2 chain."""
    # For 2x2 chain, we need 2 qubits (1 for each register)
    qc = QuantumCircuit(2, name='W(P)')
    
    # Simple implementation: encode transition probabilities
    # This is a simplified version for testing
    theta = np.arccos(np.sqrt(P[0, 0]))  # Angle for rotation
    
    # Controlled rotation based on first qubit
    qc.cry(2 * theta, 0, 1)
    
    # Swap operation
    qc.swap(0, 1)
    
    return qc


def build_phase_comparator_simple(num_qubits: int, threshold: float) -> QuantumCircuit:
    """Build a simplified phase comparator for testing."""
    qc = QuantumCircuit(num_qubits + 2, name='PhaseComp')  # +2 for ancillas
    
    # Simple threshold comparison
    # Mark states with small phase values (near 0)
    threshold_int = max(1, int(threshold * (2**num_qubits) / (2 * np.pi)))
    
    # For small systems, use direct comparison
    if num_qubits <= 3:
        # Simple approach: flip phase if MSB is 0 (small phase)
        qc.x(0)  # Flip MSB
        qc.cz(0, num_qubits)  # Controlled Z on first ancilla
        qc.x(0)  # Unflip MSB
        
        # Also handle wrap-around (phases near 1)
        # Check if all bits are 1 (large phase)
        controls = list(range(num_qubits))
        if len(controls) == 1:
            qc.cz(controls[0], num_qubits)
        elif len(controls) == 2:
            qc.ccx(controls[0], controls[1], num_qubits + 1)
            qc.cz(num_qubits + 1, num_qubits)
            qc.ccx(controls[0], controls[1], num_qubits + 1)
    
    return qc


def build_qpe_simple(walk_op: QuantumCircuit, num_ancilla: int) -> QuantumCircuit:
    """Build simplified QPE for testing."""
    system_qubits = walk_op.num_qubits
    total_qubits = num_ancilla + system_qubits
    
    qc = QuantumCircuit(total_qubits, name='QPE')
    
    # Initialize ancillas
    for i in range(num_ancilla):
        qc.h(i)
    
    # Controlled powers of walk operator
    W_gate = walk_op.to_gate()
    for j in range(num_ancilla):
        power = 2 ** j
        # Apply controlled W^power
        for _ in range(power):
            controlled_W = W_gate.control(1)
            qc.append(controlled_W, [j] + list(range(num_ancilla, total_qubits)))
    
    # QFT on ancillas
    qft = QFT(num_ancilla, do_swaps=True)
    qc.append(qft, range(num_ancilla))
    
    return qc


def build_reflection_operator_simple(walk_op: QuantumCircuit, delta: float, k: int = 1):
    """Build simplified reflection operator for testing."""
    s = int(np.ceil(np.log2(2 * np.pi / delta)))
    s = max(2, min(s, 4))  # Clamp to reasonable range
    
    system_qubits = walk_op.num_qubits
    total_qubits = s + system_qubits + 2  # +2 for comparator ancillas
    
    qc = QuantumCircuit(total_qubits, name=f'R(k={k})')
    
    for rep in range(k):
        # QPE
        qpe = build_qpe_simple(walk_op, s)
        qc.append(qpe, list(range(s + system_qubits)))
        
        # Phase comparator
        comp = build_phase_comparator_simple(s, delta)
        qc.append(comp, list(range(s)) + list(range(s + system_qubits, total_qubits)))
        
        # Inverse QPE
        qpe_inv = qpe.inverse()
        qc.append(qpe_inv, list(range(s + system_qubits)))
    
    return qc


def test_reflection_bounds():
    """Test the reflection operator bounds."""
    print("=" * 60)
    print("SIMPLIFIED THEOREM 6 TEST")
    print("=" * 60)
    
    # Create test chain
    P, delta, pi = create_simple_markov_chain()
    print(f"Markov chain P:\n{P}")
    print(f"Spectral gap δ = {delta:.4f}")
    
    # Build walk operator
    W = build_simple_walk_operator(P)
    print(f"Walk operator uses {W.num_qubits} qubits")
    
    # Test different k values
    k_values = [1, 2, 3]
    results = []
    
    print(f"\nTesting reflection operator bounds:")
    print("k | ‖(R+I)|ψ⟩‖ | Bound 2^(1-k) | Ratio")
    print("-" * 45)
    
    for k in k_values:
        # Build reflection operator
        R = build_reflection_operator_simple(W, delta, k)
        
        # Create test state (orthogonal to stationary)
        # For 2-state chain, orthogonal to uniform is (1, -1)/√2
        test_state = QuantumCircuit(2)
        test_state.x(0)  # |10⟩ 
        test_state.h(1)   # (|10⟩ + |11⟩)/√2
        test_state.z(1)   # (|10⟩ - |11⟩)/√2
        
        # Apply to full system
        full_circuit = QuantumCircuit(R.num_qubits)
        full_circuit.append(test_state, range(2))
        full_circuit.append(R, range(R.num_qubits))
        
        # Get statevector
        try:
            sv_R = Statevector(full_circuit)
            
            # Identity part
            sv_I = Statevector(test_state)
            # Pad with zeros
            padded_data = np.zeros(2**R.num_qubits, dtype=complex)
            padded_data[:len(sv_I.data)] = sv_I.data
            sv_I_padded = Statevector(padded_data)
            
            # (R + I)|ψ⟩
            sv_sum = Statevector(sv_R.data + sv_I_padded.data)
            norm = np.linalg.norm(sv_sum.data)
            
            # Theoretical bound
            bound = 2**(1 - k)
            ratio = norm / bound
            
            print(f"{k} | {norm:11.6f} | {bound:13.6f} | {ratio:5.3f}")
            
            results.append({
                'k': k,
                'norm': norm,
                'bound': bound,
                'ratio': ratio,
                'satisfies': norm <= bound * 1.2  # 20% tolerance
            })
            
        except Exception as e:
            print(f"{k} | ERROR: {str(e)[:20]}...")
            results.append({
                'k': k, 'norm': np.inf, 'bound': 2**(1-k), 
                'ratio': np.inf, 'satisfies': False
            })
    
    # Summary
    print(f"\nSummary:")
    satisfies_all = all(r['satisfies'] for r in results)
    print(f"All bounds satisfied: {'YES' if satisfies_all else 'NO'}")
    
    if results:
        valid_results = [r for r in results if np.isfinite(r['ratio'])]
        if valid_results:
            avg_ratio = np.mean([r['ratio'] for r in valid_results])
            print(f"Average ratio: {avg_ratio:.3f}")
    
    # Resource analysis
    s = int(np.ceil(np.log2(2 * np.pi / delta)))
    s = max(2, min(s, 4))
    print(f"\nResource Analysis:")
    print(f"Required ancillas s = {s}")
    print(f"Controlled-W calls per QPE ≈ 2^s = {2**s}")
    print(f"Total calls for k repetitions ≈ 2k × 2^s")
    
    for k in k_values:
        estimated_calls = 2 * k * (2**s)
        theoretical_bound = k * 2**(s + 1)
        print(f"k={k}: ~{estimated_calls} calls (bound: {theoretical_bound})")
    
    return results


def create_visual_summary(results):
    """Create visualization of results."""
    if not results or not any(np.isfinite(r['ratio']) for r in results):
        print("No valid results to visualize")
        return
    
    valid_results = [r for r in results if np.isfinite(r['ratio'])]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    k_vals = [r['k'] for r in valid_results]
    norms = [r['norm'] for r in valid_results]
    bounds = [r['bound'] for r in valid_results]
    
    ax.semilogy(k_vals, norms, 'bo-', label='‖(R+I)|ψ⟩‖', markersize=8)
    ax.semilogy(k_vals, bounds, 'r--', label='Bound 2^(1-k)', linewidth=2)
    
    ax.set_xlabel('Repetitions (k)')
    ax.set_ylabel('Norm (log scale)')
    ax.set_title('Reflection Operator Error Bounds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_vals)
    
    plt.tight_layout()
    plt.savefig('simple_theorem_6_test.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved to 'simple_theorem_6_test.png'")


if __name__ == "__main__":
    results = test_reflection_bounds()
    create_visual_summary(results)