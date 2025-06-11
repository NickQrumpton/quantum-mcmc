#!/usr/bin/env python3
"""Debug the walk operator to understand eigenvalue structure."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from qpe_real_hardware import QPEHardwareExperiment
from qiskit.quantum_info import Statevector, Operator

def debug_walk_operator():
    """Debug the walk operator eigenvalue structure."""
    
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    
    experiment = QPEHardwareExperiment(
        transition_matrix=P,
        ancilla_bits=4,
        shots=1024,
        use_simulator=True
    )
    
    print("Classical Markov chain analysis:")
    eigenvals, eigenvecs = np.linalg.eig(P.T)
    print(f"Classical eigenvalues: {eigenvals}")
    
    # Find stationary distribution
    idx = np.argmin(np.abs(eigenvals - 1.0))
    pi = np.real(eigenvecs[:, idx])
    pi = np.abs(pi) / np.sum(np.abs(pi))
    print(f"Stationary distribution: {pi}")
    
    # Build the quantum walk operator
    walk_circuit = experiment.build_walk_operator()
    print(f"\nQuantum walk circuit:")
    print(f"  Qubits: {walk_circuit.num_qubits}")
    print(f"  Depth: {walk_circuit.depth()}")
    print(f"  Gates: {walk_circuit.count_ops()}")
    
    # Convert to unitary matrix and analyze eigenvalues
    try:
        walk_unitary = Operator(walk_circuit).data
        walk_eigenvals = np.linalg.eigvals(walk_unitary)
        
        print(f"\nQuantum walk eigenvalues:")
        for i, ev in enumerate(walk_eigenvals):
            phase = np.angle(ev) / (2 * np.pi)  # Phase in units of 2π
            if phase < 0:
                phase += 1  # Convert to [0, 1] range
            print(f"  λ_{i}: {ev:.4f}, |λ|: {np.abs(ev):.4f}, phase: {phase:.4f}")
            
        # Check which eigenvalue corresponds to stationary state
        print(f"\nAnalyzing stationary eigenstate:")
        
        # The stationary state should have eigenvalue 1 (phase 0)
        # If we're seeing phase 0.5, it means eigenvalue -1
        
        # Check if there's an eigenvalue at +1
        ev_1_indices = np.where(np.abs(walk_eigenvals - 1) < 0.01)[0]
        ev_neg1_indices = np.where(np.abs(walk_eigenvals + 1) < 0.01)[0]
        
        print(f"Eigenvalues near +1: {len(ev_1_indices)} found")
        print(f"Eigenvalues near -1: {len(ev_neg1_indices)} found")
        
        if len(ev_1_indices) > 0:
            print(f"  +1 eigenvalue: {walk_eigenvals[ev_1_indices[0]]}")
        if len(ev_neg1_indices) > 0:
            print(f"  -1 eigenvalue: {walk_eigenvals[ev_neg1_indices[0]]}")
            
    except Exception as e:
        print(f"Error analyzing walk operator: {e}")
    
    # Test different initial states
    print(f"\nTesting different initial states:")
    test_states = {
        '|00⟩': Statevector.from_label('00'),
        '|01⟩': Statevector.from_label('01'), 
        '|10⟩': Statevector.from_label('10'),
        '|11⟩': Statevector.from_label('11'),
        '|++⟩': Statevector.from_label('++')
    }
    
    try:
        walk_op = Operator(walk_circuit)
        for label, state in test_states.items():
            evolved_state = state.evolve(walk_op)
            overlap = np.abs(state.inner(evolved_state))**2
            phase = np.angle(state.inner(evolved_state)) / (2 * np.pi)
            if phase < 0:
                phase += 1
            print(f"  {label}: overlap = {overlap:.4f}, phase = {phase:.4f}")
            
    except Exception as e:
        print(f"Error testing states: {e}")

if __name__ == "__main__":
    debug_walk_operator()