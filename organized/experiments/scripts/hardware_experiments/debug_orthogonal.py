#!/usr/bin/env python3
"""Debug which state gives the -1 eigenvalue (phase 0.5, bin 8)."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from qpe_real_hardware import QPEHardwareExperiment
from qiskit.quantum_info import Statevector

def find_orthogonal_state():
    """Find which state gives -1 eigenvalue."""
    
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    
    experiment = QPEHardwareExperiment(
        transition_matrix=P,
        ancilla_bits=4,
        shots=1024,
        use_simulator=True
    )
    
    print("Testing different states to find -1 eigenstate...")
    
    # Test states and their expected phases
    test_cases = [
        ('|00⟩', '00'),
        ('|01⟩', '01'), 
        ('|10⟩', '10'),
        ('|11⟩', '11'),
        ('|++⟩', 'uniform'),
        ('|+-⟩', 'custom_1'),
        ('|-+⟩', 'custom_2'),
        ('|--⟩', 'custom_3')
    ]
    
    # Run QPE on each state
    for label, state_name in test_cases:
        # Create custom state preparation function temporarily
        if state_name in ['00', '01', '10', '11']:
            # Use existing orthogonal state with modification
            pass
        elif state_name == 'uniform':
            state_name = 'stationary'  # This prepares |++⟩
        else:
            continue  # Skip custom states for now
            
        print(f"\nTesting {label}:")
        try:
            if state_name in ['00', '01', '10', '11']:
                # Modify the orthogonal state preparation temporarily
                original_prepare = experiment._prepare_test_state
                
                def custom_prepare(name):
                    if name == 'test':
                        from qiskit import QuantumRegister, QuantumCircuit
                        qr = QuantumRegister(2, 'edge')
                        qc = QuantumCircuit(qr)
                        
                        # Prepare specific computational basis state
                        if state_name == '01':
                            qc.x(qr[1])
                        elif state_name == '10':
                            qc.x(qr[0])
                        elif state_name == '11':
                            qc.x(qr[0])
                            qc.x(qr[1])
                        # '00' needs no gates
                        
                        return qc
                    else:
                        return original_prepare(name)
                
                experiment._prepare_test_state = custom_prepare
                state_to_test = 'test'
            else:
                state_to_test = state_name
            
            # Run single QPE
            result = experiment.run_hardware_qpe_single([state_to_test])
            
            if state_to_test in result['states']:
                counts = result['states'][state_to_test]['counts']
                total = sum(counts.values())
                
                # Check bin 0 and bin 8 probabilities
                bin_0 = sum(v for k, v in counts.items() if int(k[::-1], 2) % 16 == 0)
                bin_8 = sum(v for k, v in counts.items() if int(k[::-1], 2) % 16 == 8)
                
                prob_0 = bin_0 / total
                prob_8 = bin_8 / total
                
                print(f"  Bin 0 (phase 0.0): {prob_0:.3f}")
                print(f"  Bin 8 (phase 0.5): {prob_8:.3f}")
                
                if prob_8 > 0.5:
                    print(f"  ✅ FOUND -1 EIGENSTATE: {label}")
                elif prob_0 > 0.5:
                    print(f"  → +1 eigenstate")
                else:
                    print(f"  → Mixed")
                    
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    find_orthogonal_state()