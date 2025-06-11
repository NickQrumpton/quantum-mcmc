#!/usr/bin/env python3
"""
Simple QPE Validation - Single Qubit
====================================

The simplest possible QPE implementation using a single-qubit rotation
with known eigenvalues to validate the QPE methodology.

This will establish that the QPE circuit mechanics are correct
before moving to multi-qubit systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path
import json
from datetime import datetime

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

class SimpleQPE:
    """Simple single-qubit QPE for validation."""
    
    def __init__(self, ancilla_bits: int = 4, shots: int = 8192):
        self.s = ancilla_bits
        self.shots = shots
        self.backend = AerSimulator()
        
        # Define a simple single-qubit rotation with known eigenvalues
        self.theta = np.pi / 3  # 60 degrees
        self.phase = self.theta / (2 * np.pi)  # Phase in units of full turns
        
        print("Simple QPE Initialized:")
        print(f"  Rotation angle: {self.theta:.6f} rad = {np.degrees(self.theta):.1f} degrees")
        print(f"  Expected phase: {self.phase:.6f} turns")
        print(f"  Expected phase bin: {self.phase * 2**self.s:.1f}")
        print(f"  Ancilla bits: {self.s}")
        print(f"  Shots: {self.shots}")
    
    def create_rotation_circuit(self) -> QuantumCircuit:
        """Create a simple Z rotation circuit."""
        qr = QuantumRegister(1, 'qubit')
        qc = QuantumCircuit(qr, name='RZ')
        qc.rz(self.theta, qr[0])
        return qc
    
    def validate_rotation(self) -> bool:
        """Validate the rotation has correct eigenvalues."""
        circuit = self.create_rotation_circuit()
        matrix = Operator(circuit).data
        eigenvals = np.linalg.eigvals(matrix)
        
        # Expected eigenvalues: e^{±iθ/2}
        expected = [np.exp(1j * self.theta / 2), np.exp(-1j * self.theta / 2)]
        
        print(f"Rotation validation:")
        print(f"  Expected eigenvalues: {expected}")
        print(f"  Computed eigenvalues: {eigenvals}")
        
        # Check eigenvalues match (allowing for reordering)
        errors = []
        for exp in expected:
            min_error = min(abs(exp - comp) for comp in eigenvals)
            errors.append(min_error)
        
        max_error = max(errors)
        print(f"  Max error: {max_error:.2e}")
        
        return max_error < 1e-10
    
    def create_test_states(self) -> Dict[str, QuantumCircuit]:
        """Create test states for QPE."""
        states = {}
        qr = QuantumRegister(1, 'state')
        
        # |0⟩ state (eigenvalue e^{-iθ/2})
        state_0 = QuantumCircuit(qr, name='state_0')
        # No operations - |0⟩ is default
        states['state_0'] = state_0
        
        # |1⟩ state (eigenvalue e^{+iθ/2})  
        state_1 = QuantumCircuit(qr, name='state_1')
        state_1.x(qr[0])
        states['state_1'] = state_1
        
        # |+⟩ state (superposition)
        state_plus = QuantumCircuit(qr, name='state_plus')
        state_plus.h(qr[0])
        states['state_plus'] = state_plus
        
        return states
    
    def build_qpe_circuit(self, initial_state: QuantumCircuit, unitary: QuantumCircuit) -> QuantumCircuit:
        """Build single-qubit QPE circuit."""
        
        # Registers
        ancilla = QuantumRegister(self.s, 'ancilla')
        system = QuantumRegister(1, 'system')
        c_ancilla = ClassicalRegister(self.s, 'c_ancilla')
        
        qc = QuantumCircuit(ancilla, system, c_ancilla)
        
        # Initial state preparation
        qc.append(initial_state, system)
        
        # QPE
        # Step 1: Hadamard on ancilla
        for i in range(self.s):
            qc.h(ancilla[i])
        
        # Step 2: Controlled unitaries
        for j in range(self.s):
            power = 2**j
            
            # Create controlled version of the unitary
            controlled_u = unitary.control(1)
            
            # Apply U^(2^j)
            for _ in range(power):
                qc.append(controlled_u, [ancilla[j]] + list(system))
        
        # Step 3: Inverse QFT
        qft_inv = QFT(self.s, inverse=True)
        qc.append(qft_inv, ancilla)
        
        # Step 4: Measure
        qc.measure(ancilla, c_ancilla)
        
        return qc
    
    def run_experiments(self) -> Dict:
        """Run QPE experiments."""
        
        print("\n" + "="*50)
        print("RUNNING SIMPLE QPE EXPERIMENTS")
        print("="*50)
        
        # Validate rotation
        if not self.validate_rotation():
            print("❌ Rotation validation failed!")
            return None
        
        print("✅ Rotation validation passed!")
        
        # Get circuits
        rotation_circuit = self.create_rotation_circuit()
        test_states = self.create_test_states()
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'rotation_angle_rad': self.theta,
                'rotation_angle_deg': np.degrees(self.theta),
                'expected_phase': self.phase,
                'expected_bin': self.phase * 2**self.s,
                'ancilla_bits': self.s,
                'shots': self.shots
            },
            'experiments': {}
        }
        
        # Run experiments
        for state_name, state_circuit in test_states.items():
            print(f"\nRunning QPE for {state_name}...")
            
            # Build QPE circuit
            qpe_circuit = self.build_qpe_circuit(state_circuit, rotation_circuit)
            
            # Transpile and run
            transpiled = transpile(qpe_circuit, self.backend, optimization_level=3)
            job = self.backend.run(transpiled, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            # Analyze
            analysis = self._analyze_results(counts, state_name)
            
            results['experiments'][state_name] = {
                'counts': counts,
                'analysis': analysis,
                'circuit_depth': transpiled.depth(),
                'gate_count': transpiled.size()
            }
            
            print(f"  Top result: bin {analysis['top_bin']} (phase {analysis['top_phase']:.4f}) prob {analysis['top_prob']:.3f}")
            
            # Expected phase for this state
            if state_name == 'state_0':
                expected_phase = (-self.theta/2) / (2*np.pi)
                if expected_phase < 0:
                    expected_phase += 1  # Wrap to [0,1)
            elif state_name == 'state_1':
                expected_phase = (self.theta/2) / (2*np.pi)
            else:
                expected_phase = None  # Superposition
            
            if expected_phase is not None:
                expected_bin = expected_phase * 2**self.s
                phase_error = min(
                    abs(analysis['top_phase'] - expected_phase),
                    abs(analysis['top_phase'] - expected_phase + 1),
                    abs(analysis['top_phase'] - expected_phase - 1)
                )
                print(f"  Expected: phase {expected_phase:.4f} (bin {expected_bin:.1f})")
                print(f"  Phase error: {phase_error:.4f}")
                
                if phase_error < 0.1:
                    print(f"  ✅ Phase measurement correct!")
                else:
                    print(f"  ❌ Phase measurement error too large!")
        
        return results
    
    def _analyze_results(self, counts: Dict[str, int], state_name: str) -> Dict:
        """Analyze QPE results."""
        
        total = sum(counts.values())
        phase_data = []
        
        for bitstring, count in counts.items():
            bin_val = int(bitstring, 2)
            phase = bin_val / (2**self.s)
            prob = count / total
            
            phase_data.append({
                'bin': bin_val,
                'phase': phase,
                'probability': prob
            })
        
        phase_data.sort(key=lambda x: x['probability'], reverse=True)
        
        top = phase_data[0] if phase_data else {'bin': 0, 'phase': 0, 'probability': 0}
        
        return {
            'top_bin': top['bin'],
            'top_phase': top['phase'],
            'top_prob': top['probability'],
            'distribution': phase_data[:8]
        }
    
    def create_plot(self, results: Dict, save_dir: Path):
        """Create validation plot."""
        
        save_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        state_names = ['state_0', 'state_1', 'state_plus']
        
        for i, state_name in enumerate(state_names):
            ax = axes[i]
            
            if state_name in results['experiments']:
                data = results['experiments'][state_name]['analysis']['distribution']
                
                bins = [d['bin'] for d in data]
                probs = [d['probability'] for d in data]
                
                ax.bar(bins, probs, alpha=0.7)
                ax.set_xlabel('Phase Bin')
                ax.set_ylabel('Probability')
                ax.set_title(f'{state_name}')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Simple QPE Validation: θ={np.degrees(self.theta):.1f}°')
        plt.tight_layout()
        
        plt.savefig(save_dir / 'simple_qpe_validation.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_dir}")

def main():
    """Run simple QPE validation."""
    
    print("SIMPLE QPE VALIDATION")
    print("=" * 30)
    
    qpe = SimpleQPE(ancilla_bits=4, shots=8192)
    results = qpe.run_experiments()
    
    if results is None:
        print("❌ Validation failed!")
        return
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / f"simple_qpe_validation_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    qpe.create_plot(results, output_dir)
    
    print(f"\n✅ Simple QPE validation complete!")
    print(f"📁 Results: {output_dir}")

if __name__ == "__main__":
    main()