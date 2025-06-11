#!/usr/bin/env python3
"""
Minimal QPE Hardware Experiment (Reduced Dependencies)

This script runs the essential QPE experiment with minimal external dependencies.
Only requires: numpy, matplotlib, qiskit, qiskit-ibm-runtime

Usage:
    python run_qpe_minimal.py --backend ibm_brisbane
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import sys

# Check for essential imports only
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.circuit.library import QFT
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
except ImportError as e:
    print(f"❌ Missing essential Qiskit dependencies: {e}")
    print("🔧 Install with: pip install qiskit qiskit-ibm-runtime")
    sys.exit(1)

class MinimalQPEExperiment:
    """Minimal QPE experiment with basic functionality."""
    
    def __init__(self, backend_name: str, ancillas: int = 4, shots: int = 1024):
        self.backend_name = backend_name
        self.ancillas = ancillas
        self.shots = shots
        
        # 4x4 torus transition matrix  
        self.P = np.array([[0.7, 0.3], [0.4, 0.6]])
        self.phase_gap = np.pi / 4  # π/4 rad ≈ 0.7854
        
        # Initialize service
        try:
            # Try new channel first, fallback to legacy
            try:
                self.service = QiskitRuntimeService()  # Default channel
            except Exception:
                self.service = QiskitRuntimeService(channel="ibm_quantum")  # Legacy fallback
            
            self.backend = self.service.backend(backend_name)
            print(f"✅ Connected to {backend_name}")
        except Exception as e:
            print(f"❌ Failed to connect to {backend_name}: {e}")
            raise
    
    def build_qpe_circuit(self, state_type: str = "uniform") -> QuantumCircuit:
        """Build minimal QPE circuit."""
        # Registers
        ancilla = QuantumRegister(self.ancillas, 'anc')
        edge = QuantumRegister(2, 'edge')  # 2 qubits for 2-state system
        c_ancilla = ClassicalRegister(self.ancillas, 'meas')
        
        qc = QuantumCircuit(ancilla, edge, c_ancilla)
        
        # State preparation
        if state_type == "uniform":
            qc.h(edge[0])
            qc.h(edge[1])
        elif state_type == "stationary":
            # Simplified stationary state
            pi = np.array([4/7, 3/7])
            angle = 2 * np.arcsin(np.sqrt(pi[1]))
            qc.ry(angle, edge[0])
        
        # QPE: Hadamards on ancilla
        for i in range(self.ancillas):
            qc.h(ancilla[i])
        
        # Simplified controlled walk (demonstration)
        for j in range(self.ancillas):
            power = 2**j
            for _ in range(min(power, 4)):  # Limit for hardware
                qc.cx(ancilla[j], edge[0])
                qc.rz(np.pi/8, edge[0])
                qc.cx(ancilla[j], edge[1])
        
        # Inverse QFT
        qft_inv = QFT(self.ancillas, inverse=True)
        qc.append(qft_inv, ancilla)
        
        # Bit order correction (only for 4+ ancillas)
        if self.ancillas >= 4:
            qc.swap(ancilla[0], ancilla[3])
            qc.swap(ancilla[1], ancilla[2])
        elif self.ancillas == 3:
            qc.swap(ancilla[0], ancilla[2])
            # Middle qubit stays in place
        
        # Measure
        qc.measure(ancilla, c_ancilla)
        
        return qc
    
    def run_experiment(self) -> dict:
        """Run minimal QPE experiment."""
        print(f"\n🧪 Running minimal QPE on {self.backend_name}")
        print(f"   Ancillas: {self.ancillas}, Shots: {self.shots}")
        
        results = {
            'backend': self.backend_name,
            'timestamp': datetime.now().isoformat(),
            'ancillas': self.ancillas,
            'shots': self.shots,
            'phase_gap': self.phase_gap,
            'states': {}
        }
        
        states_to_test = ['uniform', 'stationary']
        
        with Session(backend=self.backend) as session:
            sampler = Sampler(mode=session)
            
            for state_name in states_to_test:
                print(f"\n--- Testing {state_name} state ---")
                
                # Build and transpile circuit
                qc = self.build_qpe_circuit(state_name)
                transpiled = transpile(qc, self.backend, optimization_level=2)
                
                print(f"  Circuit depth: {transpiled.depth()}")
                print(f"  CX gates: {transpiled.count_ops().get('cx', 0)}")
                
                # Run on hardware
                job = sampler.run([transpiled], shots=self.shots)
                result = job.result()
                
                # Extract counts
                pub_result = result[0]
                counts = pub_result.data.meas.get_counts()
                
                # Analyze results
                phases = self._extract_phases(counts)
                
                results['states'][state_name] = {
                    'counts': counts,
                    'phases': phases,
                    'circuit_depth': transpiled.depth(),
                    'cx_count': transpiled.count_ops().get('cx', 0)
                }
                
                # Print top result
                if phases:
                    top = phases[0]
                    print(f"  Top phase: bin {top['bin']}, prob {top['probability']:.3f}")
        
        return results
    
    def _extract_phases(self, counts: dict) -> list:
        """Extract phase information from counts."""
        total = sum(counts.values())
        phases = []
        
        for bitstring, count in counts.items():
            bin_val = int(bitstring[::-1], 2)  # Reverse for correct order
            phase = bin_val / (2**self.ancillas)
            prob = count / total
            
            if prob > 0.01:  # Only significant results
                phases.append({
                    'bin': bin_val,
                    'phase': phase,
                    'probability': prob,
                    'counts': count
                })
        
        return sorted(phases, key=lambda x: x['probability'], reverse=True)
    
    def create_simple_plot(self, results: dict, save_path: str = None):
        """Create simple visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        states = list(results['states'].keys())
        colors = ['blue', 'orange']
        
        for idx, (state, color) in enumerate(zip(states, colors)):
            ax = axes[idx]
            state_data = results['states'][state]
            phases = state_data['phases']
            
            if phases:
                bins = [p['bin'] for p in phases]
                probs = [p['probability'] for p in phases]
                
                ax.bar(bins, probs, color=color, alpha=0.7, edgecolor='black')
                ax.set_xlabel(f'Phase Bin (m/{2**self.ancillas})')
                ax.set_ylabel('Probability')
                ax.set_title(f'{state.title()} State')
                ax.grid(True, alpha=0.3)
                
                # Add theoretical markers
                if state == 'stationary':
                    ax.axvline(0, color='red', linestyle='--', label='Theory (bin 0)')
                    ax.legend()
        
        plt.suptitle(f'Minimal QPE Results - {results["backend"]} - {results["timestamp"][:10]}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Minimal QPE hardware experiment")
    parser.add_argument("--backend", default="ibm_brisbane", help="Backend name")
    parser.add_argument("--ancillas", type=int, default=3, help="Ancilla qubits (default: 3)")
    parser.add_argument("--shots", type=int, default=1024, help="Shots (default: 1024)")
    parser.add_argument("--save", action="store_true", help="Save results")
    
    args = parser.parse_args()
    
    print("🚀 Minimal QPE Hardware Experiment")
    print("=" * 50)
    
    # Run experiment
    try:
        exp = MinimalQPEExperiment(args.backend, args.ancillas, args.shots)
        results = exp.run_experiment()
        
        # Create visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"qpe_minimal_{args.backend}_{timestamp}.png"
        exp.create_simple_plot(results, plot_file if args.save else None)
        
        # Save results if requested
        if args.save:
            results_file = f"qpe_minimal_{args.backend}_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved: {results_file}")
        
        print("\n✅ Minimal experiment completed!")
        
    except KeyboardInterrupt:
        print("\n⏹ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()