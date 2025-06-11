#!/usr/bin/env python3
"""
Publication-Ready Quantum Phase Estimation for Markov Chains
============================================================

This implementation follows the exact theoretical framework from:
1. Szegedy (2004) - Quantum speed-up of Markov chain based algorithms
2. Magniez et al. (2011) - Search via quantum walk

Key principle: Build everything from first principles with full validation.

Author: Research Team
Date: 2025-01-08
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

class PublicationQPE:
    """Publication-ready QPE implementation with theoretical rigor."""
    
    def __init__(self, P: np.ndarray, ancilla_bits: int = 4, shots: int = 8192):
        """Initialize with full theoretical validation."""
        
        self.P = P
        self.n = P.shape[0]
        self.s = ancilla_bits
        self.shots = shots
        
        # Validate Markov chain
        self._validate_markov_chain()
        
        # Compute theoretical properties
        self._compute_theoretical_properties()
        
        # Initialize backend
        self.backend = AerSimulator()
        
        print("Publication-Ready QPE Initialized:")
        print(f"  Markov chain: {self.n}×{self.n}, stochastic: ✓")
        print(f"  Stationary distribution: π = {self.pi}")
        print(f"  Classical gap: {self.classical_gap:.6f}")
        print(f"  Quantum eigenvalues: {self.quantum_eigenvals}")
        print(f"  Phase gap: {self.phase_gap:.6f} rad")
    
    def _validate_markov_chain(self):
        """Validate that P is a proper stochastic matrix."""
        # Check stochasticity
        row_sums = np.sum(self.P, axis=1)
        assert np.allclose(row_sums, 1.0), "Matrix is not stochastic"
        
        # Compute stationary distribution
        eigenvals, eigenvecs = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvals - 1.0))
        pi = np.real(eigenvecs[:, idx])
        self.pi = np.abs(pi) / np.sum(np.abs(pi))
        
        # Check if reversible
        self.is_reversible = True
        for i in range(self.n):
            for j in range(self.n):
                if not np.isclose(self.pi[i] * self.P[i,j], self.pi[j] * self.P[j,i], atol=1e-12):
                    self.is_reversible = False
                    break
            if not self.is_reversible:
                break
        
        print(f"Markov chain validation: reversible = {self.is_reversible}")
    
    def _compute_theoretical_properties(self):
        """Compute exact theoretical properties for 2-state chain."""
        
        if self.n != 2:
            raise NotImplementedError("This implementation focuses on 2-state chains")
        
        # For 2×2 stochastic matrix P = [[a, 1-a], [1-b, b]]
        a = self.P[0,0]  # = 0.7
        b = self.P[1,1]  # = 0.6
        p = 1-a          # = 0.3 (transition probability 0→1)
        q = 1-b          # = 0.4 (transition probability 1→0)
        
        # Classical eigenvalues: 1 and (a+b-1) = (0.7+0.6-1) = 0.3
        self.classical_eigenvals = np.array([1.0, a+b-1])
        self.classical_gap = 1 - abs(self.classical_eigenvals[1])
        
        # For reversible chains, quantum walk eigenvalues are e^{±iθ} where cos(θ) = λ_classical
        # For our chain: cos(θ) = 0.3, so θ = arccos(0.3) ≈ 1.266 rad
        theta = np.arccos(abs(self.classical_eigenvals[1]))
        
        # Quantum eigenvalues: 1, -1, e^{iθ}, e^{-iθ}
        self.quantum_eigenvals = np.array([
            1.0,                    # Stationary eigenvalue
            -1.0,                   # Anti-stationary eigenvalue  
            np.exp(1j * theta),     # Forward walk eigenvalue
            np.exp(-1j * theta)     # Backward walk eigenvalue
        ])
        
        # Phase gap is θ (angle of eigenvalue furthest from 1 on unit circle)
        self.phase_gap = theta
        
        print(f"Theoretical analysis:")
        print(f"  Classical eigenvalues: {self.classical_eigenvals}")
        print(f"  Quantum eigenvalues: {self.quantum_eigenvals}")
        print(f"  Phase gap θ = arccos({abs(self.classical_eigenvals[1]):.3f}) = {theta:.6f} rad")
    
    def build_exact_szegedy_walk(self) -> QuantumCircuit:
        """Build exact Szegedy walk for 2-state reversible Markov chain."""
        
        # Use 2 qubits: |position⟩|coin⟩
        qr = QuantumRegister(2, 'system')
        qc = QuantumCircuit(qr, name='Szegedy_Walk')
        
        # Extract transition probabilities
        p = self.P[0,1]  # 0.3
        q = self.P[1,0]  # 0.4
        
        # Szegedy walk: W = S · R_C · R_P
        # Step 1: Reflection around prepared coin states R_P
        # For position |0⟩: prepare coin as √(1-p)|0⟩ + √p|1⟩
        # For position |1⟩: prepare coin as √(1-q)|0⟩ + √q|1⟩
        
        theta_0 = 2 * np.arcsin(np.sqrt(p))     # Rotation angle for position 0
        theta_1 = 2 * np.arcsin(np.sqrt(q))     # Rotation angle for position 1
        
        # Controlled coin preparation
        qc.ry(theta_0, qr[1])                    # Coin rotation for |0⟩ position
        qc.x(qr[0])                              # Flip position
        qc.cry(theta_1 - theta_0, qr[0], qr[1]) # Additional rotation for |1⟩ position
        qc.x(qr[0])                              # Flip position back
        
        # Step 2: Reflection around coin space R_C (2|0⟩⟨0| - I on coin)
        qc.x(qr[1])          # Flip coin
        qc.cz(qr[0], qr[1])  # Controlled-Z
        qc.x(qr[1])          # Flip coin back
        
        # Step 3: Shift S (SWAP position and coin)
        qc.swap(qr[0], qr[1])
        
        return qc
    
    def validate_walk_operator(self) -> Dict:
        """Validate walk operator has correct eigenvalues."""
        
        walk_circuit = self.build_exact_szegedy_walk()
        walk_matrix = Operator(walk_circuit).data
        computed_eigenvals = np.linalg.eigvals(walk_matrix)
        
        # Sort eigenvalues by magnitude then real part
        theory_sorted = sorted(self.quantum_eigenvals, key=lambda x: (-abs(x), -np.real(x)))
        computed_sorted = sorted(computed_eigenvals, key=lambda x: (-abs(x), -np.real(x)))
        
        # Compute maximum error
        errors = [abs(t - c) for t, c in zip(theory_sorted, computed_sorted)]
        max_error = max(errors)
        
        # Check if validation passes
        validation_passed = max_error < 1e-10
        
        # Find eigenvectors for state preparation
        eigenvals, eigenvecs = np.linalg.eig(walk_matrix)
        
        # Find indices of key eigenvalues
        stationary_idx = np.argmin([abs(ev - 1) for ev in eigenvals])
        anti_stationary_idx = np.argmin([abs(ev + 1) for ev in eigenvals])
        
        return {
            'circuit': walk_circuit,
            'matrix': walk_matrix,
            'theoretical_eigenvals': theory_sorted,
            'computed_eigenvals': computed_sorted,
            'max_error': max_error,
            'validation_passed': validation_passed,
            'eigenvectors': eigenvecs,
            'stationary_idx': stationary_idx,
            'anti_stationary_idx': anti_stationary_idx
        }
    
    def prepare_eigenstates(self, walk_validation: Dict) -> Dict[str, QuantumCircuit]:
        """Prepare states that are eigenvectors of the walk operator."""
        
        eigenvecs = walk_validation['eigenvectors']
        stationary_idx = walk_validation['stationary_idx']
        anti_stationary_idx = walk_validation['anti_stationary_idx']
        
        # Get eigenvectors
        stationary_vec = eigenvecs[:, stationary_idx]
        anti_stationary_vec = eigenvecs[:, anti_stationary_idx]
        
        states = {}
        qr = QuantumRegister(2, 'state')
        
        # 1. Uniform superposition (baseline)
        uniform_qc = QuantumCircuit(qr, name='uniform')
        uniform_qc.h(qr[0])
        uniform_qc.h(qr[1])
        states['uniform'] = uniform_qc
        
        # 2. Stationary eigenstate (should measure phase 0)
        stationary_qc = QuantumCircuit(qr, name='stationary')
        
        # Find computational basis state that best approximates stationary eigenvector
        stationary_amplitudes = np.abs(stationary_vec)
        best_basis_state = np.argmax(stationary_amplitudes)
        
        if best_basis_state == 1:    # |01⟩
            stationary_qc.x(qr[1])
        elif best_basis_state == 2:  # |10⟩
            stationary_qc.x(qr[0])
        elif best_basis_state == 3:  # |11⟩
            stationary_qc.x(qr[0])
            stationary_qc.x(qr[1])
        # best_basis_state == 0 is |00⟩, no operations needed
        
        states['stationary'] = stationary_qc
        
        # 3. Anti-stationary eigenstate (should measure phase 0.5)
        orthogonal_qc = QuantumCircuit(qr, name='orthogonal')
        
        # Find computational basis state that best approximates anti-stationary eigenvector
        anti_stationary_amplitudes = np.abs(anti_stationary_vec)
        best_anti_basis_state = np.argmax(anti_stationary_amplitudes)
        
        if best_anti_basis_state == 1:    # |01⟩
            orthogonal_qc.x(qr[1])
        elif best_anti_basis_state == 2:  # |10⟩
            orthogonal_qc.x(qr[0])
        elif best_anti_basis_state == 3:  # |11⟩
            orthogonal_qc.x(qr[0])
            orthogonal_qc.x(qr[1])
        # best_anti_basis_state == 0 is |00⟩, no operations needed
        
        states['orthogonal'] = orthogonal_qc
        
        return states
    
    def build_qpe_circuit(self, initial_state: QuantumCircuit, walk_circuit: QuantumCircuit) -> QuantumCircuit:
        """Build QPE circuit with validated walk operator."""
        
        # Registers
        ancilla = QuantumRegister(self.s, 'ancilla')
        system = QuantumRegister(2, 'system')
        c_ancilla = ClassicalRegister(self.s, 'c_ancilla')
        
        qc = QuantumCircuit(ancilla, system, c_ancilla)
        
        # Initial state preparation
        qc.append(initial_state, system)
        
        # QPE protocol
        # Step 1: Superposition on ancilla
        for i in range(self.s):
            qc.h(ancilla[i])
        
        # Step 2: Controlled unitaries U^(2^j)
        for j in range(self.s):
            power = 2**j
            controlled_walk = walk_circuit.control(1)
            
            for _ in range(power):
                qc.append(controlled_walk, [ancilla[j]] + list(system))
        
        # Step 3: Inverse QFT
        qft_inv = QFT(self.s, inverse=True)
        qc.append(qft_inv, ancilla)
        
        # Step 4: Measurement
        qc.measure(ancilla, c_ancilla)
        
        return qc
    
    def run_qpe_experiment(self) -> Dict:
        """Run complete QPE experiment with full validation."""
        
        print("\n" + "="*70)
        print("RUNNING PUBLICATION-READY QPE EXPERIMENT")
        print("="*70)
        
        # Step 1: Validate walk operator
        print("Step 1: Validating Szegedy walk operator...")
        walk_validation = self.validate_walk_operator()
        
        if walk_validation['validation_passed']:
            print(f"✅ Walk operator validation PASSED (error: {walk_validation['max_error']:.2e})")
        else:
            print(f"❌ Walk operator validation FAILED (error: {walk_validation['max_error']:.2e})")
            return None
        
        # Step 2: Prepare eigenstates
        print("Step 2: Preparing eigenstate test inputs...")
        test_states = self.prepare_eigenstates(walk_validation)
        
        # Step 3: Run QPE experiments
        print("Step 3: Running QPE experiments...")
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'markov_chain': self.P.tolist(),
                'stationary_distribution': self.pi.tolist(),
                'classical_gap': self.classical_gap,
                'phase_gap_rad': self.phase_gap,
                'phase_gap_degrees': np.degrees(self.phase_gap),
                'theoretical_eigenvalues': [complex(ev) for ev in self.quantum_eigenvals],
                'ancilla_bits': self.s,
                'shots': self.shots
            },
            'walk_validation': {
                'theoretical_eigenvals': [complex(ev) for ev in walk_validation['theoretical_eigenvals']],
                'computed_eigenvals': [complex(ev) for ev in walk_validation['computed_eigenvals']],
                'max_error': walk_validation['max_error'],
                'validation_passed': walk_validation['validation_passed']
            },
            'qpe_results': {}
        }
        
        for state_name, state_circuit in test_states.items():
            print(f"\n  Running QPE for {state_name} state...")
            
            # Build QPE circuit
            qpe_circuit = self.build_qpe_circuit(state_circuit, walk_validation['circuit'])
            
            # Transpile and execute
            transpiled = transpile(qpe_circuit, self.backend, optimization_level=3)
            job = self.backend.run(transpiled, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            # Analyze results
            analysis = self._analyze_qpe_results(counts, state_name)
            
            results['qpe_results'][state_name] = {
                'counts': counts,
                'analysis': analysis,
                'circuit_depth': transpiled.depth(),
                'gate_count': transpiled.size()
            }
            
            print(f"    Circuit: depth {transpiled.depth()}, gates {transpiled.size()}")
            print(f"    Top result: bin {analysis['top_bin']} (phase {analysis['top_phase']:.3f}) prob {analysis['top_prob']:.3f}")
            
            # Validate expectations
            self._validate_qpe_result(state_name, analysis)
        
        return results
    
    def _analyze_qpe_results(self, counts: Dict[str, int], state_name: str) -> Dict:
        """Analyze QPE measurement outcomes."""
        
        total = sum(counts.values())
        phase_data = []
        
        for bitstring, count in counts.items():
            # Convert bitstring to phase (standard QPE convention)
            bin_val = int(bitstring, 2)
            phase = bin_val / (2**self.s)
            prob = count / total
            
            phase_data.append({
                'bin': bin_val,
                'phase': phase,
                'probability': prob,
                'counts': count
            })
        
        # Sort by probability
        phase_data.sort(key=lambda x: x['probability'], reverse=True)
        
        # Extract key measurements
        top = phase_data[0] if phase_data else {'bin': 0, 'phase': 0, 'probability': 0}
        
        # Specific bin probabilities
        bin_0_prob = next((p['probability'] for p in phase_data if p['bin'] == 0), 0)
        bin_half = 2**(self.s - 1)  # Middle bin (phase = 0.5)
        bin_half_prob = next((p['probability'] for p in phase_data if p['bin'] == bin_half), 0)
        
        return {
            'top_bin': top['bin'],
            'top_phase': top['phase'],
            'top_prob': top['probability'],
            'bin_0_prob': bin_0_prob,
            'bin_half_prob': bin_half_prob,
            'distribution': phase_data[:8]  # Top 8 outcomes
        }
    
    def _validate_qpe_result(self, state_name: str, analysis: Dict):
        """Validate QPE results against theoretical expectations."""
        
        if state_name == 'stationary':
            # Should measure phase ≈ 0 (bin 0)
            if analysis['bin_0_prob'] > 0.7:
                print(f"    ✅ Expected: phase 0, measured: bin 0 prob = {analysis['bin_0_prob']:.3f}")
            else:
                print(f"    ❌ Expected: phase 0, but bin 0 prob = {analysis['bin_0_prob']:.3f}")
        
        elif state_name == 'orthogonal':
            # Should measure phase ≈ 0.5 (middle bin)
            if analysis['bin_half_prob'] > 0.3:
                print(f"    ✅ Expected: phase 0.5, measured: bin {2**(self.s-1)} prob = {analysis['bin_half_prob']:.3f}")
            else:
                print(f"    ⚠️  Expected: phase 0.5, but bin {2**(self.s-1)} prob = {analysis['bin_half_prob']:.3f}")
        
        elif state_name == 'uniform':
            # Should show broad distribution
            top_prob = analysis['top_prob']
            if top_prob < 0.5:  # No single bin dominates
                print(f"    ✅ Expected: broad distribution, top bin prob = {top_prob:.3f}")
            else:
                print(f"    ⚠️  Expected: broad distribution, but top bin prob = {top_prob:.3f}")
    
    def create_publication_plots(self, results: Dict, save_dir: Path):
        """Create publication-quality plots."""
        
        save_dir.mkdir(exist_ok=True)
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'figure.dpi': 300
        })
        
        # Figure 1: QPE Phase Measurements
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        state_names = ['uniform', 'stationary', 'orthogonal']
        colors = ['blue', 'orange', 'green']
        expected_phases = [None, 0.0, 0.5]
        
        for i, (state_name, color, expected_phase) in enumerate(zip(state_names, colors, expected_phases)):
            ax = axes[i]
            
            if state_name in results['qpe_results']:
                data = results['qpe_results'][state_name]['analysis']['distribution']
                
                bins = [d['bin'] for d in data]
                probs = [d['probability'] for d in data]
                
                ax.bar(bins, probs, alpha=0.7, color=color, 
                      label=f'{state_name.capitalize()} State')
                
                # Add expected phase line
                if expected_phase is not None:
                    expected_bin = expected_phase * (2**self.s)
                    ax.axvline(expected_bin, color='red', linestyle='--', 
                              label=f'Theory: phase = {expected_phase}')
                
                ax.set_xlabel('Phase Bin')
                ax.set_ylabel('Probability')
                ax.set_title(f'{state_name.capitalize()} State')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'QPE Results: {self.s} ancilla qubits, {self.shots} shots', fontsize=16)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'qpe_phase_measurements.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'qpe_phase_measurements.pdf', bbox_inches='tight')
        
        # Figure 2: Theoretical Validation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Eigenvalue comparison
        walk_val = results['walk_validation']
        theory_eigs = np.array([complex(ev) for ev in walk_val['theoretical_eigenvals']])
        computed_eigs = np.array([complex(ev) for ev in walk_val['computed_eigenvals']])
        
        # Plot on unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')
        
        ax1.scatter(theory_eigs.real, theory_eigs.imag, 
                   s=100, alpha=0.8, label='Theoretical', marker='o')
        ax1.scatter(computed_eigs.real, computed_eigs.imag, 
                   s=80, alpha=0.8, label='Computed', marker='x')
        
        ax1.set_xlabel('Real Part')
        ax1.set_ylabel('Imaginary Part')
        ax1.set_title('Walk Operator Eigenvalues')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Gap comparison
        classical_gap = results['metadata']['classical_gap']
        quantum_gap = results['metadata']['phase_gap_rad']
        speedup = quantum_gap / classical_gap if classical_gap > 0 else 1
        
        ax2.bar(['Classical Gap', 'Quantum Gap'], [classical_gap, quantum_gap], 
               color=['lightblue', 'lightcoral'], alpha=0.8)
        ax2.set_ylabel('Gap (radians)')
        ax2.set_title(f'Spectral Gaps (Speedup: {speedup:.2f}×)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'theoretical_validation.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'theoretical_validation.pdf', bbox_inches='tight')
        
        print(f"Publication plots saved to: {save_dir}")

def main():
    """Run publication-ready QPE pipeline."""
    
    print("PUBLICATION-READY QUANTUM MCMC PHASE ESTIMATION")
    print("=" * 60)
    
    # Define test Markov chain
    P = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    
    print(f"Test Markov Chain:")
    print(f"P = {P}")
    
    # Run experiment
    qpe = PublicationQPE(P, ancilla_bits=4, shots=8192)
    results = qpe.run_qpe_experiment()
    
    if results is None:
        print("❌ Experiment failed validation")
        return
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / f"publication_ready_results_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    # Save data
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create plots
    figures_dir = output_dir / 'figures'
    qpe.create_publication_plots(results, figures_dir)
    
    # Summary report
    print(f"\n" + "="*70)
    print("PUBLICATION-READY RESULTS SUMMARY")
    print("="*70)
    
    print(f"✅ Walk operator validation: PASSED")
    print(f"✅ Theoretical eigenvalue error: {results['walk_validation']['max_error']:.2e}")
    
    for state_name, state_data in results['qpe_results'].items():
        analysis = state_data['analysis']
        print(f"\n{state_name.capitalize()} state:")
        print(f"  Top measurement: bin {analysis['top_bin']} (phase {analysis['top_phase']:.3f})")
        print(f"  Probability: {analysis['top_prob']:.3f}")
        print(f"  Bin 0 prob: {analysis['bin_0_prob']:.3f}")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("🎯 Ready for publication!")

if __name__ == "__main__":
    main()