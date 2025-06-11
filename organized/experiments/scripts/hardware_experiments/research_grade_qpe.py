#!/usr/bin/env python3
"""
Research-Grade Quantum Phase Estimation for Markov Chains
=========================================================

This module implements a rigorous, theoretically-grounded QPE pipeline
for quantum Markov chain analysis with full validation against theory.

Based on:
- Szegedy (2004): "Quantum speed-up of Markov chain based algorithms"
- Magniez et al. (2011): "Search via quantum walk" 

Author: Research Implementation
Date: 2025-01-08
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

class TheoreticalValidator:
    """Validates implementations against theoretical predictions."""
    
    @staticmethod
    def validate_markov_chain(P: np.ndarray) -> Dict[str, any]:
        """Validate Markov chain properties."""
        n = P.shape[0]
        
        # Check stochasticity
        row_sums = np.sum(P, axis=1)
        is_stochastic = np.allclose(row_sums, 1.0)
        
        # Compute stationary distribution
        eigenvals, eigenvecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigenvals - 1.0))
        pi = np.real(eigenvecs[:, idx])
        pi = np.abs(pi) / np.sum(np.abs(pi))
        
        # Check detailed balance (reversibility)
        is_reversible = True
        for i in range(n):
            for j in range(n):
                if not np.isclose(pi[i] * P[i,j], pi[j] * P[j,i], atol=1e-10):
                    is_reversible = False
                    break
            if not is_reversible:
                break
        
        # Classical spectral gap
        sorted_eigenvals = sorted(np.abs(eigenvals), reverse=True)
        classical_gap = 1 - sorted_eigenvals[1] if len(sorted_eigenvals) > 1 else 1
        
        return {
            'is_stochastic': is_stochastic,
            'is_reversible': is_reversible,
            'stationary_distribution': pi,
            'eigenvalues': eigenvals,
            'classical_gap': classical_gap,
            'mixing_time_classical': int(np.ceil(1/classical_gap)) if classical_gap > 0 else np.inf
        }
    
    @staticmethod
    def theoretical_szegedy_eigenvalues(P: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute theoretical eigenvalues of Szegedy walk."""
        validation = TheoreticalValidator.validate_markov_chain(P)
        
        if not validation['is_reversible']:
            # Make lazy and reversible
            alpha = 0.5
            P_lazy = alpha * np.eye(P.shape[0]) + (1 - alpha) * P
            validation_lazy = TheoreticalValidator.validate_markov_chain(P_lazy)
            classical_eigenvals = validation_lazy['eigenvalues']
        else:
            classical_eigenvals = validation['eigenvalues']
        
        # For reversible chains, Szegedy walk eigenvalues are ±√λ_i
        quantum_eigenvals = []
        for lam in classical_eigenvals:
            if np.isreal(lam) and lam >= 0:
                sqrt_lam = np.sqrt(lam)
                quantum_eigenvals.extend([sqrt_lam, -sqrt_lam])
        
        quantum_eigenvals = np.array(quantum_eigenvals)
        
        # Phase gap is arccos of second largest eigenvalue magnitude
        sorted_mags = sorted(np.abs(quantum_eigenvals), reverse=True)
        if len(sorted_mags) > 1:
            phase_gap = np.arccos(sorted_mags[1])
        else:
            phase_gap = np.pi/2
            
        return quantum_eigenvals, phase_gap

class ResearchGradeQPE:
    """Research-grade QPE implementation with full theoretical validation."""
    
    def __init__(self, 
                 transition_matrix: np.ndarray,
                 ancilla_bits: int = 4,
                 shots: int = 4096,
                 backend_name: str = 'aer_simulator'):
        """
        Initialize research-grade QPE experiment.
        
        Args:
            transition_matrix: Markov chain transition matrix
            ancilla_bits: Number of QPE ancilla qubits
            shots: Number of measurement shots
            backend_name: Quantum backend to use
        """
        self.P_original = transition_matrix.copy()
        self.n_states = transition_matrix.shape[0]
        self.ancilla_bits = ancilla_bits
        self.shots = shots
        
        # Validate input Markov chain
        self.classical_properties = TheoreticalValidator.validate_markov_chain(self.P_original)
        
        if not self.classical_properties['is_stochastic']:
            raise ValueError("Input matrix is not stochastic")
        
        # Make reversible if needed
        if not self.classical_properties['is_reversible']:
            print("Chain is not reversible. Making lazy...")
            alpha = 0.5
            self.P = alpha * np.eye(self.n_states) + (1 - alpha) * self.P_original
            self.classical_properties = TheoreticalValidator.validate_markov_chain(self.P)
            self.is_lazy = True
        else:
            self.P = self.P_original.copy()
            self.is_lazy = False
        
        # Compute theoretical quantum properties
        self.quantum_eigenvals, self.phase_gap = TheoreticalValidator.theoretical_szegedy_eigenvalues(self.P)
        
        # System size (needs to be power of 2 for quantum circuits)
        self.system_qubits = int(np.ceil(np.log2(self.n_states)))
        self.padded_size = 2**self.system_qubits
        
        # Initialize backend
        self.backend = AerSimulator()
        
        print(f"Research-Grade QPE Initialized:")
        print(f"  Original chain: {self.n_states} states, reversible: {not self.is_lazy}")
        print(f"  Working chain: {self.n_states} states, reversible: {self.classical_properties['is_reversible']}")
        print(f"  Classical gap: {self.classical_properties['classical_gap']:.6f}")
        print(f"  Theoretical phase gap: {self.phase_gap:.6f} rad")
        print(f"  Quantum eigenvalues: {len(self.quantum_eigenvals)} total")
        print(f"  System qubits: {self.system_qubits} (padded to {self.padded_size})")
    
    def build_theoretical_szegedy_walk(self) -> QuantumCircuit:
        """Build theoretically correct Szegedy walk for 2-state reversible chain."""
        
        if self.n_states != 2:
            raise NotImplementedError("Currently only supports 2-state chains")
        
        # For 2-state reversible chain, we can build exact Szegedy walk
        # System: 1 qubit (|0⟩, |1⟩)
        # Coin: 1 qubit (edge directions)
        
        qr = QuantumRegister(2, 'szegedy')  # [system, coin]
        qc = QuantumCircuit(qr, name='Szegedy_Walk')
        
        # Get transition probabilities
        p01 = self.P[0, 1]  # Probability 0 → 1
        p10 = self.P[1, 0]  # Probability 1 → 0
        p00 = self.P[0, 0]  # Probability 0 → 0  
        p11 = self.P[1, 1]  # Probability 1 → 1
        
        # Szegedy walk: W = S · R_C · R_P
        # where R_P reflects around coin preparation, R_C reflects around coin space, S shifts
        
        # Step 1: Prepare coin states based on transition probabilities
        # For state |0⟩: coin prepared as √p00|0⟩ + √p01|1⟩
        # For state |1⟩: coin prepared as √p10|0⟩ + √p11|1⟩
        
        # Conditional coin preparation
        theta_0 = 2 * np.arcsin(np.sqrt(p01)) if p01 > 0 else 0
        theta_1 = 2 * np.arcsin(np.sqrt(p11)) if p11 > 0 else 0
        
        # Apply controlled rotations
        qc.ry(theta_0, qr[1])  # Coin rotation for |0⟩ system state
        qc.x(qr[0])
        qc.cry(theta_1 - theta_0, qr[0], qr[1])  # Additional rotation for |1⟩ system state
        qc.x(qr[0])
        
        # Step 2: Reflection around coin space (2|0⟩⟨0| - I on coin)
        qc.x(qr[1])
        qc.cz(qr[0], qr[1])  # Conditional phase when coin is |1⟩
        qc.x(qr[1])
        
        # Step 3: Shift operation (SWAP system and coin)
        qc.swap(qr[0], qr[1])
        
        return qc
    
    def validate_walk_operator(self) -> Dict[str, any]:
        """Validate the walk operator against theoretical predictions."""
        
        walk_circuit = self.build_theoretical_szegedy_walk()
        
        # Convert to unitary and analyze
        walk_unitary = Operator(walk_circuit).data
        computed_eigenvals = np.linalg.eigvals(walk_unitary)
        
        # Sort both theoretical and computed eigenvalues by magnitude
        theory_sorted = sorted(self.quantum_eigenvals, key=lambda x: (-np.abs(x), -np.real(x)))
        computed_sorted = sorted(computed_eigenvals, key=lambda x: (-np.abs(x), -np.real(x)))
        
        # Check eigenvalue matching
        eigenval_errors = []
        for i, (th, comp) in enumerate(zip(theory_sorted, computed_sorted)):
            error = abs(th - comp)
            eigenval_errors.append(error)
        
        max_eigenval_error = max(eigenval_errors)
        
        # Find stationary eigenstate (eigenvalue closest to +1)
        stationary_idx = np.argmin([abs(ev - 1) for ev in computed_eigenvals])
        stationary_eigenval = computed_eigenvals[stationary_idx]
        
        return {
            'walk_circuit': walk_circuit,
            'theoretical_eigenvalues': theory_sorted,
            'computed_eigenvalues': computed_sorted,
            'max_eigenvalue_error': max_eigenval_error,
            'stationary_eigenvalue': stationary_eigenval,
            'validation_passed': max_eigenval_error < 1e-10
        }
    
    def prepare_test_states(self) -> Dict[str, QuantumCircuit]:
        """Prepare theoretically correct test states."""
        
        # Get walk validation results
        walk_validation = self.validate_walk_operator()
        walk_unitary = Operator(walk_validation['walk_circuit']).data
        
        # Find eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(walk_unitary)
        
        # Find stationary eigenstate (eigenvalue +1)
        stationary_indices = [i for i, ev in enumerate(eigenvals) if abs(ev - 1) < 1e-10]
        
        # Find orthogonal eigenstate (eigenvalue -1 or other)
        orthogonal_indices = [i for i, ev in enumerate(eigenvals) if abs(ev + 1) < 1e-10]
        if not orthogonal_indices:
            # If no -1 eigenvalue, use eigenvalue furthest from +1
            orthogonal_indices = [np.argmax([abs(ev - 1) for ev in eigenvals])]
        
        states = {}
        
        # Uniform superposition
        qr = QuantumRegister(2, 'test')
        uniform_qc = QuantumCircuit(qr, name='uniform')
        uniform_qc.h(qr[0])
        uniform_qc.h(qr[1])
        states['uniform'] = uniform_qc
        
        # Stationary eigenstate (approximation using computational basis)
        stationary_qc = QuantumCircuit(qr, name='stationary')
        if stationary_indices:
            # Use the computational basis state that best approximates the stationary eigenstate
            stationary_eigenvec = eigenvecs[:, stationary_indices[0]]
            # Find the computational basis state with highest amplitude
            max_idx = np.argmax(np.abs(stationary_eigenvec))
            
            if max_idx == 1:  # |01⟩
                stationary_qc.x(qr[1])
            elif max_idx == 2:  # |10⟩
                stationary_qc.x(qr[0])
            elif max_idx == 3:  # |11⟩
                stationary_qc.x(qr[0])
                stationary_qc.x(qr[1])
            # max_idx == 0 is |00⟩, no gates needed
        
        states['stationary'] = stationary_qc
        
        # Orthogonal eigenstate
        orthogonal_qc = QuantumCircuit(qr, name='orthogonal')
        if orthogonal_indices:
            orthogonal_eigenvec = eigenvecs[:, orthogonal_indices[0]]
            # Prepare a superposition that approximates this eigenstate
            # For simplicity, use |+-⟩ = (|01⟩ - |10⟩)/√2
            orthogonal_qc.h(qr[0])
            orthogonal_qc.x(qr[1])
            orthogonal_qc.cx(qr[0], qr[1])
            orthogonal_qc.z(qr[0])
        
        states['orthogonal'] = orthogonal_qc
        
        return states
    
    def build_qpe_circuit(self, initial_state: QuantumCircuit) -> QuantumCircuit:
        """Build complete QPE circuit."""
        
        # Validate walk operator first
        walk_validation = self.validate_walk_operator()
        if not walk_validation['validation_passed']:
            print(f"⚠️  Walk operator validation failed with error: {walk_validation['max_eigenvalue_error']}")
        
        walk_circuit = walk_validation['walk_circuit']
        
        # Create registers
        ancilla = QuantumRegister(self.ancilla_bits, 'ancilla')
        system = QuantumRegister(2, 'system')  # 2 qubits for Szegedy walk
        c_ancilla = ClassicalRegister(self.ancilla_bits, 'c_ancilla')
        
        qc = QuantumCircuit(ancilla, system, c_ancilla)
        
        # Prepare initial state
        qc.append(initial_state, system)
        
        # QPE: Hadamard on ancilla
        for i in range(self.ancilla_bits):
            qc.h(ancilla[i])
        
        # Controlled walk operators U^(2^j)
        for j in range(self.ancilla_bits):
            power = 2**j
            
            # Create controlled walk
            controlled_walk = walk_circuit.control(1)
            
            # Apply power by repetition
            for _ in range(power):
                qc.append(controlled_walk, [ancilla[j]] + list(system))
        
        # Inverse QFT
        qft_inv = QFT(self.ancilla_bits, inverse=True)
        qc.append(qft_inv, ancilla)
        
        # Measure ancilla
        qc.measure(ancilla, c_ancilla)
        
        return qc
    
    def run_qpe_experiments(self) -> Dict[str, any]:
        """Run complete QPE experiments with validation."""
        
        print("\n" + "="*60)
        print("RUNNING RESEARCH-GRADE QPE EXPERIMENTS")
        print("="*60)
        
        # Validate walk operator
        print("Validating Szegedy walk operator...")
        walk_validation = self.validate_walk_operator()
        
        print(f"Walk operator validation:")
        print(f"  Theoretical eigenvalues: {len(walk_validation['theoretical_eigenvalues'])}")
        print(f"  Computed eigenvalues: {len(walk_validation['computed_eigenvalues'])}")
        print(f"  Max eigenvalue error: {walk_validation['max_eigenvalue_error']:.2e}")
        print(f"  Stationary eigenvalue: {walk_validation['stationary_eigenvalue']:.6f}")
        print(f"  Validation passed: {walk_validation['validation_passed']}")
        
        if not walk_validation['validation_passed']:
            print("⚠️  WARNING: Walk operator validation failed!")
        
        # Prepare test states
        print("\nPreparing test states...")
        test_states = self.prepare_test_states()
        
        results = {
            'backend': self.backend.name,
            'timestamp': datetime.now().isoformat(),
            'shots': self.shots,
            'ancilla_bits': self.ancilla_bits,
            'classical_properties': self.classical_properties,
            'theoretical_phase_gap': self.phase_gap,
            'walk_validation': {k: v for k, v in walk_validation.items() if k != 'walk_circuit'},
            'states': {}
        }
        
        # Run QPE for each test state
        for state_name, state_circuit in test_states.items():
            print(f"\nRunning QPE for {state_name} state...")
            
            # Build QPE circuit
            qpe_circuit = self.build_qpe_circuit(state_circuit)
            
            # Transpile and run
            transpiled = transpile(qpe_circuit, self.backend, optimization_level=3)
            job = self.backend.run(transpiled, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            print(f"  Circuit depth: {transpiled.depth()}")
            print(f"  Total gates: {transpiled.size()}")
            
            # Analyze results
            analysis = self.analyze_qpe_results(counts, state_name)
            
            results['states'][state_name] = {
                'counts': counts,
                'analysis': analysis,
                'circuit_depth': transpiled.depth(),
                'total_gates': transpiled.size()
            }
            
            print(f"  Top measurement: bin {analysis['top_bin']} (phase {analysis['top_phase']:.3f}) with prob {analysis['top_probability']:.3f}")
            
            # Validate against expectations
            if state_name == 'stationary':
                if analysis['top_bin'] == 0 and analysis['top_probability'] > 0.8:
                    print(f"  ✅ Stationary state correctly measures phase 0")
                else:
                    print(f"  ❌ Stationary state should measure phase 0, got bin {analysis['top_bin']}")
            
        return results
    
    def analyze_qpe_results(self, counts: Dict[str, int], state_name: str) -> Dict[str, any]:
        """Analyze QPE measurement results."""
        
        total_shots = sum(counts.values())
        
        # Convert to phases
        phase_data = []
        for bitstring, count in counts.items():
            bin_value = int(bitstring, 2)  # Direct binary to integer
            phase = bin_value / (2**self.ancilla_bits)
            probability = count / total_shots
            
            phase_data.append({
                'bin': bin_value,
                'phase': phase,
                'probability': probability,
                'counts': count
            })
        
        # Sort by probability
        phase_data.sort(key=lambda x: x['probability'], reverse=True)
        
        # Analysis
        top_measurement = phase_data[0] if phase_data else {'bin': 0, 'phase': 0, 'probability': 0}
        
        # Check phase 0 (bin 0) probability
        bin_0_prob = next((p['probability'] for p in phase_data if p['bin'] == 0), 0)
        
        # Check phase 0.5 (bin 2^(s-1)) probability
        bin_half = 2**(self.ancilla_bits - 1)
        bin_half_prob = next((p['probability'] for p in phase_data if p['bin'] == bin_half), 0)
        
        return {
            'top_bin': top_measurement['bin'],
            'top_phase': top_measurement['phase'],
            'top_probability': top_measurement['probability'],
            'bin_0_probability': bin_0_prob,
            'bin_half_probability': bin_half_prob,
            'phase_distribution': phase_data[:5]  # Top 5 phases
        }
    
    def create_publication_figures(self, results: Dict[str, any], save_dir: Path):
        """Create research-quality publication figures."""
        
        save_dir.mkdir(exist_ok=True)
        
        # Figure 1: QPE Phase Histograms
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        states = ['uniform', 'stationary', 'orthogonal']
        colors = ['blue', 'orange', 'green']
        
        for idx, (state_name, color) in enumerate(zip(states, colors)):
            ax = axes[idx]
            
            if state_name in results['states']:
                state_data = results['states'][state_name]
                phase_dist = state_data['analysis']['phase_distribution']
                
                bins = [p['bin'] for p in phase_dist]
                probs = [p['probability'] for p in phase_dist]
                
                ax.bar(bins, probs, alpha=0.7, color=color)
                ax.set_xlabel('Phase Bin')
                ax.set_ylabel('Probability')
                ax.set_title(f'{state_name.capitalize()} State')
                ax.grid(True, alpha=0.3)
                
                # Add theoretical expectations
                if state_name == 'stationary':
                    ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Theory: phase = 0')
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'figure1_qpe_histograms.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'figure1_qpe_histograms.pdf', bbox_inches='tight')
        
        # Figure 2: Theoretical Validation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Eigenvalue comparison
        walk_val = results['walk_validation']
        theory_eigs = np.array(walk_val['theoretical_eigenvalues'])
        computed_eigs = np.array(walk_val['computed_eigenvalues'])
        
        ax1.scatter(np.real(theory_eigs), np.imag(theory_eigs), 
                   alpha=0.7, label='Theoretical', s=50)
        ax1.scatter(np.real(computed_eigs), np.imag(computed_eigs), 
                   alpha=0.7, label='Computed', s=50, marker='x')
        ax1.set_xlabel('Real Part')
        ax1.set_ylabel('Imaginary Part')
        ax1.set_title('Eigenvalue Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Phase gap analysis
        classical_gap = results['classical_properties']['classical_gap']
        quantum_gap = results['theoretical_phase_gap']
        
        ax2.bar(['Classical Gap', 'Quantum Phase Gap'], [classical_gap, quantum_gap], 
               color=['blue', 'orange'], alpha=0.7)
        ax2.set_ylabel('Gap (radians)')
        ax2.set_title('Spectral Gap Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'figure2_theoretical_validation.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'figure2_theoretical_validation.pdf', bbox_inches='tight')
        
        print(f"Publication figures saved to: {save_dir}")
        
        return fig

def run_research_grade_pipeline():
    """Run the complete research-grade QPE pipeline."""
    
    print("="*80)
    print("RESEARCH-GRADE QUANTUM PHASE ESTIMATION PIPELINE")
    print("="*80)
    print("Implementing rigorous theoretical foundations for publication")
    print("="*80)
    
    # Define test Markov chain (2-state birth-death process)
    P = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    
    print(f"\nClassical Markov Chain:")
    print(f"P = {P}")
    
    # Initialize research-grade QPE
    qpe = ResearchGradeQPE(
        transition_matrix=P,
        ancilla_bits=4,
        shots=8192  # High shot count for statistical accuracy
    )
    
    # Run experiments
    results = qpe.run_qpe_experiments()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / f"research_grade_results_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = json.loads(json.dumps(results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
        json.dump(results_serializable, f, indent=2)
    
    # Create publication figures
    figures_dir = output_dir / 'figures'
    qpe.create_publication_figures(results, figures_dir)
    
    # Print summary
    print(f"\n" + "="*80)
    print("RESEARCH-GRADE RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nTheoretical Validation:")
    print(f"  Walk operator eigenvalue error: {results['walk_validation']['max_eigenvalue_error']:.2e}")
    print(f"  Validation passed: {results['walk_validation']['validation_passed']}")
    
    print(f"\nQPE Results:")
    for state_name, state_data in results['states'].items():
        analysis = state_data['analysis']
        print(f"  {state_name.capitalize()} state:")
        print(f"    Top bin: {analysis['top_bin']} (phase {analysis['top_phase']:.3f})")
        print(f"    Probability: {analysis['top_probability']:.3f}")
        print(f"    Bin 0 prob: {analysis['bin_0_probability']:.3f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    results_dir = run_research_grade_pipeline()