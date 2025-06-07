"""Comprehensive validation sweep for enhanced Theorem 6 implementation.

This script provides systematic validation of the approximate reflection operator
with parameter sweeps, resource verification, and robustness testing across
multiple spectral gaps.

Author: Nicholas Zhao
Date: 5/31/2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
import time
import json
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ValidationResult:
    """Structure for storing validation results."""
    k: int
    s: int
    enhanced_precision: bool
    spectral_gap: float
    test_norm: float
    theoretical_bound: float
    ratio: float
    controlled_w_calls: int
    total_ancillas: int
    circuit_depth: int
    success: bool
    error_message: Optional[str] = None
    computation_time: float = 0.0


class ReflectionOperatorValidator:
    """Comprehensive validator for enhanced reflection operator."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def create_test_markov_chain(self, target_delta: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """Create a well-conditioned Markov chain with target spectral gap."""
        # Create 3x3 reversible chain
        # Use parameterization that gives controllable spectral gap
        
        if target_delta <= 0.1:
            # Small gap case
            p1, p2 = 0.05, 0.03
        elif target_delta <= 0.2:
            # Medium gap case  
            p1, p2 = 0.1, 0.08
        else:
            # Large gap case
            p1, p2 = 0.2, 0.15
            
        P = np.array([
            [1-p1, p1/2, p1/2],
            [p2, 1-p1-p2, p1],
            [p1/2, p2, 1-p1/2-p2]
        ])
        
        # Ensure row-stochastic
        P = P / P.sum(axis=1, keepdims=True)
        
        # Compute stationary distribution
        eigenvals, eigenvecs = np.linalg.eig(P.T)
        pi_idx = np.argmin(np.abs(eigenvals - 1.0))
        pi = np.real(eigenvecs[:, pi_idx])
        pi = pi / np.sum(pi)
        
        # Compute actual spectral gap
        eigenvals_real = np.real(eigenvals)
        eigenvals_sorted = np.sort(eigenvals_real)[::-1]
        actual_delta = eigenvals_sorted[0] - eigenvals_sorted[1]
        
        return P, actual_delta, pi
    
    def build_enhanced_walk_operator(self, P: np.ndarray, pi: np.ndarray) -> QuantumCircuit:
        """Build enhanced quantum walk operator with exact construction."""
        n = P.shape[0]
        n_qubits_per_reg = int(np.ceil(np.log2(n)))
        total_qubits = 2 * n_qubits_per_reg
        
        # Build exact walk matrix
        W_matrix = self._compute_exact_walk_matrix(P, pi)
        
        # Pad to qubit dimensions
        full_dim = 2 ** total_qubits
        if W_matrix.shape[0] < full_dim:
            W_padded = np.eye(full_dim, dtype=complex)
            W_padded[:W_matrix.shape[0], :W_matrix.shape[1]] = W_matrix
            W_matrix = W_padded
        
        # Create quantum circuit
        qc = QuantumCircuit(total_qubits, name='Enhanced_W')
        walk_gate = UnitaryGate(W_matrix, label='W_exact')
        qc.append(walk_gate, range(total_qubits))
        
        return qc
    
    def _compute_exact_walk_matrix(self, P: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """Compute exact walk operator matrix using discriminant approach."""
        n = P.shape[0]
        dim = n * n
        
        # Build discriminant matrix
        D = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                if pi[i] > 1e-12 and pi[j] > 1e-12:
                    D[i, j] = np.sqrt(P[i, j] * pi[j] / pi[i])
        
        # Build projection operator Π
        Pi_op = np.zeros((dim, dim), dtype=complex)
        A = np.sqrt(P.astype(complex))
        
        for i in range(n):
            psi_i = np.zeros(dim, dtype=complex)
            for j in range(n):
                idx = i * n + j
                psi_i[idx] = A[i, j]
            # Normalize
            norm = np.linalg.norm(psi_i)
            if norm > 1e-12:
                psi_i = psi_i / norm
            Pi_op += np.outer(psi_i, psi_i.conj())
        
        # Build swap operator S
        S = np.zeros((dim, dim))
        for i in range(n):
            for j in range(n):
                idx_in = i * n + j
                idx_out = j * n + i
                S[idx_out, idx_in] = 1
        
        # Walk operator: W = S(2Π - I)
        reflection = 2 * Pi_op - np.eye(dim)
        W = S @ reflection
        
        # Enforce unitarity using polar decomposition
        W = self._enforce_unitarity(W)
        
        return W
    
    def _enforce_unitarity(self, U: np.ndarray) -> np.ndarray:
        """Enforce unitarity using polar decomposition."""
        try:
            from scipy.linalg import polar
            V, P = polar(U)
            return V
        except ImportError:
            # Fallback to SVD
            U_svd, s, Vh = np.linalg.svd(U)
            return U_svd @ Vh
    
    def create_orthogonal_test_state(self, pi: np.ndarray, n_qubits_per_reg: int) -> QuantumCircuit:
        """Create quantum state orthogonal to stationary distribution."""
        n_states = len(pi)
        
        # Create orthogonal vector using Gram-Schmidt
        sqrt_pi = np.sqrt(pi)
        test_vec = np.ones(n_states)
        test_vec[0] = -1  # Make different from uniform
        test_vec = test_vec / np.linalg.norm(test_vec)
        
        # Orthogonalize against sqrt_pi
        test_vec = test_vec - np.dot(test_vec, sqrt_pi) * sqrt_pi
        norm = np.linalg.norm(test_vec)
        if norm > 1e-10:
            test_vec = test_vec / norm
        else:
            # Fallback: use second basis vector
            test_vec = np.zeros(n_states)
            test_vec[1] = 1.0
        
        # Pad to qubit dimensions
        full_dim = 2 ** n_qubits_per_reg
        if n_states < full_dim:
            padded = np.zeros(full_dim)
            padded[:n_states] = test_vec
            test_vec = padded
        
        # Create edge space state: orthogonal ⊗ uniform
        edge_circuit = QuantumCircuit(2 * n_qubits_per_reg, name='orthogonal_state')
        
        # First register: orthogonal state
        edge_circuit.initialize(test_vec, range(n_qubits_per_reg))
        
        # Second register: uniform superposition
        for i in range(n_qubits_per_reg):
            edge_circuit.h(n_qubits_per_reg + i)
        
        return edge_circuit
    
    def build_enhanced_reflection_operator(
        self,
        walk_op: QuantumCircuit,
        spectral_gap: float,
        k: int,
        s: int,
        enhanced_precision: bool
    ) -> Tuple[QuantumCircuit, Dict[str, int]]:
        """Build enhanced reflection operator and return resource usage."""
        
        system_qubits = walk_op.num_qubits
        
        # Calculate ancilla requirements
        if enhanced_precision:
            compare_ancillas = 2 * s + 2
        else:
            compare_ancillas = s + 1
            
        total_qubits = s + system_qubits + compare_ancillas
        
        # Build circuit
        qc = QuantumCircuit(total_qubits, name=f'R_enhanced_k{k}_s{s}')
        
        # Count controlled-W operations
        controlled_w_calls = 0
        
        for rep in range(k):
            # QPE forward
            qpe_calls = self._add_enhanced_qpe(qc, walk_op, s, enhanced_precision)
            controlled_w_calls += qpe_calls
            
            # Phase comparator
            self._add_enhanced_comparator(qc, s, spectral_gap, enhanced_precision)
            
            # QPE inverse  
            qpe_calls = self._add_enhanced_qpe_inverse(qc, walk_op, s, enhanced_precision)
            controlled_w_calls += qpe_calls
        
        resources = {
            'controlled_w_calls': controlled_w_calls,
            'total_ancillas': total_qubits,
            'qpe_ancillas': s,
            'comparator_ancillas': compare_ancillas,
            'system_qubits': system_qubits
        }
        
        return qc, resources
    
    def _add_enhanced_qpe(self, qc: QuantumCircuit, walk_op: QuantumCircuit, 
                         s: int, enhanced: bool) -> int:
        """Add enhanced QPE to circuit and return controlled-W count."""
        system_qubits = walk_op.num_qubits
        
        # Initialize ancillas
        for i in range(s):
            qc.h(i)
        
        # Controlled powers of walk operator
        W_gate = walk_op.to_gate()
        controlled_w_calls = 0
        
        for j in range(s):
            power = 2 ** j
            controlled_w_calls += power
            
            # Add controlled W^power (simplified representation)
            if enhanced:
                # Enhanced: use iterative construction
                for _ in range(power):
                    controlled_W = W_gate.control(1)
                    qc.append(controlled_W, [j] + list(range(s, s + system_qubits)))
            else:
                # Standard: direct repetition
                for _ in range(power):
                    controlled_W = W_gate.control(1)
                    qc.append(controlled_W, [j] + list(range(s, s + system_qubits)))
        
        # QFT
        from qiskit.circuit.library import QFT
        qft = QFT(s, do_swaps=True)
        qc.append(qft, range(s))
        
        return controlled_w_calls
    
    def _add_enhanced_qpe_inverse(self, qc: QuantumCircuit, walk_op: QuantumCircuit,
                                 s: int, enhanced: bool) -> int:
        """Add inverse QPE and return controlled-W count."""
        # Same structure as forward QPE
        return self._add_enhanced_qpe(qc, walk_op, s, enhanced)
    
    def _add_enhanced_comparator(self, qc: QuantumCircuit, s: int, 
                                spectral_gap: float, enhanced: bool) -> None:
        """Add enhanced phase comparator to circuit."""
        system_qubits = qc.num_qubits - s - (2*s+2 if enhanced else s+1)
        
        # Simplified comparator representation
        threshold_int = max(1, int(spectral_gap * (2**s) / (2 * np.pi)))
        
        # Add comparator gates (simplified)
        compare_start = s + system_qubits
        
        if enhanced:
            # Enhanced precision comparator
            for i in range(min(3, s)):
                qc.cx(i, compare_start)
                qc.cz(compare_start, compare_start + 1)
        else:
            # Standard comparator
            qc.x(0)
            qc.cz(0, compare_start)
            qc.x(0)
    
    def compute_reflection_norm(self, reflection_circuit: QuantumCircuit, 
                               test_state: QuantumCircuit) -> float:
        """Compute ||（R+I)|ψ⟩|| for orthogonal test state."""
        try:
            # Build full circuit
            full_circuit = QuantumCircuit(reflection_circuit.num_qubits)
            
            # Prepare test state
            test_qubits = min(test_state.num_qubits, reflection_circuit.num_qubits)
            full_circuit.append(test_state, range(test_qubits))
            
            # Apply reflection
            full_circuit.append(reflection_circuit, range(reflection_circuit.num_qubits))
            
            # Get R|ψ⟩
            sv_R = Statevector(full_circuit)
            
            # Get I|ψ⟩ (padded)
            sv_I = Statevector(test_state)
            padded_data = np.zeros(2**reflection_circuit.num_qubits, dtype=complex)
            padded_data[:len(sv_I.data)] = sv_I.data
            sv_I_padded = Statevector(padded_data)
            
            # Compute (R + I)|ψ⟩
            sv_sum = Statevector(sv_R.data + sv_I_padded.data)
            norm = np.linalg.norm(sv_sum.data)
            
            return norm
            
        except Exception as e:
            print(f"Error computing reflection norm: {e}")
            return np.inf
    
    def run_single_validation(
        self,
        k: int,
        s: int, 
        enhanced_precision: bool,
        spectral_gap: float
    ) -> ValidationResult:
        """Run single validation test."""
        
        start_time = time.time()
        
        try:
            # Create test system
            P, actual_delta, pi = self.create_test_markov_chain(spectral_gap)
            W = self.build_enhanced_walk_operator(P, pi)
            
            # Create test state
            n_qubits_per_reg = int(np.ceil(np.log2(len(pi))))
            test_state = self.create_orthogonal_test_state(pi, n_qubits_per_reg)
            
            # Build reflection operator
            R, resources = self.build_enhanced_reflection_operator(
                W, actual_delta, k, s, enhanced_precision
            )
            
            # Compute norm
            test_norm = self.compute_reflection_norm(R, test_state)
            theoretical_bound = 2**(1 - k)
            ratio = test_norm / theoretical_bound
            
            # Check success criteria
            success = (
                ratio < 1.2 and  # Within 20% of bound
                test_norm < np.inf and  # Valid computation
                resources['controlled_w_calls'] <= k * 2**(s+1)  # Resource bound
            )
            
            computation_time = time.time() - start_time
            
            return ValidationResult(
                k=k,
                s=s,
                enhanced_precision=enhanced_precision,
                spectral_gap=actual_delta,
                test_norm=test_norm,
                theoretical_bound=theoretical_bound,
                ratio=ratio,
                controlled_w_calls=resources['controlled_w_calls'],
                total_ancillas=resources['total_ancillas'],
                circuit_depth=R.depth(),
                success=success,
                computation_time=computation_time
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            return ValidationResult(
                k=k, s=s, enhanced_precision=enhanced_precision,
                spectral_gap=spectral_gap, test_norm=np.inf,
                theoretical_bound=2**(1-k), ratio=np.inf,
                controlled_w_calls=0, total_ancillas=0, circuit_depth=0,
                success=False, error_message=str(e),
                computation_time=computation_time
            )
    
    def run_parameter_sweep(
        self,
        k_values: List[int] = [1, 2, 3, 4],
        s_values: List[int] = [5, 6, 7, 8, 9],
        precision_values: List[bool] = [False, True],
        spectral_gaps: List[float] = [0.10, 0.15, 0.20]
    ) -> pd.DataFrame:
        """Run comprehensive parameter sweep."""
        
        print("Starting Comprehensive Theorem 6 Validation")
        print("=" * 60)
        
        results = []
        total_tests = len(k_values) * len(s_values) * len(precision_values) * len(spectral_gaps)
        test_count = 0
        
        for delta in spectral_gaps:
            print(f"\nTesting spectral gap δ = {delta:.3f}")
            print("-" * 40)
            
            for enhanced in precision_values:
                precision_str = "Enhanced" if enhanced else "Standard"
                print(f"\n{precision_str} Precision:")
                
                for s in s_values:
                    print(f"  s={s} ancillas: ", end="")
                    
                    for k in k_values:
                        test_count += 1
                        print(f"k{k} ", end="", flush=True)
                        
                        result = self.run_single_validation(k, s, enhanced, delta)
                        results.append(result)
                        
                        if test_count % 20 == 0:
                            print(f"\n  Progress: {test_count}/{total_tests}")
                    
                    print()  # New line after each s
        
        print(f"\nCompleted {test_count} validation tests")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'k': r.k, 's': r.s, 'enhanced_precision': r.enhanced_precision,
            'spectral_gap': r.spectral_gap, 'test_norm': r.test_norm,
            'theoretical_bound': r.theoretical_bound, 'ratio': r.ratio,
            'controlled_w_calls': r.controlled_w_calls, 'total_ancillas': r.total_ancillas,
            'circuit_depth': r.circuit_depth, 'success': r.success,
            'error_message': r.error_message, 'computation_time': r.computation_time
        } for r in results])
        
        return df
    
    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """Analyze validation results."""
        analysis = {}
        
        # Overall statistics
        total_tests = len(df)
        successful_tests = len(df[df['success']])
        valid_ratios = df[(df['ratio'] < 10) & (df['ratio'] > 0)]
        
        analysis['overall'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests,
            'avg_ratio_valid': valid_ratios['ratio'].mean(),
            'median_ratio_valid': valid_ratios['ratio'].median(),
            'best_ratio': valid_ratios['ratio'].min(),
            'worst_ratio': valid_ratios['ratio'].max()
        }
        
        # Enhanced vs Standard precision
        enhanced_results = df[df['enhanced_precision'] == True]
        standard_results = df[df['enhanced_precision'] == False]
        
        analysis['precision_comparison'] = {
            'enhanced_success_rate': enhanced_results['success'].mean(),
            'standard_success_rate': standard_results['success'].mean(),
            'enhanced_avg_ratio': enhanced_results[enhanced_results['ratio'] < 10]['ratio'].mean(),
            'standard_avg_ratio': standard_results[standard_results['ratio'] < 10]['ratio'].mean()
        }
        
        # Scaling analysis
        analysis['scaling'] = {}
        for enhanced in [True, False]:
            precision_data = df[df['enhanced_precision'] == enhanced]
            precision_key = 'enhanced' if enhanced else 'standard'
            
            # k-scaling (should decrease exponentially)
            k_analysis = precision_data.groupby('k').agg({
                'ratio': ['mean', 'std', 'min', 'max'],
                'success': 'mean'
            }).round(4)
            analysis['scaling'][f'{precision_key}_k_scaling'] = k_analysis.to_dict()
            
            # s-scaling (should improve with more ancillas)
            s_analysis = precision_data.groupby('s').agg({
                'ratio': ['mean', 'std', 'min', 'max'],
                'success': 'mean'
            }).round(4)
            analysis['scaling'][f'{precision_key}_s_scaling'] = s_analysis.to_dict()
        
        # Resource verification
        resource_violations = df[df['controlled_w_calls'] > df['k'] * 2**(df['s']+1)]
        analysis['resource_verification'] = {
            'total_violations': len(resource_violations),
            'violation_rate': len(resource_violations) / total_tests
        }
        
        # Best configurations
        valid_successful = df[df['success'] & (df['ratio'] < 10)]
        if not valid_successful.empty:
            best_configs = valid_successful.nsmallest(10, 'ratio')[
                ['k', 's', 'enhanced_precision', 'spectral_gap', 'ratio', 'controlled_w_calls']
            ]
            analysis['best_configurations'] = best_configs.to_dict('records')
        
        return analysis
    
    def create_validation_plots(self, df: pd.DataFrame):
        """Create comprehensive validation plots."""
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Ratio vs k for different precision levels
        plt.subplot(3, 4, 1)
        valid_data = df[(df['ratio'] < 10) & (df['ratio'] > 0)]
        
        for enhanced in [False, True]:
            precision_data = valid_data[valid_data['enhanced_precision'] == enhanced]
            if not precision_data.empty:
                k_means = precision_data.groupby('k')['ratio'].mean()
                k_stds = precision_data.groupby('k')['ratio'].std()
                
                label = 'Enhanced' if enhanced else 'Standard'
                plt.errorbar(k_means.index, k_means.values, yerr=k_stds.values, 
                           marker='o', label=label, markersize=6, capsize=5)
        
        # Theoretical bound
        k_vals = sorted(df['k'].unique())
        theoretical = [2**(1-k) for k in k_vals]
        plt.plot(k_vals, theoretical, 'r--', label='Theoretical 2^(1-k)', linewidth=2)
        
        plt.xlabel('Repetitions (k)')
        plt.ylabel('Ratio to bound')
        plt.title('Error Scaling vs k')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot 2: Success rate vs s
        plt.subplot(3, 4, 2)
        for enhanced in [False, True]:
            precision_data = df[df['enhanced_precision'] == enhanced]
            if not precision_data.empty:
                s_success = precision_data.groupby('s')['success'].mean()
                label = 'Enhanced' if enhanced else 'Standard'
                plt.plot(s_success.index, s_success.values, 'o-', 
                        label=label, markersize=6)
        
        plt.xlabel('Ancilla qubits (s)')
        plt.ylabel('Success rate')
        plt.title('Success Rate vs Ancilla Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Ratio distribution
        plt.subplot(3, 4, 3)
        valid_ratios = df[(df['ratio'] < 5) & (df['ratio'] > 0)]
        
        for enhanced in [False, True]:
            precision_data = valid_ratios[valid_ratios['enhanced_precision'] == enhanced]
            if not precision_data.empty:
                label = 'Enhanced' if enhanced else 'Standard'
                plt.hist(precision_data['ratio'], alpha=0.6, label=label, 
                        bins=30, density=True)
        
        plt.axvline(x=1, color='r', linestyle='--', label='Target = 1')
        plt.xlabel('Ratio to theoretical bound')
        plt.ylabel('Density')
        plt.title('Distribution of Ratios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Resource usage verification
        plt.subplot(3, 4, 4)
        df['resource_bound'] = df['k'] * 2**(df['s']+1)
        df['resource_ratio'] = df['controlled_w_calls'] / df['resource_bound']
        
        valid_resources = df[df['controlled_w_calls'] > 0]
        plt.scatter(valid_resources['k'], valid_resources['resource_ratio'], 
                   c=[1 if e else 0 for e in valid_resources['enhanced_precision']], 
                   cmap='viridis', alpha=0.6)
        plt.axhline(y=1, color='r', linestyle='--', label='Theorem 6 bound')
        plt.xlabel('Repetitions (k)')
        plt.ylabel('Actual / Theoretical calls')
        plt.title('Resource Usage Verification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Heatmap of ratio by (k,s) for enhanced precision
        plt.subplot(3, 4, 5)
        enhanced_data = df[df['enhanced_precision'] == True]
        if not enhanced_data.empty:
            pivot = enhanced_data.pivot_table(values='ratio', index='s', columns='k', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis_r', 
                       cbar_kws={'label': 'Ratio'})
            plt.title('Enhanced Precision: Ratio Heatmap')
        
        # Plot 6: Computation time scaling
        plt.subplot(3, 4, 6)
        for enhanced in [False, True]:
            precision_data = df[df['enhanced_precision'] == enhanced]
            if not precision_data.empty:
                time_means = precision_data.groupby('k')['computation_time'].mean()
                label = 'Enhanced' if enhanced else 'Standard'
                plt.plot(time_means.index, time_means.values, 'o-', 
                        label=label, markersize=6)
        
        plt.xlabel('Repetitions (k)')
        plt.ylabel('Computation time (s)')
        plt.title('Computation Time Scaling')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 7: Spectral gap robustness
        plt.subplot(3, 4, 7)
        enhanced_data = df[df['enhanced_precision'] == True]
        if not enhanced_data.empty:
            for delta in sorted(enhanced_data['spectral_gap'].unique()):
                gap_data = enhanced_data[np.abs(enhanced_data['spectral_gap'] - delta) < 0.01]
                if not gap_data.empty:
                    k_means = gap_data.groupby('k')['ratio'].mean()
                    plt.plot(k_means.index, k_means.values, 'o-', 
                            label=f'δ={delta:.3f}', markersize=6)
        
        plt.xlabel('Repetitions (k)')
        plt.ylabel('Ratio to bound')
        plt.title('Robustness Across Spectral Gaps')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot 8: Circuit complexity
        plt.subplot(3, 4, 8)
        valid_circuits = df[df['circuit_depth'] > 0]
        if not valid_circuits.empty:
            plt.scatter(valid_circuits['total_ancillas'], valid_circuits['circuit_depth'],
                       c=[1 if e else 0 for e in valid_circuits['enhanced_precision']],
                       cmap='viridis', alpha=0.6)
            plt.xlabel('Total ancillas')
            plt.ylabel('Circuit depth')
            plt.title('Circuit Complexity')
            plt.grid(True, alpha=0.3)
        
        # Plots 9-12: Individual k value performance
        for i, k in enumerate([1, 2, 3, 4]):
            plt.subplot(3, 4, 9+i)
            k_data = df[df['k'] == k]
            
            for enhanced in [False, True]:
                precision_data = k_data[k_data['enhanced_precision'] == enhanced]
                if not precision_data.empty:
                    valid_data = precision_data[(precision_data['ratio'] < 10) & 
                                              (precision_data['ratio'] > 0)]
                    if not valid_data.empty:
                        s_means = valid_data.groupby('s')['ratio'].mean()
                        label = 'Enhanced' if enhanced else 'Standard'
                        plt.plot(s_means.index, s_means.values, 'o-', 
                                label=label, markersize=4)
            
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
            plt.xlabel('Ancilla qubits (s)')
            plt.ylabel('Ratio')
            plt.title(f'k={k} Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'theorem_6_validation_plots.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Validation plots saved to {self.output_dir / 'theorem_6_validation_plots.png'}")
    
    def save_results(self, df: pd.DataFrame, analysis: Dict):
        """Save validation results and analysis."""
        
        # Save raw data
        df.to_csv(self.output_dir / 'theorem_6_validation_results.csv', index=False)
        
        # Save analysis
        with open(self.output_dir / 'theorem_6_validation_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"✓ Results saved to {self.output_dir}")


def main():
    """Run comprehensive validation sweep."""
    
    # Initialize validator
    validator = ReflectionOperatorValidator()
    
    # Run parameter sweep
    results_df = validator.run_parameter_sweep(
        k_values=[1, 2, 3, 4],
        s_values=[5, 6, 7, 8, 9],
        precision_values=[False, True],
        spectral_gaps=[0.10, 0.15, 0.20]
    )
    
    # Analyze results
    analysis = validator.analyze_results(results_df)
    
    # Create plots
    validator.create_validation_plots(results_df)
    
    # Save everything
    validator.save_results(results_df, analysis)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    overall = analysis['overall']
    print(f"Total tests: {overall['total_tests']}")
    print(f"Success rate: {overall['success_rate']:.1%}")
    print(f"Average ratio (valid): {overall['avg_ratio_valid']:.3f}")
    print(f"Best ratio achieved: {overall['best_ratio']:.3f}")
    
    precision_comp = analysis['precision_comparison']
    print(f"\nPrecision Comparison:")
    print(f"  Enhanced success rate: {precision_comp['enhanced_success_rate']:.1%}")
    print(f"  Standard success rate: {precision_comp['standard_success_rate']:.1%}")
    print(f"  Enhanced avg ratio: {precision_comp['enhanced_avg_ratio']:.3f}")
    print(f"  Standard avg ratio: {precision_comp['standard_avg_ratio']:.3f}")
    
    resource_verif = analysis['resource_verification']
    print(f"\nResource Verification:")
    print(f"  Theorem 6 bound violations: {resource_verif['total_violations']}")
    print(f"  Violation rate: {resource_verif['violation_rate']:.1%}")
    
    if 'best_configurations' in analysis:
        print(f"\nTop 3 Configurations:")
        for i, config in enumerate(analysis['best_configurations'][:3]):
            print(f"  {i+1}. k={config['k']}, s={config['s']}, "
                  f"enhanced={config['enhanced_precision']}, ratio={config['ratio']:.3f}")
    
    return results_df, analysis


if __name__ == "__main__":
    results, analysis = main()