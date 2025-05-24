"""Unit tests for quantum phase estimation module.

This module provides comprehensive unit tests for the quantum phase estimation
implementation in quantum_mcmc.core.phase_estimation. Tests cover:
- Basic QPE functionality with known unitaries
- Statistical analysis of measurement results
- Edge cases and error handling
- Integration with quantum walk operators

The tests verify correctness on simple test cases with known eigenvalues,
robustness to various parameter settings, and proper integration with Qiskit.

Author: [Your Name]
Date: 2025-01-23
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate, RZGate, TGate, SGate
from qiskit.quantum_info import Operator

from quantum_mcmc.core.phase_estimation import (
    quantum_phase_estimation,
    analyze_qpe_results,
    estimate_required_precision,
    adaptive_qpe,
    qpe_for_quantum_walk,
    _build_qpe_circuit,
    _inverse_qft,
    _extract_phases_from_counts,
    _merge_nearby_phases,
    _compute_phase_quality,
    plot_qpe_histogram
)
from quantum_mcmc.core.quantum_walk import prepare_walk_operator
from quantum_mcmc.classical.markov_chain import build_two_state_chain, build_three_state_symmetric


class TestQuantumPhaseEstimation:
    """Test suite for quantum_phase_estimation function."""
    
    def test_diagonal_unitary_phase_extraction(self):
        """Test QPE on diagonal unitary with known phases."""
        # Create diagonal unitary with phase e^(2¿i/4) = i
        qc = QuantumCircuit(1)
        qc.rz(np.pi/2, 0)  # RZ(∏) = diag(1, e^(i∏))
        
        # Run QPE with sufficient precision
        results = quantum_phase_estimation(
            qc, num_ancilla=4, backend="statevector"
        )
        
        # Check output structure
        assert isinstance(results, dict), "QPE should return a dictionary"
        assert 'phases' in results, "Results should contain phases"
        assert 'probabilities' in results, "Results should contain probabilities"
        assert 'circuit' in results, "Results should contain the QPE circuit"
        
        # Check extracted phases
        # RZ(¿/2) has eigenvalues 1 and e^(i¿/2) = i, so phases are 0 and 0.25
        expected_phases = [0.0, 0.25]
        
        # Find closest matches
        for expected in expected_phases:
            closest = min(results['phases'], key=lambda x: abs(x - expected))
            assert abs(closest - expected) < 0.1, \
                f"Expected phase {expected}, got {closest}"
    
    def test_t_gate_phase_estimation(self):
        """Test QPE on T gate with known phase ¿/4."""
        qc = QuantumCircuit(1)
        qc.t(0)  # T gate has eigenvalues 1 and e^(i¿/4)
        
        # Need initial state preparation to get non-zero phase
        def prep_plus(qc, target):
            qc.h(target[0])  # |+È state
        
        results = quantum_phase_estimation(
            qc, num_ancilla=5, state_prep=prep_plus, backend="statevector"
        )
        
        # T gate phase should be 1/8 = 0.125
        expected_phase = 0.125
        
        # Check that we find this phase
        found_phase = False
        for phase, prob in zip(results['phases'], results['probabilities']):
            if abs(phase - expected_phase) < 0.05 and prob > 0.1:
                found_phase = True
                break
        
        assert found_phase, \
            f"T gate phase {expected_phase} not found in {results['phases'][:5]}"
    
    def test_s_gate_phase_estimation(self):
        """Test QPE on S gate with phase ¿/2."""
        qc = QuantumCircuit(1)
        qc.s(0)  # S gate has eigenvalues 1 and i
        
        def prep_plus(qc, target):
            qc.h(target[0])
        
        results = quantum_phase_estimation(
            qc, num_ancilla=4, state_prep=prep_plus, shots=2048
        )
        
        # S gate phase should be 1/4 = 0.25
        expected_phase = 0.25
        
        # With shots, check dominant phase
        assert len(results['phases']) > 0, "Should find at least one phase"
        dominant_phase = results['phases'][0]  # Already sorted by probability
        
        assert abs(dominant_phase - expected_phase) < 0.1, \
            f"S gate dominant phase should be near {expected_phase}, got {dominant_phase}"
    
    def test_precision_scaling(self):
        """Test that more ancilla qubits give better precision."""
        # Use RZ gate with irrational phase
        angle = np.pi / 3  # Phase will be 1/6 H 0.1667
        qc = QuantumCircuit(1)
        qc.rz(angle, 0)
        
        def prep_one(qc, target):
            qc.x(target[0])  # |1È state to get non-zero phase
        
        # Test with different precisions
        precisions = []
        for n_ancilla in [3, 5, 7]:
            results = quantum_phase_estimation(
                qc, num_ancilla=n_ancilla, state_prep=prep_one, 
                backend="statevector"
            )
            
            # Find phase closest to expected
            expected = angle / (2 * np.pi)
            closest = min(results['phases'], key=lambda x: abs(x - expected))
            error = abs(closest - expected)
            precisions.append(error)
        
        # Error should decrease with more ancillas
        assert precisions[1] <= precisions[0], \
            "5 ancillas should be more precise than 3"
        assert precisions[2] <= precisions[1], \
            "7 ancillas should be more precise than 5"
    
    def test_multi_qubit_unitary(self):
        """Test QPE on multi-qubit unitary."""
        # Create 2-qubit controlled rotation
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rz(np.pi/4, 1)
        qc.cx(0, 1)
        
        # This creates entangled eigenstates
        results = quantum_phase_estimation(
            qc, num_ancilla=4, backend="qiskit", shots=1024
        )
        
        # Check circuit was created with correct size
        qpe_circuit = results['circuit']
        assert qpe_circuit.num_qubits == 6, \
            f"QPE circuit should have 6 qubits (4 ancilla + 2 target), got {qpe_circuit.num_qubits}"
        
        # Should get measurement results
        assert sum(results['counts'].values()) > 0, \
            "Should have non-zero measurement counts"
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        qc = QuantumCircuit(1)
        qc.x(0)
        
        # Test invalid ancilla count
        with pytest.raises(ValueError, match="positive"):
            quantum_phase_estimation(qc, num_ancilla=0)
        
        # Test invalid backend
        with pytest.raises(ValueError, match="Backend.*not supported"):
            quantum_phase_estimation(qc, num_ancilla=3, backend="invalid")
        
        # Test non-quantum circuit input
        with pytest.raises(ValueError, match="QuantumCircuit"):
            quantum_phase_estimation("not a circuit", num_ancilla=3)
    
    def test_initial_state_options(self):
        """Test different initial state preparation methods."""
        # Create phase gate
        qc = QuantumCircuit(1)
        qc.p(np.pi/3, 0)  # Phase gate
        
        # Method 1: state_prep function
        def prep_plus(qc, target):
            qc.h(target[0])
        
        results1 = quantum_phase_estimation(
            qc, num_ancilla=4, state_prep=prep_plus, backend="statevector"
        )
        
        # Method 2: pre-built circuit
        init_circuit = QuantumCircuit(1)
        init_circuit.h(0)
        
        results2 = quantum_phase_estimation(
            qc, num_ancilla=4, initial_state=init_circuit, backend="statevector"
        )
        
        # Results should be similar
        assert results1['phases'][0] == results2['phases'][0], \
            "Different initial state methods should give same result"
    
    def test_backend_consistency(self):
        """Test that qiskit and statevector backends give consistent results."""
        qc = QuantumCircuit(1)
        qc.s(0)  # S gate
        
        def prep_plus(qc, target):
            qc.h(target[0])
        
        # Run with both backends
        results_sv = quantum_phase_estimation(
            qc, num_ancilla=4, state_prep=prep_plus, backend="statevector"
        )
        
        results_qk = quantum_phase_estimation(
            qc, num_ancilla=4, state_prep=prep_plus, backend="qiskit", shots=8192
        )
        
        # Dominant phases should match
        assert abs(results_sv['phases'][0] - results_qk['phases'][0]) < 0.1, \
            f"Backends should give similar results: {results_sv['phases'][0]} vs {results_qk['phases'][0]}"


class TestAnalyzeQPEResults:
    """Test suite for analyze_qpe_results function."""
    
    def test_basic_analysis_structure(self):
        """Test that analysis returns expected structure."""
        # Create mock results
        results = {
            'phases': np.array([0.0, 0.25, 0.5]),
            'probabilities': np.array([0.5, 0.3, 0.2]),
            'counts': {'00': 500, '01': 300, '10': 200},
            'num_ancilla': 4,
            'backend': 'qiskit',
            'shots': 1000
        }
        
        analysis = analyze_qpe_results(results)
        
        # Check output structure
        assert isinstance(analysis, dict), "Analysis should return dictionary"
        assert 'dominant_phases' in analysis, "Should identify dominant phases"
        assert 'uncertainties' in analysis, "Should compute uncertainties"
        assert 'confidence_intervals' in analysis, "Should compute confidence intervals"
        assert 'merged_phases' in analysis, "Should merge nearby phases"
        assert 'phase_quality' in analysis, "Should compute quality metrics"
    
    def test_dominant_phase_identification(self):
        """Test identification of dominant phases."""
        # Create results with clear dominant phase
        results = {
            'phases': np.array([0.125, 0.126, 0.127, 0.5, 0.9]),
            'probabilities': np.array([0.3, 0.3, 0.2, 0.1, 0.1]),
            'counts': {'001': 600, '010': 200, '111': 200},
            'num_ancilla': 3,
            'backend': 'qiskit',
            'shots': 1000
        }
        
        analysis = analyze_qpe_results(results)
        
        # Should identify high-probability phases as dominant
        assert len(analysis['dominant_phases']) >= 2, \
            "Should identify at least 2 dominant phases"
        assert 0.125 in analysis['dominant_phases'] or \
               any(abs(p - 0.125) < 0.01 for p in analysis['dominant_phases']), \
            "Should identify phase near 0.125 as dominant"
    
    def test_uncertainty_estimation(self):
        """Test uncertainty calculation for different backends."""
        # Test with shot noise
        results_shots = {
            'phases': np.array([0.25]),
            'probabilities': np.array([1.0]),
            'counts': {'01': 1000},
            'num_ancilla': 2,
            'backend': 'qiskit',
            'shots': 1000
        }
        
        analysis_shots = analyze_qpe_results(results_shots)
        
        # Test with statevector (no shot noise)
        results_sv = {
            'phases': np.array([0.25]),
            'probabilities': np.array([1.0]),
            'counts': {'01': 10000},
            'num_ancilla': 2,
            'backend': 'statevector',
            'shots': None
        }
        
        analysis_sv = analyze_qpe_results(results_sv)
        
        # Shot noise should give larger uncertainty
        assert analysis_shots['uncertainties'][0] > analysis_sv['uncertainties'][0], \
            "Shot noise should increase uncertainty"
    
    def test_phase_merging(self):
        """Test merging of nearby phases."""
        # Create results with nearby phases that should merge
        results = {
            'phases': np.array([0.124, 0.126, 0.250, 0.875, 0.877]),
            'probabilities': np.array([0.2, 0.3, 0.2, 0.15, 0.15]),
            'counts': {'001': 500, '010': 300, '100': 200},
            'num_ancilla': 3,
            'backend': 'statevector',
            'shots': None
        }
        
        analysis = analyze_qpe_results(results)
        
        # Nearby phases should be merged
        merged = analysis['merged_phases']
        assert len(merged) < len(results['phases']), \
            "Merging should reduce number of distinct phases"
        
        # Check that 0.124 and 0.126 were merged
        merged_125 = any(0.12 < p < 0.13 for p in merged)
        assert merged_125, "Phases near 0.125 should be merged"
    
    def test_phase_quality_metrics(self):
        """Test computation of quality metrics."""
        # High quality: concentrated distribution
        results_good = {
            'phases': np.array([0.25]),
            'probabilities': np.array([1.0]),
            'counts': {'0100': 1000},
            'num_ancilla': 4,
            'backend': 'qiskit',
            'shots': 1000
        }
        
        # Low quality: uniform distribution
        results_poor = {
            'phases': np.array([0.0, 0.25, 0.5, 0.75]),
            'probabilities': np.array([0.25, 0.25, 0.25, 0.25]),
            'counts': {'0000': 250, '0100': 250, '1000': 250, '1100': 250},
            'num_ancilla': 4,
            'backend': 'qiskit',
            'shots': 1000
        }
        
        analysis_good = analyze_qpe_results(results_good)
        analysis_poor = analyze_qpe_results(results_poor)
        
        # Good results should have higher concentration
        assert analysis_good['phase_quality']['concentration'] > \
               analysis_poor['phase_quality']['concentration'], \
            "Concentrated distribution should have higher quality"
        
        # Poor results should have higher entropy
        assert analysis_good['phase_quality']['entropy'] < \
               analysis_poor['phase_quality']['entropy'], \
            "Uniform distribution should have higher entropy"
    
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        results = {
            'phases': np.array([0.125, 0.5]),
            'probabilities': np.array([0.7, 0.3]),
            'counts': {'001': 700, '100': 300},
            'num_ancilla': 3,
            'backend': 'qiskit',
            'shots': 1000
        }
        
        analysis = analyze_qpe_results(results)
        
        # Check confidence intervals
        for i, (phase, ci) in enumerate(zip(analysis['dominant_phases'], 
                                           analysis['confidence_intervals'])):
            lower, upper = ci
            assert lower <= phase <= upper, \
                f"Phase {phase} should be within confidence interval {ci}"
            assert upper - lower > 0, \
                "Confidence interval should have positive width"


class TestEstimateRequiredPrecision:
    """Test suite for estimate_required_precision function."""
    
    def test_basic_accuracy_requirement(self):
        """Test precision estimation for target accuracy."""
        # 1% accuracy
        n1 = estimate_required_precision(0.01)
        assert n1 >= 7, "1% accuracy should require at least 7 ancilla qubits"
        
        # 0.1% accuracy
        n2 = estimate_required_precision(0.001)
        assert n2 >= 10, "0.1% accuracy should require at least 10 ancilla qubits"
        
        # Higher accuracy needs more qubits
        assert n2 > n1, "Better accuracy should require more qubits"
    
    def test_phase_separation_requirement(self):
        """Test precision for resolving nearby phases."""
        # Need to resolve phases 0.1 apart
        n = estimate_required_precision(0.05, phase_separation=0.1)
        
        # Should be determined by phase separation
        n_sep = int(np.ceil(np.log2(2 / 0.1)))
        assert n >= n_sep, \
            f"Should need at least {n_sep} qubits to resolve 0.1 separation"
    
    def test_combined_requirements(self):
        """Test when both accuracy and separation matter."""
        # Case 1: Accuracy dominates
        n1 = estimate_required_precision(0.001, phase_separation=0.5)
        n_acc = estimate_required_precision(0.001)
        assert n1 == n_acc, "Accuracy should dominate when separation is easy"
        
        # Case 2: Separation dominates
        n2 = estimate_required_precision(0.1, phase_separation=0.01)
        n_sep = int(np.ceil(np.log2(2 / 0.01)))
        assert n2 >= n_sep, "Separation should dominate when accuracy is easy"


class TestAdaptiveQPE:
    """Test suite for adaptive_qpe function."""
    
    def test_adaptive_convergence(self):
        """Test that adaptive QPE converges to correct phase."""
        # Simple unitary with known phase
        qc = QuantumCircuit(1)
        qc.s(0)  # Phase = 0.25
        
        def prep_plus(qc, target):
            qc.h(target[0])
        
        results = adaptive_qpe(
            qc, max_ancilla=8, confidence_threshold=0.8,
            state_prep=prep_plus, backend="statevector"
        )
        
        # Check output structure
        assert 'best_phase' in results, "Should return best phase estimate"
        assert 'confidence' in results, "Should return confidence level"
        assert 'history' in results, "Should return convergence history"
        assert 'converged' in results, "Should indicate convergence status"
        
        # Check convergence
        assert results['converged'], "Should converge for simple case"
        assert abs(results['best_phase'] - 0.25) < 0.1, \
            f"Should converge to phase near 0.25, got {results['best_phase']}"
    
    def test_adaptive_history_tracking(self):
        """Test that adaptive QPE tracks history correctly."""
        qc = QuantumCircuit(1)
        qc.t(0)
        
        def prep_plus(qc, target):
            qc.h(target[0])
        
        results = adaptive_qpe(
            qc, max_ancilla=10, confidence_threshold=0.9,
            state_prep=prep_plus, shots_per_round=200
        )
        
        history = results['history']
        
        # Check history structure
        assert len(history['num_ancilla']) == history['iterations'], \
            "Should track ancilla count for each iteration"
        assert len(history['phases']) == history['iterations'], \
            "Should track phases for each iteration"
        assert len(history['confidences']) == history['iterations'], \
            "Should track confidence for each iteration"
        
        # Ancilla count should increase
        for i in range(1, len(history['num_ancilla'])):
            assert history['num_ancilla'][i] >= history['num_ancilla'][i-1], \
                "Ancilla count should not decrease"
    
    def test_adaptive_early_stopping(self):
        """Test that adaptive QPE stops early when confident."""
        # Easy case with clear phase
        qc = QuantumCircuit(1)
        qc.z(0)  # Phase = 0.5
        
        def prep_plus(qc, target):
            qc.h(target[0])
        
        results = adaptive_qpe(
            qc, max_ancilla=12, confidence_threshold=0.8,
            state_prep=prep_plus, backend="statevector"
        )
        
        # Should stop before max_ancilla
        assert results['final_precision'] < 12, \
            "Should stop early for easy case"
        assert results['converged'], \
            "Should converge for Z gate"


class TestQPEForQuantumWalk:
    """Test suite for qpe_for_quantum_walk function."""
    
    def test_walk_operator_qpe(self):
        """Test QPE on actual quantum walk operator."""
        # Create simple 2-state Markov chain
        P = build_two_state_chain(0.3, 0.4)
        W_circuit = prepare_walk_operator(P, backend="qiskit")
        
        # Run QPE on walk operator
        results = qpe_for_quantum_walk(
            W_circuit, num_ancilla=4, backend="statevector"
        )
        
        # Check output structure
        assert 'walk_analysis' in results, \
            "Should include walk-specific analysis"
        assert 'estimated_mixing_times' in results['walk_analysis'], \
            "Should estimate mixing times"
        
        # Walk operator should have eigenvalue 1
        assert any(abs(p) < 0.1 for p in results['phases']), \
            "Walk operator should have eigenvalue 1 (phase 0)"
    
    def test_walk_qpe_with_initial_vertex(self):
        """Test QPE with specific initial vertex."""
        P = build_three_state_symmetric(0.25)
        W_circuit = prepare_walk_operator(P, backend="qiskit")
        
        # Start from vertex 0
        results = qpe_for_quantum_walk(
            W_circuit, num_ancilla=5, initial_vertex=0, shots=1024
        )
        
        # Should get valid results
        assert len(results['phases']) > 0, \
            "Should extract phases with initial vertex"
        assert results['circuit'].num_qubits > W_circuit.num_qubits, \
            "QPE circuit should have ancilla qubits"
    
    def test_spectral_gap_estimation(self):
        """Test spectral gap estimation from QPE."""
        # Chain with known spectral gap
        P = build_two_state_chain(0.2, 0.3)
        W_circuit = prepare_walk_operator(P, backend="qiskit")
        
        results = qpe_for_quantum_walk(
            W_circuit, num_ancilla=6, backend="statevector"
        )
        
        # Check spectral gap estimate
        if results['walk_analysis']['spectral_gap_estimate'] is not None:
            gap = results['walk_analysis']['spectral_gap_estimate']
            assert 0 < gap < 0.5, \
                f"Spectral gap should be in (0, 0.5), got {gap}"


class TestInternalHelpers:
    """Test suite for internal helper functions."""
    
    def test_build_qpe_circuit_structure(self):
        """Test QPE circuit construction."""
        # Simple unitary
        qc = QuantumCircuit(1)
        qc.x(0)
        
        # Build QPE circuit
        qpe_circuit = _build_qpe_circuit(qc, num_ancilla=3, num_target=1)
        
        # Check structure
        assert qpe_circuit.num_qubits == 4, \
            "QPE circuit should have 3 ancilla + 1 target qubits"
        assert qpe_circuit.num_clbits == 3, \
            "Should have classical bits for ancilla measurements"
        
        # Check for key components
        op_names = [instruction.operation.name for instruction in qpe_circuit]
        assert 'h' in op_names, "Should have Hadamard gates"
        assert 'measure' in op_names, "Should have measurements"
    
    def test_inverse_qft_unitarity(self):
        """Test that inverse QFT is unitary."""
        for n in [2, 3, 4]:
            qft_inv = _inverse_qft(n)
            
            # Convert to operator and check unitarity
            op = Operator(qft_inv)
            U = op.data
            
            # Check U U = I
            product = U.conj().T @ U
            assert np.allclose(product, np.eye(2**n)), \
                f"Inverse QFT for {n} qubits should be unitary"
    
    def test_extract_phases_from_counts(self):
        """Test phase extraction from measurement counts."""
        # Mock counts for 3-bit measurement
        counts = {
            '000': 400,  # Phase 0
            '001': 100,  # Phase 1/8
            '010': 300,  # Phase 2/8 = 1/4
            '100': 200   # Phase 4/8 = 1/2
        }
        
        phases, probs = _extract_phases_from_counts(counts, num_ancilla=3)
        
        # Check extraction
        assert len(phases) == 4, "Should extract 4 distinct phases"
        assert abs(phases[0] - 0.0) < 0.01, "Should extract phase 0"
        assert abs(phases[1] - 0.25) < 0.01, "Should extract phase 1/4"
        
        # Check probabilities sum to 1
        assert abs(sum(probs) - 1.0) < 0.001, \
            "Probabilities should sum to 1"
        
        # Check sorting by probability
        assert probs[0] >= probs[1] >= probs[2] >= probs[3], \
            "Phases should be sorted by probability"
    
    def test_merge_nearby_phases(self):
        """Test phase merging algorithm."""
        # Phases with some close together
        phases = np.array([0.12, 0.13, 0.25, 0.87, 0.88, 0.89])
        probs = np.array([0.1, 0.2, 0.3, 0.15, 0.15, 0.1])
        
        # Merge with threshold
        merged_phases, merged_probs = _merge_nearby_phases(phases, probs, threshold=0.02)
        
        # Should merge nearby phases
        assert len(merged_phases) < len(phases), \
            "Should reduce number of phases by merging"
        
        # Total probability should be preserved
        assert abs(sum(merged_probs) - sum(probs)) < 0.001, \
            "Total probability should be preserved"
        
        # Check specific merges
        # 0.12 and 0.13 should merge to ~0.127 (weighted average)
        assert any(0.125 < p < 0.135 for p in merged_phases), \
            "Should have merged phase near 0.127"
    
    def test_phase_wraparound_merging(self):
        """Test merging phases near 0/1 boundary."""
        # Phases near wraparound
        phases = np.array([0.01, 0.02, 0.98, 0.99])
        probs = np.array([0.2, 0.3, 0.3, 0.2])
        
        merged_phases, merged_probs = _merge_nearby_phases(phases, probs, threshold=0.05)
        
        # All should merge into one group
        assert len(merged_phases) == 1, \
            "Phases near 0/1 boundary should merge"
        assert abs(sum(merged_probs) - 1.0) < 0.001, \
            "Should preserve total probability"
    
    def test_compute_phase_quality(self):
        """Test phase quality metrics computation."""
        # Concentrated distribution
        counts_good = {'0010': 900, '0011': 50, '1000': 50}
        quality_good = _compute_phase_quality(counts_good, 4, np.array([0.125]))
        
        # Uniform distribution
        counts_poor = {f'{i:04b}': 62 for i in range(16)}  # 16 * 62 H 1000
        quality_poor = _compute_phase_quality(counts_poor, 4, np.array([]))
        
        # Good distribution should have lower entropy
        assert quality_good['entropy'] < quality_poor['entropy'], \
            "Concentrated distribution should have lower entropy"
        
        # Good distribution should have higher concentration
        assert quality_good['concentration'] > quality_poor['concentration'], \
            "Concentrated distribution should have higher concentration"


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_minimal_ancilla(self):
        """Test QPE with minimal number of ancilla qubits."""
        qc = QuantumCircuit(1)
        qc.z(0)  # Phase gate
        
        # Just 1 ancilla qubit
        results = quantum_phase_estimation(qc, num_ancilla=1, backend="statevector")
        
        # Should still work but with poor precision
        assert len(results['phases']) <= 2, \
            "1 ancilla can distinguish at most 2 phases"
        assert results['phases'][0] in [0.0, 0.5], \
            "1 ancilla should give phase 0 or 0.5"
    
    def test_identity_operator(self):
        """Test QPE on identity operator."""
        qc = QuantumCircuit(2)
        # Empty circuit = identity
        
        results = quantum_phase_estimation(qc, num_ancilla=3, backend="statevector")
        
        # Identity has all eigenvalues = 1 (phase 0)
        assert len(results['phases']) == 1, \
            "Identity should have single phase"
        assert abs(results['phases'][0]) < 0.01, \
            "Identity should have phase 0"
    
    def test_highly_entangled_unitary(self):
        """Test QPE on highly entangled unitary."""
        # Create maximally entangling unitary
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(np.pi/3, 1)
        qc.cx(0, 1)
        qc.h(0)
        
        # Should handle without errors
        results = quantum_phase_estimation(
            qc, num_ancilla=4, backend="qiskit", shots=2048
        )
        
        assert 'phases' in results, "Should extract phases from entangled unitary"
        assert len(results['counts']) > 0, "Should get measurement results"
    
    def test_zero_shots_error_handling(self):
        """Test handling of edge case parameters."""
        qc = QuantumCircuit(1)
        qc.x(0)
        
        # Zero shots should work with statevector
        results = quantum_phase_estimation(
            qc, num_ancilla=3, backend="statevector", shots=0
        )
        assert results['shots'] is None, \
            "Statevector backend should ignore shots parameter"
    
    def test_large_precision_request(self):
        """Test QPE with large precision request."""
        qc = QuantumCircuit(1)
        qc.t(0)
        
        # Large ancilla count
        results = quantum_phase_estimation(
            qc, num_ancilla=10, backend="statevector"
        )
        
        # Should handle large precision
        assert results['num_ancilla'] == 10, \
            "Should handle 10 ancilla qubits"
        
        # Resolution should be very fine
        resolution = 1.0 / (2 ** 10)
        assert resolution < 0.001, \
            "10 qubits should give < 0.1% resolution"
    
    def test_degenerate_phases(self):
        """Test QPE with degenerate eigenvalues."""
        # Create unitary with repeated eigenvalue
        # Using controlled-Z which has eigenvalues {1, 1, 1, -1}
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        
        results = quantum_phase_estimation(
            qc, num_ancilla=4, backend="statevector"
        )
        
        # Should find phases 0 and 0.5
        unique_phases = []
        for phase in results['phases']:
            if not any(abs(phase - p) < 0.01 for p in unique_phases):
                unique_phases.append(phase)
        
        assert len(unique_phases) <= 2, \
            "CZ gate should have at most 2 distinct phases"