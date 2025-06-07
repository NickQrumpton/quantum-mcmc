"""Quantum Phase Estimation (QPE) routines for quantum MCMC sampling.

This module implements quantum phase estimation algorithms for extracting
eigenphases of unitary operators, particularly focused on applications to
Szegedy quantum walk operators. QPE is a fundamental subroutine in quantum
algorithms that enables eigenvalue estimation with polynomial speedup.

The implementation provides flexible QPE circuits compatible with Qiskit,
supporting arbitrary unitary operators and custom initial state preparation.
Statistical analysis tools are included for interpreting measurement results
and extracting phase information with confidence intervals.

References:
    Nielsen, M. A., & Chuang, I. L. (2010). Quantum computation and quantum
    information. Cambridge University Press.
    
    Kitaev, A. Y. (1995). Quantum measurements and the Abelian stabilizer
    problem. arXiv preprint quant-ph/9511026.

Author: Nicholas Zhao  
Affiliation: Imperial College London  
Contact: nz422@ic.ac.uk
"""

from typing import Callable, Optional, Dict, List, Tuple, Union
import numpy as np
from collections import Counter
import warnings

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit.circuit import ControlledGate
from qiskit.providers import Backend
# Optional import for AerSimulator
try:
    from qiskit_aer import AerSimulator
    HAS_AER = True
except ImportError:
    AerSimulator = None
    HAS_AER = False
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def quantum_phase_estimation(
    unitary: Union[QuantumCircuit, ControlledGate],
    num_ancilla: int,
    state_prep: Optional[Callable[[QuantumCircuit, QuantumRegister], None]] = None,
    backend: str = "qiskit",
    shots: int = 1024,
    initial_state: Optional[QuantumCircuit] = None
) -> Dict[str, any]:
    """Construct and execute a quantum phase estimation circuit.
    
    Implements the standard QPE algorithm to estimate eigenphases of a unitary
    operator U. Given an eigenstate |�� with eigenvalue e^(2�i�), QPE estimates
    � with n-bit precision using n ancilla qubits.
    
    The algorithm works by:
    1. Preparing ancillas in uniform superposition
    2. Applying controlled powers of U: U^(2^k) for k=0,1,...,n-1
    3. Applying inverse QFT to extract phase information
    4. Measuring ancillas to obtain phase estimate
    
    Args:
        unitary: Unitary operator as QuantumCircuit or Gate to estimate phases for
        num_ancilla: Number of ancilla qubits (determines precision)
        state_prep: Optional function to prepare initial eigenstate on target register
        backend: Simulation backend - "qiskit" for AerSimulator, "statevector" for exact
        shots: Number of measurement shots (ignored for statevector)
        initial_state: Optional pre-built initial state circuit (overrides state_prep)
    
    Returns:
        Dictionary containing:
            - 'counts': Measurement histogram {bitstring: count}
            - 'phases': Estimated phases [0, 1) sorted by probability
            - 'probabilities': Corresponding probabilities
            - 'circuit': The QPE QuantumCircuit
            - 'raw_counts': Raw measurement data before processing
    
    Raises:
        ValueError: If inputs are invalid or incompatible
    
    Example:
        >>> # Estimate phases of a T gate
        >>> qc = QuantumCircuit(1)
        >>> qc.t(0)
        >>> results = quantum_phase_estimation(qc, num_ancilla=4, shots=1000)
        >>> print(f"Estimated phase: {results['phases'][0]:.4f}")
        Estimated phase: 0.1250
    """
    # Validate inputs
    if num_ancilla < 1:
        raise ValueError(f"Number of ancilla qubits must be positive, got {num_ancilla}")
    
    if backend not in ["qiskit", "statevector"]:
        raise ValueError(f"Backend '{backend}' not supported. Use 'qiskit' or 'statevector'")
    
    # Determine target register size
    if isinstance(unitary, QuantumCircuit):
        num_target = unitary.num_qubits
    elif hasattr(unitary, 'num_qubits'):
        num_target = unitary.num_qubits
    else:
        raise ValueError("Unitary must be a QuantumCircuit or have num_qubits attribute")
    
    # Build QPE circuit
    qpe_circuit = _build_qpe_circuit(
        unitary, num_ancilla, num_target, state_prep, initial_state
    )
    
    # Execute circuit
    if backend == "statevector":
        results = _run_statevector_simulation(qpe_circuit, num_ancilla)
    else:
        results = _run_qiskit_simulation(qpe_circuit, num_ancilla, shots)
    
    # Process results to extract phases
    phases, probabilities = _extract_phases_from_counts(
        results['counts'], num_ancilla
    )
    
    # Package results
    output = {
        'counts': results['counts'],
        'phases': phases,
        'probabilities': probabilities,
        'circuit': qpe_circuit,
        'raw_counts': results.get('raw_counts', results['counts']),
        'num_ancilla': num_ancilla,
        'backend': backend,
        'shots': shots if backend == "qiskit" else None
    }
    
    return output


def _build_qpe_circuit(
    unitary: Union[QuantumCircuit, ControlledGate],
    num_ancilla: int,
    num_target: int,
    state_prep: Optional[Callable] = None,
    initial_state: Optional[QuantumCircuit] = None
) -> QuantumCircuit:
    """Build the quantum phase estimation circuit.
    
    Constructs the full QPE circuit including state preparation,
    controlled unitary applications, and inverse QFT.
    
    Args:
        unitary: The unitary operator to analyze
        num_ancilla: Number of ancilla qubits
        num_target: Number of target qubits
        state_prep: Optional state preparation function
        initial_state: Optional pre-built initial state circuit
    
    Returns:
        qpe_circuit: Complete QPE QuantumCircuit
    """
    # Create registers
    ancilla = QuantumRegister(num_ancilla, name='ancilla')
    target = QuantumRegister(num_target, name='target')
    c_ancilla = ClassicalRegister(num_ancilla, name='c_ancilla')
    
    # Initialize circuit
    qc = QuantumCircuit(ancilla, target, c_ancilla, name='QPE')
    
    # Prepare initial state on target register
    if initial_state is not None:
        qc.append(initial_state, target[:])
    elif state_prep is not None:
        state_prep(qc, target)
    # else: leave in |0...0� state
    
    # Initialize ancillas in uniform superposition
    for i in range(num_ancilla):
        qc.h(ancilla[i])
    
    # Apply controlled powers of unitary
    # For ancilla qubit j, apply U^(2^j)
    for j in range(num_ancilla):
        # Create controlled version of U^(2^j)
        power = 2 ** j
        controlled_U = _create_controlled_power(unitary, power, j)
        
        # Apply to circuit
        qc.append(controlled_U, [ancilla[j]] + list(target[:]))
    
    # Apply inverse QFT to ancilla register
    qc.append(_inverse_qft(num_ancilla), ancilla[:])
    
    # Measure ancilla qubits
    qc.measure(ancilla, c_ancilla)
    
    return qc


def _create_controlled_power(
    unitary: Union[QuantumCircuit, ControlledGate],
    power: int,
    control_idx: int
) -> ControlledGate:
    """Create controlled version of U^power.
    
    Args:
        unitary: Base unitary operator
        power: Power to raise unitary to
        control_idx: Index for labeling
    
    Returns:
        Controlled gate implementing controlled-U^power
    """
    # Convert to gate if needed
    if isinstance(unitary, QuantumCircuit):
        U_gate = unitary.to_gate(label=f'U')
    else:
        U_gate = unitary
    
    # Create U^power by repeated application
    if power == 1:
        U_power = U_gate
    else:
        # Build circuit for U^power
        qc_power = QuantumCircuit(U_gate.num_qubits, name=f'U^{power}')
        for _ in range(power):
            qc_power.append(U_gate, range(U_gate.num_qubits))
        U_power = qc_power.to_gate()
    
    # Make it controlled
    controlled_U_power = U_power.control(1, label=f'U^{power}')
    
    return controlled_U_power


def _inverse_qft(n: int) -> QuantumCircuit:
    """Build inverse Quantum Fourier Transform circuit.
    
    The inverse QFT transforms from frequency to computational basis,
    extracting phase information encoded in the amplitudes.
    
    Args:
        n: Number of qubits
    
    Returns:
        Circuit implementing n-qubit inverse QFT
    """
    qc = QuantumCircuit(n, name='QFT ')
    
    # Reverse the order of qubits
    for i in range(n // 2):
        qc.swap(i, n - i - 1)
    
    # Apply inverse QFT gates
    for j in range(n):
        # Hadamard on qubit j
        qc.h(j)
        
        # Controlled phase rotations
        for k in range(j + 1, n):
            angle = -2 * np.pi / (2 ** (k - j + 1))
            qc.cp(angle, k, j)
    
    return qc


def _run_qiskit_simulation(
    circuit: QuantumCircuit,
    num_ancilla: int,
    shots: int
) -> Dict[str, any]:
    """Run QPE circuit on Qiskit Aer simulator.
    
    Args:
        circuit: QPE circuit to execute
        num_ancilla: Number of ancilla qubits
        shots: Number of measurement shots
    
    Returns:
        Dictionary with counts and raw data
    """
    # Use Aer simulator if available
    if not HAS_AER:
        raise ImportError("qiskit-aer is required for shot-based simulation. "
                         "Install with: pip install qiskit-aer")
    simulator = AerSimulator()
    
    # Transpile for simulator
    transpiled = transpile(circuit, simulator)
    
    # Run and get counts
    job = simulator.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    return {
        'counts': counts,
        'raw_counts': counts.copy()
    }


def _run_statevector_simulation(
    circuit: QuantumCircuit,
    num_ancilla: int
) -> Dict[str, any]:
    """Run exact statevector simulation of QPE.
    
    Args:
        circuit: QPE circuit to execute
        num_ancilla: Number of ancilla qubits
    
    Returns:
        Dictionary with probability distribution
    """
    # Remove measurements for statevector simulation
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    
    # Get statevector
    sv = Statevector(circuit_no_meas)
    
    # Extract probabilities for ancilla qubits
    probs_dict = sv.probabilities_dict(range(num_ancilla))
    
    # Convert to counts format (with fractional counts)
    total_counts = 10000  # Arbitrary large number for display
    counts = {}
    for bitstring, prob in probs_dict.items():
        if prob > 1e-10:  # Threshold small probabilities
            # Reverse bitstring to match Qiskit convention
            counts[bitstring[::-1]] = int(prob * total_counts)
    
    return {
        'counts': counts,
        'raw_counts': probs_dict
    }


def _extract_phases_from_counts(
    counts: Dict[str, int],
    num_ancilla: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract estimated phases from measurement counts.
    
    Converts measured bit strings to phase estimates in [0, 1).
    
    Args:
        counts: Measurement histogram
        num_ancilla: Number of ancilla bits
    
    Returns:
        phases: Array of unique phases sorted by probability
        probabilities: Corresponding probability estimates
    """
    total_counts = sum(counts.values())
    phase_probs = {}
    
    for bitstring, count in counts.items():
        # Convert bitstring to integer (little-endian in Qiskit)
        # Take only the ancilla bits
        ancilla_bits = bitstring[-num_ancilla:]
        value = int(ancilla_bits, 2)
        
        # Convert to phase in [0, 1)
        phase = value / (2 ** num_ancilla)
        
        # Accumulate probabilities
        if phase not in phase_probs:
            phase_probs[phase] = 0
        phase_probs[phase] += count / total_counts
    
    # Sort by probability (descending)
    sorted_phases = sorted(phase_probs.items(), key=lambda x: x[1], reverse=True)
    
    phases = np.array([p[0] for p in sorted_phases])
    probabilities = np.array([p[1] for p in sorted_phases])
    
    return phases, probabilities


def analyze_qpe_results(results: Dict[str, any]) -> Dict[str, any]:
    """Analyze QPE output to extract eigenphases with error estimates.
    
    Performs statistical analysis on QPE measurement results to:
    1. Identify dominant phases
    2. Estimate measurement uncertainties
    3. Compute confidence intervals
    4. Detect and merge nearby phases
    
    Args:
        results: Output from quantum_phase_estimation()
    
    Returns:
        Dictionary containing:
            - 'dominant_phases': Most likely phase estimates
            - 'uncertainties': Standard errors for each phase
            - 'confidence_intervals': 95% confidence intervals
            - 'merged_phases': Phases after merging nearby values
            - 'phase_quality': Quality metrics for estimates
    
    Example:
        >>> results = quantum_phase_estimation(U, num_ancilla=8)
        >>> analysis = analyze_qpe_results(results)
        >>> for phase, err in zip(analysis['dominant_phases'], 
        ...                       analysis['uncertainties']):
        ...     print(f"Phase: {phase:.4f} � {err:.4f}")
    """
    phases = results['phases']
    probs = results['probabilities']
    num_ancilla = results['num_ancilla']
    
    # Theoretical phase resolution
    resolution = 1.0 / (2 ** num_ancilla)
    
    # Find dominant phases (above threshold)
    threshold = 1.0 / (2 ** (num_ancilla // 2))  # Adaptive threshold
    dominant_mask = probs > threshold
    dominant_phases = phases[dominant_mask]
    dominant_probs = probs[dominant_mask]
    
    # Estimate uncertainties
    if results['backend'] == 'qiskit' and results['shots'] is not None:
        # Shot noise uncertainty
        shots = results['shots']
        uncertainties = np.sqrt(dominant_probs * (1 - dominant_probs) / shots)
        
        # Add discretization uncertainty
        uncertainties = np.sqrt(uncertainties**2 + (resolution/np.sqrt(12))**2)
    else:
        # For statevector, only discretization uncertainty
        uncertainties = np.full_like(dominant_phases, resolution / np.sqrt(12))
    
    # Compute confidence intervals
    z_score = 1.96  # 95% confidence
    confidence_intervals = [
        (phase - z_score * unc, phase + z_score * unc)
        for phase, unc in zip(dominant_phases, uncertainties)
    ]
    
    # Merge nearby phases
    merged_phases, merged_probs = _merge_nearby_phases(
        phases, probs, threshold=2*resolution
    )
    
    # Phase quality metrics
    quality_metrics = _compute_phase_quality(
        results['counts'], num_ancilla, dominant_phases
    )
    
    return {
        'dominant_phases': dominant_phases,
        'dominant_probabilities': dominant_probs,
        'uncertainties': uncertainties,
        'confidence_intervals': confidence_intervals,
        'merged_phases': merged_phases,
        'merged_probabilities': merged_probs,
        'phase_quality': quality_metrics,
        'resolution': resolution,
        'threshold_used': threshold
    }


def _merge_nearby_phases(
    phases: np.ndarray,
    probs: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge phases that are within threshold distance.
    
    Args:
        phases: Array of phase values
        probs: Corresponding probabilities
        threshold: Distance threshold for merging
    
    Returns:
        merged_phases: Consolidated phase values
        merged_probs: Corresponding probabilities
    """
    if len(phases) == 0:
        return phases, probs
    
    # Sort by phase value
    sorted_idx = np.argsort(phases)
    sorted_phases = phases[sorted_idx]
    sorted_probs = probs[sorted_idx]
    
    merged = []
    current_phases = [sorted_phases[0]]
    current_probs = [sorted_probs[0]]
    
    for i in range(1, len(sorted_phases)):
        # Check distance to current group (with wraparound)
        dist = min(
            abs(sorted_phases[i] - current_phases[-1]),
            1 - abs(sorted_phases[i] - current_phases[-1])
        )
        
        if dist < threshold:
            # Add to current group
            current_phases.append(sorted_phases[i])
            current_probs.append(sorted_probs[i])
        else:
            # Start new group
            weighted_phase = np.average(current_phases, weights=current_probs)
            total_prob = sum(current_probs)
            merged.append((weighted_phase % 1.0, total_prob))
            
            current_phases = [sorted_phases[i]]
            current_probs = [sorted_probs[i]]
    
    # Don't forget the last group
    if current_phases:
        weighted_phase = np.average(current_phases, weights=current_probs)
        total_prob = sum(current_probs)
        merged.append((weighted_phase % 1.0, total_prob))
    
    # Unpack and sort by probability
    merged.sort(key=lambda x: x[1], reverse=True)
    merged_phases = np.array([m[0] for m in merged])
    merged_probs = np.array([m[1] for m in merged])
    
    return merged_phases, merged_probs


def _compute_phase_quality(
    counts: Dict[str, int],
    num_ancilla: int,
    dominant_phases: np.ndarray
) -> Dict[str, float]:
    """Compute quality metrics for phase estimates.
    
    Args:
        counts: Measurement counts
        num_ancilla: Number of ancilla qubits
        dominant_phases: Identified dominant phases
    
    Returns:
        Dictionary of quality metrics
    """
    total_counts = sum(counts.values())
    
    # Shannon entropy of distribution
    probs = np.array(list(counts.values())) / total_counts
    probs = probs[probs > 0]  # Remove zeros for log
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = num_ancilla  # Maximum possible entropy
    
    # Concentration around dominant phases
    dominant_count = 0
    resolution = 1.0 / (2 ** num_ancilla)
    
    for bitstring, count in counts.items():
        ancilla_bits = bitstring[-num_ancilla:]
        value = int(ancilla_bits, 2)
        phase = value / (2 ** num_ancilla)
        
        # Check if near any dominant phase
        for dom_phase in dominant_phases:
            dist = min(abs(phase - dom_phase), 1 - abs(phase - dom_phase))
            if dist < 2 * resolution:
                dominant_count += count
                break
    
    concentration = dominant_count / total_counts if total_counts > 0 else 0
    
    return {
        'entropy': entropy,
        'normalized_entropy': entropy / max_entropy,
        'concentration': concentration,
        'effective_phases': 2 ** entropy,  # Effective number of phases
        'num_dominant': len(dominant_phases)
    }


def plot_qpe_histogram(
    results: Dict[str, any],
    analysis: Optional[Dict[str, any]] = None,
    figsize: Tuple[float, float] = (10, 6),
    show_phases: bool = True
) -> Figure:
    """Visualize QPE measurement results as histogram.
    
    Creates a bar chart of measurement outcomes with optional
    phase annotations and analysis overlays.
    
    Args:
        results: Output from quantum_phase_estimation()
        analysis: Optional output from analyze_qpe_results()
        figsize: Figure size in inches
        show_phases: Whether to show phase values on x-axis
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> results = quantum_phase_estimation(U, num_ancilla=6)
        >>> fig = plot_qpe_histogram(results)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    counts = results['counts']
    num_ancilla = results['num_ancilla']
    
    # Sort by count value
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to top outcomes for clarity
    max_outcomes = 20
    if len(sorted_counts) > max_outcomes:
        sorted_counts = sorted_counts[:max_outcomes]
        other_count = sum(count for _, count in counts.items() 
                         if _ not in dict(sorted_counts))
        if other_count > 0:
            sorted_counts.append(('other', other_count))
    
    # Prepare data
    labels = []
    values = []
    colors = []
    
    for bitstring, count in sorted_counts:
        if show_phases and bitstring != 'other':
            # Convert to phase
            ancilla_bits = bitstring[-num_ancilla:]
            value = int(ancilla_bits, 2)
            phase = value / (2 ** num_ancilla)
            labels.append(f'{bitstring}\n(�={phase:.3f})')
        else:
            labels.append(bitstring)
        
        values.append(count)
        
        # Color dominant phases differently
        if analysis and bitstring != 'other':
            ancilla_bits = bitstring[-num_ancilla:]
            value = int(ancilla_bits, 2)
            phase = value / (2 ** num_ancilla)
            
            is_dominant = any(
                abs(phase - dp) < 1/(2**(num_ancilla-1)) 
                for dp in analysis['dominant_phases']
            )
            colors.append('darkblue' if is_dominant else 'lightblue')
        else:
            colors.append('gray' if bitstring == 'other' else 'lightblue')
    
    # Create bar plot
    bars = ax.bar(range(len(labels)), values, color=colors)
    
    # Customize plot
    ax.set_xlabel('Measurement Outcome', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_title(f'QPE Results ({num_ancilla} ancilla qubits)', fontsize=14)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value}', ha='center', va='bottom', fontsize=9)
    
    # Add analysis annotations if provided
    if analysis:
        # Add text box with key results
        textstr = f"Dominant phases: {len(analysis['dominant_phases'])}\n"
        textstr += f"Resolution: {analysis['resolution']:.4f}\n"
        textstr += f"Quality: {analysis['phase_quality']['concentration']:.2%}"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig


def estimate_required_precision(
    target_accuracy: float,
    phase_separation: Optional[float] = None
) -> int:
    """Estimate number of ancilla qubits needed for target accuracy.
    
    Computes the minimum number of ancilla qubits required to achieve
    a desired phase estimation accuracy, optionally considering the
    need to resolve nearby phases.
    
    Args:
        target_accuracy: Desired accuracy for phase estimates
        phase_separation: Minimum separation between phases to resolve
    
    Returns:
        num_ancilla: Required number of ancilla qubits
    
    Example:
        >>> n = estimate_required_precision(0.01)
        >>> print(f"Need {n} ancilla qubits for 1% accuracy")
        Need 7 ancilla qubits for 1% accuracy
    """
    # Basic requirement from accuracy
    n_accuracy = int(np.ceil(np.log2(1 / target_accuracy)))
    
    if phase_separation is not None:
        # Need to resolve phases separated by phase_separation
        n_separation = int(np.ceil(np.log2(2 / phase_separation)))
        return max(n_accuracy, n_separation)
    
    return n_accuracy


def adaptive_qpe(
    unitary: Union[QuantumCircuit, ControlledGate],
    max_ancilla: int = 12,
    confidence_threshold: float = 0.95,
    state_prep: Optional[Callable] = None,
    backend: str = "qiskit",
    shots_per_round: int = 100
) -> Dict[str, any]:
    """Adaptive quantum phase estimation with iterative refinement.
    
    Implements an adaptive strategy that starts with few ancilla qubits
    and iteratively increases precision based on measurement outcomes.
    This can be more efficient when high precision is only needed for
    certain phases.
    
    Args:
        unitary: Unitary operator to analyze
        max_ancilla: Maximum number of ancilla qubits to use
        confidence_threshold: Stop when confidence exceeds this threshold
        state_prep: Optional initial state preparation
        backend: Simulation backend
        shots_per_round: Shots per iteration (for qiskit backend)
    
    Returns:
        Dictionary with final results and convergence history
    
    Example:
        >>> results = adaptive_qpe(U, max_ancilla=10)
        >>> print(f"Converged to {results['final_precision']} bits")
    """
    if isinstance(unitary, QuantumCircuit):
        num_target = unitary.num_qubits
    else:
        num_target = unitary.num_qubits
    
    history = {
        'num_ancilla': [],
        'phases': [],
        'confidences': [],
        'iterations': 0
    }
    
    # Start with coarse estimation
    current_ancilla = min(3, max_ancilla)
    best_phase = None
    best_confidence = 0
    
    while current_ancilla <= max_ancilla and best_confidence < confidence_threshold:
        # Run QPE with current precision
        results = quantum_phase_estimation(
            unitary, current_ancilla, state_prep, 
            backend, shots_per_round
        )
        
        # Analyze results
        analysis = analyze_qpe_results(results)
        
        # Track history
        history['num_ancilla'].append(current_ancilla)
        history['phases'].append(analysis['dominant_phases'])
        history['iterations'] += 1
        
        if len(analysis['dominant_phases']) > 0:
            # Update best estimate
            best_phase = analysis['dominant_phases'][0]
            best_prob = analysis['dominant_probabilities'][0]
            best_confidence = analysis['phase_quality']['concentration']
            
            history['confidences'].append(best_confidence)
            
            # Check convergence
            if best_confidence >= confidence_threshold:
                break
        else:
            history['confidences'].append(0)
        
        # Increase precision
        current_ancilla = min(current_ancilla + 2, max_ancilla)
    
    return {
        'best_phase': best_phase,
        'confidence': best_confidence,
        'final_precision': current_ancilla,
        'history': history,
        'converged': best_confidence >= confidence_threshold,
        'final_results': results,
        'final_analysis': analysis
    }


def qpe_for_quantum_walk(
    walk_operator: QuantumCircuit,
    num_ancilla: int,
    initial_vertex: Optional[int] = None,
    backend: str = "qiskit",
    shots: int = 1024
) -> Dict[str, any]:
    """Specialized QPE for quantum walk operators.
    
    Implements phase estimation specifically for Szegedy quantum walk
    operators, with appropriate initial state preparation for edge space.
    
    Args:
        walk_operator: Quantum walk operator circuit
        num_ancilla: Precision bits
        initial_vertex: Optional starting vertex for walk
        backend: Simulation backend
        shots: Number of shots
    
    Returns:
        QPE results with walk-specific analysis
    
    Example:
        >>> from quantum_mcmc.core.quantum_walk import prepare_walk_operator
        >>> W = prepare_walk_operator(P)
        >>> results = qpe_for_quantum_walk(W, num_ancilla=8)
    """
    # Prepare appropriate initial state for walk
    if initial_vertex is not None:
        # Create superposition over edges from initial vertex
        n_vertices = int(np.sqrt(walk_operator.num_qubits))
        n_qubits = int(np.ceil(np.log2(n_vertices)))
        
        def walk_state_prep(qc, target):
            # Set first register to initial vertex
            if initial_vertex > 0:
                binary = format(initial_vertex, f'0{n_qubits}b')
                for i, bit in enumerate(binary[::-1]):
                    if bit == '1':
                        qc.x(target[i])
            
            # Superposition on second register
            for i in range(n_qubits, 2*n_qubits):
                qc.h(target[i])
    else:
        walk_state_prep = None
    
    # Run standard QPE
    results = quantum_phase_estimation(
        walk_operator, num_ancilla, walk_state_prep, backend, shots
    )
    
    # Add walk-specific analysis
    analysis = analyze_qpe_results(results)
    
    # Convert phases to mixing time estimates
    mixing_times = []
    for phase in analysis['dominant_phases']:
        if phase > 0:
            # Approximate mixing time from phase gap
            gap = min(phase, 1 - phase)  # Distance to 0 or 1
            if gap > 0:
                t_mix = int(1 / gap)
                mixing_times.append(t_mix)
    
    results['walk_analysis'] = {
        'estimated_mixing_times': mixing_times,
        'spectral_gap_estimate': min(analysis['dominant_phases'][analysis['dominant_phases'] > 0])
        if any(analysis['dominant_phases'] > 0) else None
    }
    
    return results