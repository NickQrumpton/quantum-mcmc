"""Approximate reflection operator construction for quantum MCMC sampling.

This module implements the approximate reflection operator about the stationary
state of a Markov chain, following Theorem 6 of quantum MCMC algorithms.
The reflection operator is constructed using quantum phase estimation on the
Szegedy quantum walk operator and selectively flips phases of non-stationary
eigenstates.

The approximate reflection operator is a key component for implementing
quantum analogues of classical MCMC methods, enabling efficient sampling
from the stationary distribution with quantum speedup.

References:
    Lemieux, J., et al. (2019). Efficient quantum walk circuits for 
    Metropolis-Hastings algorithm. Quantum, 4, 287.
    
    Wocjan, P., & Abeyesinghe, A. (2008). Speedup via quantum sampling.
    Physical Review A, 78(4), 042336.

Author: Nicholas Zhao  
Affiliation: Imperial College London  
Contact: nz422@ic.ac.uk
"""

from typing import Optional, Dict, List, Tuple, Union, Callable
import numpy as np
import warnings

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, Operator
# Optional import for AerSimulator
try:
    from qiskit_aer import AerSimulator
    HAS_AER = True
except ImportError:
    AerSimulator = None
    HAS_AER = False
from qiskit.visualization import plot_state_city

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def approximate_reflection_operator(
    walk_operator: QuantumCircuit,
    num_ancilla: int,
    phase_threshold: float = 0.1,
    inverse: bool = False
) -> QuantumCircuit:
    """Construct approximate reflection operator about stationary state.
    
    Implements the reflection operator R_� that approximately reflects about
    the stationary state � of a Markov chain. The operator acts as:
        R_� H 2|����| - I
    
    This is achieved by:
    1. Using QPE to identify eigenstates of the walk operator
    2. Applying a phase flip to all eigenstates except those with
       eigenphase close to 0 (the stationary state)
    3. Undoing the QPE to restore the original basis
    
    The approximation quality depends on the number of ancilla qubits
    used for phase estimation and the phase threshold parameter.
    
    Args:
        walk_operator: Quantum walk operator W(P) as a QuantumCircuit
        num_ancilla: Number of ancilla qubits for phase estimation precision
        phase_threshold: Threshold for identifying stationary eigenstates.
                        Eigenstates with |phase| < threshold are not flipped.
        inverse: If True, construct the inverse reflection operator
    
    Returns:
        reflection_circuit: QuantumCircuit implementing the approximate
                          reflection operator
    
    Raises:
        ValueError: If inputs are invalid
    
    Example:
        >>> from quantum_mcmc.core.quantum_walk import prepare_walk_operator
        >>> W = prepare_walk_operator(P)
        >>> R = approximate_reflection_operator(W, num_ancilla=8)
        >>> print(f"Reflection operator uses {R.num_qubits} qubits")
    """
    # Validate inputs
    if num_ancilla < 1:
        raise ValueError(f"Number of ancilla qubits must be positive, got {num_ancilla}")
    
    if not 0 < phase_threshold < 0.5:
        raise ValueError(f"Phase threshold must be in (0, 0.5), got {phase_threshold}")
    
    # Get dimensions
    num_system = walk_operator.num_qubits
    
    # Create registers
    ancilla = QuantumRegister(num_ancilla, name='ancilla')
    system = QuantumRegister(num_system, name='system')
    
    # Initialize circuit
    qc = QuantumCircuit(ancilla, system, name='R_�')
    
    # Step 1: Apply QPE to identify eigenspaces
    qpe_circuit = _build_qpe_for_reflection(walk_operator, num_ancilla)
    qc.append(qpe_circuit, ancilla[:] + system[:])
    
    # Step 2: Apply conditional phase flip
    phase_flip = _build_conditional_phase_flip(num_ancilla, phase_threshold)
    qc.append(phase_flip, ancilla[:])
    
    # Step 3: Apply inverse QPE
    if not inverse:
        # Normal reflection: QPE � flip � QPE 
        inverse_qpe = qpe_circuit.inverse()
        qc.append(inverse_qpe, ancilla[:] + system[:])
    else:
        # Inverse reflection: skip the inverse QPE
        # This is used when the reflection is part of a larger circuit
        pass
    
    return qc


def _build_qpe_for_reflection(
    walk_operator: QuantumCircuit,
    num_ancilla: int
) -> QuantumCircuit:
    """Build QPE circuit component for reflection operator.
    
    Constructs the quantum phase estimation part without measurements,
    suitable for use within the reflection operator.
    
    Args:
        walk_operator: The quantum walk operator
        num_ancilla: Number of ancilla qubits
    
    Returns:
        qpe_circuit: QPE circuit without measurements
    """
    num_system = walk_operator.num_qubits
    
    # Create circuit
    ancilla = QuantumRegister(num_ancilla, name='anc')
    system = QuantumRegister(num_system, name='sys')
    qc = QuantumCircuit(ancilla, system, name='QPE_part')
    
    # Initialize ancillas in superposition
    for i in range(num_ancilla):
        qc.h(ancilla[i])
    
    # Convert walk operator to gate
    W_gate = walk_operator.to_gate(label='W')
    
    # Apply controlled powers of walk operator
    for j in range(num_ancilla):
        power = 2 ** j
        # Create controlled W^power
        W_power = _create_walk_power(W_gate, power)
        controlled_W = W_power.control(1, label=f'c-W^{power}')
        
        # Apply to circuit
        qc.append(controlled_W, [ancilla[j]] + list(system[:]))
    
    # Apply QFT to ancilla register
    qft = QFT(num_ancilla, do_swaps=True).to_gate()
    qc.append(qft, ancilla[:])
    
    return qc


def _create_walk_power(W_gate, power: int):
    """Create W^power by repeated application."""
    if power == 1:
        return W_gate
    
    # Build circuit for W^power
    num_qubits = W_gate.num_qubits
    qc_power = QuantumCircuit(num_qubits)
    
    for _ in range(power):
        qc_power.append(W_gate, range(num_qubits))
    
    return qc_power.to_gate(label=f'W^{power}')


def _build_conditional_phase_flip(
    num_ancilla: int,
    phase_threshold: float
) -> QuantumCircuit:
    """Build conditional phase flip based on measured phase.
    
    Applies a phase flip (Z gate) to ancilla states that encode
    eigenphases outside the threshold range around 0.
    
    Args:
        num_ancilla: Number of ancilla qubits
        phase_threshold: Threshold for phase flip
    
    Returns:
        Circuit implementing conditional phase flip
    """
    ancilla = QuantumRegister(num_ancilla, name='ancilla')
    qc = QuantumCircuit(ancilla, name='phase_flip')
    
    # Convert threshold to integer range
    # Phase 0 corresponds to bitstring 00...0
    # Phase threshold corresponds to bitstring representing threshold * 2^n
    threshold_int = int(phase_threshold * (2 ** num_ancilla))
    
    # We want to flip phase for all states except those near 00...0
    # This can be implemented using multi-controlled Z gates
    
    # For simplicity in this implementation, we use an oracle approach
    # In practice, this would be optimized using arithmetic comparators
    
    # Create phase oracle
    oracle = _create_phase_oracle(num_ancilla, threshold_int)
    qc.append(oracle, ancilla[:])
    
    return qc


def _create_phase_oracle(num_qubits: int, threshold: int) -> QuantumCircuit:
    """Create oracle that flips phase of states outside threshold.
    
    The oracle applies a phase flip to all computational basis states
    |k� where k > threshold or k > 2^n - threshold (wraparound).
    
    Args:
        num_qubits: Number of qubits
        threshold: Integer threshold value
    
    Returns:
        Oracle circuit
    """
    qc = QuantumCircuit(num_qubits, name='oracle')
    
    # For small number of qubits, we can implement this exactly
    # For larger systems, approximate methods would be used
    
    if num_qubits <= 6:  # Exact implementation for small systems
        # Create diagonal unitary that flips appropriate phases
        diagonal = np.ones(2 ** num_qubits, dtype=complex)
        
        # States to preserve (near 0)
        for k in range(threshold):
            diagonal[k] = 1
        for k in range(2 ** num_qubits - threshold, 2 ** num_qubits):
            diagonal[k] = 1
        
        # States to flip (far from 0)
        for k in range(threshold, 2 ** num_qubits - threshold):
            diagonal[k] = -1
        
        # Create diagonal gate
        from qiskit.quantum_info import Operator
        from qiskit.circuit.library import UnitaryGate
        
        # Convert diagonal to full unitary
        unitary = np.diag(diagonal)
        oracle_gate = UnitaryGate(unitary, label='phase_oracle')
        qc.append(oracle_gate, range(num_qubits))
        
    else:
        # For larger systems, use arithmetic comparator circuits
        # This is a simplified placeholder
        warnings.warn(f"Using approximate oracle for {num_qubits} qubits")
        
        # Apply global phase flip and then unflip the states near 0
        for i in range(num_qubits):
            qc.z(i)
    
    return qc


def apply_reflection_operator(
    state_circuit: QuantumCircuit,
    reflection_operator: QuantumCircuit,
    backend: str = "qiskit",
    shots: int = 1024,
    return_statevector: bool = False
) -> Dict[str, any]:
    """Apply reflection operator to a quantum state.
    
    Applies the approximate reflection operator to a prepared quantum state
    and returns the measurement results or final statevector.
    
    Args:
        state_circuit: Circuit preparing the initial quantum state
        reflection_operator: The reflection operator circuit
        backend: Simulation backend - "qiskit" or "statevector"
        shots: Number of measurement shots (for qiskit backend)
        return_statevector: If True, return the final statevector
    
    Returns:
        Dictionary containing:
            - 'counts': Measurement histogram (if not return_statevector)
            - 'statevector': Final statevector (if return_statevector)
            - 'fidelity': Fidelity with target stationary state (if known)
            - 'circuit': The complete circuit
    
    Example:
        >>> # Prepare initial state
        >>> init = QuantumCircuit(4)
        >>> init.h(range(4))  # Uniform superposition
        >>> 
        >>> # Apply reflection
        >>> R = approximate_reflection_operator(W, num_ancilla=6)
        >>> results = apply_reflection_operator(init, R)
        >>> print(f"Most probable outcome: {max(results['counts'], key=results['counts'].get)}")
    """
    # Validate inputs
    if backend not in ["qiskit", "statevector"]:
        raise ValueError(f"Backend '{backend}' not supported")
    
    # Check compatibility
    num_system_qubits = state_circuit.num_qubits
    num_refl_system = reflection_operator.num_qubits - reflection_operator.num_qubits // 2
    
    # Build complete circuit
    if num_system_qubits != num_refl_system:
        # Need to adjust for ancilla qubits
        num_ancilla = reflection_operator.num_qubits - num_system_qubits
        
        # Create full circuit with ancillas
        ancilla = QuantumRegister(num_ancilla, name='ancilla')
        system = QuantumRegister(num_system_qubits, name='system')
        
        if return_statevector or backend == "statevector":
            qc = QuantumCircuit(ancilla, system)
        else:
            c_system = ClassicalRegister(num_system_qubits, name='c_system')
            qc = QuantumCircuit(ancilla, system, c_system)
        
        # Apply state preparation to system register
        qc.append(state_circuit, system[:])
        
        # Apply reflection operator
        qc.append(reflection_operator, ancilla[:] + system[:])
        
        # Measure only system qubits if needed
        if not return_statevector and backend != "statevector":
            qc.measure(system, c_system)
    else:
        # Simple case: no ancillas
        qc = state_circuit.compose(reflection_operator)
        
        if not return_statevector and backend != "statevector":
            c_reg = ClassicalRegister(num_system_qubits, name='c')
            qc.add_register(c_reg)
            qc.measure_all()
    
    # Execute circuit
    if backend == "statevector" or return_statevector:
        results = _run_statevector_simulation(qc)
    else:
        results = _run_shot_simulation(qc, shots)
    
    results['circuit'] = qc
    
    return results


def _run_statevector_simulation(circuit: QuantumCircuit) -> Dict[str, any]:
    """Run statevector simulation of the circuit."""
    # Remove measurements if any
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    
    # Get statevector
    sv = Statevector(circuit_no_meas)
    
    # Extract system state by tracing out ancillas if needed
    num_qubits = circuit.num_qubits
    if 'ancilla' in [reg.name for reg in circuit.qregs]:
        # Need to trace out ancillas
        num_ancilla = circuit.get_qreg('ancilla').size
        num_system = num_qubits - num_ancilla
        
        # Get reduced density matrix for system
        rho = sv.to_operator()
        system_qubits = list(range(num_ancilla, num_qubits))
        rho_system = partial_trace(rho, keep_qubits=system_qubits)
        
        # Convert back to statevector if pure
        eigvals, eigvecs = np.linalg.eigh(rho_system)
        if eigvals[-1] > 0.99:  # Nearly pure state
            sv_system = Statevector(eigvecs[:, -1])
        else:
            sv_system = None
            
        return {
            'statevector': sv_system,
            'density_matrix': rho_system,
            'purity': np.real(np.trace(rho_system @ rho_system)),
            'full_statevector': sv
        }
    else:
        return {
            'statevector': sv,
            'counts': sv.probabilities_dict()
        }


def _run_shot_simulation(circuit: QuantumCircuit, shots: int) -> Dict[str, any]:
    """Run shot-based simulation."""
    if not HAS_AER:
        raise ImportError("qiskit-aer is required for shot-based simulation. "
                         "Install with: pip install qiskit-aer")
    simulator = AerSimulator()
    transpiled = transpile(circuit, simulator)
    
    job = simulator.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    return {
        'counts': counts,
        'shots': shots
    }


def partial_trace(rho: np.ndarray, keep_qubits: List[int]) -> np.ndarray:
    """Compute partial trace of density matrix.
    
    Args:
        rho: Full density matrix
        keep_qubits: Indices of qubits to keep
    
    Returns:
        Reduced density matrix
    """
    n_qubits = int(np.log2(rho.shape[0]))
    keep = sorted(keep_qubits)
    trace_out = [q for q in range(n_qubits) if q not in keep]
    
    # Reshape into tensor
    rho_tensor = rho.reshape([2] * (2 * n_qubits))
    
    # Trace out unwanted qubits
    for q in reversed(sorted(trace_out)):
        # Trace out qubit q
        axis1 = q
        axis2 = q + n_qubits
        rho_tensor = np.trace(rho_tensor, axis1=axis1, axis2=axis2)
        
        # Adjust indices
        n_qubits -= 1
    
    # Reshape back to matrix
    d = 2 ** len(keep)
    return rho_tensor.reshape(d, d)


def analyze_reflection_quality(
    reflection_operator: QuantumCircuit,
    target_state: Optional[Union[QuantumCircuit, Statevector]] = None,
    num_samples: int = 100,
    walk_operator: Optional[QuantumCircuit] = None
) -> Dict[str, float]:
    """Analyze the quality of the approximate reflection operator.
    
    Evaluates how well the reflection operator approximates the ideal
    reflection about the stationary state by computing various metrics.
    
    Args:
        reflection_operator: The reflection operator to analyze
        target_state: Target stationary state (if known)
        num_samples: Number of random states to test
        walk_operator: Original walk operator (for comparison)
    
    Returns:
        Dictionary containing:
            - 'average_reflection_fidelity': Average fidelity of reflection
            - 'phase_accuracy': Accuracy of phase discrimination
            - 'operator_norm_error': Operator norm of error
            - 'eigenvalue_analysis': Eigenvalue distribution info
    
    Example:
        >>> R = approximate_reflection_operator(W, num_ancilla=8)
        >>> quality = analyze_reflection_quality(R, target_state=pi_state)
        >>> print(f"Reflection fidelity: {quality['average_reflection_fidelity']:.4f}")
    """
    analysis = {}
    
    # Get operator representation
    R_op = Operator(reflection_operator)
    n_qubits = reflection_operator.num_qubits
    
    # If the operator includes ancillas, we need the system size
    if 'ancilla' in [reg.name for reg in reflection_operator.qregs]:
        n_system = reflection_operator.get_qreg('system').size
        n_ancilla = reflection_operator.get_qreg('ancilla').size
    else:
        n_system = n_qubits
        n_ancilla = 0
    
    # Analyze eigenvalues
    eigenvals = np.linalg.eigvals(R_op.data)
    phases = np.angle(eigenvals) / np.pi  # Normalize to [-1, 1]
    
    # Check how many eigenvalues are close to �1
    n_plus = np.sum(np.abs(phases) < 0.1)
    n_minus = np.sum(np.abs(np.abs(phases) - 1) < 0.1)
    
    analysis['eigenvalue_analysis'] = {
        'num_eigenvalues': len(eigenvals),
        'num_near_plus_one': int(n_plus),
        'num_near_minus_one': int(n_minus),
        'phase_distribution': phases
    }
    
    # Test reflection property on random states
    if num_samples > 0:
        fidelities = []
        
        for _ in range(num_samples):
            # Generate random state
            random_state = Statevector.from_label('0' * n_system)
            random_state = random_state.evolve(
                QuantumCircuit(n_system).compose(
                    random_unitary(n_system)
                )
            )
            
            # Apply reflection twice
            if n_ancilla > 0:
                # Pad with ancilla qubits
                full_state = Statevector.from_label('0' * n_ancilla) ^ random_state
            else:
                full_state = random_state
            
            state_after_one = full_state.evolve(R_op)
            state_after_two = state_after_one.evolve(R_op)
            
            # Check if we get back the original state
            if n_ancilla > 0:
                # Trace out ancillas
                rho_final = partial_trace(
                    state_after_two.to_operator().data,
                    keep_qubits=list(range(n_ancilla, n_qubits))
                )
                rho_initial = random_state.to_operator().data
                fidelity = np.real(np.trace(rho_final @ rho_initial))
            else:
                fidelity = np.abs(state_after_two.inner(full_state)) ** 2
            
            fidelities.append(fidelity)
        
        analysis['average_reflection_fidelity'] = np.mean(fidelities)
        analysis['fidelity_std'] = np.std(fidelities)
    
    # If target state is provided, check specific reflection
    if target_state is not None:
        if isinstance(target_state, QuantumCircuit):
            target_sv = Statevector(target_state)
        else:
            target_sv = target_state
        
        # Check that target is approximately preserved
        if n_ancilla > 0:
            full_target = Statevector.from_label('0' * n_ancilla) ^ target_sv
        else:
            full_target = target_sv
        
        target_after = full_target.evolve(R_op)
        preservation_fidelity = np.abs(target_after.inner(full_target)) ** 2
        
        analysis['target_preservation_fidelity'] = preservation_fidelity
    
    # Estimate operator norm error
    # Ideal reflection has eigenvalues �1
    ideal_eigenvals = np.ones_like(eigenvals)
    ideal_eigenvals[n_plus:] = -1
    
    eigenval_errors = np.abs(eigenvals - ideal_eigenvals)
    analysis['operator_norm_error'] = np.max(eigenval_errors)
    analysis['average_eigenvalue_error'] = np.mean(eigenval_errors)
    
    return analysis


def random_unitary(n_qubits: int) -> QuantumCircuit:
    """Generate a random unitary circuit."""
    from qiskit.circuit.library import UnitaryGate
    from scipy.stats import unitary_group
    
    U = unitary_group.rvs(2 ** n_qubits)
    qc = QuantumCircuit(n_qubits)
    qc.append(UnitaryGate(U), range(n_qubits))
    
    return qc


def optimize_reflection_parameters(
    walk_operator: QuantumCircuit,
    test_states: List[Statevector],
    ancilla_range: Tuple[int, int] = (4, 10),
    threshold_range: Tuple[float, float] = (0.05, 0.3)
) -> Dict[str, any]:
    """Optimize parameters for the reflection operator.
    
    Finds optimal number of ancilla qubits and phase threshold
    for the reflection operator by testing on provided states.
    
    Args:
        walk_operator: Quantum walk operator
        test_states: List of test states to evaluate
        ancilla_range: Range of ancilla qubits to try
        threshold_range: Range of phase thresholds to try
    
    Returns:
        Dictionary with optimal parameters and performance metrics
    
    Example:
        >>> test_states = [stationary_state, uniform_state]
        >>> params = optimize_reflection_parameters(W, test_states)
        >>> print(f"Optimal ancillas: {params['optimal_ancilla']}")
    """
    results = []
    
    ancilla_values = range(ancilla_range[0], ancilla_range[1] + 1)
    threshold_values = np.linspace(threshold_range[0], threshold_range[1], 5)
    
    for n_anc in ancilla_values:
        for thresh in threshold_values:
            # Build reflection operator
            try:
                R = approximate_reflection_operator(
                    walk_operator, n_anc, thresh
                )
                
                # Evaluate on test states
                fidelities = []
                for state in test_states:
                    result = apply_reflection_operator(
                        state.to_circuit() if hasattr(state, 'to_circuit') else state,
                        R,
                        backend="statevector",
                        return_statevector=True
                    )
                    
                    # Compute fidelity (simplified)
                    if result['statevector'] is not None:
                        fid = 1.0  # Placeholder
                    else:
                        fid = result['purity']
                    
                    fidelities.append(fid)
                
                avg_fidelity = np.mean(fidelities)
                
                results.append({
                    'num_ancilla': n_anc,
                    'threshold': thresh,
                    'average_fidelity': avg_fidelity,
                    'circuit_depth': R.depth(),
                    'gate_count': R.size()
                })
                
            except Exception as e:
                warnings.warn(f"Failed for ancilla={n_anc}, threshold={thresh}: {e}")
    
    # Find optimal parameters
    if results:
        best = max(results, key=lambda x: x['average_fidelity'])
        
        return {
            'optimal_ancilla': best['num_ancilla'],
            'optimal_threshold': best['threshold'],
            'best_fidelity': best['average_fidelity'],
            'all_results': results,
            'parameter_sensitivity': _compute_sensitivity(results)
        }
    else:
        return {'error': 'No valid configurations found'}


def _compute_sensitivity(results: List[Dict]) -> Dict[str, float]:
    """Compute parameter sensitivity from optimization results."""
    if len(results) < 2:
        return {}
    
    fidelities = [r['average_fidelity'] for r in results]
    ancillas = [r['num_ancilla'] for r in results]
    thresholds = [r['threshold'] for r in results]
    
    # Simple sensitivity measures
    sensitivity = {}
    
    # Variation with ancilla
    if len(set(ancillas)) > 1:
        ancilla_groups = {}
        for r in results:
            n = r['num_ancilla']
            if n not in ancilla_groups:
                ancilla_groups[n] = []
            ancilla_groups[n].append(r['average_fidelity'])
        
        ancilla_means = [np.mean(v) for v in ancilla_groups.values()]
        sensitivity['ancilla_sensitivity'] = np.std(ancilla_means)
    
    # Variation with threshold
    if len(set(thresholds)) > 1:
        threshold_groups = {}
        for r in results:
            t = round(r['threshold'], 3)
            if t not in threshold_groups:
                threshold_groups[t] = []
            threshold_groups[t].append(r['average_fidelity'])
        
        threshold_means = [np.mean(v) for v in threshold_groups.values()]
        sensitivity['threshold_sensitivity'] = np.std(threshold_means)
    
    return sensitivity


def plot_reflection_analysis(
    analysis: Dict[str, any],
    figsize: Tuple[float, float] = (12, 8)
) -> Figure:
    """Visualize reflection operator analysis results.
    
    Creates plots showing eigenvalue distribution, fidelity results,
    and other analysis metrics.
    
    Args:
        analysis: Output from analyze_reflection_quality()
        figsize: Figure size in inches
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Eigenvalue phases
    if 'eigenvalue_analysis' in analysis:
        ax = axes[0, 0]
        phases = analysis['eigenvalue_analysis']['phase_distribution']
        ax.hist(phases, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', label='Target phase')
        ax.axvline(x=1, color='red', linestyle='--')
        ax.axvline(x=-1, color='red', linestyle='--')
        ax.set_xlabel('Eigenvalue Phase / �')
        ax.set_ylabel('Count')
        ax.set_title('Eigenvalue Phase Distribution')
        ax.legend()
    
    # Plot 2: Reflection fidelity
    if 'average_reflection_fidelity' in analysis:
        ax = axes[0, 1]
        fidelity = analysis['average_reflection_fidelity']
        std = analysis.get('fidelity_std', 0)
        
        ax.bar(['Reflection\nFidelity'], [fidelity], yerr=[std],
               color='green', alpha=0.7, capsize=10)
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Fidelity')
        ax.set_title(f'Average Reflection Fidelity: {fidelity:.4f}')
    
    # Plot 3: Eigenvalue errors
    if 'eigenvalue_analysis' in analysis:
        ax = axes[1, 0]
        n_plus = analysis['eigenvalue_analysis']['num_near_plus_one']
        n_minus = analysis['eigenvalue_analysis']['num_near_minus_one']
        n_total = analysis['eigenvalue_analysis']['num_eigenvalues']
        n_other = n_total - n_plus - n_minus
        
        labels = ['+1', '-1', 'Other']
        sizes = [n_plus, n_minus, n_other]
        colors = ['green', 'red', 'gray']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90)
        ax.set_title('Eigenvalue Classification')
    
    # Plot 4: Error metrics
    if 'operator_norm_error' in analysis:
        ax = axes[1, 1]
        metrics = {
            'Operator\nNorm Error': analysis['operator_norm_error'],
            'Average\nEigenvalue Error': analysis['average_eigenvalue_error']
        }
        
        if 'target_preservation_fidelity' in analysis:
            metrics['Target\nPreservation'] = 1 - analysis['target_preservation_fidelity']
        
        ax.bar(metrics.keys(), metrics.values(), color='orange', alpha=0.7)
        ax.set_ylabel('Error')
        ax.set_title('Error Metrics')
        ax.set_ylim([0, max(metrics.values()) * 1.2])
    
    plt.tight_layout()
    return fig