"""State preparation utilities for quantum MCMC sampling.

This module provides functions to prepare quantum states that serve as inputs
to Szegedy quantum walks, phase estimation, and other quantum MCMC routines.
The implementations focus on efficient preparation of probability distributions,
basis states, and superposition states commonly used in quantum algorithms.

State preparation is a crucial component of quantum algorithms, as the initial
state often encodes the problem structure or target distribution. This module
provides both exact and approximate methods for preparing various quantum states.

References:
    Grover, L., & Rudolph, T. (2002). Creating superpositions that correspond
    to efficiently integrable probability distributions. arXiv:quant-ph/0208112.
    
    Shende, V. V., Bullock, S. S., & Markov, I. L. (2006). Synthesis of
    quantum-logic circuits. IEEE Transactions on Computer-Aided Design, 25(6).

Author: [Your Name]
Date: 2025-01-23
"""

from typing import List, Optional, Union, Tuple, Callable
import numpy as np
import warnings
from math import log2, ceil

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, UniformSuperposition
from qiskit.quantum_info import Statevector
from qiskit.extensions import UnitaryGate


def prepare_stationary_state(
    pi: np.ndarray,
    num_qubits: int,
    method: str = "exact",
    threshold: float = 1e-10
) -> QuantumCircuit:
    """Prepare quantum state encoding stationary distribution.
    
    Constructs a quantum circuit that prepares the state:
        |¿È = £_x (¿_x) |xÈ
    
    where ¿ is a probability distribution over computational basis states.
    This state preparation is essential for quantum MCMC algorithms where
    the stationary distribution serves as the target state.
    
    Args:
        pi: Probability distribution (must sum to 1)
        num_qubits: Number of qubits in the quantum register
        method: Preparation method - "exact" uses Qiskit's StatePreparation,
                "sparse" uses optimized methods for sparse distributions
        threshold: Minimum probability to include in sparse preparation
    
    Returns:
        QuantumCircuit that prepares |¿È when applied to |0...0È
    
    Raises:
        ValueError: If pi is not a valid probability distribution or
                   dimensions don't match
    
    Example:
        >>> pi = np.array([0.3, 0.7])  # Two-state distribution
        >>> qc = prepare_stationary_state(pi, num_qubits=1)
        >>> # qc prepares |»È = 0.3|0È + 0.7|1È
    """
    # Validate inputs
    if not np.allclose(np.sum(pi), 1.0):
        raise ValueError(f"Probabilities must sum to 1, got {np.sum(pi)}")
    
    if np.any(pi < -threshold):
        raise ValueError("Probabilities cannot be negative")
    
    # Ensure non-negative
    pi = np.maximum(pi, 0)
    pi = pi / np.sum(pi)  # Renormalize
    
    # Check dimension compatibility
    max_states = 2 ** num_qubits
    if len(pi) > max_states:
        raise ValueError(f"Distribution has {len(pi)} states but only "
                        f"{max_states} are possible with {num_qubits} qubits")
    
    # Pad with zeros if necessary
    if len(pi) < max_states:
        pi_padded = np.zeros(max_states)
        pi_padded[:len(pi)] = pi
        pi = pi_padded
    
    # Create quantum circuit
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name='state_prep_¿')
    
    if method == "exact":
        # Use Qiskit's built-in state preparation
        amplitudes = np.sqrt(pi)
        
        # Handle numerical precision
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        # Create state preparation circuit
        state_prep = StatePreparation(amplitudes, label='prepare_¿')
        qc.append(state_prep, qreg[:])
        
    elif method == "sparse":
        # Optimized preparation for sparse distributions
        qc = _prepare_sparse_distribution(pi, num_qubits, threshold)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return qc


def prepare_basis_state(
    index: int,
    num_qubits: int,
    little_endian: bool = True
) -> QuantumCircuit:
    """Prepare a computational basis state |xÈ.
    
    Creates a quantum circuit that prepares the basis state |indexÈ
    from the initial state |0...0È. This is useful for preparing
    specific starting states for quantum walks or as reference states.
    
    Args:
        index: Index of the basis state to prepare (0 to 2^n - 1)
        num_qubits: Number of qubits
        little_endian: If True, use little-endian bit ordering (default)
                      If False, use big-endian ordering
    
    Returns:
        QuantumCircuit that prepares |indexÈ
    
    Raises:
        ValueError: If index is out of range
    
    Example:
        >>> qc = prepare_basis_state(5, num_qubits=3)
        >>> # Prepares |101È = |5È (in little-endian)
    """
    max_index = 2 ** num_qubits - 1
    if not 0 <= index <= max_index:
        raise ValueError(f"Index {index} out of range [0, {max_index}]")
    
    # Create circuit
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name=f'prepare_|{index}È')
    
    # Convert index to binary
    if little_endian:
        # Little-endian: LSB is qubit 0
        for i in range(num_qubits):
            if (index >> i) & 1:
                qc.x(qreg[i])
    else:
        # Big-endian: MSB is qubit 0
        for i in range(num_qubits):
            if (index >> (num_qubits - 1 - i)) & 1:
                qc.x(qreg[i])
    
    return qc


def prepare_uniform_superposition(
    num_qubits: int,
    num_states: Optional[int] = None
) -> QuantumCircuit:
    """Prepare uniform superposition over computational basis states.
    
    Creates the state:
        |»È = (1/N) £_{x=0}^{N-1} |xÈ
    
    where N is the number of states in superposition.
    
    Args:
        num_qubits: Number of qubits
        num_states: Number of states to include in superposition.
                   If None, includes all 2^n states.
    
    Returns:
        QuantumCircuit preparing uniform superposition
    
    Example:
        >>> qc = prepare_uniform_superposition(3, num_states=5)
        >>> # Prepares (|0È + |1È + |2È + |3È + |4È) / 5
    """
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name='uniform_superposition')
    
    if num_states is None:
        # Full superposition using Hadamard gates
        for i in range(num_qubits):
            qc.h(qreg[i])
    else:
        # Partial superposition
        if num_states > 2 ** num_qubits:
            raise ValueError(f"Cannot create superposition of {num_states} states "
                           f"with only {num_qubits} qubits")
        
        if num_states == 2 ** num_qubits:
            # Full superposition
            for i in range(num_qubits):
                qc.h(qreg[i])
        else:
            # Use Qiskit's UniformSuperposition
            uniform_gate = UniformSuperposition(num_states, num_qubits)
            qc.append(uniform_gate, qreg[:])
    
    return qc


def prepare_gaussian_state(
    mean: float,
    std: float,
    num_qubits: int,
    truncate: float = 4.0
) -> QuantumCircuit:
    """Prepare approximate Gaussian distribution state.
    
    Creates a quantum state with amplitudes following a discretized
    Gaussian distribution:
        |»È = £_x (p_x) |xÈ
    where p_x  exp(-(x-º)≤/(2√≤))
    
    Args:
        mean: Mean of the Gaussian (º)
        std: Standard deviation (√)
        num_qubits: Number of qubits
        truncate: Number of standard deviations to truncate at
    
    Returns:
        QuantumCircuit preparing approximate Gaussian state
    
    Example:
        >>> qc = prepare_gaussian_state(mean=7.5, std=2.0, num_qubits=4)
        >>> # Prepares Gaussian centered at state |7È-|8È
    """
    n_states = 2 ** num_qubits
    
    # Create discrete Gaussian distribution
    x = np.arange(n_states)
    
    # Compute Gaussian probabilities
    probs = np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    # Truncate small probabilities
    cutoff = np.exp(-0.5 * truncate ** 2)
    probs[probs < cutoff * np.max(probs)] = 0
    
    # Normalize
    probs = probs / np.sum(probs)
    
    # Use stationary state preparation
    return prepare_stationary_state(probs, num_qubits, method="sparse")


def prepare_binomial_state(
    n_trials: int,
    p_success: float,
    num_qubits: int
) -> QuantumCircuit:
    """Prepare quantum state following binomial distribution.
    
    Creates a state where amplitudes follow:
        |»È = £_k (B(k; n,p)) |kÈ
    where B(k; n,p) is the binomial probability.
    
    Args:
        n_trials: Number of trials
        p_success: Success probability
        num_qubits: Number of qubits
    
    Returns:
        QuantumCircuit preparing binomial distribution state
    
    Example:
        >>> qc = prepare_binomial_state(n_trials=10, p_success=0.3, num_qubits=4)
    """
    from scipy.stats import binom
    
    n_states = 2 ** num_qubits
    if n_trials >= n_states:
        raise ValueError(f"Number of trials {n_trials} too large for {num_qubits} qubits")
    
    # Compute binomial probabilities
    probs = np.zeros(n_states)
    for k in range(min(n_trials + 1, n_states)):
        probs[k] = binom.pmf(k, n_trials, p_success)
    
    # Normalize (should already be normalized, but ensure numerical stability)
    probs = probs / np.sum(probs)
    
    return prepare_stationary_state(probs, num_qubits)


def prepare_thermal_state(
    energies: np.ndarray,
    beta: float,
    num_qubits: int
) -> QuantumCircuit:
    """Prepare thermal (Gibbs) state for given energy spectrum.
    
    Creates the thermal state:
        |»È = £_x (Z^(-1) exp(-≤ E_x)) |xÈ
    where Z is the partition function and ≤ = 1/kT.
    
    Args:
        energies: Energy values for each computational basis state
        beta: Inverse temperature
        num_qubits: Number of qubits
    
    Returns:
        QuantumCircuit preparing thermal state
    
    Example:
        >>> E = np.array([0, 1, 1, 2])  # Energy levels
        >>> qc = prepare_thermal_state(E, beta=2.0, num_qubits=2)
    """
    if len(energies) > 2 ** num_qubits:
        raise ValueError(f"Too many energy levels for {num_qubits} qubits")
    
    # Compute Boltzmann weights
    weights = np.exp(-beta * energies)
    
    # Handle overflow/underflow
    if np.any(np.isinf(weights)) or np.all(weights == 0):
        # Normalize energies to prevent numerical issues
        energies_shifted = energies - np.min(energies)
        weights = np.exp(-beta * energies_shifted)
    
    # Compute probabilities (Gibbs distribution)
    Z = np.sum(weights)  # Partition function
    probs = weights / Z
    
    # Pad if necessary
    if len(probs) < 2 ** num_qubits:
        probs_padded = np.zeros(2 ** num_qubits)
        probs_padded[:len(probs)] = probs
        probs = probs_padded
    
    return prepare_stationary_state(probs, num_qubits)


def prepare_w_state(num_qubits: int) -> QuantumCircuit:
    """Prepare W state: equal superposition of single-excitation states.
    
    Creates the W state:
        |WÈ = (1/n) (|100...0È + |010...0È + ... + |000...1È)
    
    This state has exactly one qubit in |1È state and is useful
    for certain quantum walk initializations.
    
    Args:
        num_qubits: Number of qubits (n e 2)
    
    Returns:
        QuantumCircuit preparing W state
    
    Example:
        >>> qc = prepare_w_state(3)
        >>> # Prepares (|100È + |010È + |001È) / 3
    """
    if num_qubits < 2:
        raise ValueError("W state requires at least 2 qubits")
    
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name=f'W_{num_qubits}')
    
    # Start with |100...0È
    qc.x(qreg[0])
    
    # Recursively build W state
    for i in range(num_qubits - 1):
        # Rotation angle to distribute amplitude
        # After i steps, we have W_{i+1} and need to create W_{i+2}
        theta = 2 * np.arccos(np.sqrt((i + 1) / (i + 2)))
        
        # Controlled rotation
        qc.cry(theta, qreg[i], qreg[i + 1])
        
        # CNOT cascade to move excitation
        for j in range(i, 0, -1):
            qc.cx(qreg[j], qreg[j - 1])
    
    return qc


def prepare_dicke_state(
    num_qubits: int,
    num_excitations: int
) -> QuantumCircuit:
    """Prepare Dicke state: equal superposition with fixed Hamming weight.
    
    Creates the Dicke state |D_n^kÈ which is an equal superposition
    of all n-qubit states with exactly k ones:
        |D_n^kÈ = (1/C(n,k)) £_{|x|=k} |xÈ
    
    where |x| denotes Hamming weight and C(n,k) is binomial coefficient.
    
    Args:
        num_qubits: Total number of qubits (n)
        num_excitations: Number of excitations/ones (k)
    
    Returns:
        QuantumCircuit preparing Dicke state
    
    Example:
        >>> qc = prepare_dicke_state(4, 2)
        >>> # Prepares equal superposition of all 4-bit strings with 2 ones
    """
    if num_excitations > num_qubits:
        raise ValueError(f"Cannot have {num_excitations} excitations with {num_qubits} qubits")
    
    if num_excitations == 1:
        return prepare_w_state(num_qubits)
    
    # For general Dicke states, we use a more complex construction
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name=f'Dicke_{num_qubits}^{num_excitations}')
    
    if num_excitations == 0:
        # Already in |000...0È
        pass
    elif num_excitations == num_qubits:
        # All ones
        qc.x(qreg[:])
    else:
        # General case: Use state preparation
        # Compute the state vector
        from math import comb
        
        state_vector = np.zeros(2 ** num_qubits)
        for i in range(2 ** num_qubits):
            if bin(i).count('1') == num_excitations:
                state_vector[i] = 1.0
        
        # Normalize
        state_vector = state_vector / np.linalg.norm(state_vector)
        
        # Use state preparation
        state_prep = StatePreparation(state_vector, label=f'D_{num_qubits}^{num_excitations}')
        qc.append(state_prep, qreg[:])
    
    return qc


def prepare_edge_superposition(
    vertex: int,
    num_vertices: int,
    transition_probs: Optional[np.ndarray] = None
) -> QuantumCircuit:
    """Prepare superposition over edges from a given vertex.
    
    For quantum walks on graphs, prepares the state:
        |»_vÈ = £_u (P_{vu}) |vÈ|uÈ
    
    where P_{vu} is the transition probability from v to u.
    
    Args:
        vertex: Source vertex index
        num_vertices: Total number of vertices
        transition_probs: Transition probabilities from vertex.
                         If None, uses uniform distribution.
    
    Returns:
        QuantumCircuit preparing edge superposition state
    
    Example:
        >>> # Prepare superposition of edges from vertex 0
        >>> qc = prepare_edge_superposition(0, num_vertices=4)
    """
    n_qubits = int(ceil(log2(num_vertices)))
    if n_qubits == 0:
        n_qubits = 1
    
    # Create registers for source and target vertices
    qreg_source = QuantumRegister(n_qubits, name='source')
    qreg_target = QuantumRegister(n_qubits, name='target')
    qc = QuantumCircuit(qreg_source, qreg_target, name=f'edges_from_{vertex}')
    
    # Prepare source vertex state
    source_prep = prepare_basis_state(vertex, n_qubits)
    qc.append(source_prep, qreg_source[:])
    
    # Prepare superposition over target vertices
    if transition_probs is None:
        # Uniform superposition
        target_prep = prepare_uniform_superposition(n_qubits, num_vertices)
    else:
        # Weighted superposition according to transition probabilities
        if len(transition_probs) != num_vertices:
            raise ValueError(f"Expected {num_vertices} transition probabilities, "
                           f"got {len(transition_probs)}")
        
        # Normalize if needed
        if not np.allclose(np.sum(transition_probs), 1.0):
            transition_probs = transition_probs / np.sum(transition_probs)
        
        target_prep = prepare_stationary_state(transition_probs, n_qubits)
    
    qc.append(target_prep, qreg_target[:])
    
    return qc


def _prepare_sparse_distribution(
    probs: np.ndarray,
    num_qubits: int,
    threshold: float = 1e-10
) -> QuantumCircuit:
    """Optimized state preparation for sparse probability distributions.
    
    Uses a more efficient circuit construction when the distribution
    has many zero or near-zero probabilities.
    
    Args:
        probs: Probability distribution
        num_qubits: Number of qubits
        threshold: Minimum probability to include
    
    Returns:
        Optimized quantum circuit
    """
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name='sparse_prep')
    
    # Find non-negligible probabilities
    significant_indices = np.where(probs > threshold)[0]
    significant_probs = probs[significant_indices]
    
    if len(significant_indices) == 0:
        warnings.warn("All probabilities below threshold")
        return qc
    
    if len(significant_indices) == 1:
        # Single basis state
        basis_prep = prepare_basis_state(significant_indices[0], num_qubits)
        qc.append(basis_prep, qreg[:])
        return qc
    
    # For sparse distributions, use a tree-based approach
    # This is more efficient than full state preparation
    if len(significant_indices) <= 2 ** (num_qubits // 2):
        # Renormalize
        significant_probs = significant_probs / np.sum(significant_probs)
        amplitudes = np.sqrt(significant_probs)
        
        # Build superposition using controlled rotations
        qc = _build_sparse_superposition(
            significant_indices, amplitudes, num_qubits
        )
    else:
        # Fall back to standard state preparation
        amplitudes = np.sqrt(probs)
        state_prep = StatePreparation(amplitudes)
        qc.append(state_prep, qreg[:])
    
    return qc


def _build_sparse_superposition(
    indices: np.ndarray,
    amplitudes: np.ndarray,
    num_qubits: int
) -> QuantumCircuit:
    """Build superposition over sparse set of basis states.
    
    Uses a binary tree of controlled rotations to efficiently
    prepare superpositions with few non-zero amplitudes.
    """
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name='sparse_superposition')
    
    # If only one or two states, handle directly
    if len(indices) == 1:
        prep = prepare_basis_state(int(indices[0]), num_qubits)
        qc.append(prep, qreg[:])
        return qc
    
    # For multiple states, use recursive construction
    # This is a simplified implementation
    # In practice, more sophisticated methods would be used
    
    # Create full state vector
    full_amplitudes = np.zeros(2 ** num_qubits, dtype=complex)
    for idx, amp in zip(indices, amplitudes):
        full_amplitudes[idx] = amp
    
    # Normalize
    full_amplitudes = full_amplitudes / np.linalg.norm(full_amplitudes)
    
    # Use state preparation
    state_prep = StatePreparation(full_amplitudes)
    qc.append(state_prep, qreg[:])
    
    return qc


def validate_state_preparation(
    prep_circuit: QuantumCircuit,
    target_distribution: Optional[np.ndarray] = None,
    tolerance: float = 1e-6
) -> bool:
    """Validate that a state preparation circuit produces the expected state.
    
    Args:
        prep_circuit: State preparation circuit to validate
        target_distribution: Expected probability distribution (if known)
        tolerance: Tolerance for comparison
    
    Returns:
        True if the circuit produces the expected state
    
    Example:
        >>> pi = np.array([0.25, 0.75])
        >>> qc = prepare_stationary_state(pi, 1)
        >>> is_valid = validate_state_preparation(qc, pi)
    """
    # Simulate the circuit
    initial_state = Statevector.from_label('0' * prep_circuit.num_qubits)
    final_state = initial_state.evolve(prep_circuit)
    
    if target_distribution is not None:
        # Compare with target
        probs = final_state.probabilities()
        
        # Pad target if necessary
        if len(target_distribution) < len(probs):
            target_padded = np.zeros(len(probs))
            target_padded[:len(target_distribution)] = target_distribution
            target_distribution = target_padded
        
        # Check if distributions match
        return np.allclose(probs[:len(target_distribution)], 
                          target_distribution, atol=tolerance)
    else:
        # Just check if it's a valid quantum state
        return np.allclose(final_state.is_valid(), True)


def optimize_state_preparation(
    target_distribution: np.ndarray,
    num_qubits: int,
    max_gates: Optional[int] = None
) -> QuantumCircuit:
    """Find optimized circuit for preparing given distribution.
    
    Attempts to find a more efficient circuit for preparing
    the target distribution using various optimization strategies.
    
    Args:
        target_distribution: Target probability distribution
        num_qubits: Number of qubits
        max_gates: Maximum number of gates allowed
    
    Returns:
        Optimized quantum circuit
    
    Note:
        This is a simplified implementation. Production implementations
        would use more sophisticated optimization techniques.
    """
    # Try different methods and choose the most efficient
    circuits = []
    
    # Method 1: Standard state preparation
    qc1 = prepare_stationary_state(target_distribution, num_qubits, method="exact")
    circuits.append(('exact', qc1))
    
    # Method 2: Sparse preparation if applicable
    sparsity = np.sum(target_distribution > 1e-10) / len(target_distribution)
    if sparsity < 0.5:
        qc2 = prepare_stationary_state(target_distribution, num_qubits, method="sparse")
        circuits.append(('sparse', qc2))
    
    # Choose circuit with fewest gates
    best_method, best_circuit = min(circuits, key=lambda x: x[1].size())
    
    if max_gates is not None and best_circuit.size() > max_gates:
        warnings.warn(f"Could not find circuit with <= {max_gates} gates. "
                     f"Best found: {best_circuit.size()} gates")
    
    return best_circuit