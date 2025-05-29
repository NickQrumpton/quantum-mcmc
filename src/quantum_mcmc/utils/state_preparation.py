"""State preparation utilities for quantum MCMC sampling.

This module provides functions to prepare quantum states that serve as inputs
to Szegedy quantum walks, phase estimation, and other quantum MCMC routines.
The implementations focus on efficient preparation of probability distributions,
basis states, and superposition states commonly used in quantum algorithms.

State preparation is a crucial component of quantum algorithms, as the initial
state often encodes the problem structure or target distribution. This module
provides both exact and approximate methods for preparing various quantum states.

Mathematical Framework:
For a probability distribution π over states {0, 1, ..., n-1}, we prepare:
    |π⟩ = Σₓ √π(x) |x⟩

For Szegedy quantum walks, we work with edge superpositions:
    |ψᵥ⟩ = Σᵤ √P(v,u) |v⟩|u⟩

References:
    Grover, L., & Rudolph, T. (2002). Creating superpositions that correspond
    to efficiently integrable probability distributions. arXiv:quant-ph/0208112.
    
    Shende, V. V., Bullock, S. S., & Markov, I. L. (2006). Synthesis of
    quantum-logic circuits. IEEE Transactions on Computer-Aided Design, 25(6).
    
    Szegedy, M. (2004). Quantum speed-up of Markov chain based algorithms.
    FOCS 2004: 32-41.

Author: Nicholas Zhao  
Affiliation: Imperial College London  
Contact: nz422@ic.ac.uk
"""

from typing import List, Optional, Union, Tuple, Callable, Dict, Any
import numpy as np
import warnings
from math import log2, ceil, sqrt, pi as math_pi
import scipy.sparse as sp
from scipy.stats import entropy

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import StatePreparation

# UniformSuperposition is not available in all Qiskit versions
try:
    from qiskit.circuit.library import UniformSuperposition
    HAS_UNIFORM_SUPERPOSITION = True
except ImportError:
    UniformSuperposition = None
    HAS_UNIFORM_SUPERPOSITION = False
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.extensions import UnitaryGate
from qiskit.circuit import Parameter, ParameterVector


def prepare_stationary_state(
    pi: np.ndarray,
    num_qubits: Optional[int] = None,
    method: str = "exact",
    threshold: float = 1e-10
) -> QuantumCircuit:
    """Prepare quantum state encoding stationary distribution.
    
    Constructs a quantum circuit that prepares the state:
        |π⟩ = Σₓ √π(x) |x⟩
    
    where π is a probability distribution over computational basis states.
    This state preparation is essential for quantum MCMC algorithms where
    the stationary distribution serves as the target state.
    
    Mathematical Details:
        For a stationary distribution π with π(x) ≥ 0 and Σₓ π(x) = 1,
        we prepare |π⟩ with amplitude √π(x) for basis state |x⟩.
    
    Args:
        pi: Probability distribution (must sum to 1)
        num_qubits: Number of qubits in the quantum register. If None,
                   computed as ceil(log₂(len(π)))
        method: Preparation method - "exact" uses Qiskit's StatePreparation,
                "sparse" uses optimized methods for sparse distributions,
                "tree" uses binary tree decomposition
        threshold: Minimum probability to include in sparse preparation
    
    Returns:
        QuantumCircuit that prepares |π⟩ when applied to |0...0⟩
    
    Raises:
        ValueError: If π is not a valid probability distribution or
                   dimensions don't match
    
    Example:
        >>> pi = np.array([0.3, 0.7])  # Two-state distribution
        >>> qc = prepare_stationary_state(pi, num_qubits=1)
        >>> # qc prepares |π⟩ = √0.3|0⟩ + √0.7|1⟩
        
        >>> # For sparse distributions
        >>> pi_sparse = np.array([0.8, 0.0, 0.0, 0.2])
        >>> qc = prepare_stationary_state(pi_sparse, method="sparse")
    """
    # Input validation
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 1:
        raise ValueError("Distribution must be a 1D array")
    
    if not np.allclose(np.sum(pi), 1.0, atol=1e-10):
        raise ValueError(f"Probabilities must sum to 1, got {np.sum(pi):.10f}")
    
    if np.any(pi < -threshold):
        raise ValueError("Probabilities cannot be negative")
    
    # Clean up numerical errors
    pi = np.maximum(pi, 0)
    pi = pi / np.sum(pi)  # Renormalize
    
    # Determine number of qubits
    if num_qubits is None:
        num_qubits = max(1, int(ceil(log2(len(pi)))))
    
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
    qc = QuantumCircuit(qreg, name='state_prep_π')
    
    if method == "exact":
        # Use Qiskit's built-in state preparation
        amplitudes = np.sqrt(pi)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        state_prep = StatePreparation(amplitudes, label='prepare_π')
        qc.append(state_prep, qreg[:])
        
    elif method == "sparse":
        # Optimized preparation for sparse distributions
        qc = _prepare_sparse_distribution(pi, num_qubits, threshold)
        
    elif method == "tree":
        # Binary tree decomposition for specific cases
        qc = _prepare_tree_decomposition(pi, num_qubits)
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'exact', 'sparse', 'tree'")
    
    return qc


def prepare_basis_state(
    index: int,
    num_qubits: int,
    little_endian: bool = True
) -> QuantumCircuit:
    """Prepare a computational basis state |x⟩.
    
    Creates a quantum circuit that prepares the basis state |index⟩
    from the initial state |0...0⟩. This is useful for preparing
    specific starting states for quantum walks or as reference states.
    
    Mathematical Details:
        Prepares |index⟩ = |bₙ₋₁...b₁b₀⟩ where bᵢ is the i-th bit 
        of the binary representation of index.
    
    Args:
        index: Index of the basis state to prepare (0 to 2ⁿ - 1)
        num_qubits: Number of qubits
        little_endian: If True, use little-endian bit ordering (default)
                      If False, use big-endian ordering
    
    Returns:
        QuantumCircuit that prepares |index⟩
    
    Raises:
        ValueError: If index is out of range
    
    Example:
        >>> qc = prepare_basis_state(5, num_qubits=3)
        >>> # Prepares |101⟩ = |5⟩ (in little-endian)
        
        >>> qc = prepare_basis_state(5, num_qubits=3, little_endian=False)
        >>> # Prepares |101⟩ with MSB as qubit 0
    """
    # Input validation
    if not isinstance(index, int) or index < 0:
        raise ValueError("Index must be a non-negative integer")
    
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("Number of qubits must be a positive integer")
    
    max_index = 2 ** num_qubits - 1
    if index > max_index:
        raise ValueError(f"Index {index} out of range [0, {max_index}]")
    
    # Create circuit
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name=f'prepare_|{index}⟩')
    
    # Convert index to binary and apply X gates
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
        |ψ⟩ = (1/√N) Σₓ₌₀^{N-1} |x⟩
    
    where N is the number of states in superposition.
    
    Mathematical Details:
        For N = 2ⁿ, uses Hadamard gates: H⊗ⁿ|0⟩⊗ⁿ
        For general N, uses amplitude encoding with uniform amplitudes.
    
    Args:
        num_qubits: Number of qubits
        num_states: Number of states to include in superposition.
                   If None, includes all 2ⁿ states.
    
    Returns:
        QuantumCircuit preparing uniform superposition
        
    Raises:
        ValueError: If num_states > 2^num_qubits
    
    Example:
        >>> qc = prepare_uniform_superposition(3, num_states=5)
        >>> # Prepares (|0⟩ + |1⟩ + |2⟩ + |3⟩ + |4⟩) / √5
        
        >>> qc = prepare_uniform_superposition(2)  # Full superposition
        >>> # Prepares (|0⟩ + |1⟩ + |2⟩ + |3⟩) / 2
    """
    # Input validation
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("Number of qubits must be a positive integer")
    
    max_states = 2 ** num_qubits
    if num_states is not None:
        if not isinstance(num_states, int) or num_states < 1:
            raise ValueError("Number of states must be a positive integer")
        if num_states > max_states:
            raise ValueError(f"Cannot create superposition of {num_states} states "
                           f"with only {num_qubits} qubits")
    
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name='uniform_superposition')
    
    if num_states is None or num_states == max_states:
        # Full superposition using Hadamard gates
        for i in range(num_qubits):
            qc.h(qreg[i])
    else:
        # Partial superposition
        if HAS_UNIFORM_SUPERPOSITION:
            try:
                # Try using Qiskit's UniformSuperposition
                uniform_gate = UniformSuperposition(num_states, num_qubits)
                qc.append(uniform_gate, qreg[:])
            except:
                # Fallback: manual construction
                uniform_probs = np.zeros(max_states)
                uniform_probs[:num_states] = 1.0 / num_states
                amplitudes = np.sqrt(uniform_probs)
                state_prep = StatePreparation(amplitudes)
                qc.append(state_prep, qreg[:])
        else:
            # Manual construction when UniformSuperposition is not available
            uniform_probs = np.zeros(max_states)
            uniform_probs[:num_states] = 1.0 / num_states
            amplitudes = np.sqrt(uniform_probs)
            state_prep = StatePreparation(amplitudes)
            qc.append(state_prep, qreg[:])
    
    return qc


def prepare_edge_superposition(
    adjacency_matrix: Optional[np.ndarray] = None,
    edge_list: Optional[List[Tuple[int, int]]] = None,
    num_vertices: Optional[int] = None,
    weights: Optional[np.ndarray] = None
) -> QuantumCircuit:
    """Prepare uniform superposition over all graph edges for Szegedy quantum walks.
    
    For quantum walks on graphs, prepares the state:
        |E⟩ = (1/√|E|) Σ_{(u,v)∈E} |u⟩|v⟩
        
    Or with weights:
        |E⟩ = Σ_{(u,v)∈E} √w(u,v) |u⟩|v⟩
    
    This is the standard initial state for Szegedy quantum walks, where
    the first register encodes the source vertex and the second register
    encodes the target vertex.
    
    Mathematical Details:
        For an undirected graph G = (V,E), we prepare a superposition over
        all edges encoded as |source⟩|target⟩. For directed graphs, we
        include all directed edges. The state is normalized to unit length.
    
    Args:
        adjacency_matrix: Adjacency matrix of the graph (symmetric for undirected)
        edge_list: List of edges as (source, target) tuples
        num_vertices: Number of vertices (required if using edge_list)
        weights: Edge weights (if None, uniform weights are used)
    
    Returns:
        QuantumCircuit preparing edge superposition state with two registers
        
    Raises:
        ValueError: If neither adjacency_matrix nor edge_list is provided,
                   or if dimensions are inconsistent
    
    Example:
        >>> # From adjacency matrix
        >>> adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Triangle graph
        >>> qc = prepare_edge_superposition(adjacency_matrix=adj)
        >>> # Prepares (|01⟩ + |02⟩ + |10⟩ + |12⟩ + |20⟩ + |21⟩) / √6
        
        >>> # From edge list
        >>> edges = [(0, 1), (1, 2), (2, 0)]  # Directed cycle
        >>> qc = prepare_edge_superposition(edge_list=edges, num_vertices=3)
    """
    # Input validation and edge extraction
    if adjacency_matrix is not None:
        adj = np.asarray(adjacency_matrix)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        
        num_vertices = adj.shape[0]
        
        # Extract edges from adjacency matrix
        edges = []
        edge_weights = []
        for i in range(num_vertices):
            for j in range(num_vertices):
                if adj[i, j] > 0:  # Edge exists
                    edges.append((i, j))
                    edge_weights.append(adj[i, j] if weights is None else weights[len(edges)-1])
                    
    elif edge_list is not None:
        if num_vertices is None:
            # Infer number of vertices
            num_vertices = max(max(edge) for edge in edge_list) + 1
        
        edges = list(edge_list)
        if weights is None:
            edge_weights = [1.0] * len(edges)
        else:
            if len(weights) != len(edges):
                raise ValueError(f"Number of weights ({len(weights)}) must match "
                               f"number of edges ({len(edges)})")
            edge_weights = list(weights)
    else:
        raise ValueError("Must provide either adjacency_matrix or edge_list")
    
    if len(edges) == 0:
        raise ValueError("Graph has no edges")
    
    # Normalize weights to probabilities
    edge_weights = np.array(edge_weights)
    edge_weights = edge_weights / np.sum(edge_weights)
    
    # Determine number of qubits per register
    n_qubits_per_reg = max(1, int(ceil(log2(num_vertices))))
    
    # Create registers
    qreg_source = QuantumRegister(n_qubits_per_reg, name='source')
    qreg_target = QuantumRegister(n_qubits_per_reg, name='target')
    qc = QuantumCircuit(qreg_source, qreg_target, name='edge_superposition')
    
    # Prepare edge superposition using state preparation
    if len(edges) == 1:
        # Single edge case
        source, target = edges[0]
        source_prep = prepare_basis_state(source, n_qubits_per_reg)
        target_prep = prepare_basis_state(target, n_qubits_per_reg)
        qc.append(source_prep, qreg_source[:])
        qc.append(target_prep, qreg_target[:])
        
    else:
        # Multiple edges: prepare joint state
        total_qubits = 2 * n_qubits_per_reg
        joint_amplitudes = np.zeros(2 ** total_qubits)
        
        for (source, target), weight in zip(edges, edge_weights):
            # Convert edge to joint state index
            # Use little-endian: target bits first, then source bits
            joint_index = target + source * (2 ** n_qubits_per_reg)
            joint_amplitudes[joint_index] = sqrt(weight)
        
        # Normalize
        joint_amplitudes = joint_amplitudes / np.linalg.norm(joint_amplitudes)
        
        # Prepare joint state
        state_prep = StatePreparation(joint_amplitudes, label='edge_prep')
        qc.append(state_prep, qreg_source[:] + qreg_target[:])
    
    return qc


def prepare_hamiltonian_eigenstate(
    hamiltonian: np.ndarray,
    eigenvalue_index: int = 0,
    num_qubits: Optional[int] = None
) -> QuantumCircuit:
    """Prepare eigenstate of a given Hamiltonian matrix.
    
    Prepares the eigenstate |ψₖ⟩ corresponding to the k-th eigenvalue
    of the Hamiltonian H, where H|ψₖ⟩ = λₖ|ψₖ⟩.
    
    This is useful for preparing ground states or thermal states
    in quantum simulation applications.
    
    Mathematical Details:
        For Hamiltonian H with eigendecomposition H = UΛU†,
        prepares |ψₖ⟩ = Σⱼ Uⱼₖ |j⟩ (k-th column of U).
    
    Args:
        hamiltonian: Hermitian matrix representing the Hamiltonian
        eigenvalue_index: Index of the eigenvalue (0 = ground state)
        num_qubits: Number of qubits (inferred from matrix size if None)
    
    Returns:
        QuantumCircuit preparing the specified eigenstate
        
    Raises:
        ValueError: If Hamiltonian is not Hermitian or dimensions are wrong
    
    Example:
        >>> # Pauli-Z Hamiltonian
        >>> H = np.array([[1, 0], [0, -1]])
        >>> qc = prepare_hamiltonian_eigenstate(H, eigenvalue_index=0)
        >>> # Prepares ground state |1⟩ (eigenvalue -1)
    """
    # Input validation
    H = np.asarray(hamiltonian, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("Hamiltonian must be a square matrix")
    
    # Check if Hermitian
    if not np.allclose(H, H.conj().T, atol=1e-10):
        raise ValueError("Hamiltonian must be Hermitian")
    
    n_states = H.shape[0]
    if num_qubits is None:
        num_qubits = max(1, int(ceil(log2(n_states))))
    
    if eigenvalue_index < 0 or eigenvalue_index >= n_states:
        raise ValueError(f"Eigenvalue index {eigenvalue_index} out of range [0, {n_states-1}]")
    
    # Compute eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(H)
    
    # Sort by eigenvalue (ascending)
    sort_indices = np.argsort(eigenvals)
    eigenvals = eigenvals[sort_indices]
    eigenvecs = eigenvecs[:, sort_indices]
    
    # Get the desired eigenstate
    eigenstate = eigenvecs[:, eigenvalue_index]
    
    # Pad to full Hilbert space if necessary
    if n_states < 2 ** num_qubits:
        padded_state = np.zeros(2 ** num_qubits, dtype=complex)
        padded_state[:n_states] = eigenstate
        eigenstate = padded_state
    
    # Prepare the state
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name=f'eigenstate_{eigenvalue_index}')
    
    # Use state preparation
    state_prep = StatePreparation(eigenstate, label=f'ψ_{eigenvalue_index}')
    qc.append(state_prep, qreg[:])
    
    return qc


def prepare_gaussian_state(
    mean: float,
    std: float,
    num_qubits: int,
    domain: Optional[Tuple[float, float]] = None,
    truncate: float = 4.0
) -> QuantumCircuit:
    """Prepare approximate discrete Gaussian distribution state.
    
    Creates a quantum state with amplitudes following a discretized
    Gaussian distribution:
        |ψ⟩ = Σₓ √p(x) |x⟩
    where p(x) ∝ exp(-(x-μ)²/(2σ²))
    
    Mathematical Details:
        Discretizes the continuous Gaussian over the domain, then
        normalizes to create a probability distribution. Useful for
        lattice-based sampling problems.
    
    Args:
        mean: Mean of the Gaussian (μ)
        std: Standard deviation (σ)
        num_qubits: Number of qubits
        domain: (min, max) values for discretization. If None, uses [0, 2^n-1]
        truncate: Number of standard deviations to truncate at
    
    Returns:
        QuantumCircuit preparing approximate Gaussian state
        
    Raises:
        ValueError: If parameters are invalid
    
    Example:
        >>> qc = prepare_gaussian_state(mean=7.5, std=2.0, num_qubits=4)
        >>> # Prepares discrete Gaussian centered around state |7⟩-|8⟩
    """
    # Input validation
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("Number of qubits must be a positive integer")
    
    if std <= 0:
        raise ValueError("Standard deviation must be positive")
    
    if truncate <= 0:
        raise ValueError("Truncation parameter must be positive")
    
    n_states = 2 ** num_qubits
    
    # Set domain
    if domain is None:
        x_values = np.arange(n_states)
    else:
        x_min, x_max = domain
        x_values = np.linspace(x_min, x_max, n_states)
    
    # Compute Gaussian probabilities
    probs = np.exp(-0.5 * ((x_values - mean) / std) ** 2)
    
    # Apply truncation
    cutoff_value = np.exp(-0.5 * truncate ** 2)
    probs[probs < cutoff_value * np.max(probs)] = 0
    
    # Normalize to probability distribution
    if np.sum(probs) == 0:
        raise ValueError("All probabilities are zero after truncation")
    
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
        |ψ⟩ = Σₖ √B(k; n,p) |k⟩
    where B(k; n,p) is the binomial probability mass function.
    
    Mathematical Details:
        B(k; n,p) = C(n,k) * p^k * (1-p)^(n-k)
        where C(n,k) is the binomial coefficient.
    
    Args:
        n_trials: Number of trials
        p_success: Success probability (0 ≤ p ≤ 1)
        num_qubits: Number of qubits
    
    Returns:
        QuantumCircuit preparing binomial distribution state
        
    Raises:
        ValueError: If parameters are invalid
    
    Example:
        >>> qc = prepare_binomial_state(n_trials=10, p_success=0.3, num_qubits=4)
        >>> # Prepares binomial distribution with n=10, p=0.3
    """
    # Input validation
    if not isinstance(n_trials, int) or n_trials < 0:
        raise ValueError("Number of trials must be a non-negative integer")
    
    if not 0 <= p_success <= 1:
        raise ValueError("Success probability must be in [0, 1]")
    
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("Number of qubits must be a positive integer")
    
    n_states = 2 ** num_qubits
    if n_trials >= n_states:
        raise ValueError(f"Number of trials {n_trials} too large for {num_qubits} qubits")
    
    # Compute binomial probabilities
    from scipy.stats import binom
    
    probs = np.zeros(n_states)
    for k in range(min(n_trials + 1, n_states)):
        probs[k] = binom.pmf(k, n_trials, p_success)
    
    # Ensure normalization (should already be normalized)
    probs = probs / np.sum(probs)
    
    return prepare_stationary_state(probs, num_qubits)


def prepare_thermal_state(
    energies: np.ndarray,
    beta: float,
    num_qubits: Optional[int] = None
) -> QuantumCircuit:
    """Prepare thermal (Gibbs) state for given energy spectrum.
    
    Creates the thermal state:
        |ψ⟩ = Σₓ √(Z⁻¹ exp(-β Eₓ)) |x⟩
    where Z is the partition function and β = 1/(kT).
    
    Mathematical Details:
        Gibbs distribution: p(x) = exp(-βE_x) / Z
        Partition function: Z = Σₓ exp(-βE_x)
        Temperature relation: β = 1/(k_B T)
    
    Args:
        energies: Energy values for each computational basis state
        beta: Inverse temperature (1/kT)
        num_qubits: Number of qubits (inferred from energies if None)
    
    Returns:
        QuantumCircuit preparing thermal state
        
    Raises:
        ValueError: If parameters are invalid
    
    Example:
        >>> E = np.array([0, 1, 1, 2])  # Energy levels
        >>> qc = prepare_thermal_state(E, beta=2.0, num_qubits=2)
        >>> # Prepares thermal state at inverse temperature β=2
    """
    # Input validation
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 1:
        raise ValueError("Energies must be a 1D array")
    
    if beta < 0:
        raise ValueError("Inverse temperature β must be non-negative")
    
    if num_qubits is None:
        num_qubits = max(1, int(ceil(log2(len(energies)))))
    
    if len(energies) > 2 ** num_qubits:
        raise ValueError(f"Too many energy levels ({len(energies)}) for {num_qubits} qubits")
    
    # Compute Boltzmann weights with numerical stability
    if beta == 0:
        # Infinite temperature: uniform distribution
        weights = np.ones(len(energies))
    else:
        # Shift energies to prevent overflow
        energy_shift = np.min(energies)
        shifted_energies = energies - energy_shift
        weights = np.exp(-beta * shifted_energies)
    
    # Handle numerical edge cases
    if np.any(np.isinf(weights)) or np.all(weights == 0):
        # Use log-space computation
        log_weights = -beta * (energies - np.min(energies))
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
    
    # Compute probabilities (Gibbs distribution)
    Z = np.sum(weights)  # Partition function
    if Z == 0:
        raise ValueError("Partition function is zero - all states have infinite energy")
    
    probs = weights / Z
    
    # Pad to full Hilbert space if necessary
    if len(probs) < 2 ** num_qubits:
        probs_padded = np.zeros(2 ** num_qubits)
        probs_padded[:len(probs)] = probs
        probs = probs_padded
    
    return prepare_stationary_state(probs, num_qubits)


def prepare_w_state(num_qubits: int) -> QuantumCircuit:
    """Prepare W state: equal superposition of single-excitation states.
    
    Creates the W state:
        |W⟩ = (1/√n) (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)
    
    This state has exactly one qubit in |1⟩ state and is maximally
    entangled with respect to certain measures. Useful for certain
    quantum walk initializations.
    
    Mathematical Details:
        |Wₙ⟩ = (1/√n) Σᵢ₌₀ⁿ⁻¹ |0...010...0⟩ᵢ
        where the |1⟩ is in position i.
    
    Args:
        num_qubits: Number of qubits (n ≥ 2)
    
    Returns:
        QuantumCircuit preparing W state
        
    Raises:
        ValueError: If num_qubits < 2
    
    Example:
        >>> qc = prepare_w_state(3)
        >>> # Prepares (|100⟩ + |010⟩ + |001⟩) / √3
    """
    if not isinstance(num_qubits, int) or num_qubits < 2:
        raise ValueError("W state requires at least 2 qubits")
    
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name=f'W_{num_qubits}')
    
    # Recursive construction of W state
    # Start with |100...0⟩
    qc.x(qreg[0])
    
    # Use recursive formula for W state preparation
    for i in range(num_qubits - 1):
        # Rotation angle to distribute amplitude correctly
        # P(excited | next qubit) = 1/(n-i)
        theta = 2 * np.arccos(sqrt((i + 1) / (i + 2)))
        
        # Controlled rotation
        qc.cry(theta, qreg[i], qreg[i + 1])
        
        # CNOT cascade to propagate the excitation
        for j in range(i, 0, -1):
            qc.cx(qreg[j], qreg[j - 1])
    
    return qc


def prepare_dicke_state(
    num_qubits: int,
    num_excitations: int
) -> QuantumCircuit:
    """Prepare Dicke state: equal superposition with fixed Hamming weight.
    
    Creates the Dicke state |D_n^k⟩ which is an equal superposition
    of all n-qubit states with exactly k ones:
        |D_n^k⟩ = (1/√C(n,k)) Σ_{|x|=k} |x⟩
    
    where |x| denotes Hamming weight and C(n,k) is the binomial coefficient.
    
    Mathematical Details:
        The Dicke state is the symmetric subspace of the full Hilbert space
        with fixed total magnetization in spin systems.
    
    Args:
        num_qubits: Total number of qubits (n)
        num_excitations: Number of excitations/ones (k)
    
    Returns:
        QuantumCircuit preparing Dicke state
        
    Raises:
        ValueError: If k > n or parameters are invalid
    
    Example:
        >>> qc = prepare_dicke_state(4, 2)
        >>> # Prepares equal superposition of all 4-bit strings with 2 ones
        >>> # (|0011⟩ + |0101⟩ + |0110⟩ + |1001⟩ + |1010⟩ + |1100⟩) / √6
    """
    # Input validation
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("Number of qubits must be a positive integer")
    
    if not isinstance(num_excitations, int) or num_excitations < 0:
        raise ValueError("Number of excitations must be non-negative")
    
    if num_excitations > num_qubits:
        raise ValueError(f"Cannot have {num_excitations} excitations with {num_qubits} qubits")
    
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name=f'Dicke_{num_qubits}^{num_excitations}')
    
    # Special cases
    if num_excitations == 0:
        # Already in |000...0⟩
        pass
    elif num_excitations == num_qubits:
        # All ones: |111...1⟩
        qc.x(qreg[:])
    elif num_excitations == 1:
        # W state
        w_circuit = prepare_w_state(num_qubits)
        qc.append(w_circuit, qreg[:])
    else:
        # General case: use state preparation
        from math import comb
        
        # Build state vector
        state_vector = np.zeros(2 ** num_qubits)
        for i in range(2 ** num_qubits):
            if bin(i).count('1') == num_excitations:
                state_vector[i] = 1.0
        
        # Normalize
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
        
        # Use state preparation
        if np.any(state_vector != 0):
            state_prep = StatePreparation(state_vector, label=f'D_{num_qubits}^{num_excitations}')
            qc.append(state_prep, qreg[:])
    
    return qc


def _prepare_sparse_distribution(
    probs: np.ndarray,
    num_qubits: int,
    threshold: float = 1e-10
) -> QuantumCircuit:
    """Optimized state preparation for sparse probability distributions.
    
    Uses a more efficient circuit construction when the distribution
    has many zero or near-zero probabilities by focusing on the
    significant amplitudes only.
    
    Args:
        probs: Probability distribution
        num_qubits: Number of qubits
        threshold: Minimum probability to include
    
    Returns:
        Optimized quantum circuit
    """
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name='sparse_prep')
    
    # Find significant probabilities
    significant_mask = probs > threshold
    significant_indices = np.where(significant_mask)[0]
    significant_probs = probs[significant_mask]
    
    if len(significant_indices) == 0:
        warnings.warn("All probabilities below threshold, returning |0⟩ state")
        return qc
    
    if len(significant_indices) == 1:
        # Single basis state
        basis_prep = prepare_basis_state(int(significant_indices[0]), num_qubits)
        qc.append(basis_prep, qreg[:])
        return qc
    
    # Multiple significant states
    # Renormalize significant probabilities
    significant_probs = significant_probs / np.sum(significant_probs)
    
    # For very sparse distributions, use a tree-based approach
    sparsity_ratio = len(significant_indices) / (2 ** num_qubits)
    
    if sparsity_ratio < 0.25:  # Less than 25% of states are significant
        # Use sparse construction
        qc = _build_sparse_superposition(significant_indices, np.sqrt(significant_probs), num_qubits)
    else:
        # Use standard state preparation
        full_amplitudes = np.sqrt(probs)
        state_prep = StatePreparation(full_amplitudes, label='sparse_prep')
        qc.append(state_prep, qreg[:])
    
    return qc


def _prepare_tree_decomposition(
    probs: np.ndarray,
    num_qubits: int
) -> QuantumCircuit:
    """Prepare state using binary tree decomposition.
    
    Uses a binary tree of controlled rotations for more efficient
    preparation of certain probability distributions.
    """
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name='tree_prep')
    
    # Convert to amplitudes
    amplitudes = np.sqrt(probs)
    
    # Use recursive tree construction
    _apply_tree_rotations(qc, qreg, amplitudes, 0, len(amplitudes), 0)
    
    return qc


def _apply_tree_rotations(
    qc: QuantumCircuit,
    qreg: QuantumRegister,
    amplitudes: np.ndarray,
    start: int,
    end: int,
    qubit_index: int
) -> None:
    """Recursively apply rotations for tree decomposition."""
    if end - start <= 1 or qubit_index >= len(qreg):
        return
    
    mid = (start + end) // 2
    
    # Compute left and right norms
    left_norm = np.linalg.norm(amplitudes[start:mid])
    right_norm = np.linalg.norm(amplitudes[mid:end])
    total_norm = np.sqrt(left_norm**2 + right_norm**2)
    
    if total_norm > 1e-12:
        # Rotation angle
        theta = 2 * np.arccos(left_norm / total_norm)
        
        # Apply rotation
        qc.ry(theta, qreg[qubit_index])
        
        # Recursively apply to left and right branches
        if left_norm > 1e-12:
            normalized_left = amplitudes[start:mid] / left_norm
            _apply_tree_rotations(qc, qreg, normalized_left, 0, mid - start, qubit_index + 1)
        
        if right_norm > 1e-12:
            normalized_right = amplitudes[mid:end] / right_norm
            # Apply X gate to access right branch
            qc.x(qreg[qubit_index])
            _apply_tree_rotations(qc, qreg, normalized_right, 0, end - mid, qubit_index + 1)
            qc.x(qreg[qubit_index])  # Undo X gate


def _build_sparse_superposition(
    indices: np.ndarray,
    amplitudes: np.ndarray,
    num_qubits: int
) -> QuantumCircuit:
    """Build superposition over sparse set of basis states.
    
    Uses efficient construction for superpositions with few non-zero amplitudes.
    """
    qreg = QuantumRegister(num_qubits, name='q')
    qc = QuantumCircuit(qreg, name='sparse_superposition')
    
    if len(indices) == 1:
        # Single state
        prep = prepare_basis_state(int(indices[0]), num_qubits)
        qc.append(prep, qreg[:])
        return qc
    
    # Build full amplitude vector
    full_amplitudes = np.zeros(2 ** num_qubits, dtype=complex)
    for idx, amp in zip(indices, amplitudes):
        full_amplitudes[int(idx)] = amp
    
    # Normalize
    norm = np.linalg.norm(full_amplitudes)
    if norm > 0:
        full_amplitudes = full_amplitudes / norm
    
    # Use state preparation
    state_prep = StatePreparation(full_amplitudes, label='sparse')
    qc.append(state_prep, qreg[:])
    
    return qc


def validate_state_preparation(
    prep_circuit: QuantumCircuit,
    target_distribution: Optional[np.ndarray] = None,
    tolerance: float = 1e-6
) -> bool:
    """Validate that a state preparation circuit produces the expected state.
    
    Simulates the circuit and checks if the resulting state matches
    the expected probability distribution.
    
    Args:
        prep_circuit: State preparation circuit to validate
        target_distribution: Expected probability distribution (if known)
        tolerance: Tolerance for comparison
    
    Returns:
        True if the circuit produces the expected state within tolerance
        
    Raises:
        RuntimeError: If simulation fails
    
    Example:
        >>> pi = np.array([0.25, 0.75])
        >>> qc = prepare_stationary_state(pi, 1)
        >>> is_valid = validate_state_preparation(qc, pi)
        >>> assert is_valid
    """
    try:
        # Simulate the circuit
        initial_state = Statevector.from_label('0' * prep_circuit.num_qubits)
        final_state = initial_state.evolve(prep_circuit)
        
        if target_distribution is not None:
            # Compare with target distribution
            simulated_probs = final_state.probabilities()
            
            # Pad target if necessary
            target = np.asarray(target_distribution)
            if len(target) < len(simulated_probs):
                target_padded = np.zeros(len(simulated_probs))
                target_padded[:len(target)] = target
                target = target_padded
            elif len(target) > len(simulated_probs):
                target = target[:len(simulated_probs)]
            
            # Check if distributions match
            return np.allclose(simulated_probs, target, atol=tolerance)
        else:
            # Just check if it's a valid normalized quantum state
            return final_state.is_valid()
            
    except Exception as e:
        raise RuntimeError(f"State validation failed: {str(e)}")


def optimize_state_preparation(
    target_distribution: np.ndarray,
    num_qubits: Optional[int] = None,
    max_gates: Optional[int] = None,
    methods: Optional[List[str]] = None
) -> QuantumCircuit:
    """Find optimized circuit for preparing given distribution.
    
    Attempts to find the most efficient circuit for preparing
    the target distribution using various optimization strategies.
    
    Args:
        target_distribution: Target probability distribution
        num_qubits: Number of qubits (inferred if None)
        max_gates: Maximum number of gates allowed
        methods: List of methods to try ['exact', 'sparse', 'tree']
    
    Returns:
        Optimized quantum circuit with minimum gate count
        
    Raises:
        ValueError: If no suitable circuit is found
    
    Example:
        >>> pi = np.array([0.8, 0.0, 0.0, 0.2])  # Sparse distribution
        >>> qc = optimize_state_preparation(pi, max_gates=50)
        >>> # Returns most efficient circuit (likely 'sparse' method)
    """
    # Input validation
    target = np.asarray(target_distribution, dtype=float)
    if not np.allclose(np.sum(target), 1.0, atol=1e-10):
        raise ValueError("Target must be a probability distribution")
    
    if num_qubits is None:
        num_qubits = max(1, int(ceil(log2(len(target)))))
    
    if methods is None:
        methods = ['exact', 'sparse', 'tree']
    
    # Try different methods and collect results
    circuits = []
    
    for method in methods:
        try:
            if method == 'exact':
                qc = prepare_stationary_state(target, num_qubits, method='exact')
                circuits.append((method, qc, qc.size()))
                
            elif method == 'sparse':
                # Check if sparse method is appropriate
                sparsity = np.sum(target > 1e-10) / len(target)
                if sparsity < 0.75:  # Only use sparse for reasonably sparse distributions
                    qc = prepare_stationary_state(target, num_qubits, method='sparse')
                    circuits.append((method, qc, qc.size()))
                    
            elif method == 'tree':
                qc = prepare_stationary_state(target, num_qubits, method='tree')
                circuits.append((method, qc, qc.size()))
                
        except Exception as e:
            warnings.warn(f"Method '{method}' failed: {str(e)}")
            continue
    
    if not circuits:
        raise ValueError("No preparation method succeeded")
    
    # Select circuit with minimum gate count
    best_method, best_circuit, best_size = min(circuits, key=lambda x: x[2])
    
    # Check gate count constraint
    if max_gates is not None and best_size > max_gates:
        # Try to find alternative within constraint
        valid_circuits = [(m, c, s) for m, c, s in circuits if s <= max_gates]
        if valid_circuits:
            best_method, best_circuit, best_size = min(valid_circuits, key=lambda x: x[2])
        else:
            warnings.warn(f"Could not find circuit with ≤ {max_gates} gates. "
                         f"Best found: {best_size} gates using '{best_method}' method")
    
    return best_circuit


def compute_state_preparation_fidelity(
    prep_circuit: QuantumCircuit,
    target_state: np.ndarray
) -> float:
    """Compute fidelity between prepared state and target state.
    
    Args:
        prep_circuit: State preparation circuit
        target_state: Target state vector (amplitudes)
    
    Returns:
        Fidelity F = |⟨ψ_target|ψ_prepared⟩|²
    """
    # Simulate prepared state
    initial_state = Statevector.from_label('0' * prep_circuit.num_qubits)
    prepared_state = initial_state.evolve(prep_circuit)
    
    # Ensure target state is normalized
    target_normalized = target_state / np.linalg.norm(target_state)
    
    # Pad if necessary
    if len(target_normalized) < 2 ** prep_circuit.num_qubits:
        target_padded = np.zeros(2 ** prep_circuit.num_qubits, dtype=complex)
        target_padded[:len(target_normalized)] = target_normalized
        target_normalized = target_padded
    
    target_statevector = Statevector(target_normalized)
    
    return state_fidelity(prepared_state, target_statevector)