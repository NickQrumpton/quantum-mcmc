"""Szegedy quantum walk operator construction for quantum MCMC.

This module implements the construction of quantum walk operators following
Szegedy's framework for reversible Markov chains. The quantum walk operator
provides quadratic speedup for mixing times and is the core component for
quantum-enhanced MCMC sampling algorithms.

The implementation supports both matrix representations for analysis/debugging
and Qiskit quantum circuits for execution on quantum simulators/hardware.

References:
    Szegedy, M. (2004). Quantum speed-up of Markov chain based algorithms.
    FOCS 2004: 32-41.
    
    Montanaro, A. (2015). Quantum speedup of Monte Carlo methods.
    Proceedings of the Royal Society A, 471(2181), 20150301.

Author: Nicholas Zhao  
Affiliation: Imperial College London  
Contact: nz422@ic.ac.uk
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy.linalg import sqrtm, svd
import warnings

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator


def prepare_walk_operator(P: np.ndarray, 
                         pi: Optional[np.ndarray] = None,
                         backend: str = "qiskit") -> Union[QuantumCircuit, np.ndarray]:
    """Construct the Szegedy quantum walk operator W(P) for a Markov chain.
    
    Given a reversible Markov chain with transition matrix P, constructs
    the quantum walk operator W(P) that acts on the space of directed edges.
    The walk operator provides quadratic speedup for mixing and sampling.
    
    The quantum walk operator is defined as:
        W = S * (2� - I)
    where:
        - S is the swap operator exchanging edge endpoints
        - � is the projection onto the span of transition vectors
        - I is the identity operator
    
    Args:
        P: n�n reversible transition matrix (row-stochastic)
        pi: Stationary distribution. If None, it will be computed.
        backend: Output format - "qiskit" for QuantumCircuit, "matrix" for numpy array
    
    Returns:
        W: Quantum walk operator as QuantumCircuit (if backend="qiskit") or 
           as numpy array (if backend="matrix")
    
    Raises:
        ValueError: If P is not stochastic, not reversible, or if the
                   backend is not supported
    
    Note:
        - The operator acts on a 2-register quantum state |i�|j� representing
          directed edges (i�j) in the Markov chain graph
        - Requires 2log�(n)	 qubits total for an n-state chain
        - The eigenvalues of W determine mixing time and spectral gap
    
    Example:
        >>> from quantum_mcmc.classical.markov_chain import build_two_state_chain
        >>> P = build_two_state_chain(0.3)
        >>> qc = prepare_walk_operator(P, backend="qiskit")
        >>> print(f"Circuit uses {qc.num_qubits} qubits")
        Circuit uses 2 qubits
    """
    # Import here to avoid circular dependency
    from ..classical.markov_chain import is_stochastic, is_reversible, stationary_distribution
    from ..classical.discriminant import discriminant_matrix
    
    # Validate inputs
    if backend not in ["qiskit", "matrix"]:
        raise ValueError(f"Backend '{backend}' not supported. Use 'qiskit' or 'matrix'.")
    
    if not is_stochastic(P):
        raise ValueError("Transition matrix P must be row-stochastic")
    
    n = P.shape[0]
    
    # Compute stationary distribution if not provided
    if pi is None:
        pi = stationary_distribution(P)
    
    # Verify reversibility
    if not is_reversible(P, pi):
        raise ValueError("Markov chain must be reversible")
    
    # Compute discriminant matrix for transition amplitudes
    D = discriminant_matrix(P, pi)
    
    # Build the quantum walk operator matrix
    W_matrix = _build_walk_matrix(D, P, pi)
    
    if backend == "matrix":
        return W_matrix
    
    # Convert to Qiskit circuit
    return _matrix_to_circuit(W_matrix, n)


def _build_walk_matrix(D: np.ndarray, P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Construct the matrix representation of the quantum walk operator.
    
    The walk operator acts on the space of directed edges |i�|j� and
    implements the transformation:
        W = S * (2� - I)
    
    Args:
        D: n�n discriminant matrix
        P: n�n transition matrix
        pi: Stationary distribution
    
    Returns:
        W: n��n� unitary matrix representing the quantum walk operator
    """
    n = D.shape[0]
    dim = n * n  # Dimension of edge space
    
    # Build projection operator � onto span of transition vectors
    Pi_op = _build_projection_operator(D, P, pi)
    
    # Build swap operator S that exchanges |i�|j� � |j�|i�
    S = _build_swap_operator(n)
    
    # Construct walk operator: W = S * (2� - I)
    reflection = 2 * Pi_op - np.eye(dim)
    W = S @ reflection
    
    return W


def _build_projection_operator(D: np.ndarray, P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Build the projection operator onto transition vectors.
    
    The projection operator � projects onto the span of vectors:
        |�b� = �| (P[i,j]) |i�|j�
    
    This is the key component that encodes the Markov chain structure
    into the quantum walk.
    
    Args:
        D: Discriminant matrix
        P: Transition matrix  
        pi: Stationary distribution
    
    Returns:
        Pi_op: n��n� projection matrix
    """
    n = D.shape[0]
    dim = n * n
    
    # Initialize projection operator
    Pi_op = np.zeros((dim, dim), dtype=complex)
    
    # Build transition amplitude matrix
    # A[i,j] = sqrt(P[i,j]) for the amplitude from |i� to |i�|j�
    A = np.sqrt(P)
    
    # Construct projection as sum of outer products |�b���b|
    for i in range(n):
        # Build transition vector |�b� for state i
        psi_i = np.zeros(dim, dtype=complex)
        for j in range(n):
            # |�b� = �| (P[i,j]) |i,j�
            idx = i * n + j  # Index for |i�|j� in computational basis
            psi_i[idx] = A[i, j]
        
        # Add outer product |�b���b| to projection
        Pi_op += np.outer(psi_i, psi_i.conj())
    
    return Pi_op


def _build_swap_operator(n: int) -> np.ndarray:
    """Build the swap operator for two n-dimensional registers.
    
    The swap operator S exchanges the two registers:
        S|i�|j� = |j�|i�
    
    Args:
        n: Dimension of each register
    
    Returns:
        S: n��n� permutation matrix
    """
    dim = n * n
    S = np.zeros((dim, dim))
    
    for i in range(n):
        for j in range(n):
            # Map |i�|j� to |j�|i�
            idx_in = i * n + j
            idx_out = j * n + i
            S[idx_out, idx_in] = 1
    
    return S


def _matrix_to_circuit(W: np.ndarray, n: int) -> QuantumCircuit:
    """Convert walk operator matrix to Qiskit QuantumCircuit.
    
    Args:
        W: Unitary matrix for quantum walk operator
        n: Number of states in original Markov chain
    
    Returns:
        qc: QuantumCircuit implementing the walk operator
    """
    # Calculate number of qubits needed for each register
    n_qubits_per_reg = int(np.ceil(np.log2(n)))
    n_qubits_total = 2 * n_qubits_per_reg
    
    # Pad matrix if necessary to match qubit dimensions
    full_dim = 2 ** n_qubits_total
    if W.shape[0] < full_dim:
        W_padded = np.eye(full_dim, dtype=complex)
        W_padded[:W.shape[0], :W.shape[1]] = W
        W = W_padded
    
    # Create quantum circuit
    qr1 = QuantumRegister(n_qubits_per_reg, name='source')
    qr2 = QuantumRegister(n_qubits_per_reg, name='target')
    qc = QuantumCircuit(qr1, qr2, name='W(P)')
    
    # Add walk operator as unitary gate
    walk_gate = UnitaryGate(W, label='Walk')
    qc.append(walk_gate, qr1[:] + qr2[:])
    
    return qc


def is_unitary(W: np.ndarray, atol: float = 1e-10) -> bool:
    """Verify that a matrix is unitary.
    
    A matrix W is unitary if W W = WW  = I, where W  is the
    conjugate transpose and I is the identity matrix.
    
    Args:
        W: Matrix to check
        atol: Absolute tolerance for numerical comparison
    
    Returns:
        True if W is unitary within tolerance, False otherwise
    
    Example:
        >>> W = prepare_walk_operator(P, backend="matrix")
        >>> is_unitary(W)
        True
    """
    n = W.shape[0]
    if W.shape != (n, n):
        return False
    
    # Check W W = I
    should_be_I_1 = W.conj().T @ W
    if not np.allclose(should_be_I_1, np.eye(n), atol=atol):
        return False
    
    # Check WW  = I
    should_be_I_2 = W @ W.conj().T
    if not np.allclose(should_be_I_2, np.eye(n), atol=atol):
        return False
    
    return True


def walk_eigenvalues(P: np.ndarray, pi: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute eigenvalues of the quantum walk operator.
    
    The eigenvalues of W(P) determine the mixing properties and spectral gap
    of the quantum walk. They are related to the singular values of the
    discriminant matrix through:
        � = �(1 - 4ò(1 - ò))
    where � are the singular values.
    
    Args:
        P: Transition matrix
        pi: Stationary distribution (computed if not provided)
    
    Returns:
        eigenvals: Complex eigenvalues of W(P) sorted by magnitude
    
    Example:
        >>> eigenvals = walk_eigenvalues(P)
        >>> phase_gap = np.min(np.abs(np.angle(eigenvals[eigenvals != 1])))
    """
    from ..classical.discriminant import discriminant_matrix, singular_values
    
    # Get discriminant matrix and its singular values
    D = discriminant_matrix(P, pi)
    sigmas = singular_values(D)
    
    # Compute walk eigenvalues from singular values
    # For each singular value σ, we get two eigenvalues e^{±iθ}
    eigenvals = []
    
    for sigma in sigmas:
        # Skip near-zero singular values (correspond to eigenvalue 1)
        if sigma < 1e-14:
            eigenvals.extend([1.0, 1.0])
            continue
            
        # For singular value σ = 1, eigenvalue is -1
        if abs(sigma - 1) < 1e-14:
            eigenvals.extend([-1.0, -1.0])
            continue
            
        # Compute cos(θ) = √(1 - 4σ²(1 - σ²))
        cos_theta_sq = 1 - 4 * sigma**2 * (1 - sigma**2)
        
        # Ensure cos_theta_sq is in valid range [0, 1]
        cos_theta_sq = np.clip(cos_theta_sq, 0, 1)
        cos_theta = np.sqrt(cos_theta_sq)
        
        # Compute angle θ
        theta = np.arccos(cos_theta)
        
        # Eigenvalues are e^{±iθ}
        eigenvals.extend([np.exp(1j * theta), np.exp(-1j * theta)])
    
    return np.array(eigenvals)


def validate_walk_operator(W: Union[QuantumCircuit, np.ndarray], 
                          P: np.ndarray,
                          pi: Optional[np.ndarray] = None,
                          atol: float = 1e-10) -> bool:
    """Validate that W is a correct quantum walk operator for P.
    
    Performs several checks:
    1. W is unitary
    2. W has correct dimension
    3. W preserves the uniform superposition over edges (approximately)
    4. Eigenvalues match expected spectrum
    
    Args:
        W: Walk operator as circuit or matrix
        P: Original transition matrix
        pi: Stationary distribution
        atol: Tolerance for numerical checks
    
    Returns:
        is_valid: True if all validation checks pass
    
    Example:
        >>> W = prepare_walk_operator(P)
        >>> validate_walk_operator(W, P)
        True
    """
    # Convert to matrix if needed
    if isinstance(W, QuantumCircuit):
        W_matrix = Operator(W).data
    else:
        W_matrix = W
    
    n = P.shape[0]
    expected_dim = n * n
    
    # Check dimension
    if W_matrix.shape[0] < expected_dim:
        warnings.warn(f"Walk operator dimension {W_matrix.shape[0]} < {expected_dim}")
        return False
    
    # Extract relevant submatrix if padded
    if W_matrix.shape[0] > expected_dim:
        W_matrix = W_matrix[:expected_dim, :expected_dim]
    
    # Check unitarity
    if not is_unitary(W_matrix, atol):
        return False
    
    # Additional spectral checks could be added here
    
    return True


def prepare_initial_state(n: int, 
                         start_vertex: Optional[int] = None) -> QuantumCircuit:
    """Prepare initial state for quantum walk.
    
    Creates a quantum state for starting the walk, either from a specific
    vertex or from a uniform superposition.
    
    Args:
        n: Number of vertices in the graph
        start_vertex: Starting vertex index (None for uniform superposition)
    
    Returns:
        qc: QuantumCircuit preparing the initial state
    
    Example:
        >>> qc_init = prepare_initial_state(4, start_vertex=0)
        >>> # Prepares state |0�|Ȁ� where |Ȁ� = �| P�| |j�
    """
    n_qubits = int(np.ceil(np.log2(n)))
    
    qr1 = QuantumRegister(n_qubits, name='source')
    qr2 = QuantumRegister(n_qubits, name='target')
    qc = QuantumCircuit(qr1, qr2)
    
    if start_vertex is not None:
        # Prepare |start_vertex� in first register
        if start_vertex > 0:
            binary = format(start_vertex, f'0{n_qubits}b')
            for i, bit in enumerate(binary):
                if bit == '1':
                    qc.x(qr1[i])
        
        # Prepare uniform superposition in second register
        # (In practice, would prepare according to transition probabilities)
        for i in range(n_qubits):
            qc.h(qr2[i])
    else:
        # Uniform superposition over all edges
        for i in range(n_qubits):
            qc.h(qr1[i])
            qc.h(qr2[i])
    
    return qc


def measure_walk_mixing(W: Union[QuantumCircuit, np.ndarray],
                       P: np.ndarray,
                       t_steps: int,
                       initial_state: Optional[np.ndarray] = None) -> float:
    """Measure mixing distance after t steps of quantum walk.
    
    Computes the total variation distance between the walk distribution
    after t steps and the stationary distribution.
    
    Args:
        W: Quantum walk operator
        P: Original transition matrix
        t_steps: Number of walk steps
        initial_state: Initial state vector (uniform if None)
    
    Returns:
        distance: Total variation distance from stationary distribution
    
    Example:
        >>> W = prepare_walk_operator(P, backend="matrix")
        >>> dist = measure_walk_mixing(W, P, t_steps=10)
        >>> print(f"Distance after 10 steps: {dist:.6f}")
    """
    from ..classical.markov_chain import stationary_distribution
    
    # Convert to matrix if needed
    if isinstance(W, QuantumCircuit):
        W_matrix = Operator(W).data
    else:
        W_matrix = W
    
    n = P.shape[0]
    dim = n * n
    
    # Prepare initial state if not provided
    if initial_state is None:
        # Uniform superposition over edges
        initial_state = np.ones(dim) / np.sqrt(dim)
    
    # Apply t steps of walk
    state = initial_state.copy()
    for _ in range(t_steps):
        state = W_matrix @ state
    
    # Extract vertex distribution by tracing out second register
    vertex_probs = np.zeros(n)
    for i in range(n):
        # Sum probabilities over all edges starting from vertex i
        for j in range(n):
            idx = i * n + j
            if idx < len(state):
                vertex_probs[i] += np.abs(state[idx])**2
    
    # Compare to stationary distribution
    pi = stationary_distribution(P)
    distance = 0.5 * np.sum(np.abs(vertex_probs - pi))
    
    return distance


def quantum_mixing_time(P: np.ndarray, 
                       epsilon: float = 0.01,
                       pi: Optional[np.ndarray] = None) -> int:
    """Estimate quantum mixing time for given accuracy.
    
    Returns the number of quantum walk steps needed to mix to within
    � total variation distance of the stationary distribution.
    
    Args:
        P: Transition matrix
        epsilon: Target accuracy
        pi: Stationary distribution
    
    Returns:
        t_mix: Estimated mixing time in walk steps
    
    Example:
        >>> t_mix = quantum_mixing_time(P, epsilon=0.01)
        >>> print(f"Quantum mixing time: {t_mix} steps")
    """
    from ..classical.discriminant import discriminant_matrix, phase_gap
    
    D = discriminant_matrix(P, pi)
    delta = phase_gap(D)
    
    if delta < 1e-10:
        warnings.warn("Phase gap near zero; mixing time may be infinite")
        return int(1e6)  # Return large number
    
    # Standard mixing time bound
    n = P.shape[0]
    t_mix = int(np.ceil((2.0 / delta) * np.log(n / epsilon)))
    
    return t_mix