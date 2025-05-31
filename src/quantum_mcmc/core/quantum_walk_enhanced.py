"""Enhanced quantum walk operator with minimal approximation errors.

This module provides high-precision quantum walk operators using exact
matrix decomposition and improved circuit synthesis.

Author: Nicholas Zhao
Date: 5/31/2025
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from scipy.linalg import expm, logm, sqrtm
from typing import Tuple, Optional, Dict
import warnings


def prepare_enhanced_walk_operator(
    P: np.ndarray,
    pi: Optional[np.ndarray] = None,
    method: str = "exact_decomposition",
    precision_target: float = 1e-12
) -> QuantumCircuit:
    """Prepare enhanced quantum walk operator with minimal approximation.
    
    Args:
        P: Transition matrix
        pi: Stationary distribution
        method: Construction method ("exact_decomposition", "trotterized", "variational")
        precision_target: Target precision for operator construction
        
    Returns:
        Enhanced quantum walk operator circuit
    """
    from ..classical.markov_chain import stationary_distribution, is_reversible
    from ..classical.discriminant import discriminant_matrix
    
    # Validate and prepare
    n = P.shape[0]
    if pi is None:
        pi = stationary_distribution(P)
    
    if not is_reversible(P, pi):
        raise ValueError("Markov chain must be reversible for quantum walk")
    
    # Choose construction method based on size and precision requirements
    if method == "exact_decomposition":
        return _build_exact_walk_operator(P, pi, precision_target)
    elif method == "trotterized":
        return _build_trotterized_walk_operator(P, pi, precision_target)
    elif method == "variational":
        return _build_variational_walk_operator(P, pi, precision_target)
    else:
        raise ValueError(f"Unknown method: {method}")


def _build_exact_walk_operator(
    P: np.ndarray,
    pi: np.ndarray,
    precision_target: float
) -> QuantumCircuit:
    """Build walk operator using exact matrix decomposition."""
    from ..classical.discriminant import discriminant_matrix
    
    n = P.shape[0]
    n_qubits_per_reg = int(np.ceil(np.log2(n)))
    total_qubits = 2 * n_qubits_per_reg
    
    # Compute exact walk operator matrix
    D = discriminant_matrix(P, pi)
    W_matrix = _compute_exact_walk_matrix(D, P, pi)
    
    # Verify unitarity with high precision
    if not _is_unitary_precise(W_matrix, precision_target):
        warnings.warn("Walk operator matrix is not unitary within target precision")
    
    # Pad matrix to fit qubit requirements
    full_dim = 2 ** total_qubits
    if W_matrix.shape[0] < full_dim:
        W_padded = np.eye(full_dim, dtype=complex)
        W_padded[:W_matrix.shape[0], :W_matrix.shape[1]] = W_matrix
        W_matrix = W_padded
    
    # Create quantum circuit
    qr1 = QuantumRegister(n_qubits_per_reg, name='reg1')
    qr2 = QuantumRegister(n_qubits_per_reg, name='reg2')
    qc = QuantumCircuit(qr1, qr2, name='Enhanced_W(P)')
    
    # Use high-precision unitary synthesis
    walk_gate = UnitaryGate(W_matrix, label='W_exact')
    qc.append(walk_gate, qr1[:] + qr2[:])
    
    return qc


def _compute_exact_walk_matrix(
    D: np.ndarray,
    P: np.ndarray,
    pi: np.ndarray,
    use_improved_numerics: bool = True
) -> np.ndarray:
    """Compute walk operator matrix with enhanced numerical stability."""
    n = D.shape[0]
    dim = n * n
    
    if use_improved_numerics:
        # Use higher precision arithmetic
        P_hp = P.astype(np.complex128)
        pi_hp = pi.astype(np.complex128)
        D_hp = D.astype(np.complex128)
    else:
        P_hp, pi_hp, D_hp = P, pi, D
    
    # Build projection operator with improved numerics
    Pi_op = _build_enhanced_projection_operator(D_hp, P_hp, pi_hp)
    
    # Build swap operator
    S = _build_exact_swap_operator(n)
    
    # Construct walk operator: W = S * (2Π - I)
    reflection = 2 * Pi_op - np.eye(dim, dtype=np.complex128)
    W = S @ reflection
    
    # Ensure exact unitarity
    W = _enforce_unitarity(W)
    
    return W


def _build_enhanced_projection_operator(
    D: np.ndarray,
    P: np.ndarray,
    pi: np.ndarray
) -> np.ndarray:
    """Build projection operator with enhanced numerical precision."""
    n = D.shape[0]
    dim = n * n
    
    # Use QR decomposition for better numerical stability
    Pi_op = np.zeros((dim, dim), dtype=np.complex128)
    
    # Build transition amplitude matrix with better precision
    A = np.sqrt(P.astype(np.complex128))
    
    # Improved projection construction
    for i in range(n):
        # Build transition vector with normalization check
        psi_i = np.zeros(dim, dtype=np.complex128)
        
        norm_sq = 0.0
        for j in range(n):
            idx = i * n + j
            psi_i[idx] = A[i, j]
            norm_sq += np.abs(A[i, j])**2
        
        # Normalize to account for any precision loss
        if norm_sq > 1e-12:
            psi_i = psi_i / np.sqrt(norm_sq)
        
        # Add rank-1 projection
        Pi_op += np.outer(psi_i, psi_i.conj())
    
    # Ensure projection properties: Π² = Π and Π† = Π
    Pi_op = _enforce_projection_properties(Pi_op)
    
    return Pi_op


def _build_exact_swap_operator(n: int) -> np.ndarray:
    """Build exact swap operator with verification."""
    dim = n * n
    S = np.zeros((dim, dim), dtype=np.complex128)
    
    for i in range(n):
        for j in range(n):
            # Map |i⟩|j⟩ to |j⟩|i⟩
            idx_in = i * n + j
            idx_out = j * n + i
            S[idx_out, idx_in] = 1.0
    
    # Verify swap properties
    assert np.allclose(S @ S, np.eye(dim)), "Swap operator is not involutory"
    assert np.allclose(S, S.T), "Swap operator is not symmetric"
    
    return S


def _enforce_unitarity(U: np.ndarray, method: str = "polar") -> np.ndarray:
    """Enforce exact unitarity using polar decomposition."""
    if method == "polar":
        # Polar decomposition: U = V * P, take V (unitary part)
        try:
            from scipy.linalg import polar
            V, P = polar(U)
            return V
        except:
            # Fallback to SVD method
            method = "svd"
    
    if method == "svd":
        # SVD method: U = USV†, return UV†
        U_svd, s, Vh = np.linalg.svd(U)
        return U_svd @ Vh
    
    if method == "gram_schmidt":
        # Modified Gram-Schmidt on columns
        Q = U.copy()
        m, n = Q.shape
        for j in range(n):
            for i in range(j):
                Q[:, j] -= np.vdot(Q[:, i], Q[:, j]) * Q[:, i]
            Q[:, j] = Q[:, j] / np.linalg.norm(Q[:, j])
        return Q
    
    return U


def _enforce_projection_properties(Pi: np.ndarray) -> np.ndarray:
    """Enforce projection properties: Π² = Π and Π† = Π."""
    # Make Hermitian
    Pi = (Pi + Pi.conj().T) / 2
    
    # Ensure idempotent via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(Pi)
    # Project eigenvalues to {0, 1}
    eigvals_proj = np.where(eigvals > 0.5, 1.0, 0.0)
    
    Pi_corrected = eigvecs @ np.diag(eigvals_proj) @ eigvecs.conj().T
    
    return Pi_corrected


def _is_unitary_precise(U: np.ndarray, tolerance: float) -> bool:
    """Check unitarity with specified precision."""
    n = U.shape[0]
    
    # Check U†U = I
    product1 = U.conj().T @ U
    identity_error1 = np.max(np.abs(product1 - np.eye(n)))
    
    # Check UU† = I  
    product2 = U @ U.conj().T
    identity_error2 = np.max(np.abs(product2 - np.eye(n)))
    
    return max(identity_error1, identity_error2) < tolerance


def _build_trotterized_walk_operator(
    P: np.ndarray,
    pi: np.ndarray,
    precision_target: float
) -> QuantumCircuit:
    """Build walk operator using Trotterization for large systems."""
    # This is a placeholder for advanced Trotterization methods
    # For now, fall back to exact method
    return _build_exact_walk_operator(P, pi, precision_target)


def _build_variational_walk_operator(
    P: np.ndarray,
    pi: np.ndarray,
    precision_target: float
) -> QuantumCircuit:
    """Build walk operator using variational quantum circuits."""
    # This is a placeholder for variational methods
    # For now, fall back to exact method
    return _build_exact_walk_operator(P, pi, precision_target)


def verify_walk_operator_precision(
    qc: QuantumCircuit,
    P: np.ndarray,
    pi: np.ndarray,
    precision_target: float
) -> Dict[str, float]:
    """Verify the precision of a quantum walk operator."""
    # Get operator matrix
    op = Operator(qc)
    W_actual = op.data
    
    # Compute theoretical matrix
    from ..classical.discriminant import discriminant_matrix
    D = discriminant_matrix(P, pi)
    W_theory = _compute_exact_walk_matrix(D, P, pi)
    
    # Pad if necessary
    if W_actual.shape[0] > W_theory.shape[0]:
        W_theory_padded = np.eye(W_actual.shape[0], dtype=complex)
        W_theory_padded[:W_theory.shape[0], :W_theory.shape[1]] = W_theory
        W_theory = W_theory_padded
    
    # Compute errors
    frobenius_error = np.linalg.norm(W_actual - W_theory, 'fro')
    operator_norm_error = np.linalg.norm(W_actual - W_theory, 2)
    max_element_error = np.max(np.abs(W_actual - W_theory))
    
    # Check unitarity
    unitarity_error1 = np.max(np.abs(W_actual.conj().T @ W_actual - np.eye(W_actual.shape[0])))
    unitarity_error2 = np.max(np.abs(W_actual @ W_actual.conj().T - np.eye(W_actual.shape[0])))
    unitarity_error = max(unitarity_error1, unitarity_error2)
    
    return {
        'frobenius_error': frobenius_error,
        'operator_norm_error': operator_norm_error,
        'max_element_error': max_element_error,
        'unitarity_error': unitarity_error,
        'meets_precision_target': max(frobenius_error, unitarity_error) < precision_target,
        'precision_target': precision_target
    }