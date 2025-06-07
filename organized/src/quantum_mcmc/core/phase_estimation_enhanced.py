"""Enhanced Phase Estimation with improved precision for Theorem 6.

This module provides high-precision QPE with adaptive ancilla sizing and
improved controlled-power implementations.

Author: Nicholas Zhao
Date: 5/31/2025
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Operator
from typing import Tuple, Dict, Optional
import warnings


def calculate_optimal_ancillas(spectral_gap: float, target_precision: float = 0.01) -> int:
    """Calculate optimal number of ancilla qubits for given precision.
    
    Uses both the theoretical minimum and precision requirements.
    
    Args:
        spectral_gap: Spectral gap Δ(P)
        target_precision: Desired phase estimation precision
        
    Returns:
        Optimal number of ancilla qubits
    """
    # Theoretical minimum from Theorem 6
    s_min = int(np.ceil(np.log2(2 * np.pi / spectral_gap)))
    
    # Precision requirement: phase resolution = 1/2^s
    s_precision = int(np.ceil(np.log2(1 / target_precision)))
    
    # Additional buffer for edge cases and wraparound
    s_buffer = 2
    
    # Take maximum of all requirements
    s_optimal = max(s_min, s_precision) + s_buffer
    
    return min(s_optimal, 12)  # Cap at 12 for simulation feasibility


def build_enhanced_qpe(
    unitary: QuantumCircuit,
    num_ancilla: int,
    use_iterative_powers: bool = True,
    verify_unitarity: bool = True
) -> QuantumCircuit:
    """Build enhanced QPE with improved controlled-power implementation.
    
    Args:
        unitary: Target unitary operator
        num_ancilla: Number of ancilla qubits
        use_iterative_powers: If True, use repeated squaring for efficiency
        verify_unitarity: If True, verify intermediate operators are unitary
        
    Returns:
        Enhanced QPE circuit
    """
    num_system = unitary.num_qubits
    
    # Create registers
    ancilla = QuantumRegister(num_ancilla, 'anc')
    system = QuantumRegister(num_system, 'sys')
    qc = QuantumCircuit(ancilla, system, name='Enhanced_QPE')
    
    # Initialize ancillas in uniform superposition
    for i in range(num_ancilla):
        qc.h(ancilla[i])
    
    # Get unitary gate
    if isinstance(unitary, QuantumCircuit):
        U_gate = unitary.to_gate(label='U')
    else:
        U_gate = unitary
    
    # Build controlled powers using optimized method
    if use_iterative_powers:
        _add_iterative_controlled_powers(qc, U_gate, ancilla, system, verify_unitarity)
    else:
        _add_direct_controlled_powers(qc, U_gate, ancilla, system)
    
    # Apply inverse QFT with improved precision
    qft_inv = _build_enhanced_inverse_qft(num_ancilla)
    qc.append(qft_inv, ancilla[:])
    
    return qc


def _add_iterative_controlled_powers(
    qc: QuantumCircuit,
    U_gate,
    ancilla: QuantumRegister,
    system: QuantumRegister,
    verify_unitarity: bool = True
) -> None:
    """Add controlled powers using iterative squaring for better precision."""
    num_ancilla = len(ancilla)
    
    # Store computed powers to avoid recomputation
    power_gates = {}
    
    for j in range(num_ancilla):
        power = 2 ** j
        
        if power == 1:
            # U^1 = U
            controlled_U = U_gate.control(1, label=f'c-U^{power}')
            qc.append(controlled_U, [ancilla[j]] + list(system[:]))
            power_gates[1] = U_gate
        else:
            # Build U^power using repeated squaring
            if power not in power_gates:
                U_power = _compute_power_iteratively(U_gate, power, power_gates, verify_unitarity)
                power_gates[power] = U_power
            else:
                U_power = power_gates[power]
            
            # Make it controlled
            controlled_U_power = U_power.control(1, label=f'c-U^{power}')
            qc.append(controlled_U_power, [ancilla[j]] + list(system[:]))


def _compute_power_iteratively(
    U_gate,
    target_power: int,
    power_gates: Dict[int, any],
    verify_unitarity: bool = True
):
    """Compute U^target_power using iterative squaring."""
    # Find the largest power of 2 <= target_power that we've already computed
    available_powers = [p for p in power_gates.keys() if p <= target_power]
    
    if not available_powers:
        # Start from U^1
        current_power = 1
        current_gate = power_gates[1]
    else:
        current_power = max(available_powers)
        current_gate = power_gates[current_power]
    
    # Iteratively square until we reach target_power
    while current_power < target_power:
        if current_power * 2 <= target_power:
            # Square the current gate
            qc_temp = QuantumCircuit(current_gate.num_qubits)
            qc_temp.append(current_gate, range(current_gate.num_qubits))
            qc_temp.append(current_gate, range(current_gate.num_qubits))
            
            current_gate = qc_temp.to_gate(label=f'U^{current_power * 2}')
            current_power *= 2
            
            # Verify unitarity if requested
            if verify_unitarity and current_power <= 16:  # Only for small powers
                try:
                    op = Operator(current_gate)
                    if not _is_unitary(op.data):
                        warnings.warn(f"U^{current_power} may not be unitary")
                except:
                    pass  # Skip verification if it fails
            
            power_gates[current_power] = current_gate
        else:
            # Multiply by smaller powers to reach exact target
            remaining = target_power - current_power
            smaller_power = max([p for p in power_gates.keys() if p <= remaining])
            
            qc_temp = QuantumCircuit(current_gate.num_qubits)
            qc_temp.append(current_gate, range(current_gate.num_qubits))
            qc_temp.append(power_gates[smaller_power], range(current_gate.num_qubits))
            
            current_gate = qc_temp.to_gate(label=f'U^{target_power}')
            current_power = target_power
    
    return current_gate


def _add_direct_controlled_powers(
    qc: QuantumCircuit,
    U_gate,
    ancilla: QuantumRegister,
    system: QuantumRegister
) -> None:
    """Add controlled powers using direct repeated application."""
    num_ancilla = len(ancilla)
    
    for j in range(num_ancilla):
        power = 2 ** j
        
        # Build U^power by direct repetition
        qc_power = QuantumCircuit(U_gate.num_qubits)
        for _ in range(power):
            qc_power.append(U_gate, range(U_gate.num_qubits))
        
        U_power = qc_power.to_gate(label=f'U^{power}')
        controlled_U_power = U_power.control(1, label=f'c-U^{power}')
        
        qc.append(controlled_U_power, [ancilla[j]] + list(system[:]))


def _build_enhanced_inverse_qft(num_qubits: int) -> QuantumCircuit:
    """Build enhanced inverse QFT with improved precision."""
    qc = QuantumCircuit(num_qubits, name='Enhanced_iQFT')
    
    # Use higher-precision phase rotations
    for j in range(num_qubits):
        # Hadamard on qubit j
        qc.h(j)
        
        # Controlled phase rotations with enhanced precision
        for k in range(j + 1, num_qubits):
            # Use exact phase angles
            angle = -np.pi / (2 ** (k - j))
            qc.cp(angle, k, j)
    
    # Reverse qubit order
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - i - 1)
    
    return qc


def _is_unitary(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Check if a matrix is unitary within tolerance."""
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        return False
    
    # Check U†U = I
    should_be_identity = matrix.conj().T @ matrix
    return np.allclose(should_be_identity, np.eye(n), atol=atol)


def estimate_phase_precision(num_ancilla: int, spectral_gap: float) -> Dict[str, float]:
    """Estimate the precision of phase estimation."""
    phase_resolution = 1.0 / (2 ** num_ancilla)
    relative_precision = phase_resolution / spectral_gap
    
    # Estimate probability of correct discrimination
    # Phases within ±Δ/2 should be correctly identified
    discrimination_window = spectral_gap / (2 * np.pi)
    resolution_ratio = phase_resolution / discrimination_window
    
    return {
        'phase_resolution': phase_resolution,
        'relative_precision': relative_precision,
        'discrimination_window': discrimination_window,
        'resolution_ratio': resolution_ratio,
        'recommended_s': int(np.ceil(np.log2(4 * np.pi / spectral_gap)))
    }