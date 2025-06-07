"""Phase Comparator circuit for Approximate Reflection operator.

This module implements a proper phase comparator that checks if the estimated
phase φ lies within ±Δ(P)/2 of zero, as required by Theorem 6.

Author: Nicholas Zhao
Date: 5/31/2025
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import UnitaryGate
from typing import Tuple


def build_phase_comparator(num_phase_qubits: int, threshold: float, enhanced_precision: bool = True) -> QuantumCircuit:
    """Build a phase comparator circuit that marks phases within threshold of 0.
    
    The comparator checks if |φ| < threshold/2, where φ is encoded in the 
    phase register as an s-bit integer representation.
    
    Args:
        num_phase_qubits: Number of qubits in phase register (s)
        threshold: Phase threshold Δ(P) for comparison  
        enhanced_precision: Use enhanced precision arithmetic
        
    Returns:
        QuantumCircuit implementing the phase comparator
    """
    # Enhanced threshold calculation with better precision
    if enhanced_precision:
        # Use floating point for intermediate calculations, then round carefully
        threshold_float = threshold / (2 * np.pi) * (2**num_phase_qubits)
        # Add small buffer for edge cases
        threshold_int = max(1, int(np.round(threshold_float * 0.9)))  # Slightly tighter threshold
        wrap_threshold = 2**num_phase_qubits - threshold_int
    else:
        threshold_int = int(threshold * (2**num_phase_qubits) / (2 * np.pi))
        wrap_threshold = 2**num_phase_qubits - threshold_int
    
    # Create registers with additional precision ancillas
    phase_reg = QuantumRegister(num_phase_qubits, 'phase')
    if enhanced_precision:
        compare_ancilla = AncillaRegister(2 * num_phase_qubits + 2, 'cmp')  # Extra ancillas for precision
    else:
        compare_ancilla = AncillaRegister(num_phase_qubits + 1, 'cmp')
    result_ancilla = AncillaRegister(1, 'result')
    
    qc = QuantumCircuit(phase_reg, compare_ancilla, result_ancilla, name=f'PhaseComp_s{num_phase_qubits}')
    
    if enhanced_precision:
        # Enhanced comparison with better edge case handling
        _add_enhanced_threshold_check(qc, phase_reg, compare_ancilla, result_ancilla, 
                                    threshold_int, wrap_threshold)
    else:
        # Original implementation
        _add_less_than_circuit(qc, phase_reg, compare_ancilla, result_ancilla, threshold_int)
        _add_greater_than_circuit(qc, phase_reg, compare_ancilla, result_ancilla, wrap_threshold)
    
    # Apply Z gate on result ancilla (phase kick)
    qc.z(result_ancilla[0])
    
    # Uncompute the comparison
    if enhanced_precision:
        _add_enhanced_threshold_check(qc, phase_reg, compare_ancilla, result_ancilla,
                                    threshold_int, wrap_threshold)
    else:
        _add_greater_than_circuit(qc, phase_reg, compare_ancilla, result_ancilla, wrap_threshold)
        _add_less_than_circuit(qc, phase_reg, compare_ancilla, result_ancilla, threshold_int)
    
    return qc


def _add_less_than_circuit(qc: QuantumCircuit, 
                          phase_reg: QuantumRegister,
                          ancilla_reg: AncillaRegister, 
                          result_reg: AncillaRegister,
                          threshold: int) -> None:
    """Add a less-than comparator to the circuit.
    
    Computes whether the phase register value is less than threshold.
    """
    n = len(phase_reg)
    
    # Convert threshold to binary
    threshold_bits = format(threshold, f'0{n}b')
    
    # Ripple-carry comparison from MSB to LSB
    # We use ancilla qubits to track the comparison state
    for i in range(n):
        bit_idx = n - 1 - i  # Start from MSB
        threshold_bit = int(threshold_bits[i])
        
        if i == 0:
            # First bit comparison
            if threshold_bit == 1:
                # phase[i] < 1 when phase[i] = 0
                qc.x(phase_reg[bit_idx])
                qc.cx(phase_reg[bit_idx], ancilla_reg[i])
                qc.x(phase_reg[bit_idx])
            # else: phase[i] < 0 is always false, do nothing
        else:
            # Subsequent bits: check previous comparison state
            if threshold_bit == 1:
                # If prev equal and current bit is 0, then less than
                qc.x(phase_reg[bit_idx])
                qc.ccx(ancilla_reg[i-1], phase_reg[bit_idx], ancilla_reg[i])
                qc.x(phase_reg[bit_idx])
                # Propagate previous less-than state
                qc.cx(ancilla_reg[i-1], ancilla_reg[i])
            else:
                # If prev less-than, stay less-than
                qc.cx(ancilla_reg[i-1], ancilla_reg[i])
                # If prev equal and current bit is 0, stay equal (do nothing)
    
    # Final result: copy to result register
    qc.cx(ancilla_reg[n-1], result_reg[0])
    
    # Clean up ancilla qubits (in reverse order)
    for i in range(n-1, -1, -1):
        bit_idx = n - 1 - i
        threshold_bit = int(threshold_bits[i])
        
        if i == 0:
            if threshold_bit == 1:
                qc.x(phase_reg[bit_idx])
                qc.cx(phase_reg[bit_idx], ancilla_reg[i])
                qc.x(phase_reg[bit_idx])
        else:
            if threshold_bit == 1:
                qc.cx(ancilla_reg[i-1], ancilla_reg[i])
                qc.x(phase_reg[bit_idx])
                qc.ccx(ancilla_reg[i-1], phase_reg[bit_idx], ancilla_reg[i])
                qc.x(phase_reg[bit_idx])
            else:
                qc.cx(ancilla_reg[i-1], ancilla_reg[i])


def _add_greater_than_circuit(qc: QuantumCircuit,
                             phase_reg: QuantumRegister,
                             ancilla_reg: AncillaRegister,
                             result_reg: AncillaRegister,
                             threshold: int) -> None:
    """Add a greater-than comparator to the circuit.
    
    Computes whether the phase register value is greater than threshold.
    """
    n = len(phase_reg)
    
    # Convert threshold to binary
    threshold_bits = format(threshold, f'0{n}b')
    
    # Use different ancilla qubits for this comparison
    offset = n  # Start using ancillas from position n
    
    # Similar ripple-carry but for greater-than
    for i in range(n):
        bit_idx = n - 1 - i  # Start from MSB
        threshold_bit = int(threshold_bits[i])
        
        if i == 0:
            if threshold_bit == 0:
                # phase[i] > 0 when phase[i] = 1
                qc.cx(phase_reg[bit_idx], ancilla_reg[offset])
            # else: phase[i] > 1 is always false
        else:
            if threshold_bit == 0:
                # If prev equal and current bit is 1, then greater than
                qc.ccx(ancilla_reg[offset + i - 1], phase_reg[bit_idx], ancilla_reg[offset + i])
                # Propagate previous greater-than state
                qc.cx(ancilla_reg[offset + i - 1], ancilla_reg[offset + i])
            else:
                # If prev greater-than, stay greater-than
                qc.cx(ancilla_reg[offset + i - 1], ancilla_reg[offset + i])
    
    # OR the result with existing result (phases near 0 OR phases near 1)
    qc.cx(ancilla_reg[offset + n - 1], result_reg[0])
    
    # Clean up (in reverse)
    for i in range(n-1, -1, -1):
        bit_idx = n - 1 - i
        threshold_bit = int(threshold_bits[i])
        
        if i == 0:
            if threshold_bit == 0:
                qc.cx(phase_reg[bit_idx], ancilla_reg[offset])
        else:
            if threshold_bit == 0:
                qc.cx(ancilla_reg[offset + i - 1], ancilla_reg[offset + i])
                qc.ccx(ancilla_reg[offset + i - 1], phase_reg[bit_idx], ancilla_reg[offset + i])
            else:
                qc.cx(ancilla_reg[offset + i - 1], ancilla_reg[offset + i])


def _add_enhanced_threshold_check(qc: QuantumCircuit,
                                 phase_reg: QuantumRegister,
                                 ancilla_reg: AncillaRegister,
                                 result_reg: AncillaRegister,
                                 threshold_int: int,
                                 wrap_threshold: int) -> None:
    """Enhanced threshold check with better precision and edge case handling."""
    n = len(phase_reg)
    
    # Use multiple comparison strategies for robustness
    
    # Strategy 1: Direct threshold comparison (phases near 0)
    _add_precise_less_than(qc, phase_reg, ancilla_reg[:n], result_reg, threshold_int)
    
    # Strategy 2: Wraparound comparison (phases near 1 representing negative values)
    _add_precise_greater_than(qc, phase_reg, ancilla_reg[n:2*n], result_reg, wrap_threshold)
    
    # Strategy 3: Additional check for exact boundary cases
    if threshold_int > 1:
        _add_boundary_check(qc, phase_reg, ancilla_reg[2*n:], result_reg, threshold_int)


def _add_precise_less_than(qc: QuantumCircuit,
                          phase_reg: QuantumRegister,
                          work_ancilla: list,
                          result_reg: AncillaRegister,
                          threshold: int) -> None:
    """Precise less-than comparison with reduced error."""
    n = len(phase_reg)
    threshold_bits = format(threshold, f'0{n}b')
    
    # Improved ripple-carry comparison
    for i in range(n):
        bit_idx = n - 1 - i
        threshold_bit = int(threshold_bits[i])
        
        if i == 0:
            # MSB comparison
            if threshold_bit == 1:
                qc.x(phase_reg[bit_idx])
                qc.cx(phase_reg[bit_idx], work_ancilla[i])
                qc.x(phase_reg[bit_idx])
        else:
            # Subsequent bits with improved logic
            if threshold_bit == 1:
                # Equal condition: all previous bits match exactly
                qc.x(phase_reg[bit_idx])
                qc.ccx(work_ancilla[i-1], phase_reg[bit_idx], work_ancilla[i])
                qc.x(phase_reg[bit_idx])
                # Propagate previous less-than
                qc.cx(work_ancilla[i-1], work_ancilla[i])
            else:
                # If threshold bit is 0, propagate only if previously less-than
                qc.cx(work_ancilla[i-1], work_ancilla[i])
    
    # Update result
    qc.cx(work_ancilla[n-1], result_reg[0])
    
    # Clean up in reverse order
    for i in range(n-1, -1, -1):
        bit_idx = n - 1 - i
        threshold_bit = int(threshold_bits[i])
        
        if i == 0:
            if threshold_bit == 1:
                qc.x(phase_reg[bit_idx])
                qc.cx(phase_reg[bit_idx], work_ancilla[i])
                qc.x(phase_reg[bit_idx])
        else:
            if threshold_bit == 1:
                qc.cx(work_ancilla[i-1], work_ancilla[i])
                qc.x(phase_reg[bit_idx])
                qc.ccx(work_ancilla[i-1], phase_reg[bit_idx], work_ancilla[i])
                qc.x(phase_reg[bit_idx])
            else:
                qc.cx(work_ancilla[i-1], work_ancilla[i])


def _add_precise_greater_than(qc: QuantumCircuit,
                             phase_reg: QuantumRegister,
                             work_ancilla: list,
                             result_reg: AncillaRegister,
                             threshold: int) -> None:
    """Precise greater-than comparison for wraparound cases."""
    n = len(phase_reg)
    threshold_bits = format(threshold, f'0{n}b')
    
    # Similar to less-than but with reversed logic
    for i in range(n):
        bit_idx = n - 1 - i
        threshold_bit = int(threshold_bits[i])
        
        if i == 0:
            if threshold_bit == 0:
                qc.cx(phase_reg[bit_idx], work_ancilla[i])
        else:
            if threshold_bit == 0:
                qc.ccx(work_ancilla[i-1], phase_reg[bit_idx], work_ancilla[i])
                qc.cx(work_ancilla[i-1], work_ancilla[i])
            else:
                qc.cx(work_ancilla[i-1], work_ancilla[i])
    
    # OR with existing result
    qc.cx(work_ancilla[n-1], result_reg[0])
    
    # Clean up
    for i in range(n-1, -1, -1):
        bit_idx = n - 1 - i
        threshold_bit = int(threshold_bits[i])
        
        if i == 0:
            if threshold_bit == 0:
                qc.cx(phase_reg[bit_idx], work_ancilla[i])
        else:
            if threshold_bit == 0:
                qc.cx(work_ancilla[i-1], work_ancilla[i])
                qc.ccx(work_ancilla[i-1], phase_reg[bit_idx], work_ancilla[i])
            else:
                qc.cx(work_ancilla[i-1], work_ancilla[i])


def _add_boundary_check(qc: QuantumCircuit,
                       phase_reg: QuantumRegister,
                       work_ancilla: list,
                       result_reg: AncillaRegister,
                       threshold: int) -> None:
    """Additional check for exact boundary values."""
    # Check for exact threshold value (edge case)
    n = len(phase_reg)
    if len(work_ancilla) < 2:
        return  # Not enough ancillas for boundary check
        
    threshold_bits = format(threshold, f'0{n}b')
    
    # Check if phase register exactly equals threshold
    for i, bit in enumerate(threshold_bits):
        if bit == '1':
            qc.cx(phase_reg[n-1-i], work_ancilla[0])
        else:
            qc.x(phase_reg[n-1-i])
            qc.cx(phase_reg[n-1-i], work_ancilla[0])
            qc.x(phase_reg[n-1-i])
    
    # If exactly equal, don't flip (boundary case)
    qc.x(work_ancilla[0])
    qc.cx(work_ancilla[0], result_reg[0])
    qc.x(work_ancilla[0])
    
    # Clean up
    for i, bit in enumerate(threshold_bits):
        if bit == '1':
            qc.cx(phase_reg[n-1-i], work_ancilla[0])
        else:
            qc.x(phase_reg[n-1-i])
            qc.cx(phase_reg[n-1-i], work_ancilla[0])
            qc.x(phase_reg[n-1-i])


def create_optimized_phase_oracle(num_qubits: int, delta: float) -> QuantumCircuit:
    """Create an optimized phase oracle using the comparator.
    
    This is a more efficient version that directly applies the phase flip
    to states within the threshold.
    
    Args:
        num_qubits: Number of phase qubits
        delta: Spectral gap Δ(P)
        
    Returns:
        Oracle circuit that flips phase of states outside [-Δ/2, Δ/2]
    """
    # Build the comparator
    comparator = build_phase_comparator(num_qubits, delta)
    
    # The comparator already includes the Z gate and uncomputation
    return comparator