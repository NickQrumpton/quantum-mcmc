#!/usr/bin/env python3
"""
Script to save IBM Quantum credentials.
Run this once to save your token.
"""

from qiskit_ibm_runtime import QiskitRuntimeService

# Your IBM Quantum token
token = 'a6d61aaedc08e2f162ed1dedb40e9c904fa9352dc6e443a762ea0878e9bc5f5ed01115aede9ddbf041a1cc35d5d7395024233bf5f9b5968a2adcb4fd18ad98ca'

# Save the account
try:
    QiskitRuntimeService.save_account(
        channel='ibm_quantum',
        token=token,
        overwrite=True
    )
    print("✓ IBM Quantum credentials saved successfully!")
    
    # Test the connection
    service = QiskitRuntimeService(channel='ibm_quantum')
    backends = service.backends()
    
    print(f"\n✓ Successfully connected to IBM Quantum")
    print(f"  Found {len(backends)} backends")
    
    # Show available real devices
    real_backends = [b for b in backends if not b.configuration().simulator]
    print(f"\nAvailable real quantum devices:")
    for backend in real_backends[:10]:
        print(f"  - {backend.name}: {backend.num_qubits} qubits")
        
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nPossible issues:")
    print("1. Invalid token")
    print("2. Network connection issues")
    print("3. IBM Quantum service is down")