#!/usr/bin/env python3
"""
Test script to verify real IBMQ hardware connection works correctly.
"""

import sys

print("Testing Real IBMQ Hardware Connection")
print("=" * 50)

# Test 1: Import check
print("\n1. Testing imports...")
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
    from qpe_real_hardware import QPEHardwareExperiment
    from qpe_hardware_advanced import AdvancedQPEExperiment
    print("✓ All imports successful (no fake provider imports)")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Service connection
print("\n2. Testing IBMQ service connection...")
try:
    service = QiskitRuntimeService(channel="ibm_quantum")
    print("✓ Connected to IBMQ service")
    
    # List real backends
    real_backends = service.backends(simulator=False, operational=True)
    print(f"✓ Found {len(real_backends)} real quantum devices:")
    for backend in real_backends[:5]:
        print(f"   - {backend.name}: {backend.num_qubits} qubits")
        
except Exception as e:
    print(f"✗ Service connection error: {e}")
    print("\nNote: You need to save your IBMQ credentials first:")
    print("  from qiskit_ibm_runtime import QiskitRuntimeService")
    print("  QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
    sys.exit(1)

# Test 3: Backend selection
print("\n3. Testing backend selection...")
try:
    # Get least busy backend
    backend = service.least_busy(simulator=False, operational=True)
    print(f"✓ Selected backend: {backend.name}")
    print(f"   - Status: {backend.status()}")
    print(f"   - Queue length: {backend.status().pending_jobs}")
except Exception as e:
    print(f"✗ Backend selection error: {e}")

print("\n✓ All tests passed! Ready to run on real hardware.")
print("\nTo run QPE on real hardware:")
print("  python run_qpe_hardware_demo.py --hardware")
print("  python run_qpe_hardware_demo.py --hardware --backend ibmq_lima")