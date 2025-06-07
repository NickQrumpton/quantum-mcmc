#!/usr/bin/env python3
"""
Test script to check Qiskit installation and IBMQ connection.
"""

import sys

print("Testing Qiskit installation...")
print("-" * 40)

# Check Python version
print(f"Python version: {sys.version}")

# Test imports
modules_to_test = [
    'numpy',
    'matplotlib',
    'qiskit',
    'qiskit_ibm_runtime',
    'qiskit_aer'
]

missing_modules = []
available_modules = []

for module in modules_to_test:
    try:
        __import__(module)
        available_modules.append(module)
        print(f"✓ {module} is installed")
    except ImportError:
        missing_modules.append(module)
        print(f"✗ {module} is NOT installed")

if missing_modules:
    print("\nMissing required modules!")
    print("Please install them using:")
    print(f"pip install {' '.join(missing_modules)}")
else:
    print("\n✓ All required modules are installed!")
    
    # Try to connect to IBMQ
    print("\nTesting IBMQ connection...")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        # Try with the provided token
        token = 'a6d61aaedc08e2f162ed1dedb40e9c904fa9352dc6e443a762ea0878e9bc5f5ed01115aede9ddbf041a1cc35d5d7395024233bf5f9b5968a2adcb4fd18ad98ca'
        
        # Save account
        QiskitRuntimeService.save_account(
            channel='ibm_quantum',
            token=token,
            overwrite=True
        )
        
        # Test connection
        service = QiskitRuntimeService(channel='ibm_quantum')
        backends = list(service.backends())
        
        print(f"✓ Successfully connected to IBM Quantum!")
        print(f"  Available backends: {len(backends)}")
        
        # Show some backends
        for backend in backends[:5]:
            print(f"  - {backend.name}")
            
    except Exception as e:
        print(f"✗ Error connecting to IBMQ: {e}")
        print("\nThis might be due to:")
        print("1. Invalid token")
        print("2. Network issues")
        print("3. IBMQ service downtime")