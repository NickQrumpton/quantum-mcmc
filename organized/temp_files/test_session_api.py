#!/usr/bin/env python3
"""
Test Session API to find correct syntax
"""

from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2

# Connect
service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.least_busy(simulator=False, operational=True)

print(f"Backend: {backend.name}")

# Check Session constructor
print("\nSession constructor parameters:")
import inspect
sig = inspect.signature(Session.__init__)
print(sig)

# Check SamplerV2 constructor  
print("\nSamplerV2 constructor parameters:")
sig = inspect.signature(SamplerV2.__init__)
print(sig)