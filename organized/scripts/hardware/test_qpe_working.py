#!/usr/bin/env python3
"""
Minimal test to verify QPE works on hardware
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler

print("Testing QPE on IBM Quantum Hardware")
print("=" * 50)

# Connect to service
try:
    service = QiskitRuntimeService(channel='ibm_quantum')
    backend = service.least_busy(simulator=False, operational=True)
    print(f"✓ Connected to: {backend.name}")
except Exception as e:
    print(f"✗ Connection error: {e}")
    exit(1)

# Create minimal QPE circuit
print("\nCreating QPE circuit...")
q = QuantumRegister(3, 'q')
c = ClassicalRegister(2, 'c')
qc = QuantumCircuit(q, c)

# Simple QPE pattern
qc.h(q[0])
qc.h(q[1])
qc.x(q[2])  # Target state |1>

# Controlled operations
qc.cp(np.pi/2, q[0], q[2])
qc.cp(np.pi, q[1], q[2])

# Simplified inverse QFT
qc.h(q[1])
qc.cp(-np.pi/2, q[0], q[1])
qc.h(q[0])

# Measure
qc.measure([q[0], q[1]], c)

print(f"Circuit created: {qc.depth()} depth, {qc.size()} gates")

# Transpile
transpiled = transpile(qc, backend, optimization_level=3)
print(f"Transpiled: {transpiled.depth()} depth")

# Run on hardware
print(f"\nSubmitting to {backend.name}...")
with Session(backend=backend) as session:
    sampler = Sampler(mode=session)
    
    # Run with shots
    job = sampler.run([transpiled], shots=1024)
    print(f"Job ID: {job.job_id()}")
    
    # Wait for results
    print("Waiting for results...")
    result = job.result()
    
    # Get counts
    pub_result = result[0]
    counts = pub_result.data.c.get_counts()
    
    print("\nResults:")
    print("Outcome | Count")
    print("-" * 20)
    for outcome, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{outcome:>7} | {count:4d}")

print("\n✓ Success!")