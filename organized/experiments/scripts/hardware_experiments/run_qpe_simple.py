#!/usr/bin/env python3
"""
Simple QPE script that works with current Qiskit IBM Runtime API.
"""

import numpy as np
from datetime import datetime
from pathlib import Path

# Step 1: Save credentials
print("Setting up IBM Quantum credentials...")
from qiskit_ibm_runtime import QiskitRuntimeService

token = 'a6d61aaedc08e2f162ed1dedb40e9c904fa9352dc6e443a762ea0878e9bc5f5ed01115aede9ddbf041a1cc35d5d7395024233bf5f9b5968a2adcb4fd18ad98ca'

try:
    QiskitRuntimeService.save_account(
        channel='ibm_quantum',
        token=token,
        overwrite=True
    )
    print("✓ Credentials saved successfully!")
except Exception as e:
    print(f"Error saving credentials: {e}")

# Step 2: Connect to service
print("\nConnecting to IBM Quantum...")
try:
    service = QiskitRuntimeService(channel='ibm_quantum')
    print("✓ Connected to IBM Quantum service")
    
    # List available backends
    backends = service.backends(simulator=False, operational=True)
    print(f"\nAvailable quantum devices: {len(backends)}")
    for b in backends[:5]:
        print(f"  - {b.name}: {b.num_qubits} qubits")
    
    # Select backend
    backend = service.least_busy(simulator=False, operational=True)
    print(f"\n✓ Selected backend: {backend.name}")
    
except Exception as e:
    print(f"Error connecting to IBM Quantum: {e}")
    print("\nFalling back to simulator...")
    backend = None

# Step 3: Run simple QPE experiment
if backend:
    print("\nRunning QPE experiment...")
    
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
    from qiskit.circuit.library import QFT
    
    # Create simple 2-qubit QPE circuit
    ancilla = QuantumRegister(2, 'ancilla')
    work = QuantumRegister(2, 'work')
    c = ClassicalRegister(2, 'c')
    
    qc = QuantumCircuit(ancilla, work, c)
    
    # Initialize work register in superposition
    qc.h(work[0])
    qc.h(work[1])
    
    # QPE: Hadamard on ancilla
    qc.h(ancilla[0])
    qc.h(ancilla[1])
    
    # Simple controlled operations (placeholder for walk operator)
    qc.cp(np.pi/4, ancilla[0], work[0])
    qc.cp(np.pi/2, ancilla[1], work[1])
    
    # Inverse QFT on ancilla
    qc.h(ancilla[1])
    qc.cp(-np.pi/2, ancilla[0], ancilla[1])
    qc.h(ancilla[0])
    qc.swap(ancilla[0], ancilla[1])
    
    # Measure
    qc.measure(ancilla, c)
    
    print(f"Circuit depth: {qc.depth()}")
    print(f"Circuit gates: {qc.size()}")
    
    # Transpile for backend
    transpiled = transpile(qc, backend, optimization_level=3)
    print(f"Transpiled depth: {transpiled.depth()}")
    
    # Run on hardware
    print("\nSubmitting job to quantum device...")
    
    # For SamplerV2, we pass shots directly to run()
    shots = 1024
    
    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        
        # Run with shots parameter
        job = sampler.run([transpiled], shots=shots)
        print(f"Job ID: {job.job_id()}")
        print("Waiting for results...")
        
        result = job.result()
        
        # Get counts from SamplerV2 result
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        
        print("\nResults:")
        print("Bitstring | Count")
        print("-" * 25)
        total = sum(counts.values())
        for bitstring, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            prob = count / total
            print(f"{bitstring:>9} | {count:4d} ({prob:.3f})")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"qpe_results_{backend.name}_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"QPE Results on {backend.name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Shots: 1024\n\n")
            f.write("Bitstring | Count | Probability\n")
            f.write("-" * 35 + "\n")
            for bitstring, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                prob = count / total
                f.write(f"{bitstring:>9} | {count:4d} | {prob:.4f}\n")
        
        print(f"\n✓ Results saved to: {results_file}")
        
else:
    print("\nNo quantum backend available. Please check your credentials and network connection.")