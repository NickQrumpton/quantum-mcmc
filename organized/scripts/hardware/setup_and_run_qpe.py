#!/usr/bin/env python3
"""
Setup IBMQ credentials and run QPE experiment on real quantum hardware.
"""

import numpy as np
from datetime import datetime
from pathlib import Path

# First, save the IBMQ credentials
from qiskit_ibm_runtime import QiskitRuntimeService

# Save the account with the provided token
token = 'a6d61aaedc08e2f162ed1dedb40e9c904fa9352dc6e443a762ea0878e9bc5f5ed01115aede9ddbf041a1cc35d5d7395024233bf5f9b5968a2adcb4fd18ad98ca'

try:
    QiskitRuntimeService.save_account(
        channel='ibm_quantum',
        token=token,
        overwrite=True
    )
    print("✓ IBMQ credentials saved successfully")
except Exception as e:
    print(f"Error saving credentials: {e}")
    exit(1)

# Now run the QPE experiment
from qpe_real_hardware import QPEHardwareExperiment
from qpe_hardware_advanced import AdvancedQPEExperiment

print("\n" + "="*60)
print("Running QPE on IBM Quantum Hardware")
print("="*60 + "\n")

# Define the 2-state Markov chain
P = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

print("Markov chain transition matrix:")
print(P)
print()

# Initialize the service
try:
    service = QiskitRuntimeService(channel='ibm_quantum')
    print("✓ Connected to IBM Quantum service")
    
    # List available backends
    backends = service.backends(simulator=False, operational=True)
    print(f"\nAvailable quantum devices: {len(backends)}")
    for b in backends[:5]:  # Show first 5
        print(f"  - {b.name}: {b.configuration().n_qubits} qubits")
    
except Exception as e:
    print(f"Error connecting to IBMQ: {e}")
    exit(1)

# Create advanced experiment for backend profiling
print("\nProfileing backends to find the best available device...")
experiment = AdvancedQPEExperiment(
    transition_matrix=P,
    ancilla_bits=3,  # Keep small for hardware
    shots=1024  # Reduced shots for faster execution
)

# Profile and select best backend
try:
    backend_df = experiment.profile_backends(service, min_qubits=5)
    
    if len(backend_df) == 0:
        print("No suitable backends available. The queue might be full.")
        print("Trying least busy backend...")
        backend = service.least_busy(simulator=False, operational=True)
        backend_name = backend.name
    else:
        # Select the best backend based on score
        backend_name = backend_df.iloc[0]['Backend']
        print(f"\n✓ Selected backend: {backend_name}")
        print(f"  - Error rate: {backend_df.iloc[0]['CX Error']:.4f}")
        print(f"  - Qubits: {backend_df.iloc[0]['Qubits']}")
    
except Exception as e:
    print(f"Error selecting backend: {e}")
    # Fallback to least busy
    backend = service.least_busy(simulator=False, operational=True)
    backend_name = backend.name
    print(f"Using fallback backend: {backend_name}")

# Run the experiment with error mitigation
print(f"\nSubmitting QPE job to {backend_name}...")
print("This may take 5-20 minutes depending on the queue...")

try:
    # Run with basic error mitigation for faster results
    results = experiment.run_with_error_mitigation(
        service,
        backend_name,
        test_states=['uniform', 'stationary'],
        mitigation_strategy='basic'  # Use basic for faster execution
    )
    
    print("\n✓ Job completed successfully!")
    
    # Display results
    print("\nResults Summary:")
    print("-" * 40)
    
    for state_name, state_data in results['states'].items():
        print(f"\n{state_name.upper()} state results:")
        print(f"  Circuit depth: {state_data['circuit_metadata']['depth']}")
        print(f"  CX gates: {state_data['circuit_metadata']['cx_gates']}")
        
        # Show top phases
        print(f"\n  Top measured phases:")
        for i, phase_data in enumerate(state_data['phases'][:3]):
            phase = phase_data['phase']
            prob = phase_data['probability']
            eigenvalue = np.exp(2j * np.pi * phase)
            print(f"    {i+1}. Phase: {phase:.4f} (probability: {prob:.3f})")
            print(f"       → Eigenvalue: {eigenvalue:.4f}")
    
    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(f"qpe_hardware_results_{backend_name}_{timestamp}.png")
    
    print(f"\nGenerating visualization...")
    experiment.visualize_hardware_comparison(results, save_path=save_path)
    
    # Save detailed results
    import json
    results_file = f"qpe_hardware_results_{backend_name}_{timestamp}.json"
    
    # Convert complex numbers for JSON serialization
    def convert_complex(obj):
        if isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert_complex)
    
    print(f"\n✓ Results saved:")
    print(f"  - Visualization: {save_path}")
    print(f"  - Raw data: {results_file}")
    
    # Compare with theoretical values
    print("\nComparison with Theory:")
    print("-" * 40)
    # For 2-state chain
    p, q = P[0, 1], P[1, 0]
    theoretical_eigenvalues = [1.0, 1 - 2 * np.sqrt(p * q)]
    print(f"Theoretical eigenvalues: {theoretical_eigenvalues}")
    print(f"Phase gap: {2 * np.sqrt(p * q):.4f}")
    
except Exception as e:
    print(f"\nError running on hardware: {e}")
    print("\nTrying with simulator fallback...")
    
    # Fallback to simulator
    sim_experiment = QPEHardwareExperiment(
        transition_matrix=P,
        ancilla_bits=3,
        shots=4096,
        use_simulator=True
    )
    
    results = sim_experiment.run_hardware_qpe(
        test_states=['uniform', 'stationary'],
        error_mitigation_level=1
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(f"qpe_simulator_results_{timestamp}.png")
    sim_experiment.visualize_results(results, save_path=save_path)
    print(f"\nSimulator results saved to: {save_path}")

print("\n" + "="*60)
print("Experiment completed!")
print("="*60)