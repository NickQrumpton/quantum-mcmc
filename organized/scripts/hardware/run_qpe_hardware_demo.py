#!/usr/bin/env python3
"""
Simple demo script to run QPE experiments on quantum hardware or simulators.

This script provides an easy way to test the QPE implementation with
automatic fallback to simulators if hardware is not available.

Usage:
    python run_qpe_hardware_demo.py [--hardware] [--backend NAME]
"""

import numpy as np
import argparse
from pathlib import Path
import sys
from datetime import datetime

# Check if required modules are available
try:
    from qpe_real_hardware import QPEHardwareExperiment
    from qpe_hardware_advanced import AdvancedQPEExperiment
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError as e:
    print("Error: Required modules not found.")
    print("Please install requirements first:")
    print("  pip install qiskit qiskit-ibm-runtime qiskit-aer matplotlib pandas")
    sys.exit(1)


def check_ibmq_credentials():
    """Check if IBMQ credentials are configured."""
    try:
        service = QiskitRuntimeService(channel="ibm_quantum")
        backends = service.backends()
        return True, service, len(backends)
    except Exception as e:
        return False, None, 0


def run_simulator_demo():
    """Run QPE demo using noisy simulator."""
    print("\n" + "="*60)
    print("Running QPE Demo with Noisy Simulator")
    print("="*60 + "\n")
    
    # Simple 2-state Markov chain
    P = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    
    print("Markov chain transition matrix:")
    print(P)
    print()
    
    # Create experiment with simulator
    experiment = QPEHardwareExperiment(
        transition_matrix=P,
        ancilla_bits=3,
        shots=4096,
        use_simulator=True
    )
    
    # Run QPE
    print("Running QPE experiments on simulator...")
    results = experiment.run_hardware_qpe(
        test_states=['uniform', 'stationary', 'orthogonal'],
        error_mitigation_level=1
    )
    
    # Compare with theory
    print("\nComparing results with theory...")
    comparison = experiment.compare_with_theory(results)
    
    print(f"\nTheoretical eigenvalues: {comparison['theoretical_eigenvalues']}")
    print("\nMeasured phases by state:")
    for state_name, state_data in results['states'].items():
        phases = state_data['phases']
        print(f"\n{state_name} state:")
        for i, phase in enumerate(phases[:3]):  # Top 3 phases
            eigenvalue = np.exp(2j * np.pi * phase)
            print(f"  Phase {i+1}: {phase:.4f} → eigenvalue: {eigenvalue:.4f}")
    
    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(f"qpe_simulator_demo_{timestamp}.png")
    experiment.visualize_results(results, save_path=save_path)
    
    # Save results
    experiment.save_results(results, f"qpe_simulator_demo_{timestamp}.json")
    
    print(f"\nResults saved to:")
    print(f"  - Plot: {save_path}")
    print(f"  - Data: qpe_simulator_demo_{timestamp}.json")


def run_hardware_demo(service, backend_name=None, ancillas=4, repeats=3, shots=4096):
    """Run QPE demo on real quantum hardware."""
    print("\n" + "="*60)
    print("Running QPE Demo on Real Quantum Hardware")
    print("="*60 + "\n")
    
    # 8-cycle Markov chain for publication results
    P = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    
    # Create hardware experiment with specified parameters
    experiment = QPEHardwareExperiment(
        transition_matrix=P,
        ancilla_bits=ancillas,
        shots=shots,
        repeats=repeats,
        use_simulator=False,
        backend_name=backend_name
    )
    
    # Profile backends if no specific backend requested
    if backend_name is None:
        print("Profiling available quantum backends...")
        backend_df = experiment.profile_backends(service, min_qubits=5)
        
        if len(backend_df) == 0:
            print("No suitable quantum backends available.")
            return
        
        # Select least busy backend
        backend_name = backend_df.iloc[0]['Backend']
        print(f"\nSelected backend: {backend_name}")
    
    # Run with basic error mitigation (faster)
    print(f"\nSubmitting job to {backend_name}...")
    print("This may take several minutes depending on queue times...")
    
    try:
        results = experiment.run_with_error_mitigation(
            service,
            backend_name,
            test_states=['uniform'],  # Just one state for demo
            mitigation_strategy='basic'
        )
        
        # Visualize results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(f"qpe_hardware_demo_{backend_name}_{timestamp}.png")
        experiment.visualize_hardware_comparison(results, save_path=save_path)
        
        print(f"\nResults saved to: {save_path}")
        
        # Show key results
        uniform_results = results['states']['uniform']
        print("\nTop measured phases:")
        for i, phase_data in enumerate(uniform_results['phases'][:3]):
            phase = phase_data['phase']
            prob = phase_data['probability']
            eigenvalue = np.exp(2j * np.pi * phase)
            print(f"  {i+1}. Phase: {phase:.4f}, Probability: {prob:.3f}, Eigenvalue: {eigenvalue:.4f}")
            
    except Exception as e:
        print(f"\nError running on hardware: {e}")
        print("Falling back to simulator...")
        run_simulator_demo()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run QPE experiments on quantum hardware or simulators"
    )
    parser.add_argument(
        "--hardware", 
        action="store_true",
        help="Use real quantum hardware (requires IBMQ credentials)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Specific backend name to use (e.g., ibm_brisbane)"
    )
    parser.add_argument(
        "--ancillas",
        type=int,
        default=4,
        help="Number of ancilla qubits for QPE precision (default: 4)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of independent runs for error analysis (default: 3)"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Number of measurement shots per run (default: 4096)"
    )
    
    args = parser.parse_args()
    
    if args.hardware:
        # Check credentials
        has_creds, service, n_backends = check_ibmq_credentials()
        
        if not has_creds:
            print("\nError: IBMQ credentials not found.")
            print("\nTo use real hardware, you need to:")
            print("1. Get a free IBM Quantum account at https://quantum.ibm.com/")
            print("2. Save your API token using:")
            print("\n    from qiskit_ibm_runtime import QiskitRuntimeService")
            print("    QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
            print("\nFalling back to simulator...")
            run_simulator_demo()
        else:
            print(f"\nIBMQ credentials found. {n_backends} backends available.")
            run_hardware_demo(service, args.backend, args.ancillas, args.repeats, args.shots)
    else:
        # Default: use simulator
        run_simulator_demo()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()