#!/usr/bin/env python3
"""Quick test of the corrected QPE implementation."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from qpe_real_hardware import QPEHardwareExperiment

def test_qpe_corrections():
    """Test that QPE now gives correct results."""
    
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    
    experiment = QPEHardwareExperiment(
        transition_matrix=P,
        ancilla_bits=4,
        shots=4096,
        repeats=1,  # Single run for quick test
        use_simulator=True
    )
    
    print("Testing corrected QPE implementation...")
    print(f"Phase gap: {experiment.phase_gap:.6f} rad (π/2)")
    
    # Test single run
    results = experiment.run_hardware_qpe_single(['stationary', 'orthogonal'])
    
    print("\nResults:")
    for state_name, state_data in results['states'].items():
        print(f"\n{state_name.capitalize()} state:")
        
        counts = state_data['counts']
        total = sum(counts.values())
        
        # Find top 3 outcomes
        sorted_outcomes = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for bitstring, count in sorted_outcomes:
            prob = count / total
            bin_val = int(bitstring[::-1], 2) % 16
            phase = bin_val / 16
            print(f"  Bin {bin_val:2d}: {prob:.3f} (phase = {phase:.3f})")
        
        # Check specific expectations
        if state_name == 'stationary':
            bin_0_count = sum(v for k, v in counts.items() if int(k[::-1], 2) % 16 == 0)
            bin_0_prob = bin_0_count / total
            print(f"  Bin 0 probability: {bin_0_prob:.3f}")
            if bin_0_prob > 0.5:
                print(f"  ✅ SUCCESS: Stationary state peaks at bin 0!")
            else:
                print(f"  ❌ Still not working: bin 0 prob = {bin_0_prob:.3f}")
                
        elif state_name == 'orthogonal':
            bin_8_count = sum(v for k, v in counts.items() if int(k[::-1], 2) % 16 == 8)
            bin_8_prob = bin_8_count / total
            print(f"  Bin 8 probability: {bin_8_prob:.3f}")
            if bin_8_prob > 0.3:
                print(f"  ✅ SUCCESS: Orthogonal state peaks at bin 8!")
            else:
                print(f"  ⚠️  Bin 8 prob = {bin_8_prob:.3f}")

if __name__ == "__main__":
    test_qpe_corrections()