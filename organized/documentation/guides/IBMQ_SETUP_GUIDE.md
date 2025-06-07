# IBM Quantum Hardware Setup Guide

This guide will help you configure your environment to run QPE experiments on real IBM quantum hardware.

## Prerequisites

1. IBM Quantum account (free tier available)
2. Python 3.8 or higher
3. Qiskit Runtime installed

## Step 1: Create IBM Quantum Account

1. Go to https://quantum.ibm.com/
2. Click "Sign up" and create a free account
3. After logging in, go to your account page

## Step 2: Get Your API Token

1. In IBM Quantum dashboard, click on your profile icon
2. Select "Account settings"
3. Find and copy your API token

## Step 3: Install Required Packages

```bash
pip install qiskit qiskit-ibm-runtime qiskit-aer matplotlib pandas
```

## Step 4: Save Your Credentials

Run this Python code once to save your credentials:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Replace 'YOUR_API_TOKEN' with your actual token
QiskitRuntimeService.save_account(
    channel='ibm_quantum',
    token='YOUR_API_TOKEN',
    overwrite=True
)

# Verify the account is saved
service = QiskitRuntimeService()
print("Available backends:", [b.name for b in service.backends()])
```

## Step 5: Run QPE on Hardware

### Basic Usage

```python
from qpe_real_hardware import QPEHardwareExperiment
import numpy as np

# Define your Markov chain
P = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Create experiment (will use real hardware)
experiment = QPEHardwareExperiment(
    transition_matrix=P,
    ancilla_bits=3,  # Keep small for hardware
    shots=4096,
    use_simulator=False  # Set to False for real hardware
)

# Run experiment
results = experiment.run_hardware_qpe()

# Visualize results
experiment.visualize_results(results)
```

### Advanced Usage with Error Mitigation

```python
from qpe_hardware_advanced import AdvancedQPEExperiment
from qiskit_ibm_runtime import QiskitRuntimeService

# Initialize service
service = QiskitRuntimeService()

# Create advanced experiment
experiment = AdvancedQPEExperiment(
    transition_matrix=P,
    ancilla_bits=3,
    shots=4096
)

# Profile available backends
backend_df = experiment.profile_backends(service)
print(backend_df)

# Choose backend (or let it auto-select)
backend_name = backend_df.iloc[0]['Backend']  # Best backend

# Run with error mitigation
results = experiment.run_with_error_mitigation(
    service,
    backend_name,
    test_states=['uniform', 'stationary'],
    mitigation_strategy='advanced'  # or 'basic', 'comprehensive'
)
```

## Available Backends

### Free Tier (IBM Quantum Network)
- Limited to 10 minutes monthly runtime
- Access to 5-7 qubit devices
- Example backends: `ibmq_lima`, `ibmq_belem`, `ibmq_quito`

### Premium Access
- More qubits (up to 127)
- Better error rates
- Priority queue access

## Hardware Considerations

### 1. Qubit Requirements
- QPE needs: (edge_qubits + ancilla_qubits) total qubits
- For 2-state Markov chain: 2 + ancilla_bits
- For n-state chain: 2*ceil(log2(n)) + ancilla_bits

### 2. Circuit Depth
- Real hardware has limited coherence time
- Keep circuits shallow (depth < 100 preferred)
- Use fewer ancilla bits for hardware

### 3. Error Rates
- Typical CNOT error: 0.5-2%
- Readout error: 1-5%
- Use error mitigation for better results

## Error Mitigation Strategies

### Basic (Level 1)
- Readout error mitigation
- Fast, minimal overhead

### Advanced (Level 2)
- Zero-noise extrapolation (ZNE)
- Readout mitigation
- Better accuracy, more shots needed

### Comprehensive (Level 3)
- ZNE + PEC (Probabilistic Error Cancellation)
- Best accuracy
- Significant overhead (10-100x more shots)

## Troubleshooting

### "No backends available"
- Check your account has access to quantum devices
- Verify your credentials are correct
- Try during off-peak hours

### "Queue timeout"
- Hardware queues can be long
- Use `least_busy()` to find available backend
- Consider using simulator for testing

### "Circuit too deep"
- Reduce ancilla bits
- Use circuit optimization
- Try simpler Markov chains

## Example: Complete Workflow

```python
import numpy as np
from pathlib import Path
from qpe_hardware_advanced import AdvancedQPEExperiment
from qiskit_ibm_runtime import QiskitRuntimeService

# 1. Setup
P = np.array([[0.7, 0.3], [0.4, 0.6]])
service = QiskitRuntimeService()

# 2. Create experiment
exp = AdvancedQPEExperiment(P, ancilla_bits=3, shots=4096)

# 3. Find best backend
backends = exp.profile_backends(service)
best_backend = backends.iloc[0]['Backend']

# 4. Run with error mitigation
results = exp.run_with_error_mitigation(
    service, 
    best_backend,
    test_states=['uniform', 'stationary'],
    mitigation_strategy='advanced'
)

# 5. Visualize and save
exp.visualize_hardware_comparison(
    results,
    save_path=Path('my_qpe_results.png')
)

# 6. Compare with simulator
sim_exp = AdvancedQPEExperiment(P, ancilla_bits=3, shots=4096)
# ... run simulator for comparison
```

## Best Practices

1. **Start Small**: Test with 2-3 ancilla bits first
2. **Use Caching**: Save results for expensive runs
3. **Monitor Queues**: Check backend status before submitting
4. **Batch Jobs**: Group multiple circuits in one submission
5. **Error Mitigation**: Always use at least basic mitigation

## Additional Resources

- [IBM Quantum Documentation](https://docs.quantum.ibm.com/)
- [Qiskit Textbook](https://qiskit.org/textbook/)
- [Error Mitigation Guide](https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials/Error-Suppression-and-Error-Mitigation.html)