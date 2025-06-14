# Quantum Phase Estimation Experiment Report

## Experiment Details
- **Timestamp**: 20250608_184441
- **Markov Chain**: 2×2 reversible
- **Stationary Distribution**: π = [0.57142857 0.42857143]
- **Classical Spectral Gap**: 0.700000
- **QPE Parameters**: 4 ancilla qubits, 8192 shots

## Walk Operator Validation
- **Validation Passed**: False
- **Computed Eigenvalues**: 4 total

## QPE Results Summary

### state_00
- **Top Result**: Bin 1 (phase 0.0625) with probability 0.342
- **Bin 0 Probability**: 0.006
- **Bin 8 Probability**: 0.002
- **Circuit Complexity**: Depth 198, Gates 243

### state_01
- **Top Result**: Bin 1 (phase 0.0625) with probability 0.194
- **Bin 0 Probability**: 0.005
- **Bin 8 Probability**: 0.007
- **Circuit Complexity**: Depth 198, Gates 244

### state_10
- **Top Result**: Bin 1 (phase 0.0625) with probability 0.338
- **Bin 0 Probability**: 0.006
- **Bin 8 Probability**: 0.003
- **Circuit Complexity**: Depth 198, Gates 243

### state_11
- **Top Result**: Bin 11 (phase 0.6875) with probability 0.189
- **Bin 0 Probability**: 0.006
- **Bin 8 Probability**: 0.007
- **Circuit Complexity**: Depth 198, Gates 244

### uniform
- **Top Result**: Bin 11 (phase 0.6875) with probability 0.224
- **Bin 0 Probability**: 0.008
- **Bin 8 Probability**: 0.008
- **Circuit Complexity**: Depth 198, Gates 244


## Files Generated
- **Raw Data**: `data/qpe_results.json`
- **Individual States**: `data/{state_name}_counts.json`
- **Figures**: `figures/qpe_results.{png,pdf}`
- **This Report**: `reports/experiment_report.md`

## Conclusion
⚠️  Experiment completed with validation warnings

Generated by Production QPE Pipeline
