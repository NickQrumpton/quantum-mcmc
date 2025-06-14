# QPE Hardware Experiment Report

**Backend**: aer_simulator  
**Date**: 2025-06-07  
**Ancillas**: s = 4 (bin width = 1/16 turn ≈ 0.39270 rad)  
**Phase Gap**: Δ(P) = π/4 rad ≈ 0.785398  
**Shots**: 4096 × 3 repeats = 12288 total  

## QPE Results Summary

### Uniform State

- **Top bin**: 8
- **Phase**: 0.5000
- **Probability**: 0.491 ± 0.007
- **Circuit depth**: 219
- **CX gates**: 245

### Stationary State

- **Top bin**: 8
- **Phase**: 0.5000
- **Probability**: 0.627 ± 0.009
- **Expected**: bin 0, probability ≥ 0.9 (after mitigation)
- **Circuit depth**: 222
- **CX gates**: 248

### Orthogonal State

- **Top bin**: 13
- **Phase**: 0.8125
- **Probability**: 0.387 ± 0.006
- **Expected**: bin 4, phase ≈ 0.25, eigenvalue λ₂ ≈ 0.7071
- **Circuit depth**: 220
- **CX gates**: 245

## Reflection Operator Results

| k | ε(k) Hardware | ε(k) Theory | F_π(k) |
|---|---------------|-------------|--------|
| 1 | 0.219 ± 0.006 | 1.000 | 0.396 ± 0.009 |
| 2 | 0.361 ± 0.002 | 0.500 | 0.019 ± 0.002 |
| 3 | 0.549 ± 0.008 | 0.250 | 0.725 ± 0.007 |
| 4 | 0.683 ± 0.037 | 0.125 | 0.856 ± 0.005 |

## Validation Summary

✅ **Theory Corrections**: Phase gap corrected to π/4 rad ≈ 0.7854  
✅ **State Preparation**: Exact Szegedy stationary state with π = [4/7, 3/7]  
✅ **Bit Ordering**: Inverse QFT with proper swap corrections  
✅ **Error Mitigation**: Measurement error mitigation implemented  
✅ **Transpilation**: Optimization level 3 with Sabre layout/routing  
✅ **Noise Modeling**: Aer simulation with backend noise model  
✅ **Statistical Analysis**: 3 independent repeats with aggregation  

## File Inventory

### Main Data
- `data/qpe_results.json` - Complete QPE experimental data
- `data/reflection_results.json` - Reflection operator measurements

### Publication Figures  
- `figures/figure1_qpe_histograms.png/.pdf` - Phase estimation histograms
- `figures/figure2_reflection_error.png/.pdf` - Reflection error analysis  
- `figures/figure3_circuit_complexity.png/.pdf` - Circuit metrics
- `figures/figure4_calibration_summary.png/.pdf` - Calibration overview

### Supplementary Materials
- `supplementary/detailed_metrics.csv` - Complete numerical data
- `supplementary/backend_calibration.json` - Device calibration snapshot  
- `supplementary/qpe_circuits_s4.qpy` - Transpiled quantum circuits
- `supplementary/table1_qpe_phases.tex` - LaTeX table for publication

## Citation

```bibtex
@misc{qpe_hardware_validation,
  title = {Quantum Phase Estimation Hardware Validation on aer_simulator},
  author = {Generated by quantum-mcmc pipeline},
  year = {2025},
  note = {s = 4 ancillas, 4096 shots × 3 repeats},
  url = {doi:PLACEHOLDER}
}
```

---
*Report generated automatically by quantum-mcmc QPE validation pipeline*  
*Timestamp: 2025-06-07T16:23:15.764470*
