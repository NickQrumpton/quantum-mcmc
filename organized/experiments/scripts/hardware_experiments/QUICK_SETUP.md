# Quick Setup Guide for QPE Hardware Experiments

## 🔧 Installation

### Option 1: Quick Install (Recommended)
```bash
cd organized/scripts/hardware/
python install_deps.py
```

### Option 2: Manual Install
```bash
pip install numpy matplotlib pandas seaborn qiskit qiskit-aer qiskit-ibm-runtime
```

### Option 3: Full Requirements
```bash
cd organized/
pip install -r requirements.txt
```

## 🔑 IBM Quantum Setup

1. **Get IBM Quantum Account** (free): https://quantum.ibm.com/
2. **Save your token**:
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN_HERE')
   ```

## 🧪 Quick Tests

### Test Dependencies
```bash
python install_deps.py
```

### Test Setup (No Hardware)
```bash
python run_complete_qpe_experiment.py --dry-run
```

### Test Connection
```bash
python run_qpe_minimal.py --backend ibm_brisbane --ancillas 3 --shots 100
```

## 🚀 Run Experiments

### Complete Publication Experiment
```bash
python run_complete_qpe_experiment.py --backend ibm_brisbane --ancillas 4 --repeats 3 --shots 4096
```

### Quick Hardware Test
```bash
python run_qpe_minimal.py --backend ibm_brisbane --ancillas 3 --shots 1024
```

### Simulator Demo
```bash
python run_qpe_hardware_demo.py --ancillas 4 --repeats 3
```

## 📊 Expected Output

The complete experiment creates:
```
qpe_publication_ibm_brisbane_YYYYMMDD_HHMMSS/
├── EXPERIMENT_REPORT.md
├── data/
│   ├── qpe_results.json
│   └── reflection_results.json  
├── figures/
│   ├── figure1_qpe_histograms.png/.pdf
│   ├── figure2_reflection_error.png/.pdf
│   ├── figure3_circuit_complexity.png/.pdf
│   └── figure4_calibration_summary.png/.pdf
└── supplementary/
    ├── detailed_metrics.csv
    ├── backend_calibration.json
    ├── qpe_circuits_s4.qpy
    └── table1_qpe_phases.tex
```

## ❌ Troubleshooting

### Import Errors
- **seaborn missing**: Run `pip install seaborn`
- **qiskit-ibm-runtime missing**: Run `pip install qiskit-ibm-runtime`
- **Multiple missing**: Run `python install_deps.py`

### Connection Errors
- **No token**: Set up IBM Quantum account and save token
- **Backend not found**: Check available backends with `QiskitRuntimeService().backends()`
- **Queue full**: Try different backend or reduce shots/repeats

### Hardware Errors
- **Circuit too deep**: Reduce ancillas or use minimal script
- **Job timeout**: Reduce shots or check queue status
- **Calibration errors**: Backend properties might be unavailable (non-critical)

## 📞 Support

- Check hardware queue: https://quantum.ibm.com/services/resources
- IBM Quantum docs: https://docs.quantum.ibm.com/
- Qiskit tutorials: https://learning.quantum.ibm.com/

## ✅ Validation

After running, you should see:
- ✅ Phase gap = π/2 rad ≈ 1.2566 (not 0.6928)
- ✅ Stationary state peaks at bin 0
- ✅ Orthogonal state peaks at bin 5  
- ✅ Reflection errors ε(k) ≈ 2^(1-k)
- ✅ Circuit depth ~290-295, CX count ~430-435

## 🎯 Success Criteria

**Main QPE Results:**
- Stationary bin=0 probability > 0.7 (raw), > 0.9 (mitigated)
- Orthogonal bin=5 probability > 0.5 (raw), > 0.8 (mitigated)
- Uniform distribution roughly flat across bins

**Reflection Operator:**
- ε(1) ≈ 1.00 ± 0.01
- ε(2) ≈ 0.50 ± 0.02  
- ε(3) ≈ 0.25 ± 0.02
- ε(4) ≈ 0.125 ± 0.02

**Circuit Metrics:**
- Depth after optimization: 290-300
- CX gates after optimization: 430-440
- All circuits transpile successfully with optimization_level=3