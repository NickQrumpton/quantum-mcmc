# Quantum MCMC Project - Organized Structure

## 📁 Folder Organization

### 🚀 **Quick Start**
To run QPE on real quantum hardware:
```bash
cd organized/scripts/hardware/
python save_ibmq_credentials.py    # Run once to save credentials
python run_qpe_simple.py           # Run QPE experiment
```

---

## 📂 **Directory Structure**

### **organized/scripts/** - All Executable Scripts
```
hardware/                           # ⭐ READY TO USE - Real Quantum Hardware
├── qpe_real_hardware.py           # Basic QPE on real hardware
├── qpe_hardware_advanced.py       # Advanced QPE with error mitigation
├── run_qpe_simple.py              # ⭐ START HERE - Simple working example
├── run_qpe_hardware_demo.py       # Demo with multiple options
├── save_ibmq_credentials.py       # ⭐ RUN FIRST - Save IBM token
├── setup_and_run_qpe.py          # Complete automated setup
└── test_*.py                      # Connection tests

theorem6/                          # Theorem 6 Implementations
├── theorem6_final_implementation.py
├── theorem6_qiskit_complete.py
└── theorem6_8cycle_corrected_study.py

benchmarks/                        # Benchmark Scripts
├── benchmark_classical_vs_quantum.py
├── create_benchmark_plots.py
└── generate_theorem6_figures.py
```

### **organized/results/** - All Results & Data
```
hardware/                          # Real quantum hardware results
└── (Results from hardware runs will be saved here)

benchmarks/                        # Benchmark comparison results
└── quantum_mcmc_results_updated/

theorem6/                          # Theorem 6 validation results
└── theorem6_8cycle_corrected_results/

figures/                           # Key publication figures
├── figure_1_qpe_discrimination.png
├── figure_2_reflection_analysis.png
├── figure_3_complete_summary.png
├── theorem6_parameter_analysis.png
└── theorem6_resource_scaling.png

final_results/                     # Main publication results
├── benchmark_data.csv
├── theorem_6_validation_results.csv
└── crypto_benchmark_results.pdf
```

### **organized/documentation/** - All Documentation
```
guides/                            # Setup & Usage Guides
├── FINAL_FIXES_SUMMARY.md         # ⭐ Latest API fixes
├── IBMQ_SETUP_GUIDE.md           # ⭐ How to setup IBM Quantum
└── HARDWARE_UPDATE_SUMMARY.md     # Hardware compatibility notes

reports/                           # Research Reports
├── FINAL_IMPLEMENTATION_SUMMARY.md
├── PUBLICATION_RESULTS_SUMMARY.md
├── THEOREM6_VALIDATION_COMPLETE_REPORT.md
└── VALIDATION_COMPLETE_SUMMARY.md

api_reference/                     # Technical Documentation
├── api_reference.md
├── theory.md
└── use_cases.md
```

### **organized/src/** - Source Code Package
```
quantum_mcmc/                      # Main Python package
├── classical/                     # Classical MCMC implementations
├── core/                         # Quantum walk & phase estimation
├── utils/                        # Analysis & visualization tools
└── __init__.py

quantum_mcmc.egg-info/            # Package metadata
```

### **organized/tests/** - Test Suite
```
integration/                       # End-to-end tests
├── test_end_to_end_pipeline.py
test_*.py                         # Unit tests
```

### **organized/examples/** - Working Examples
```
simple_2state_mcmc.py             # Simple 2-state example
imhk_lattice_gaussian.py          # IMHK example
benchmark_results.ipynb           # Results notebook
```

### **organized/notebooks/** - Jupyter Notebooks
```
tutorial_quantum_mcmc.ipynb       # Tutorial notebook
qpe_walk_demo.ipynb              # QPE demonstration
spectral_analysis.ipynb          # Spectral analysis
```

---

## 🎯 **What to Use**

### **For Running QPE on Quantum Hardware:**
1. **Setup**: `organized/scripts/hardware/save_ibmq_credentials.py`
2. **Run**: `organized/scripts/hardware/run_qpe_simple.py`
3. **Advanced**: `organized/scripts/hardware/qpe_hardware_advanced.py`

### **For Benchmark Comparisons:**
1. **Scripts**: `organized/scripts/benchmarks/`
2. **Results**: `organized/results/benchmarks/`

### **For Research/Publication:**
1. **Figures**: `organized/results/figures/`
2. **Reports**: `organized/documentation/reports/`
3. **Final Data**: `organized/results/final_results/`

---

## 🗑️ **Archived Files**

Non-essential files moved to `organized/archive/`:
- Old/duplicate implementations
- Intermediate test files
- Development artifacts
- Superseded versions

**Temp files** in `temp_files/`:
- API testing scripts
- Installation helpers

---

## 🚀 **Quick Commands**

```bash
# Setup IBM Quantum credentials (run once)
cd organized/scripts/hardware/
python save_ibmq_credentials.py

# Run QPE on real quantum hardware
python run_qpe_simple.py

# Test connection
python test_qpe_working.py

# Advanced experiment with error mitigation
python qpe_hardware_advanced.py

# View results
ls ../../../results/hardware/
```

---

## 📋 **File Status**

✅ **Ready to Use**: All scripts in `organized/scripts/hardware/`
✅ **Results**: Organized by category in `organized/results/`
✅ **Documentation**: Complete guides in `organized/documentation/`
✅ **Source Code**: Clean package in `organized/src/`
🗂️ **Archived**: Old files safely stored in `organized/archive/`

All files are now organized for easy navigation and clear workflow!