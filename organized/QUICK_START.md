# 🚀 Quick Start Guide

## Run QPE on Real Quantum Hardware in 3 Steps

### Step 1: Setup (One-time only)
```bash
cd scripts/hardware/
python save_ibmq_credentials.py
```

### Step 2: Test Connection
```bash
python test_qpe_working.py
```

### Step 3: Run Experiment
```bash
python run_qpe_simple.py
```

## Expected Output:
```
Setting up IBM Quantum credentials...
✓ Credentials saved successfully!

Connecting to IBM Quantum...
✓ Connected to IBM Quantum service
✓ Selected backend: ibm_brisbane

Running QPE experiment...
Circuit created: 7 depth, 10 gates
Transpiled: 29 depth

Submitting to ibm_brisbane...
Job ID: ch1a2b3c...
Waiting for results...

Results:
Outcome | Count
--------------------
     00 |  512
     01 |  256
     10 |  128
     11 |  128

✓ Success!
```

## What's Available:

### 📁 **scripts/hardware/** - Quantum Hardware
- `run_qpe_simple.py` - ⭐ **Start here**
- `qpe_hardware_advanced.py` - Advanced features
- `save_ibmq_credentials.py` - ⭐ **Setup credentials**

### 📁 **results/** - Your Results
- `hardware/` - Hardware experiment results
- `figures/` - Publication-quality figures
- `final_results/` - Main research results

### 📁 **documentation/** - Help & Guides
- `guides/IBMQ_SETUP_GUIDE.md` - Detailed setup
- `guides/FINAL_FIXES_SUMMARY.md` - Latest updates

## Need Help?
- Connection issues → Check `documentation/guides/IBMQ_SETUP_GUIDE.md`
- API errors → See `documentation/guides/FINAL_FIXES_SUMMARY.md`
- Results analysis → Browse `results/` folders