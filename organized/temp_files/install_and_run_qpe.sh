#!/bin/bash
# Script to install requirements and run QPE on quantum hardware

echo "=================================================="
echo "QPE on IBM Quantum Hardware - Setup and Execution"
echo "=================================================="
echo

# Step 1: Install required packages
echo "Step 1: Installing required packages..."
echo "This may take a few minutes..."
pip3 install numpy matplotlib pandas qiskit qiskit-ibm-runtime qiskit-aer

if [ $? -ne 0 ]; then
    echo "Error: Failed to install packages"
    echo "Please try running manually:"
    echo "  pip3 install numpy matplotlib pandas qiskit qiskit-ibm-runtime qiskit-aer"
    exit 1
fi

echo
echo "✓ Packages installed successfully!"
echo

# Step 2: Run the QPE experiment
echo "Step 2: Running QPE experiment on quantum hardware..."
python3 setup_and_run_qpe.py

echo
echo "Script completed!"