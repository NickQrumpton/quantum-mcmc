#!/usr/bin/env python3
"""
Quick dependency installer for QPE hardware experiments.
This script installs the essential packages needed to run the QPE experiments.
"""

import subprocess
import sys
from pathlib import Path

# Essential packages for QPE hardware experiments
ESSENTIAL_PACKAGES = [
    "numpy>=1.21.0",
    "matplotlib>=3.4.0", 
    "pandas>=1.3.0",
    "seaborn>=0.11.0",
    "qiskit>=0.45.0",
    "qiskit-aer>=0.13.0",
    "qiskit-ibm-runtime>=0.15.0"
]

def install_package(package):
    """Install a single package using pip."""
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def check_package(package_name):
    """Check if a package is already installed."""
    try:
        __import__(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False

def main():
    print("🔧 Installing essential dependencies for QPE hardware experiments...")
    print("=" * 60)
    
    # Check which packages are missing
    missing_packages = []
    
    package_map = {
        "numpy": "numpy",
        "matplotlib": "matplotlib", 
        "pandas": "pandas",
        "seaborn": "seaborn",
        "qiskit": "qiskit",
        "qiskit-aer": "qiskit_aer",
        "qiskit-ibm-runtime": "qiskit_ibm_runtime"
    }
    
    for pkg_name, import_name in package_map.items():
        if not check_package(import_name):
            # Find the full package spec
            for full_pkg in ESSENTIAL_PACKAGES:
                if full_pkg.startswith(pkg_name):
                    missing_packages.append(full_pkg)
                    break
    
    if not missing_packages:
        print("✅ All essential packages are already installed!")
        return
    
    print(f"📦 Found {len(missing_packages)} missing packages:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    
    print("\n🚀 Installing missing packages...")
    
    # Install missing packages
    success_count = 0
    for package in missing_packages:
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 60)
    if success_count == len(missing_packages):
        print("🎉 All packages installed successfully!")
        print("\n🧪 You can now run the QPE experiments:")
        print("  python run_complete_qpe_experiment.py --dry-run")
    else:
        failed_count = len(missing_packages) - success_count
        print(f"⚠️  {success_count}/{len(missing_packages)} packages installed successfully")
        print(f"❌ {failed_count} packages failed to install")
        print("\n🔧 You may need to install failed packages manually:")
        print("  pip install <package_name>")

if __name__ == "__main__":
    main()