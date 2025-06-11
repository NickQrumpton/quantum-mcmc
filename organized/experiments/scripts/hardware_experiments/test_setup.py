#!/usr/bin/env python3
"""
Test IBM Quantum Setup and Dependencies

This script tests your setup and provides guidance for fixing issues.
"""

import sys
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("🔍 Testing imports...")
    
    missing = []
    
    # Test core packages
    try:
        import numpy as np
        print("  ✅ numpy")
    except ImportError:
        missing.append("numpy")
        print("  ❌ numpy")
    
    try:
        import matplotlib.pyplot as plt
        print("  ✅ matplotlib")
    except ImportError:
        missing.append("matplotlib")
        print("  ❌ matplotlib")
    
    try:
        import pandas as pd
        print("  ✅ pandas")
    except ImportError:
        missing.append("pandas")
        print("  ❌ pandas")
    
    try:
        import seaborn as sns
        print("  ✅ seaborn")
    except ImportError:
        missing.append("seaborn")
        print("  ❌ seaborn")
    
    # Test Qiskit
    try:
        import qiskit
        print(f"  ✅ qiskit (version {qiskit.__version__})")
    except ImportError:
        missing.append("qiskit")
        print("  ❌ qiskit")
    
    try:
        import qiskit_aer
        print("  ✅ qiskit-aer")
    except ImportError:
        missing.append("qiskit-aer")
        print("  ❌ qiskit-aer")
    
    try:
        import qiskit_ibm_runtime
        print("  ✅ qiskit-ibm-runtime")
    except ImportError:
        missing.append("qiskit-ibm-runtime")
        print("  ❌ qiskit-ibm-runtime")
    
    # Test local modules
    try:
        from qpe_real_hardware import QPEHardwareExperiment
        print("  ✅ qpe_real_hardware")
    except ImportError as e:
        print(f"  ❌ qpe_real_hardware: {e}")
    
    try:
        from plot_qpe_publication import QPEPublicationPlotter
        print("  ✅ plot_qpe_publication")
    except ImportError as e:
        print(f"  ❌ plot_qpe_publication: {e}")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Fix: python install_deps.py")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def test_ibm_credentials():
    """Test IBM Quantum credentials."""
    print("\n🔑 Testing IBM Quantum credentials...")
    
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        # Try to initialize service
        try:
            service = QiskitRuntimeService()
            print("  ✅ Service initialized (default channel)")
        except Exception as e1:
            try:
                service = QiskitRuntimeService(channel="ibm_quantum")
                print("  ✅ Service initialized (legacy channel)")
            except Exception as e2:
                print(f"  ❌ Service initialization failed")
                print(f"     Default: {e1}")
                print(f"     Legacy: {e2}")
                return False, None
        
        # Try to list backends
        try:
            backends = service.backends()
            print(f"  ✅ Found {len(backends)} backends")
            
            # List operational backends
            operational = [b for b in backends if b.status().operational and not b.configuration().simulator]
            if operational:
                print(f"  ✅ {len(operational)} operational quantum backends:")
                for backend in operational[:5]:  # Show first 5
                    qubits = backend.configuration().n_qubits
                    print(f"     - {backend.name}: {qubits} qubits")
                if len(operational) > 5:
                    print(f"     ... and {len(operational) - 5} more")
            else:
                print("  ⚠️  No operational quantum backends found")
            
            return True, service
            
        except Exception as e:
            print(f"  ❌ Cannot list backends: {e}")
            return False, None
            
    except ImportError:
        print("  ❌ qiskit_ibm_runtime not installed")
        return False, None

def test_specific_backend(service, backend_name="ibm_brisbane"):
    """Test connection to a specific backend."""
    if not service:
        return False
    
    print(f"\n🖥️  Testing connection to {backend_name}...")
    
    try:
        backend = service.backend(backend_name)
        config = backend.configuration()
        
        print(f"  ✅ Connected to {backend.name}")
        print(f"     Qubits: {config.n_qubits}")
        print(f"     Simulator: {config.simulator}")
        
        # Test status
        try:
            status = backend.status()
            print(f"     Operational: {status.operational}")
            print(f"     Jobs in queue: {status.pending_jobs}")
        except Exception:
            print("     Status: unavailable")
        
        # Test properties (this was causing the original error)
        try:
            props = backend.properties()
            if props:
                print(f"     Properties: available")
                # Test gate error extraction safely
                gate_count = 0
                for gate in props.gates:
                    if gate.gate not in ['reset', 'measure', 'delay']:
                        gate_count += 1
                print(f"     Gates with error data: {gate_count}")
            else:
                print(f"     Properties: unavailable")
        except Exception as e:
            print(f"     Properties error: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Cannot connect to {backend_name}: {e}")
        return False

def setup_credentials_guide():
    """Provide credential setup guide."""
    print("\n🔧 IBM Quantum Setup Guide:")
    print("1. Get free account: https://quantum.ibm.com/")
    print("2. Find your API token in Account settings")
    print("3. Save token in Python:")
    print()
    print("   from qiskit_ibm_runtime import QiskitRuntimeService")
    print("   QiskitRuntimeService.save_account(")
    print("       channel='ibm_quantum',")
    print("       token='YOUR_TOKEN_HERE'")
    print("   )")
    print()
    print("4. Test connection: python test_setup.py")

def test_simulator():
    """Test simulator functionality."""
    print("\n🧮 Testing simulator...")
    
    try:
        import numpy as np
        from qpe_real_hardware import QPEHardwareExperiment
        
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        
        exp = QPEHardwareExperiment(
            P, ancilla_bits=3, shots=100, repeats=1, 
            use_simulator=True  # Use simulator
        )
        
        print("  ✅ Simulator initialization successful")
        print(f"     Backend: {exp.backend.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Simulator test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 QPE Hardware Setup Test")
    print("=" * 50)
    
    # Test 1: Imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Fix import issues first, then re-run this test")
        return
    
    # Test 2: IBM Quantum credentials
    creds_ok, service = test_ibm_credentials()
    
    if creds_ok:
        # Test 3: Specific backend
        backend_ok = test_specific_backend(service)
    else:
        setup_credentials_guide()
        backend_ok = False
    
    # Test 4: Simulator
    sim_ok = test_simulator()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"  Imports: {'✅' if imports_ok else '❌'}")
    print(f"  IBM Credentials: {'✅' if creds_ok else '❌'}")
    print(f"  Backend Access: {'✅' if backend_ok else '❌'}")
    print(f"  Simulator: {'✅' if sim_ok else '❌'}")
    
    if imports_ok and sim_ok:
        print("\n🎉 You can run simulator experiments!")
        print("   python run_qpe_hardware_demo.py --ancillas 3")
    
    if imports_ok and creds_ok and backend_ok:
        print("\n🎉 You can run hardware experiments!")
        print("   python run_complete_qpe_experiment.py --backend ibm_brisbane --ancillas 3 --repeats 1 --shots 1024")
    
    if not creds_ok:
        print("\n🔧 Set up IBM Quantum credentials to access hardware")

if __name__ == "__main__":
    main()