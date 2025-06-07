#!/usr/bin/env python3
"""
Check the current Options API structure
"""

from qiskit_ibm_runtime import Options

# Create options object
options = Options()

# Print all attributes
print("Options attributes:")
print(dir(options))

# Try to inspect the structure
print("\nOptions structure:")
for attr in dir(options):
    if not attr.startswith('_'):
        try:
            value = getattr(options, attr)
            print(f"  {attr}: {type(value)} = {value}")
        except:
            print(f"  {attr}: <unable to access>")

# Check if it has model_fields
if hasattr(options, 'model_fields'):
    print("\nModel fields:")
    for field_name, field_info in options.model_fields.items():
        print(f"  {field_name}: {field_info}")