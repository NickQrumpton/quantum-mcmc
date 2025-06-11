#!/usr/bin/env python3
"""Run discriminant matrix tests directly."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the tests
if __name__ == "__main__":
    from tests.test_discriminant import *
    import unittest
    unittest.main(verbosity=2)