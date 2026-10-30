#!/usr/bin/env python
"""
Environment validation script for the hybrid quantum-classical
speech recognition project.

Imports all critical libraries, prints versions, reports hardware
availability, and exits nonzero if any required dependency is missing.

Usage:
    python scripts/check_environment.py
"""

import sys

REQUIRED = {
    "Python": None,  # checked separately
    "torch": None,
    "transformers": None,
    "numpy": None,
    "pandas": None,
    "sklearn": None,
    "scipy": None,
    "librosa": None,
    "soundfile": None,
    "matplotlib": None,
    "qiskit": None,
    "qiskit_machine_learning": None,
}

OPTIONAL = [
    "qiskit_aer",
    "pennylane",
]


def main() -> int:
    print("=" * 60)
    print("ENVIRONMENT VALIDATION")
    print("=" * 60)

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"\nPython version: {py_ver}")
    if sys.version_info < (3, 10):
        print("  [FAIL] Python 3.10+ required")
        return 1
    else:
        print("  [OK]")

    # Required libraries
    failures = []
    print("\n--- Required Libraries ---")
    for name in REQUIRED:
        if name == "Python":
            continue
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"{name:30s} {ver:20s} [OK]")
            REQUIRED[name] = ver
        except ImportError:
            print(f"{name:30s} {'MISSING':20s} [FAIL]")
            failures.append(name)

    # Optional libraries
    print("\n--- Optional Libraries ---")
    for name in OPTIONAL:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"{name:30s} {ver:20s} [OK]")
        except ImportError:
            print(f"{name:30s} {'not installed':20s} [SKIP]")

    # Hardware
    print("\n--- Hardware ---")
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available:  {cuda_available}")
    if cuda_available:
        print(f"CUDA device:     {torch.cuda.get_device_name(0)}")
        print(f"CUDA version:    {torch.version.cuda}")
    else:
        print("  (CPU-only mode -- acceptable for this project)")

    # Quantum backend
    print("\n--- Quantum Backend ---")
    try:
        from qiskit.primitives import StatevectorSampler
        print("StatevectorSampler:  available [OK]")
    except ImportError:
        print("StatevectorSampler:  MISSING [WARN]")
        # Not fatal -- FidelityQuantumKernel may use its own default

    try:
        import importlib
        qiskit_aer = importlib.import_module("qiskit_aer")
        _ = getattr(qiskit_aer, "AerSimulator")
        print("AerSimulator:        available [OK]")
    except (ImportError, AttributeError):
        print("AerSimulator:        not installed [SKIP]")

    # Key quantum-ML components
    print("\n--- Quantum ML Components ---")
    try:
        from qiskit.circuit.library import ZZFeatureMap
        print("ZZFeatureMap:            available [OK]")
    except ImportError:
        print("ZZFeatureMap:            MISSING [FAIL]")
        failures.append("ZZFeatureMap")

    try:
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        print("FidelityQuantumKernel:   available [OK]")
    except ImportError:
        print("FidelityQuantumKernel:   MISSING [FAIL]")
        failures.append("FidelityQuantumKernel")

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print(f"ENVIRONMENT VALIDATION: FAIL")
        print(f"Missing: {', '.join(failures)}")
        return 1
    else:
        print("ENVIRONMENT VALIDATION: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
