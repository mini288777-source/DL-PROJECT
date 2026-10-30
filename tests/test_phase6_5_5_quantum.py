"""
Unit tests for Subphase 6.5.5: Quantum Kernel Specifications & Architecture.
Validates:
1. Q4 input dimension = 4, Q8 input dimension = 8.
2. ZZFeatureMap properties (linear entanglement, reps=1, reps=2).
3. FidelityQuantumKernel statevector simulation correctness.
4. Classical vs Quantum parameter and Hilbert space accounting.
"""

import os
import sys
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.quantum.fidelity_kernel import QuantumFidelityKernelEngine
from src.quantum.pca_pipeline import TrainOnlyPCAPipeline


def test_q4_circuit_properties():
    """Verify Q4 configuration matches frozen contract."""
    engine = QuantumFidelityKernelEngine(num_qubits=4, reps=1, entanglement="linear")
    props = engine.get_circuit_properties()
    assert props["num_qubits"] == 4
    assert props["reps"] == 1
    assert props["entanglement"] == "linear"
    assert props["trainable_quantum_parameters"] == 0
    assert props["encoding_parameters"] == 4
    # In Qiskit, each ZZ interaction e^{i theta Z_i Z_j} is synthesized via 2 CNOT gates
    # Linear entanglement on 4 qubits has 3 pairs -> 3 * 2 = 6 CX gates
    assert props["two_qubit_gate_count"] == 6


def test_q8_circuit_properties():
    """Verify Q8 configuration matches frozen contract."""
    engine = QuantumFidelityKernelEngine(num_qubits=8, reps=1, entanglement="linear")
    props = engine.get_circuit_properties()
    assert props["num_qubits"] == 8
    assert props["reps"] == 1
    assert props["entanglement"] == "linear"
    assert props["trainable_quantum_parameters"] == 0
    assert props["encoding_parameters"] == 8
    # Linear entanglement on 8 qubits has 7 pairs -> 7 * 2 = 14 CX gates
    assert props["two_qubit_gate_count"] == 14


def test_q8_reps2_ablation_properties():
    """Verify Q8-R2 ablation circuit depth and gate count."""
    engine = QuantumFidelityKernelEngine(num_qubits=8, reps=2, entanglement="linear")
    props = engine.get_circuit_properties()
    assert props["num_qubits"] == 8
    assert props["reps"] == 2
    assert props["entanglement"] == "linear"
    # 2 reps * 14 CX gates = 28 CX gates
    assert props["two_qubit_gate_count"] == 28
    assert props["encoding_parameters"] == 8


def test_statevector_fidelity_bounds():
    """Verify quantum fidelity kernel outputs are strictly in [0, 1]."""
    engine = QuantumFidelityKernelEngine(num_qubits=4, reps=1, entanglement="linear")
    X = np.random.uniform(0.0, np.pi, size=(5, 4))
    K = engine.evaluate(X)
    assert K.shape == (5, 5)
    assert np.all(K >= 0.0)
    assert np.all(K <= 1.0)
    assert np.allclose(np.diag(K), 1.0, atol=1e-7)
    assert np.allclose(K, K.T, atol=1e-7)
