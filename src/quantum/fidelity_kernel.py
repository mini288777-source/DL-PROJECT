"""
Quantum Fidelity Kernel Engine.

Constructs ZZ Pauli feature maps, computes state-fidelity quantum kernels
via local ideal statevector simulation, performs matrix validation,
and manages persistent caching.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector


class QuantumFidelityKernelEngine:
    """
    Engine for evaluating, validating, and caching Fidelity Quantum Kernels
    generated from ZZ second-order Pauli feature maps.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        reps: int = 1,
        entanglement: str = "linear",
    ):
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement

        # Build feature map circuit
        self.feature_map = zz_feature_map(
            feature_dimension=self.num_qubits,
            reps=self.reps,
            entanglement=self.entanglement,
        )

        # Precompute circuit properties
        ops = self.feature_map.count_ops()
        self.two_qubit_gate_count = int(ops.get("cx", 0))
        self.one_qubit_gate_count = int(sum(v for k, v in ops.items() if k != "cx"))
        self.total_gate_count = int(sum(ops.values()))
        self.circuit_depth = int(self.feature_map.depth())
        self.trainable_quantum_parameters = 0  # Fixed non-variational feature map
        self.encoding_parameters = int(self.feature_map.num_parameters)

    def compute_statevectors(
        self,
        X: np.ndarray,
        chunk_size: int = 100,
    ) -> np.ndarray:
        """
        Compute statevector representation Psi for each coordinate row in X.

        Args:
            X: Quantum coordinate array of shape (M, num_qubits).
            chunk_size: Number of samples to process per batch.

        Returns:
            Complex statevector array of shape (M, 2^num_qubits).
        """
        M = X.shape[0]
        dim_hilbert = 2**self.num_qubits
        statevectors = np.zeros((M, dim_hilbert), dtype=np.complex128)

        for start_idx in range(0, M, chunk_size):
            end_idx = min(start_idx + chunk_size, M)
            for i in range(start_idx, end_idx):
                bound_circuit = self.feature_map.assign_parameters(X[i])
                sv = Statevector.from_instruction(bound_circuit)
                statevectors[i] = sv.data

        return statevectors

    def evaluate(
        self,
        X_left: np.ndarray,
        X_right: Optional[np.ndarray] = None,
        chunk_size: int = 100,
    ) -> np.ndarray:
        """
        Evaluate quantum fidelity kernel matrix K(X_left, X_right).
        If X_right is None, computes symmetric training matrix K(X_left, X_left).

        Args:
            X_left: Array of shape (M, num_qubits).
            X_right: Array of shape (P, num_qubits) or None.
            chunk_size: Batch size for statevector computation.

        Returns:
            Kernel matrix of shape (M, P) or (M, M).
        """
        is_symmetric = X_right is None

        psi_left = self.compute_statevectors(X_left, chunk_size=chunk_size)
        if is_symmetric:
            psi_right = psi_left
        else:
            psi_right = self.compute_statevectors(X_right, chunk_size=chunk_size)

        # Transition amplitude overlap matrix: S = psi_left @ psi_right^\dagger
        # Fidelity: K_ij = |<psi_i | psi_j>|^2 = |S_ij|^2
        amplitude_matrix = psi_left @ psi_right.conj().T
        K = np.abs(amplitude_matrix) ** 2

        if is_symmetric:
            # Enforce exact mathematical symmetry and diagonal = 1.0
            K = 0.5 * (K + K.T)
            np.fill_diagonal(K, 1.0)

        # Numerical bounds cleanup
        K = np.clip(K, 0.0, 1.0)
        return K

    def validate_kernel_matrix(
        self,
        K: np.ndarray,
        is_training: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform matrix diagnostics on a computed quantum kernel.

        Args:
            K: Kernel matrix.
            is_training: Whether K is the square training kernel matrix.

        Returns:
            Dictionary of diagnostic metrics.
        """
        is_finite = bool(np.isfinite(K).all())
        val_min = float(np.min(K))
        val_max = float(np.max(K))

        diag_err = 0.0
        sym_err = 0.0
        min_eig = 0.0
        max_eig = 0.0
        condition_number = 0.0
        trace_val = 0.0
        mean_off_diag = 0.0
        std_off_diag = 0.0

        if is_training:
            n = K.shape[0]
            diag = np.diag(K)
            diag_err = float(np.max(np.abs(diag - 1.0)))
            sym_err = float(np.max(np.abs(K - K.T)))
            trace_val = float(np.trace(K))

            off_diag_mask = ~np.eye(n, dtype=bool)
            if np.any(off_diag_mask):
                off_diag_vals = K[off_diag_mask]
                mean_off_diag = float(np.mean(off_diag_vals))
                std_off_diag = float(np.std(off_diag_vals))

            # Eigenvalues of symmetric training matrix
            eigs = np.linalg.eigvalsh(K)
            min_eig = float(np.min(eigs))
            max_eig = float(np.max(eigs))
            condition_number = float(max_eig / max(min_eig, 1e-12))

        return {
            "shape": list(K.shape),
            "is_finite": is_finite,
            "min_value": val_min,
            "max_value": val_max,
            "symmetry_error": sym_err,
            "diagonal_error": diag_err,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": condition_number,
            "trace": trace_val,
            "mean_off_diagonal": mean_off_diag,
            "std_off_diagonal": std_off_diag,
            "is_training": is_training,
        }

    def get_circuit_properties(self) -> Dict[str, Any]:
        """Return circuit complexity metrics for parameter/compute accounting."""
        return {
            "num_qubits": self.num_qubits,
            "reps": self.reps,
            "entanglement": self.entanglement,
            "circuit_depth": self.circuit_depth,
            "one_qubit_gate_count": self.one_qubit_gate_count,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "total_gate_count": self.total_gate_count,
            "trainable_quantum_parameters": self.trainable_quantum_parameters,
            "encoding_parameters": self.encoding_parameters,
        }


# --- Persistent Kernel Caching Utilities ---


def get_kernel_cache_paths(
    cache_dir: str,
    speaker_id: str,
    sample_n: int,
    qubit_count: int,
    reps: int = 1,
) -> Dict[str, str]:
    """Generate canonical filenames for cached kernel matrices."""
    base_prefix = f"{speaker_id}_N{sample_n}_q{qubit_count}_r{reps}"
    return {
        "train_kernel": os.path.join(cache_dir, f"{base_prefix}_train.npy"),
        "test_kernel": os.path.join(cache_dir, f"{base_prefix}_test.npy"),
        "metadata": os.path.join(cache_dir, f"{base_prefix}_meta.json"),
    }


def save_cached_kernels(
    cache_dir: str,
    speaker_id: str,
    sample_n: int,
    qubit_count: int,
    K_train: np.ndarray,
    K_test: np.ndarray,
    metadata: Dict[str, Any],
    reps: int = 1,
) -> None:
    """Save training and test kernel matrices alongside provenance metadata."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = get_kernel_cache_paths(cache_dir, speaker_id, sample_n, qubit_count, reps)

    np.save(paths["train_kernel"], K_train.astype(np.float64))
    np.save(paths["test_kernel"], K_test.astype(np.float64))

    meta_payload = {
        "speaker_id": speaker_id,
        "sample_n": sample_n,
        "qubit_count": qubit_count,
        "reps": reps,
        "train_shape": list(K_train.shape),
        "test_shape": list(K_test.shape),
        "timestamp": time.time(),
        **metadata,
    }
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)


def load_cached_kernels(
    cache_dir: str,
    speaker_id: str,
    sample_n: int,
    qubit_count: int,
    reps: int = 1,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Load cached kernel matrices if they exist and are valid."""
    paths = get_kernel_cache_paths(cache_dir, speaker_id, sample_n, qubit_count, reps)
    if (
        os.path.exists(paths["train_kernel"])
        and os.path.exists(paths["test_kernel"])
        and os.path.exists(paths["metadata"])
    ):
        try:
            K_train = np.load(paths["train_kernel"])
            K_test = np.load(paths["test_kernel"])
            with open(paths["metadata"], "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return K_train, K_test, metadata
        except Exception:
            return None
    return None
