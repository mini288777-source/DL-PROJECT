"""
Reproducibility utilities for the hybrid quantum-classical speech recognition project.

Sets deterministic seeds for Python, NumPy, PyTorch, and scikit-learn workflows.

IMPORTANT NOTES ON DETERMINISM:
- CPU PyTorch operations are fully deterministic with these settings.
- GPU operations may have non-deterministic kernels even with seeds set.
  torch.use_deterministic_algorithms(True) is attempted but may raise errors
  for certain operations (e.g., scatter_add). If so, set warn_only=True.
- Qiskit statevector simulation is mathematically exact (no sampling noise)
  when using StatevectorSampler, so it is inherently deterministic.
- Qiskit shot-based simulation (e.g., AerSimulator with shots) will vary
  unless the simulator seed is set separately per-run.
- librosa resampling is deterministic for identical inputs.
"""

import os
import random
import numpy as np
import torch


from typing import Any, Dict


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> Dict[str, Any]:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic_torch: If True, enable PyTorch deterministic mode
            and disable CuDNN benchmarking. May reduce GPU performance.

    Returns:
        Dictionary summarizing the seed configuration applied.
    """
    # Python stdlib
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    config: Dict[str, Any] = {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch_manual_seed": True,
        "torch_cuda_seed": torch.cuda.is_available(),
        "pythonhashseed": str(seed),
        "deterministic_torch": False,
        "cudnn_benchmark": None,
        "cudnn_deterministic": None,
    }

    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True)
            config["deterministic_torch"] = True
        except Exception:
            # Some operations don't support deterministic mode; fall back
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                config["deterministic_torch"] = "warn_only"
            except Exception:
                config["deterministic_torch"] = False

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            config["cudnn_benchmark"] = False
            config["cudnn_deterministic"] = True

    return config


def get_numpy_rng(seed: int = 42) -> np.random.Generator:
    """
    Create a NumPy Generator with a specific seed.
    Preferred over legacy np.random.seed() for isolated reproducible streams.

    Args:
        seed: Integer seed value.

    Returns:
        A numpy.random.Generator instance.
    """
    return np.random.default_rng(seed)
