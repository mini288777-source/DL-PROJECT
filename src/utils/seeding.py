"""
Seeding and Determinism Utility.

Phase 3: Provides unified random seed configuration across Python,
NumPy, PyTorch, and scikit-learn to guarantee strict reproducibility.
"""

import os
import random
from typing import Any, Dict
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> Dict[str, Any]:
    """
    Set random seeds across Python random, NumPy, PyTorch CPU/CUDA, and environment.

    Args:
        seed: Fixed integer seed (default: 42).
        deterministic_torch: Whether to enforce deterministic PyTorch operations.

    Returns:
        Dictionary detailing the applied seeding configuration.
    """
    # 1. Python random stdlib
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. Environment hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 4. PyTorch CPU & GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cfg: Dict[str, Any] = {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch_manual_seed": True,
        "torch_cuda": torch.cuda.is_available(),
        "pythonhashseed": str(seed),
        "deterministic_torch": False,
    }

    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True)
            cfg["deterministic_torch"] = True
        except Exception:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                cfg["deterministic_torch"] = "warn_only"
            except Exception:
                cfg["deterministic_torch"] = False

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            cfg["cudnn_deterministic"] = True

    return cfg


def get_rng(seed: int = 42) -> np.random.Generator:
    """Get an isolated NumPy random generator."""
    return np.random.default_rng(seed)
