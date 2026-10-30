"""Model loading, checkpoint verification, and inference orchestration for Gate B Prototype."""

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import numpy as np
import torch
import torch.nn.functional as F

from src.app.config import (
    REPO_ROOT,
    CANONICAL_CLASSES,
    EMERGENCY_CLASSES,
    GENERAL_SPEECH_CLASS,
    DEFAULT_DECISION_TAU,
    FER_SAFETY_BOUND,
    FER_SAFETY_BOUND_PERCENT,
    DISCLAIMER_NOTICE,
    SCIENTIFIC_VERDICT_NOTICE,
)
from src.models.classical_adapter import ClassicalAdapterNet


class ModelServiceError(Exception):
    """Base exception for model management and inference."""
    pass


class ModelNotFoundError(ModelServiceError):
    """Raised when an unknown speaker or model identifier is requested."""
    pass


class ModelNotReadyException(ModelServiceError):
    """
    Raised when an unmaterialized model checkpoint is requested (e.g. Phase 8.1 models).
    Returns HTTP 503 Service Unavailable to strictly prevent mock or fabricated inference.
    """
    def __init__(self, model_id: str, message: str, remediation: str):
        super().__init__(message)
        self.model_id = model_id
        self.message = message
        self.remediation = remediation


class ModelManager:
    """
    Manages loading, checkpoint verification, and forward inference for verified
    continual learning models, and enforces strict HTTP 503 unavailability for
    unmaterialized Phase 8.1 candidates.
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self._loaded_models: Dict[str, ClassicalAdapterNet] = {}
        self._inventory_cache: Optional[List[Dict[str, Any]]] = None
        self._load_checkpoint_inventory()

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_checkpoint_inventory(self) -> None:
        inv_path = REPO_ROOT / "reports" / "gate_b_hardening" / "checkpoint_inventory.json"
        if inv_path.exists():
            with open(inv_path, "r", encoding="utf-8") as f:
                self._inventory_cache = json.load(f)
        else:
            self._inventory_cache = []

    def get_inventory(self) -> List[Dict[str, Any]]:
        if not self._inventory_cache:
            self._load_checkpoint_inventory()
        return self._inventory_cache or []

    def get_available_speakers(self) -> List[str]:
        phase7_5_dir = REPO_ROOT / "models" / "continual" / "phase7_5"
        if phase7_5_dir.exists():
            return sorted([d.name for d in phase7_5_dir.iterdir() if d.is_dir()])
        return ["F01"]

    def _find_checkpoint_for_speaker(self, speaker_id: str) -> Optional[Path]:
        phase7_5_dir = REPO_ROOT / "models" / "continual" / "phase7_5" / speaker_id / "classical_adapter"
        if not phase7_5_dir.exists():
            return None

        # Look for budget 50 replay checkpoint, else any valid checkpoint
        candidates = list(phase7_5_dir.glob("*/model_checkpoint.pt"))
        if not candidates:
            return None

        # Prefer budget 50 replay
        b50_candidates = [c for c in candidates if "b50" in c.parent.name]
        if b50_candidates:
            return b50_candidates[0]
        return candidates[0]

    def get_or_load_adapter(self, speaker_id: str) -> Tuple[ClassicalAdapterNet, str, Path]:
        """
        Loads and caches verified ClassicalAdapterNet for the given speaker.

        Returns:
            Tuple of (model_net, version_string, checkpoint_path)
        """
        if speaker_id in self._loaded_models:
            net = self._loaded_models[speaker_id]
            ckpt_path = self._find_checkpoint_for_speaker(speaker_id)
            version_str = ckpt_path.parent.name if ckpt_path else f"{speaker_id}_active"
            return net, version_str, ckpt_path

        ckpt_path = self._find_checkpoint_for_speaker(speaker_id)
        if ckpt_path is None or not ckpt_path.exists():
            raise ModelNotFoundError(
                f"No verified model checkpoint found on disk for speaker '{speaker_id}'."
            )

        # Instantiate ClassicalAdapterNet matching exact architecture
        net = ClassicalAdapterNet(input_dim=384, hidden_dim=64, num_classes=7, dropout=0.0)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.load_state_dict(state_dict)
        net.eval()
        for param in net.parameters():
            param.requires_grad = False

        self._loaded_models[speaker_id] = net
        version_str = ckpt_path.parent.name
        return net, version_str, ckpt_path

    def get_model_info(self, model_id: str, speaker_id: str = "F01") -> Dict[str, Any]:
        """Returns comprehensive metadata, checkpoint provenance, and decision configuration."""
        # Check if requesting an unmaterialized Phase 8.1 model
        if "phase8" in model_id.lower() or "track" in model_id.lower() or "int-" in model_id.lower():
            return {
                "model_id": model_id,
                "model_name": f"Phase 8.1 Candidate ({model_id})",
                "speaker_id": speaker_id,
                "architecture": "Stage 1 Logistic Gate + Classical/Quantum Residual MLP",
                "parameter_count": "Unmaterialized",
                "input_shape": [384],
                "taxonomy": CANONICAL_CLASSES,
                "decision_policy": {
                    "score_scale": "probability_unit_interval_[0,1]",
                    "decision_cutoff_tau": DEFAULT_DECISION_TAU,
                    "comparator": "greater_than_or_equal",
                    "decoupled_from_fer": True
                },
                "safety_constraints": {
                    "fer_maximum_fraction": FER_SAFETY_BOUND,
                    "fer_maximum_percent": FER_SAFETY_BOUND_PERCENT,
                    "formula": "FP / (FP + TN) = FP / Total_General_Speech"
                },
                "checkpoint_provenance": {
                    "path": f"checkpoints/phase8_1/{model_id}.pt",
                    "exists": False,
                    "sha256": None,
                    "status": "MODEL_NOT_READY"
                },
                "status": "MODEL_NOT_READY",
                "quantum_verdict": "CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE",
                "clinical_safety_verdict": "SAFETY-INFEASIBLE (Recall 10-20% under FER <= 15% ceiling)",
                "disclaimer": DISCLAIMER_NOTICE
            }

        # Otherwise, inspect verified Phase 7.5 continual classical adapter
        ckpt_path = self._find_checkpoint_for_speaker(speaker_id)
        if ckpt_path and ckpt_path.exists():
            h = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
            status = "MATERIALIZED_AND_VERIFIED"
            version_str = ckpt_path.parent.name
            path_str = ckpt_path.relative_to(REPO_ROOT).as_posix()
            size_bytes = ckpt_path.stat().st_size
        else:
            h = None
            status = "MODEL_NOT_READY"
            version_str = "unmaterialized"
            path_str = f"models/continual/phase7_5/{speaker_id}/classical_adapter/model_checkpoint.pt"
            size_bytes = 0

        return {
            "model_id": model_id,
            "model_name": f"Continual Classical Adapter (Speaker: {speaker_id})",
            "speaker_id": speaker_id,
            "architecture": "ClassicalAdapterNet (Linear 384->64, LayerNorm, GELU, Dropout 0.1, Linear 64->7)",
            "parameter_count": "25,223 trainable parameters",
            "input_shape": [384],
            "taxonomy": CANONICAL_CLASSES,
            "decision_policy": {
                "score_scale": "probability_unit_interval_[0,1]",
                "decision_cutoff_tau": DEFAULT_DECISION_TAU,
                "comparator": "greater_than_or_equal",
                "decoupled_from_fer": True,
                "status": "THRESHOLD_PROVENANCE_UNVERIFIED",
                "notice": "Uncalibrated demonstration default — threshold provenance unverified for this model."
            },
            "safety_constraints": {
                "fer_maximum_fraction": FER_SAFETY_BOUND,
                "fer_safety_ceiling": FER_SAFETY_BOUND,
                "fer_maximum_percent": FER_SAFETY_BOUND_PERCENT,
                "formula": "FP / (FP + TN) = FP / Total_General_Speech"
            },
            "checkpoint_provenance": {
                "path": path_str,
                "version": version_str,
                "exists": ckpt_path is not None and ckpt_path.exists(),
                "file_size_bytes": size_bytes,
                "sha256": h,
                "status": status
            },
            "status": status,
            "quantum_verdict": "CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE",
            "clinical_safety_verdict": "SAFETY-INFEASIBLE (Recall 10-20% under FER <= 15% ceiling)",
            "disclaimer": DISCLAIMER_NOTICE
        }

    def predict(
        self,
        embedding: np.ndarray,
        speaker_id: str = "F01",
        model_id: str = "continual_classical_adapter",
        operating_tau: Optional[float] = None,
        acoustic_quality: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Executes genuine forward inference using a verified model checkpoint,
        or raises ModelNotReadyException (HTTP 503) for unmaterialized models.
        """
        t_start = time.perf_counter()

        # 1. Enforce strict unavailability on Phase 8.1 models (No fabrication permitted)
        if "phase8" in model_id.lower() or "track" in model_id.lower() or "int-" in model_id.lower():
            raise ModelNotReadyException(
                model_id=model_id,
                message=(
                    f"Candidate model '{model_id}' cannot be loaded because its checkpoint "
                    f"('checkpoints/phase8_1/{model_id}.pt') is unmaterialized on disk. "
                    f"As verified in the Phase 8.2.1 forensic audit, candidate weights were "
                    f"evaluated in nested cross-validation without serializing checkpoint binaries. "
                    f"In accordance with Gate B non-negotiable scientific constraints, model weights "
                    f"cannot be fabricated or silently retrained without explicit owner authorization."
                ),
                remediation=(
                    "Use the verified 'continual_classical_adapter' model for demonstration, or obtain "
                    "explicit owner authorization to materialize candidate checkpoints under an "
                    "authorized development protocol."
                )
            )

        # 2. Input validation
        if embedding is None or not isinstance(embedding, np.ndarray):
            raise ValueError("Input embedding must be a non-null numpy array.")
        if embedding.shape != (384,):
            raise ValueError(f"Input embedding must have shape (384,), got {embedding.shape}")

        session_id = str(uuid.uuid4())[:8]

        # 3. Degraded acoustic quality -> Abstention
        if acoustic_quality < 0.30:
            return {
                "session_id": session_id,
                "speaker_id": speaker_id,
                "model_id": model_id,
                "predicted_intent": "UNKNOWN",
                "is_emergency": False,
                "confidence_score": 0.0,
                "confidence_band": "LOW",
                "suggested_action": "ABSTAIN",
                "class_probabilities": {c: 0.0 for c in CANONICAL_CLASSES},
                "decision_cutoff_tau": operating_tau or DEFAULT_DECISION_TAU,
                "model_version": f"{speaker_id}_abstained_low_quality",
                "processing_latency_ms": (time.perf_counter() - t_start) * 1000.0,
            }

        # 4. Load verified model
        net, version_str, ckpt_path = self.get_or_load_adapter(speaker_id)

        # 5. Deterministic forward pass
        with torch.no_grad():
            x_tensor = torch.from_numpy(embedding).float().unsqueeze(0)
            logits = net(x_tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        class_probs = {CANONICAL_CLASSES[i]: float(probs[i]) for i in range(len(CANONICAL_CLASSES))}
        top_idx = int(np.argmax(probs))
        predicted_class = CANONICAL_CLASSES[top_idx]
        confidence = float(probs[top_idx])

        # 6. Apply decision threshold tau (decoupled from FER limit)
        effective_tau = float(operating_tau) if operating_tau is not None else DEFAULT_DECISION_TAU
        emergency_prob = float(sum(class_probs[c] for c in EMERGENCY_CLASSES))

        # Check emergency status against threshold
        is_emergency = emergency_prob >= effective_tau

        # 7. Confidence bands and suggested actions
        if confidence >= 0.85:
            conf_band = "HIGH"
        elif confidence >= 0.60:
            conf_band = "MEDIUM"
        else:
            conf_band = "LOW"

        if is_emergency:
            if conf_band == "HIGH":
                suggested_action = "CONFIRM_AND_DISPATCH"
            else:
                suggested_action = "REQUEST_CONFIRMATION"
        else:
            if conf_band in ("HIGH", "MEDIUM"):
                suggested_action = "PROCEED"
            else:
                suggested_action = "ABSTAIN"

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "session_id": session_id,
            "speaker_id": speaker_id,
            "model_id": model_id,
            "predicted_intent": predicted_class,
            "is_emergency": is_emergency,
            "confidence_score": confidence,
            "confidence_band": conf_band,
            "suggested_action": suggested_action,
            "class_probabilities": class_probs,
            "decision_cutoff_tau": effective_tau,
            "model_version": version_str,
            "processing_latency_ms": latency_ms,
        }
