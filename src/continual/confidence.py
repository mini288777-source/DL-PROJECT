"""
Model-Agnostic Confidence Representation & Validation Engine (Phase 7.2).
Provides:
- Explicit separation between raw model scores, calibrated confidence, and decision confidence.
- Deterministic confidence validation (rejecting NaN, Inf, and out-of-bounds probabilities).
- Prediction uncertainty quantification (margin, entropy, max probability).
- Configuration-driven confidence banding.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import yaml


class ConfidenceType(str, Enum):
    """Semantic origin of the confidence measurement."""
    SOFTMAX_PROBABILITY = "SOFTMAX_PROBABILITY"
    DECISION_MARGIN = "DECISION_MARGIN"
    DISTANCE_TO_BOUNDARY = "DISTANCE_TO_BOUNDARY"
    QUANTUM_KERNEL_ALIGNMENT = "QUANTUM_KERNEL_ALIGNMENT"
    HEURISTIC_SCORE = "HEURISTIC_SCORE"


class CalibrationStatus(str, Enum):
    """Calibration state of the confidence score. No calibration is claimed unless evaluated."""
    UNCALIBRATED = "UNCALIBRATED"
    PLATT_SCALED = "PLATT_SCALED"
    ISOTONIC_SCALED = "ISOTONIC_SCALED"
    TEMPERATURE_SCALED = "TEMPERATURE_SCALED"


class ConfidenceBand(str, Enum):
    """Categorical confidence level dictated by policy configuration."""
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    MEDIUM_CONFIDENCE = "MEDIUM_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INVALID_CONFIDENCE = "INVALID_CONFIDENCE"


def load_confidence_policy(
    policy_path: str = "config/continual/confidence_policy.yaml"
) -> Dict[str, Any]:
    """Load confidence policy configuration with fallback to canonical defaults."""
    if os.path.exists(policy_path):
        with open(policy_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # Default fallback if config file missing
    return {
        "policy_metadata": {
            "policy_version": "v1.0.0_research_draft",
            "policy_id": "continual_confidence_policy_default",
        },
        "confidence_bands": {
            "high_threshold": 0.85,
            "medium_threshold": 0.65,
        },
        "abstention_policy": {
            "general_speech": {"accept_threshold": 0.70, "confirm_threshold": 0.50},
            "emergency_speech": {"accept_threshold": 0.85, "confirm_threshold": 0.65},
        },
        "uncertainty_policy": {
            "min_top1_top2_margin": 0.15,
            "max_normalized_entropy": 0.75,
        },
        "safety_invariants": {
            "allow_self_training_without_verification": False,
            "emergency_requires_user_confirmation": True,
            "quarantine_degraded_acoustic_quality": True,
            "min_acoustic_quality_score": 0.50,
        },
    }


class ConfidenceValidator:
    """Deterministic validation of confidence values against mathematical and scientific bounds."""

    @staticmethod
    def validate(
        confidence_value: Any,
        confidence_type: ConfidenceType = ConfidenceType.SOFTMAX_PROBABILITY,
    ) -> Tuple[bool, List[str]]:
        """
        Validate confidence measurement.
        Returns:
            (is_valid: bool, validation_notes: List[str])
        """
        notes: List[str] = []

        if confidence_value is None:
            return False, ["Confidence value is None"]

        if not isinstance(confidence_value, (int, float)):
            return False, [f"Confidence value is of non-numeric type: {type(confidence_value).__name__}"]

        # Mathematical validity checks
        if math.isnan(confidence_value):
            return False, ["Confidence value is NaN"]

        if math.isinf(confidence_value):
            return False, ["Confidence value is Infinite"]

        # Probability semantic checks
        if confidence_type == ConfidenceType.SOFTMAX_PROBABILITY:
            if confidence_value < 0.0:
                notes.append(f"Probability cannot be negative: {confidence_value}")
            if confidence_value > 1.0:
                notes.append(f"Probability cannot exceed 1.0: {confidence_value}")

        return (len(notes) == 0), notes


class UncertaintyCalculator:
    """Computes distribution uncertainty metrics when full posterior logits/probabilities are available."""

    @staticmethod
    def compute_from_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
        """
        Compute standard uncertainty metrics from a multi-class probability dictionary.
        Returns dict with:
            - max_prob: highest probability
            - top1_top2_margin: difference between top-1 and top-2
            - entropy: Shannon entropy in bits
            - normalized_entropy: entropy divided by log2(K)
        """
        if not probs:
            return {
                "max_prob": 0.0,
                "top1_top2_margin": 0.0,
                "entropy": 0.0,
                "normalized_entropy": 0.0,
            }

        sorted_probs = sorted(probs.values(), reverse=True)
        max_prob = sorted_probs[0]
        margin = (max_prob - sorted_probs[1]) if len(sorted_probs) > 1 else max_prob

        k = len(sorted_probs)
        entropy = 0.0
        for p in sorted_probs:
            if p > 1e-12:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(k) if k > 1 else 1.0
        norm_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0

        return {
            "max_prob": float(max_prob),
            "top1_top2_margin": float(margin),
            "entropy": float(entropy),
            "normalized_entropy": float(norm_entropy),
        }


@dataclass
class ConfidenceRepresentation:
    """
    Structured, auditable confidence measurement.
    Encapsulates raw score, calibrated probability, confidence band, and validity.
    """
    raw_score: float
    confidence_value: float
    confidence_type: ConfidenceType = ConfidenceType.SOFTMAX_PROBABILITY
    calibration_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    confidence_band: ConfidenceBand = ConfidenceBand.LOW_CONFIDENCE
    is_valid: bool = True
    validation_notes: List[str] = field(default_factory=list)
    source_model: str = "v_base"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @classmethod
    def create(
        cls,
        raw_score: float,
        confidence_value: float,
        source_model: str,
        confidence_type: ConfidenceType = ConfidenceType.SOFTMAX_PROBABILITY,
        calibration_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED,
        policy: Optional[Dict[str, Any]] = None,
    ) -> "ConfidenceRepresentation":
        """Factory constructor that executes deterministic validation and assigns confidence band."""
        if policy is None:
            policy = load_confidence_policy()

        is_valid, notes = ConfidenceValidator.validate(confidence_value, confidence_type)

        if not is_valid:
            band = ConfidenceBand.INVALID_CONFIDENCE
        else:
            bands_cfg = policy.get("confidence_bands", {})
            high_thresh = bands_cfg.get("high_threshold", 0.85)
            med_thresh = bands_cfg.get("medium_threshold", 0.65)

            if confidence_value >= high_thresh:
                band = ConfidenceBand.HIGH_CONFIDENCE
            elif confidence_value >= med_thresh:
                band = ConfidenceBand.MEDIUM_CONFIDENCE
            else:
                band = ConfidenceBand.LOW_CONFIDENCE

        return cls(
            raw_score=raw_score,
            confidence_value=confidence_value,
            confidence_type=confidence_type,
            calibration_status=calibration_status,
            confidence_band=band,
            is_valid=is_valid,
            validation_notes=notes,
            source_model=source_model,
        )
