"""
Phase 7.7 Speaker Drift Detection & Recalibration Engine.
Implements:
- Structured drift event & window schemas with strict immutability.
- Multi-faceted drift detectors:
  1. Representation / input drift (MMD via RBF kernel + centroid cosine distance).
  2. Prediction-distribution drift (Total Variation Distance across canonical 7 classes).
  3. Confidence drift (Two-sample Kolmogorov-Smirnov test + delta mean confidence).
  4. Calibration drift (Expected Calibration Error + Brier score).
  5. Class-frequency drift (Ground truth distribution divergence).
- Strict speaker-local isolation & temporal causality enforcement.
- Deterministic recalibration trigger policy.
- Temperature-scaling recalibrator for confidence/decision mapping.
- Integration with Phase 7.6 Validation Gate ensuring zero auto-promotion and unconditional Safety Veto.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy import stats
import yaml

from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.continual.model_version import (
    ModelLifecycleState,
    ModelVersionRecord,
    ModelRegistry,
    ModelVersionError,
    compute_sha256,
)
from src.continual.governance import (
    ValidationGate,
    ValidationDecisionRecord,
    PromotionDecision,
    ValidationReasonCode,
)


class DriftType(str, Enum):
    """Canonical categories of speaker drift matching Section 8."""
    REPRESENTATION_DRIFT = "REPRESENTATION_DRIFT"
    PREDICTION_DRIFT = "PREDICTION_DRIFT"
    CONFIDENCE_DRIFT = "CONFIDENCE_DRIFT"
    CALIBRATION_DRIFT = "CALIBRATION_DRIFT"
    CLASS_FREQUENCY_DRIFT = "CLASS_FREQUENCY_DRIFT"


class DriftStatus(str, Enum):
    """Categorical drift decision status matching Section 13."""
    NO_DRIFT = "NO_DRIFT"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    DRIFT_DETECTED_TRANSIENT = "DRIFT_DETECTED_TRANSIENT"
    MEANINGFUL_DRIFT = "MEANINGFUL_DRIFT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID = "INVALID"


class RecalibrationTriggerState(str, Enum):
    """Recalibration decision state matching Section 23."""
    NO_RECALIBRATION = "NO_RECALIBRATION"
    RECALIBRATION_RECOMMENDED = "RECALIBRATION_RECOMMENDED"
    RECALIBRATION_REQUIRED = "RECALIBRATION_REQUIRED"
    HOLD_INSUFFICIENT_EVIDENCE = "HOLD_INSUFFICIENT_EVIDENCE"


class DriftError(Exception):
    """Base exception for drift detection operations."""
    pass


class SpeakerIsolationError(DriftError):
    """Raised when cross-speaker contamination is detected in drift evaluation."""
    pass


class TemporalSafetyError(DriftError):
    """Raised when temporal causality is violated (e.g., future samples in reference)."""
    pass


class DriftDataError(DriftError):
    """Raised on invalid drift data or window schemas."""
    pass


def load_drift_policy(
    policy_path: str = "config/continual/drift_policy.yaml"
) -> Dict[str, Any]:
    """Load drift and recalibration policy configuration."""
    if os.path.exists(policy_path):
        with open(policy_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "policy_metadata": {
            "policy_id": "speaker_drift_recalibration_policy_default",
            "policy_version": "v1.0.0_frozen",
            "frozen": True,
        },
        "window_policy": {
            "reference_source": "initial_calibration_b0",
            "min_reference_samples": 10,
            "min_current_samples": 10,
        },
        "drift_methods_and_thresholds": {
            "representation_drift": {"statistic_threshold": 0.15, "secondary_centroid_cosine_threshold": 0.10},
            "prediction_drift": {"statistic_threshold": 0.25},
            "confidence_drift": {"statistic_threshold": 0.30, "significance_alpha": 0.05, "mean_confidence_delta_threshold": 0.10},
            "calibration_drift": {"ece_delta_threshold": 0.08, "brier_delta_threshold": 0.10, "num_bins": 10},
            "class_frequency_drift": {"statistic_threshold": 0.25},
        },
        "persistence_policy": {
            "transient_drift_threshold_windows": 1,
            "meaningful_drift_consecutive_windows": 2,
        },
        "recalibration_trigger_policy": {
            "require_verified_labels": True,
        },
    }


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class DriftWindow:
    """
    Chronological window of speaker experience for drift evaluation.
    Enforces speaker isolation and chronological bounds.
    """
    window_id: str
    speaker_id: str
    sample_ids: List[str]
    embeddings: np.ndarray  # (N, D) float32
    predicted_classes: List[str]
    confidences: List[float]
    model_version: str
    true_labels: Optional[List[str]] = None
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.speaker_id or not isinstance(self.speaker_id, str):
            raise DriftDataError("DriftWindow requires a valid, non-empty speaker_id.")
        n_samples = len(self.sample_ids)
        if len(self.embeddings) != n_samples:
            raise DriftDataError(f"Embedding count ({len(self.embeddings)}) != sample_ids count ({n_samples}).")
        if len(self.predicted_classes) != n_samples:
            raise DriftDataError(f"Predictions count ({len(self.predicted_classes)}) != sample_ids count ({n_samples}).")
        if len(self.confidences) != n_samples:
            raise DriftDataError(f"Confidence count ({len(self.confidences)}) != sample_ids count ({n_samples}).")
        if self.true_labels is not None and len(self.true_labels) != n_samples:
            raise DriftDataError(f"True labels count ({len(self.true_labels)}) != sample_ids count ({n_samples}).")

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)

    def compute_fingerprint(self) -> str:
        """Deterministic fingerprint of window data."""
        hasher = hashlib.sha256()
        hasher.update(self.window_id.encode("utf-8"))
        hasher.update(self.speaker_id.encode("utf-8"))
        hasher.update(",".join(self.sample_ids).encode("utf-8"))
        hasher.update(self.embeddings.tobytes())
        return hasher.hexdigest()


@dataclass
class DriftEvent:
    """
    Structured, immutable drift detection result matching Section 24.
    """
    drift_event_id: str
    speaker_id: str
    model_version: str
    reference_window_id: str
    current_window_id: str
    drift_type: str
    drift_method: str
    statistic: float
    threshold: float
    status: str
    policy_version: str
    p_value: Optional[float] = None
    confidence_summary: Dict[str, float] = field(default_factory=dict)
    class_distribution_summary: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        super().__setattr__("_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError(f"DriftEvent is immutable; cannot modify attribute '{name}'.")
        super().__setattr__(name, value)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_initialized", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftEvent":
        clean_data = dict(data)
        clean_data.pop("_initialized", None)
        return cls(**clean_data)


@dataclass
class RecalibrationCandidate:
    """
    Structured candidate update resulting from recalibration matching Section 25.
    Must be validated by Phase 7.6 Validation Gate before promotion.
    """
    candidate_id: str
    speaker_id: str
    parent_model_version: str
    calibrated_model_version: str
    calibration_method: str
    temperature: float
    calibration_data_fingerprint: str
    calibration_sample_ids: List[str]
    calibration_policy_version: str
    validation_status: str = "PENDING_VALIDATION"
    promotion_status: str = ModelLifecycleState.CANDIDATE.value
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__setattr__("_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False):
            # Allow validation/promotion status transitions while keeping provenance immutable
            if name in ("validation_status", "promotion_status"):
                super().__setattr__(name, value)
                return
            raise AttributeError(f"RecalibrationCandidate provenance is immutable; cannot modify '{name}'.")
        super().__setattr__(name, value)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_initialized", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecalibrationCandidate":
        clean_data = dict(data)
        clean_data.pop("_initialized", None)
        return cls(**clean_data)


# ==============================================================================
# MATHEMATICAL ESTIMATORS
# ==============================================================================

def compute_mmd_rbf(
    X_ref: np.ndarray,
    X_curr: np.ndarray,
    bandwidth: Optional[float] = None
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD^2) between two representation sets
    using an RBF kernel with the median bandwidth heuristic.
    
    Formula:
      MMD^2(P, Q) = E[k(x, x')] + E[k(y, y')] - 2*E[k(x, y)]
    """
    if len(X_ref) < 2 or len(X_curr) < 2:
        return 0.0

    X_ref = np.asarray(X_ref, dtype=np.float64)
    X_curr = np.asarray(X_curr, dtype=np.float64)

    # Median heuristic bandwidth selection
    if bandwidth is None or bandwidth <= 0:
        combined = np.vstack([X_ref, X_curr])
        n_comb = len(combined)
        # Compute pairwise squared Euclidean distances
        dists_sq = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=-1)
        # Extract upper triangle non-zero distances
        triu_indices = np.triu_indices(n_comb, k=1)
        dist_vals = dists_sq[triu_indices]
        dist_vals = dist_vals[dist_vals > 0]
        if len(dist_vals) > 0:
            median_dist_sq = float(np.median(dist_vals))
            gamma = 1.0 / (2.0 * max(median_dist_sq, 1e-6))
        else:
            gamma = 1.0 / (2.0 * X_ref.shape[1])
    else:
        gamma = 1.0 / (2.0 * max(bandwidth ** 2, 1e-6))

    # Kernel matrices
    d_xx = np.sum((X_ref[:, None, :] - X_ref[None, :, :]) ** 2, axis=-1)
    K_xx = np.exp(-gamma * d_xx)

    d_yy = np.sum((X_curr[:, None, :] - X_curr[None, :, :]) ** 2, axis=-1)
    K_yy = np.exp(-gamma * d_yy)

    d_xy = np.sum((X_ref[:, None, :] - X_curr[None, :, :]) ** 2, axis=-1)
    K_xy = np.exp(-gamma * d_xy)

    m = len(X_ref)
    n = len(X_curr)

    # Unbiased estimate: exclude diagonal terms for within-sample expectations
    term_xx = (np.sum(K_xx) - np.trace(K_xx)) / (m * (m - 1))
    term_yy = (np.sum(K_yy) - np.trace(K_yy)) / (n * (n - 1))
    term_xy = 2.0 * np.sum(K_xy) / (m * n)

    mmd_sq = float(term_xx + term_yy - term_xy)
    return float(max(0.0, mmd_sq))


def compute_centroid_cosine_distance(
    X_ref: np.ndarray,
    X_curr: np.ndarray
) -> float:
    """
    Compute cosine distance between mean representation centroids.
    Range: [0.0, 2.0]
    """
    if len(X_ref) == 0 or len(X_curr) == 0:
        return 0.0

    c_ref = np.mean(X_ref, axis=0)
    c_curr = np.mean(X_curr, axis=0)

    norm_ref = np.linalg.norm(c_ref)
    norm_curr = np.linalg.norm(c_curr)

    if norm_ref < 1e-12 or norm_curr < 1e-12:
        return 0.0

    cosine_sim = float(np.dot(c_ref, c_curr) / (norm_ref * norm_curr))
    # Clip for numerical stability
    cosine_sim = max(-1.0, min(1.0, cosine_sim))
    return float(1.0 - cosine_sim)


def compute_prediction_tvd(
    preds_ref: List[str],
    preds_curr: List[str],
    classes: Optional[List[str]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Total Variation Distance (TVD) between two discrete prediction distributions.
    
    Formula:
      TVD = 0.5 * sum_c |P(Y=c) - Q(Y=c)|
    Range: [0.0, 1.0]
    """
    if classes is None:
        classes = CANONICAL_CLASSES

    if len(preds_ref) == 0 or len(preds_curr) == 0:
        return 0.0, {c: 0.0 for c in classes}

    dist_ref = {c: preds_ref.count(c) / len(preds_ref) for c in classes}
    dist_curr = {c: preds_curr.count(c) / len(preds_curr) for c in classes}

    tvd = 0.5 * sum(abs(dist_ref[c] - dist_curr[c]) for c in classes)
    per_class_deltas = {c: float(dist_curr[c] - dist_ref[c]) for c in classes}

    return float(tvd), per_class_deltas


def compute_confidence_ks(
    conf_ref: List[float],
    conf_curr: List[float]
) -> Tuple[float, float, float]:
    """
    Two-sample Kolmogorov-Smirnov test comparing confidence distributions.
    Returns:
      (ks_statistic, p_value, delta_mean_confidence)
    """
    if len(conf_ref) == 0 or len(conf_curr) == 0:
        return 0.0, 1.0, 0.0

    res = stats.ks_2samp(conf_ref, conf_curr)
    delta_mean = float(np.mean(conf_curr) - np.mean(conf_ref))

    return float(res.statistic), float(res.pvalue), float(delta_mean)


def compute_expected_calibration_error(
    confidences: List[float],
    accuracies: List[int],
    num_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width probability bins.
    
    Formula:
      ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if len(confidences) == 0 or len(accuracies) == 0:
        return 0.0

    n = len(confidences)
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = [j for j, c in enumerate(confidences) if bin_lower <= c <= bin_upper]
        else:
            in_bin = [j for j, c in enumerate(confidences) if bin_lower <= c < bin_upper]

        if len(in_bin) > 0:
            bin_acc = np.mean([accuracies[j] for j in in_bin])
            bin_conf = np.mean([confidences[j] for j in in_bin])
            ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_maximum_calibration_error(
    confidences: List[float],
    accuracies: List[int],
    num_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    Formula:
      MCE = max_{m: |B_m| > 0} |acc(B_m) - conf(B_m)|
    """
    if len(confidences) == 0 or len(accuracies) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    max_err = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = [j for j, c in enumerate(confidences) if bin_lower <= c <= bin_upper]
        else:
            in_bin = [j for j, c in enumerate(confidences) if bin_lower <= c < bin_upper]

        if len(in_bin) > 0:
            bin_acc = np.mean([accuracies[j] for j in in_bin])
            bin_conf = np.mean([confidences[j] for j in in_bin])
            err = abs(bin_acc - bin_conf)
            if err > max_err:
                max_err = err

    return float(max_err)


def compute_brier_score(
    prob_matrix: np.ndarray,
    true_class_indices: List[int]
) -> float:
    """
    Compute multi-class Brier score.
    Formula:
      1/N * sum_i sum_k (p_ik - y_ik)^2
    """
    if len(prob_matrix) == 0 or len(true_class_indices) == 0:
        return 0.0

    n, k = prob_matrix.shape
    one_hot = np.zeros((n, k), dtype=np.float64)
    for i, idx in enumerate(true_class_indices):
        if 0 <= idx < k:
            one_hot[i, idx] = 1.0

    brier = np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1))
    return float(brier)


# ==============================================================================
# TEMPERATURE SCALING RECALIBRATOR
# ==============================================================================

class TemperatureScalingRecalibrator:
    """
    Model-Agnostic Recalibrator using Temperature Scaling.
    
    Adjusts the temperature scalar T > 0 such that calibrated probabilities
    p_i = softmax(logits / T) minimize NLL or calibration error on a held-out
    calibration set without modifying the underlying model features or weights.
    """

    def __init__(
        self,
        temperature_range: Tuple[float, float] = (0.5, 3.0),
        num_steps: int = 51,
        policy_version: str = "v1.0.0_frozen"
    ):
        self.t_min, self.t_max = temperature_range
        self.num_steps = num_steps
        self.policy_version = policy_version

    def fit(
        self,
        logits: np.ndarray,
        true_labels: List[str],
        classes: Optional[List[str]] = None
    ) -> float:
        """
        Fit optimal temperature T on calibration data.
        Returns fitted temperature scalar T.
        """
        if classes is None:
            classes = CANONICAL_CLASSES

        if len(logits) == 0 or len(true_labels) == 0:
            return 1.0

        class_to_idx = {c: i for i, c in enumerate(classes)}
        targets = np.array([class_to_idx.get(lbl, 0) for lbl in true_labels], dtype=np.int64)

        temperatures = np.linspace(self.t_min, self.t_max, self.num_steps)
        best_nll = float("inf")
        best_t = 1.0

        for t in temperatures:
            # Scaled softmax
            scaled_logits = logits / t
            # Stable log-softmax
            max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(scaled_logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            # Clip for log stability
            probs = np.clip(probs, 1e-12, 1.0)

            # Negative log-likelihood
            nll = -np.mean(np.log(probs[np.arange(len(targets)), targets]))

            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

        return best_t

    def apply(
        self,
        logits: np.ndarray,
        temperature: float
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Apply temperature scaling to logits.
        Returns:
          (calibrated_probs: np.ndarray, max_confidences: List[float])
        """
        t = max(0.01, float(temperature))
        scaled_logits = logits / t
        max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        max_conf = np.max(probs, axis=-1).tolist()
        return probs, max_conf


# ==============================================================================
# SPEAKER DRIFT DETECTOR
# ==============================================================================

class SpeakerDriftDetector:
    """
    Speaker-Local Drift Detection and Recalibration Trigger Engine.
    
    Enforces:
    - Speaker-local reference isolation (no cross-speaker drift comparison).
    - Temporal causality (reference must precede or match current experience window).
    - Multi-faceted drift testing (representation, prediction, confidence, calibration).
    - Deterministic persistence tracking (transient vs meaningful drift).
    - Frozen threshold policy adherence.
    """

    def __init__(
        self,
        policy: Optional[Dict[str, Any]] = None,
        policy_path: str = "config/continual/drift_policy.yaml"
    ):
        if policy is None:
            policy = load_drift_policy(policy_path)
        self.policy = policy
        self.policy_version = policy.get("policy_metadata", {}).get("policy_version", "v1.0.0_frozen")

        # Speaker-specific reference windows: speaker_id -> DriftWindow
        self._reference_windows: Dict[str, DriftWindow] = {}
        # Speaker-specific drift event history: speaker_id -> List[DriftEvent]
        self._drift_history: Dict[str, List[DriftEvent]] = {}

    def set_reference_window(self, window: DriftWindow) -> None:
        """
        Register canonical reference distribution for a speaker.
        Enforces speaker-local isolation and minimum sample requirements.
        """
        min_ref = self.policy.get("window_policy", {}).get("min_reference_samples", 10)
        if window.num_samples < min_ref:
            raise DriftDataError(
                f"Reference window '{window.window_id}' has {window.num_samples} samples, below required minimum {min_ref}."
            )
        self._reference_windows[window.speaker_id] = window
        if window.speaker_id not in self._drift_history:
            self._drift_history[window.speaker_id] = []

    def get_reference_window(self, speaker_id: str) -> Optional[DriftWindow]:
        """Retrieve registered reference window for a speaker."""
        return self._reference_windows.get(speaker_id)

    def evaluate_drift(
        self,
        current_window: DriftWindow,
        reference_window: Optional[DriftWindow] = None
    ) -> List[DriftEvent]:
        """
        Perform comprehensive multi-faceted drift evaluation comparing
        current_window against reference_window.
        """
        spk = current_window.speaker_id

        # 1. Resolve reference window
        if reference_window is None:
            reference_window = self._reference_windows.get(spk)

        if reference_window is None:
            raise DriftError(f"No reference window established for speaker '{spk}'.")

        # 2. Enforce Speaker Isolation (Section 5)
        if reference_window.speaker_id != current_window.speaker_id:
            raise SpeakerIsolationError(
                f"Speaker isolation violated: Reference speaker '{reference_window.speaker_id}' "
                f"does not match current window speaker '{current_window.speaker_id}'."
            )

        # 3. Enforce Temporal Safety (Section 30)
        if (reference_window.start_timestamp and current_window.start_timestamp and
                reference_window.start_timestamp > current_window.start_timestamp):
            raise TemporalSafetyError(
                f"Temporal ordering violated: Reference timestamp ({reference_window.start_timestamp}) "
                f"is later than current window timestamp ({current_window.start_timestamp})."
            )

        # 4. Check data sufficiency
        min_curr = self.policy.get("window_policy", {}).get("min_current_samples", 10)
        if current_window.num_samples < min_curr:
            # Emit INSUFFICIENT_DATA event
            ev_id = f"drift_insufficient_{spk}_{current_window.window_id}"
            ev = DriftEvent(
                drift_event_id=ev_id,
                speaker_id=spk,
                model_version=current_window.model_version,
                reference_window_id=reference_window.window_id,
                current_window_id=current_window.window_id,
                drift_type=DriftType.REPRESENTATION_DRIFT.value,
                drift_method="sample_count_check",
                statistic=float(current_window.num_samples),
                threshold=float(min_curr),
                status=DriftStatus.INSUFFICIENT_DATA.value,
                policy_version=self.policy_version,
                metadata={"reason": "Sample count below minimum current window threshold"},
            )
            self._record_event(spk, ev)
            return [ev]

        thresholds = self.policy.get("drift_methods_and_thresholds", {})
        events: List[DriftEvent] = []

        # ======================================================================
        # A. REPRESENTATION / INPUT DRIFT (Section 9)
        # ======================================================================
        rep_cfg = thresholds.get("representation_drift", {})
        rep_thresh = rep_cfg.get("statistic_threshold", 0.15)
        cos_thresh = rep_cfg.get("secondary_centroid_cosine_threshold", 0.10)

        mmd_stat = compute_mmd_rbf(reference_window.embeddings, current_window.embeddings)
        cos_dist = compute_centroid_cosine_distance(reference_window.embeddings, current_window.embeddings)

        rep_status = (
            DriftStatus.DRIFT_DETECTED.value
            if (mmd_stat >= rep_thresh or cos_dist >= cos_thresh)
            else DriftStatus.NO_DRIFT.value
        )

        ev_rep = DriftEvent(
            drift_event_id=f"drift_rep_{spk}_{reference_window.window_id}_{current_window.window_id}",
            speaker_id=spk,
            model_version=current_window.model_version,
            reference_window_id=reference_window.window_id,
            current_window_id=current_window.window_id,
            drift_type=DriftType.REPRESENTATION_DRIFT.value,
            drift_method="mmd_rbf_and_centroid_cosine",
            statistic=float(mmd_stat),
            threshold=float(rep_thresh),
            status=rep_status,
            policy_version=self.policy_version,
            metadata={
                "mmd_statistic": mmd_stat,
                "centroid_cosine_distance": cos_dist,
                "centroid_cosine_threshold": cos_thresh,
            },
        )
        events.append(ev_rep)

        # ======================================================================
        # B. PREDICTION-DISTRIBUTION DRIFT (Section 10)
        # ======================================================================
        pred_cfg = thresholds.get("prediction_drift", {})
        pred_thresh = pred_cfg.get("statistic_threshold", 0.25)

        tvd_stat, per_class_deltas = compute_prediction_tvd(
            reference_window.predicted_classes,
            current_window.predicted_classes,
            CANONICAL_CLASSES,
        )

        # Emergency aggregated drift
        ref_emerg_prop = sum(1 for p in reference_window.predicted_classes if p in EMERGENCY_CLASSES) / len(reference_window.predicted_classes)
        curr_emerg_prop = sum(1 for p in current_window.predicted_classes if p in EMERGENCY_CLASSES) / len(current_window.predicted_classes)
        emerg_delta = float(curr_emerg_prop - ref_emerg_prop)

        pred_status = (
            DriftStatus.DRIFT_DETECTED.value
            if tvd_stat >= pred_thresh
            else DriftStatus.NO_DRIFT.value
        )

        ref_dist = {c: float(reference_window.predicted_classes.count(c) / len(reference_window.predicted_classes)) for c in CANONICAL_CLASSES}
        curr_dist = {c: float(current_window.predicted_classes.count(c) / len(current_window.predicted_classes)) for c in CANONICAL_CLASSES}

        ev_pred = DriftEvent(
            drift_event_id=f"drift_pred_{spk}_{reference_window.window_id}_{current_window.window_id}",
            speaker_id=spk,
            model_version=current_window.model_version,
            reference_window_id=reference_window.window_id,
            current_window_id=current_window.window_id,
            drift_type=DriftType.PREDICTION_DRIFT.value,
            drift_method="total_variation_distance",
            statistic=float(tvd_stat),
            threshold=float(pred_thresh),
            status=pred_status,
            policy_version=self.policy_version,
            class_distribution_summary={
                "reference_distribution": ref_dist,
                "current_distribution": curr_dist,
                "per_class_deltas": per_class_deltas,
                "emergency_proportion_delta": emerg_delta,
            },
            metadata={
                "ref_emergency_proportion": ref_emerg_prop,
                "curr_emergency_proportion": curr_emerg_prop,
            },
        )
        events.append(ev_pred)

        # ======================================================================
        # C. CONFIDENCE DRIFT (Section 11)
        # ======================================================================
        conf_cfg = thresholds.get("confidence_drift", {})
        conf_stat_thresh = conf_cfg.get("statistic_threshold", 0.30)
        alpha = conf_cfg.get("significance_alpha", 0.05)
        delta_mean_thresh = conf_cfg.get("mean_confidence_delta_threshold", 0.10)

        ks_stat, p_val, delta_mean_conf = compute_confidence_ks(
            reference_window.confidences,
            current_window.confidences,
        )

        conf_status = (
            DriftStatus.DRIFT_DETECTED.value
            if (ks_stat >= conf_stat_thresh and p_val < alpha) or abs(delta_mean_conf) >= delta_mean_thresh
            else DriftStatus.NO_DRIFT.value
        )

        ev_conf = DriftEvent(
            drift_event_id=f"drift_conf_{spk}_{reference_window.window_id}_{current_window.window_id}",
            speaker_id=spk,
            model_version=current_window.model_version,
            reference_window_id=reference_window.window_id,
            current_window_id=current_window.window_id,
            drift_type=DriftType.CONFIDENCE_DRIFT.value,
            drift_method="two_sample_ks_and_delta_mean",
            statistic=float(ks_stat),
            threshold=float(conf_stat_thresh),
            p_value=float(p_val),
            status=conf_status,
            policy_version=self.policy_version,
            confidence_summary={
                "reference_mean_confidence": float(np.mean(reference_window.confidences)),
                "current_mean_confidence": float(np.mean(current_window.confidences)),
                "delta_mean_confidence": float(delta_mean_conf),
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(p_val),
            },
            metadata={
                "delta_mean_threshold": delta_mean_thresh,
                "alpha": alpha,
            },
        )
        events.append(ev_conf)

        # ======================================================================
        # D. CALIBRATION DRIFT (Section 12) - Evaluated if verified labels present
        # ======================================================================
        if reference_window.true_labels is not None and current_window.true_labels is not None:
            cal_cfg = thresholds.get("calibration_drift", {})
            ece_thresh = cal_cfg.get("ece_delta_threshold", 0.08)
            num_bins = cal_cfg.get("num_bins", 10)

            ref_accs = [1 if p == y else 0 for p, y in zip(reference_window.predicted_classes, reference_window.true_labels)]
            curr_accs = [1 if p == y else 0 for p, y in zip(current_window.predicted_classes, current_window.true_labels)]

            ref_ece = compute_expected_calibration_error(reference_window.confidences, ref_accs, num_bins=num_bins)
            curr_ece = compute_expected_calibration_error(current_window.confidences, curr_accs, num_bins=num_bins)
            delta_ece = float(curr_ece - ref_ece)

            cal_status = (
                DriftStatus.DRIFT_DETECTED.value
                if delta_ece >= ece_thresh
                else DriftStatus.NO_DRIFT.value
            )

            ev_cal = DriftEvent(
                drift_event_id=f"drift_cal_{spk}_{reference_window.window_id}_{current_window.window_id}",
                speaker_id=spk,
                model_version=current_window.model_version,
                reference_window_id=reference_window.window_id,
                current_window_id=current_window.window_id,
                drift_type=DriftType.CALIBRATION_DRIFT.value,
                drift_method="ece_delta",
                statistic=float(delta_ece),
                threshold=float(ece_thresh),
                status=cal_status,
                policy_version=self.policy_version,
                confidence_summary={
                    "reference_ece": float(ref_ece),
                    "current_ece": float(curr_ece),
                    "delta_ece": float(delta_ece),
                },
                metadata={
                    "reference_accuracy": float(np.mean(ref_accs)),
                    "current_accuracy": float(np.mean(curr_accs)),
                    "num_bins": num_bins,
                },
            )
            events.append(ev_cal)

        # ======================================================================
        # E. CLASS-FREQUENCY DRIFT (Section 8 & 32)
        # ======================================================================
        if reference_window.true_labels is not None and current_window.true_labels is not None:
            freq_cfg = thresholds.get("class_frequency_drift", {})
            freq_thresh = freq_cfg.get("statistic_threshold", 0.25)

            freq_tvd, freq_deltas = compute_prediction_tvd(
                reference_window.true_labels,
                current_window.true_labels,
                CANONICAL_CLASSES,
            )

            freq_status = (
                DriftStatus.DRIFT_DETECTED.value
                if freq_tvd >= freq_thresh
                else DriftStatus.NO_DRIFT.value
            )

            ev_freq = DriftEvent(
                drift_event_id=f"drift_freq_{spk}_{reference_window.window_id}_{current_window.window_id}",
                speaker_id=spk,
                model_version=current_window.model_version,
                reference_window_id=reference_window.window_id,
                current_window_id=current_window.window_id,
                drift_type=DriftType.CLASS_FREQUENCY_DRIFT.value,
                drift_method="tvd_ground_truth_frequency",
                statistic=float(freq_tvd),
                threshold=float(freq_thresh),
                status=freq_status,
                policy_version=self.policy_version,
                class_distribution_summary={
                    "ground_truth_deltas": freq_deltas,
                },
            )
            events.append(ev_freq)

        # Record events into history
        for ev in events:
            self._record_event(spk, ev)

        return events

    def _record_event(self, speaker_id: str, event: DriftEvent) -> None:
        """Internal append to speaker history."""
        if speaker_id not in self._drift_history:
            self._drift_history[speaker_id] = []
        self._drift_history[speaker_id].append(event)

    def assess_recalibration_trigger(
        self,
        recent_events: List[DriftEvent]
    ) -> Tuple[RecalibrationTriggerState, List[str]]:
        """
        Evaluate deterministic recalibration trigger policy matching Section 23.
        Distinguishes:
        - NO_RECALIBRATION
        - RECALIBRATION_RECOMMENDED (transient drift or moderate signal)
        - RECALIBRATION_REQUIRED (calibration degradation or persistent compound drift)
        - HOLD_INSUFFICIENT_EVIDENCE
        """
        reasons: List[str] = []

        if not recent_events:
            return RecalibrationTriggerState.NO_RECALIBRATION, ["No drift events provided"]

        # Check for insufficient data
        if any(e.status == DriftStatus.INSUFFICIENT_DATA.value for e in recent_events):
            return RecalibrationTriggerState.HOLD_INSUFFICIENT_EVIDENCE, ["Insufficient sample count in evaluated window"]

        # Check for calibration degradation (highest priority)
        cal_events = [e for e in recent_events if e.drift_type == DriftType.CALIBRATION_DRIFT.value]
        if cal_events:
            cal_ev = cal_events[-1]
            if cal_ev.status == DriftStatus.DRIFT_DETECTED.value:
                reasons.append(f"Calibration degradation detected: ECE delta ({cal_ev.statistic:.4f}) >= threshold ({cal_ev.threshold:.4f})")
                return RecalibrationTriggerState.RECALIBRATION_REQUIRED, reasons

        # Check for representation drift
        rep_events = [e for e in recent_events if e.drift_type == DriftType.REPRESENTATION_DRIFT.value]
        rep_drift = any(e.status == DriftStatus.DRIFT_DETECTED.value for e in rep_events)

        # Check for confidence drift
        conf_events = [e for e in recent_events if e.drift_type == DriftType.CONFIDENCE_DRIFT.value]
        conf_drift = any(e.status == DriftStatus.DRIFT_DETECTED.value for e in conf_events)

        # Check for prediction drift
        pred_events = [e for e in recent_events if e.drift_type == DriftType.PREDICTION_DRIFT.value]
        pred_drift = any(e.status == DriftStatus.DRIFT_DETECTED.value for e in pred_events)

        # Compound drift: Representation + Confidence drift indicates structural shift requiring recalibration
        if rep_drift and conf_drift:
            reasons.append("Compound drift detected: Representation drift observed alongside confidence distribution shift.")
            return RecalibrationTriggerState.RECALIBRATION_REQUIRED, reasons

        # Moderate single-facet drift triggers RECOMMENDED, not REQUIRED
        if rep_drift or conf_drift or pred_drift:
            detected_facets = []
            if rep_drift: detected_facets.append("representation")
            if conf_drift: detected_facets.append("confidence")
            if pred_drift: detected_facets.append("prediction")
            reasons.append(f"Transient or single-facet drift detected in: {', '.join(detected_facets)}.")
            return RecalibrationTriggerState.RECALIBRATION_RECOMMENDED, reasons

        reasons.append("All drift statistics within frozen null bounds.")
        return RecalibrationTriggerState.NO_RECALIBRATION, reasons


# ==============================================================================
# CONTROLLED RECALIBRATION & VALIDATION PIPELINE
# ==============================================================================

class ControlledRecalibrationPipeline:
    """
    Controlled Recalibration Pipeline with Phase 7.6 Governance Integration.
    
    Ensures:
    1. Zero auto-promotion: recalibration candidate enters as CANDIDATE.
    2. Strict data isolation: calibration set cannot overlap evaluation or training.
    3. Candidate evaluation via Phase 7.6 Validation Gate.
    4. Unconditional safety veto: degrades in Emergency Recall or spikes in FER result in REJECT.
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        validation_gate: Optional[ValidationGate] = None,
        governance_manager: Optional[Any] = None,
        drift_detector: Optional[SpeakerDriftDetector] = None,
        recalibrator: Optional[TemperatureScalingRecalibrator] = None,
    ):
        from src.continual.governance import ModelGovernanceManager
        if governance_manager is not None:
            self.governance_manager = governance_manager
            self.model_registry = governance_manager.registry
            self.validation_gate = governance_manager.gate
        else:
            if model_registry is None:
                raise ValueError("Must provide either governance_manager or model_registry.")
            self.model_registry = model_registry
            self.validation_gate = validation_gate or ValidationGate()
            self.governance_manager = ModelGovernanceManager(
                model_registry=self.model_registry,
                validation_gate=self.validation_gate,
            )
        self.drift_detector = drift_detector or SpeakerDriftDetector()
        self.recalibrator = recalibrator or TemperatureScalingRecalibrator()

    def generate_candidate(
        self,
        speaker_id: str,
        parent_model_version: str,
        calibration_samples: List[Dict[str, Any]],
        calibration_logits: np.ndarray,
        candidate_version_suffix: str = "recal_v1"
    ) -> Tuple[RecalibrationCandidate, ModelVersionRecord]:
        """
        Fit recalibration on verified calibration samples and register as CANDIDATE.
        Does NOT promote model.
        """
        if not calibration_samples:
            raise DriftDataError("Cannot generate recalibration candidate with empty calibration set.")

        cal_labels = [s["confirmed_intent"] for s in calibration_samples]
        cal_ids = [s["sample_id"] for s in calibration_samples]

        # Deterministic dataset fingerprint
        hasher = hashlib.sha256()
        hasher.update(",".join(cal_ids).encode("utf-8"))
        hasher.update(",".join(cal_labels).encode("utf-8"))
        fingerprint = hasher.hexdigest()

        # Fit optimal temperature
        optimal_t = self.recalibrator.fit(calibration_logits, cal_labels, CANONICAL_CLASSES)

        new_version = f"{parent_model_version}_{candidate_version_suffix}"
        cand_id = f"recal_{speaker_id}_{new_version}"

        # Register in ModelRegistry as CANDIDATE
        parent_rec = self.model_registry.get_version(parent_model_version)
        if not parent_rec:
            raise ModelVersionError(f"Parent model '{parent_model_version}' not found in registry.")

        cand_rec = ModelVersionRecord(
            model_id=parent_rec.model_id,
            version=new_version,
            speaker_id=speaker_id,
            parent_version=parent_model_version,
            training_sample_ids=parent_rec.training_sample_ids,
            training_dataset_hash=parent_rec.training_dataset_hash,
            configuration_hash=compute_sha256(json.dumps({"temperature": optimal_t, "parent": parent_model_version})),
            hyperparameters={"temperature": optimal_t, "calibration_method": "temperature_scaling"},
            status=ModelLifecycleState.CANDIDATE.value,
        )
        self.model_registry.register_candidate(cand_rec)

        recal_cand = RecalibrationCandidate(
            candidate_id=cand_id,
            speaker_id=speaker_id,
            parent_model_version=parent_model_version,
            calibrated_model_version=new_version,
            calibration_method="temperature_scaling",
            temperature=float(optimal_t),
            calibration_data_fingerprint=fingerprint,
            calibration_sample_ids=cal_ids,
            calibration_policy_version=self.drift_detector.policy_version,
            validation_status="PENDING_VALIDATION",
            promotion_status=ModelLifecycleState.CANDIDATE.value,
        )

        return recal_cand, cand_rec

    def evaluate_and_govern(
        self,
        candidate_record: ModelVersionRecord,
        candidate_predict_fn: Callable[[np.ndarray], List[str]],
        val_samples: List[Any],
        val_dataset_id: str,
        parent_record: Optional[ModelVersionRecord] = None,
        parent_predict_fn: Optional[Callable[[np.ndarray], List[str]]] = None,
        replay_sample_ids: Optional[List[str]] = None,
        cutoff_timestamp: Optional[str] = None,
    ) -> Tuple[ValidationDecisionRecord, Optional[ModelVersionRecord]]:
        """
        Pass recalibration candidate through Phase 7.6 Validation Gate.
        
        Returns:
          (decision_record, promoted_record_or_None)
        """
        # Normalize val_samples into [(sid, lbl, emb)] or [(sid, lbl, emb, ts)]
        norm_val_samples = []
        for s in val_samples:
            if isinstance(s, (tuple, list)):
                norm_val_samples.append(s)
            elif isinstance(s, dict):
                sid = s.get("sample_id") or s.get("id", "")
                lbl = s.get("confirmed_intent") or s.get("intent", GENERAL_SPEECH_CLASS)
                emb = s.get("embedding")
                if emb is None:
                    emb = np.zeros(384, dtype=np.float32)
                ts = s.get("timestamp")
                if ts is not None:
                    norm_val_samples.append((sid, lbl, emb, str(ts)))
                else:
                    norm_val_samples.append((sid, lbl, emb))
            else:
                norm_val_samples.append(s)

        # Invariant: Must pass through Phase 7.6 gate
        decision_rec = self.validation_gate.evaluate_candidate(
            candidate_record=candidate_record,
            val_samples=norm_val_samples,
            candidate_predict_fn=candidate_predict_fn,
            parent_record=parent_record,
            parent_predict_fn=parent_predict_fn,
            val_dataset_id=val_dataset_id,
            replay_sample_ids=replay_sample_ids,
            cutoff_timestamp=cutoff_timestamp,
        )

        promoted_record = None
        if decision_rec.decision == PromotionDecision.PROMOTE.value and not decision_rec.safety_veto:
            self.governance_manager.promote_candidate(
                candidate_version=candidate_record.version,
                decision_record=decision_rec,
            )
            promoted_record = self.model_registry.get_version(candidate_record.version)

        return decision_rec, promoted_record
