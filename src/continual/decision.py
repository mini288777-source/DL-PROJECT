"""
Structured Decision & Abstention Engine for Continual Personalization (Phase 7.2).
Evaluates prediction trustworthiness, enforces risk-aware abstention policies,
and determines verification requirements.

CRITICAL INVARIANTS:
1. Prediction != Confidence != Verification != Ground Truth != Learning Eligibility.
2. High confidence NEVER creates ground truth or automatic verification.
3. Emergency communication enforces strictly risk-averse confirmation policies.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional
import uuid

from src.continual.confidence import (
    ConfidenceRepresentation,
    ConfidenceBand,
    load_confidence_policy,
)
from src.continual.taxonomy import is_valid_intent, is_emergency_intent


class ActionDecision(str, Enum):
    """Routing and execution decision for an inference event."""
    ACCEPT = "ACCEPT"    # Prediction is confident enough to route
    CONFIRM = "CONFIRM"  # Prediction requires user / caregiver confirmation
    ABSTAIN = "ABSTAIN"  # Prediction is withheld due to uncertainty or safety risk


class SuggestedAction(str, Enum):
    """Actionable machine-readable recommendation for Phase 7.3 human-in-the-loop."""
    ROUTE_PREDICTION = "ROUTE_PREDICTION"
    REQUEST_USER_CONFIRMATION = "REQUEST_USER_CONFIRMATION"
    REQUEST_CLINICAL_REVIEW = "REQUEST_CLINICAL_REVIEW"
    ESCALATE_SAFETY = "ESCALATE_SAFETY"
    FALLBACK_TO_CARE_SERVICE = "FALLBACK_TO_CARE_SERVICE"


class PrioritizationTag(str, Enum):
    """Active-learning priority indicators for Phase 7.3."""
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    HIGH_EMERGENCY_RISK = "HIGH_EMERGENCY_RISK"
    NOVEL_SPEAKER_PATTERN = "NOVEL_SPEAKER_PATTERN"
    DISTRIBUTIONALLY_UNUSUAL = "DISTRIBUTIONALLY_UNUSUAL"


@dataclass
class ConfidenceDecision:
    """
    Structured, immutable decision object recording confidence assessment,
    abstention choices, verification requirements, and policy provenance.
    """
    decision_id: str
    experience_id: str
    speaker_id: str
    session_id: str
    model_version: str

    predicted_label: str

    raw_confidence: float
    confidence_type: str
    confidence_status: str
    confidence_band: str

    uncertainty_measures: Dict[str, float] = field(default_factory=dict)
    uncertainty_status: str = "AVAILABLE"

    decision: str = ActionDecision.ABSTAIN.value
    decision_reason: str = "UNINITIALIZED"

    verification_required: bool = True
    learning_eligible: bool = False

    suggested_action: str = SuggestedAction.REQUEST_USER_CONFIRMATION.value
    prioritization_tags: List[str] = field(default_factory=list)

    policy_version: str = "v1.0.0_research_draft"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision object to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfidenceDecision":
        """Instantiate decision object from dictionary."""
        return cls(**data)


class DecisionEngine:
    """
    Deterministic rule engine that maps confidence and uncertainty measurements
    to risk-aware actions (ACCEPT / CONFIRM / ABSTAIN) and verification requirements.
    """

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy or load_confidence_policy()
        self.policy_version = self.policy.get("policy_metadata", {}).get(
            "policy_version", "v1.0.0_research_draft"
        )

    def evaluate(
        self,
        predicted_label: str,
        confidence_rep: ConfidenceRepresentation,
        speaker_id: str,
        session_id: str = "session_default",
        experience_id: Optional[str] = None,
        model_version: str = "v_base",
        uncertainty_measures: Optional[Dict[str, float]] = None,
        quality_score: float = 1.0,
    ) -> ConfidenceDecision:
        """
        Evaluate prediction trustworthiness and generate an auditable ConfidenceDecision.
        """
        if experience_id is None:
            experience_id = f"exp_{uuid.uuid4().hex[:12]}"

        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        unc_measures = uncertainty_measures or {}
        prioritization_tags: List[str] = []

        abstention_cfg = self.policy.get("abstention_policy", {})
        gen_cfg = abstention_cfg.get("general_speech", {"accept_threshold": 0.70, "confirm_threshold": 0.50})
        em_cfg = abstention_cfg.get("emergency_speech", {"accept_threshold": 0.85, "confirm_threshold": 0.65})

        # Check taxonomic validity
        if not is_valid_intent(predicted_label):
            return ConfidenceDecision(
                decision_id=decision_id,
                experience_id=experience_id,
                speaker_id=speaker_id,
                session_id=session_id,
                model_version=model_version,
                predicted_label=predicted_label,
                raw_confidence=confidence_rep.confidence_value,
                confidence_type=confidence_rep.confidence_type.value,
                confidence_status="INVALID_TAXONOMY",
                confidence_band=ConfidenceBand.INVALID_CONFIDENCE.value,
                uncertainty_measures=unc_measures,
                uncertainty_status="UNAVAILABLE",
                decision=ActionDecision.ABSTAIN.value,
                decision_reason="INVALID_TAXONOMY_LABEL",
                verification_required=True,
                learning_eligible=False,
                suggested_action=SuggestedAction.REQUEST_CLINICAL_REVIEW.value,
                prioritization_tags=["INVALID_TAXONOMY"],
                policy_version=self.policy_version,
            )

        # Check confidence mathematical validity
        if not confidence_rep.is_valid:
            return ConfidenceDecision(
                decision_id=decision_id,
                experience_id=experience_id,
                speaker_id=speaker_id,
                session_id=session_id,
                model_version=model_version,
                predicted_label=predicted_label,
                raw_confidence=confidence_rep.confidence_value if isinstance(confidence_rep.confidence_value, (int, float)) else 0.0,
                confidence_type=confidence_rep.confidence_type.value,
                confidence_status="INVALID_CONFIDENCE",
                confidence_band=ConfidenceBand.INVALID_CONFIDENCE.value,
                uncertainty_measures=unc_measures,
                uncertainty_status="UNAVAILABLE",
                decision=ActionDecision.ABSTAIN.value,
                decision_reason="INVALID_CONFIDENCE_MEASUREMENT",
                verification_required=True,
                learning_eligible=False,
                suggested_action=SuggestedAction.REQUEST_CLINICAL_REVIEW.value,
                prioritization_tags=["INVALID_CONFIDENCE"],
                policy_version=self.policy_version,
            )

        conf_val = confidence_rep.confidence_value
        is_emergency = is_emergency_intent(predicted_label)

        # Prioritization tag assignment for active learning
        if conf_val < self.policy.get("prioritization_policy", {}).get("low_confidence_tag_threshold", 0.65):
            prioritization_tags.append(PrioritizationTag.LOW_CONFIDENCE.value)

        margin = unc_measures.get("top1_top2_margin", 1.0)
        if margin < self.policy.get("prioritization_policy", {}).get("high_uncertainty_margin_threshold", 0.15):
            prioritization_tags.append(PrioritizationTag.HIGH_UNCERTAINTY.value)

        if is_emergency:
            prioritization_tags.append(PrioritizationTag.HIGH_EMERGENCY_RISK.value)

        # Acoustic degradation safeguard
        if quality_score < self.policy.get("safety_invariants", {}).get("min_acoustic_quality_score", 0.50):
            return ConfidenceDecision(
                decision_id=decision_id,
                experience_id=experience_id,
                speaker_id=speaker_id,
                session_id=session_id,
                model_version=model_version,
                predicted_label=predicted_label,
                raw_confidence=conf_val,
                confidence_type=confidence_rep.confidence_type.value,
                confidence_status="VALID",
                confidence_band=confidence_rep.confidence_band.value,
                uncertainty_measures=unc_measures,
                uncertainty_status="AVAILABLE",
                decision=ActionDecision.ABSTAIN.value,
                decision_reason="DEGRADED_ACOUSTIC_QUALITY",
                verification_required=True,
                learning_eligible=False,
                suggested_action=SuggestedAction.FALLBACK_TO_CARE_SERVICE.value,
                prioritization_tags=prioritization_tags,
                policy_version=self.policy_version,
            )

        # Emergency Decision Pathway
        if is_emergency:
            accept_thresh = em_cfg.get("accept_threshold", 0.85)
            confirm_thresh = em_cfg.get("confirm_threshold", 0.65)

            if conf_val >= accept_thresh:
                # Emergency routing accepted, but verification is still required for future learning!
                action = ActionDecision.ACCEPT.value
                reason = "HIGH_CONFIDENCE_EMERGENCY_ACCEPTED"
                action_sugg = SuggestedAction.ROUTE_PREDICTION.value
                verif_req = True
            elif conf_val >= confirm_thresh:
                action = ActionDecision.CONFIRM.value
                reason = "EMERGENCY_CONFIRMATION_REQUIRED"
                action_sugg = SuggestedAction.REQUEST_USER_CONFIRMATION.value
                verif_req = True
            else:
                action = ActionDecision.ABSTAIN.value
                reason = "LOW_CONFIDENCE_EMERGENCY_ABSTAIN"
                action_sugg = SuggestedAction.ESCALATE_SAFETY.value
                verif_req = True

        # General Speech Decision Pathway
        else:
            accept_thresh = gen_cfg.get("accept_threshold", 0.70)
            confirm_thresh = gen_cfg.get("confirm_threshold", 0.50)

            if conf_val >= accept_thresh:
                action = ActionDecision.ACCEPT.value
                reason = "HIGH_CONFIDENCE_GENERAL_SPEECH_ACCEPTED"
                action_sugg = SuggestedAction.ROUTE_PREDICTION.value
                verif_req = False
            elif conf_val >= confirm_thresh:
                action = ActionDecision.CONFIRM.value
                reason = "MEDIUM_CONFIDENCE_GENERAL_SPEECH_CONFIRM"
                action_sugg = SuggestedAction.REQUEST_USER_CONFIRMATION.value
                verif_req = True
            else:
                action = ActionDecision.ABSTAIN.value
                reason = "LOW_CONFIDENCE_GENERAL_SPEECH_ABSTAIN"
                action_sugg = SuggestedAction.FALLBACK_TO_CARE_SERVICE.value
                verif_req = True

        # CRITICAL SAFETY INVARIANT: An unverified prediction is NEVER learning-eligible!
        learning_eligible = False

        return ConfidenceDecision(
            decision_id=decision_id,
            experience_id=experience_id,
            speaker_id=speaker_id,
            session_id=session_id,
            model_version=model_version,
            predicted_label=predicted_label,
            raw_confidence=conf_val,
            confidence_type=confidence_rep.confidence_type.value,
            confidence_status="VALID",
            confidence_band=confidence_rep.confidence_band.value,
            uncertainty_measures=unc_measures,
            uncertainty_status="AVAILABLE",
            decision=action,
            decision_reason=reason,
            verification_required=verif_req,
            learning_eligible=learning_eligible,
            suggested_action=action_sugg,
            prioritization_tags=prioritization_tags,
            policy_version=self.policy_version,
        )
