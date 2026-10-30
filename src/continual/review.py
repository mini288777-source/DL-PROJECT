"""
Human-in-the-Loop Review Data Structures, Validation & State Machine (Phase 7.3).
Provides:
- Strongly-typed ReviewRequest and ReviewResponse objects.
- Deterministic review state lifecycle (PENDING_REVIEW -> ACCEPTED / CORRECTED / REJECTED / EXPIRED).
- Canonical taxonomy enforcement for user corrections.
- Ground-truth safety barriers ensuring prediction confidence NEVER establishes verification.
- Speaker, model, and policy provenance tracking.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple
import uuid
import yaml

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.taxonomy import (
    CANONICAL_CLASSES,
    is_valid_intent,
    is_emergency_intent,
)
from src.continual.decision import (
    ActionDecision,
    ConfidenceDecision,
    PrioritizationTag,
)


class ReviewState(str, Enum):
    """Lifecycle state of an active human-in-the-loop review request."""
    PENDING_REVIEW = "PENDING_REVIEW"
    ACCEPTED = "ACCEPTED"
    CORRECTED = "CORRECTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ReviewAction(str, Enum):
    """Explicit human action taken in response to a review request."""
    ACCEPT = "ACCEPT"    # Confirms predicted label as ground truth
    CORRECT = "CORRECT"  # Replaces with a valid canonical intent
    REJECT = "REJECT"    # Excludes from learning dataset
    INVALID = "INVALID"  # Unintelligible or corrupted audio


class ReviewPriority(str, Enum):
    """Deterministic scheduling priority for active learning."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class VerifierType(str, Enum):
    """Role classification of the human reviewer matching protocol contract."""
    USER = "user_explicit"
    CLINICIAN = "clinician_verified"
    CAREGIVER = "clinician_verified"
    RESEARCHER = "user_explicit"
    ORACLE = "oracle_ground_truth"


class ReviewValidationError(Exception):
    """Raised when review request or response fails contract validation."""
    pass


class ReviewConflictError(Exception):
    """Raised when an already finalized review receives a conflicting action."""
    pass


class InvalidReviewStateTransitionError(Exception):
    """Raised when attempting an invalid review lifecycle transition."""
    pass


class ReviewNotFoundError(Exception):
    """Raised when a review request is not found by ID."""
    pass


def load_review_policy(policy_path: Optional[str] = None) -> Dict[str, Any]:
    """Load research review policy configuration."""
    if policy_path is None:
        policy_path = os.path.join("config", "continual", "review_policy.yaml")
    if not os.path.exists(policy_path):
        return {
            "policy_metadata": {
                "policy_id": "continual_review_policy_default",
                "policy_version": "v1.0.0_research_draft",
            },
            "prioritization_policy": {
                "default_priority": "MEDIUM",
                "high_priority_tags": [
                    "HIGH_EMERGENCY_RISK",
                    "HIGH_UNCERTAINTY",
                    "LOW_CONFIDENCE",
                ],
                "high_priority_decisions": ["ABSTAIN"],
                "low_priority_decisions": ["ACCEPT"],
            },
            "safety_invariants": {
                "ground_truth_requires_explicit_verification": True,
                "allow_automatic_label_promotion": False,
                "enforce_canonical_taxonomy": True,
            },
        }
    with open(policy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def determine_review_priority(
    predicted_label: str,
    confidence: float,
    decision: str = ActionDecision.CONFIRM.value,
    prioritization_tags: Optional[List[str]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Deterministic rule-based prioritization for review scheduling.
    Evaluates emergency status, confidence bands, uncertainty tags, and abstention states.
    """
    cfg = policy or load_review_policy()
    p_cfg = cfg.get("prioritization_policy", {})
    tags = set(prioritization_tags or [])

    # 1. High priority triggers:
    # - Abstention decision
    # - High uncertainty / low confidence tags
    # - Emergency intent with confidence below acceptance threshold
    if decision == ActionDecision.ABSTAIN.value or decision in p_cfg.get("high_priority_decisions", []):
        return ReviewPriority.HIGH.value

    high_tags = set(p_cfg.get("high_priority_tags", []))
    if any(t in high_tags for t in tags):
        return ReviewPriority.HIGH.value

    if is_emergency_intent(predicted_label) and confidence < 0.85:
        return ReviewPriority.HIGH.value

    # 2. Low priority triggers:
    # - Clear accepted non-emergency prediction
    if decision == ActionDecision.ACCEPT.value and not is_emergency_intent(predicted_label) and confidence >= 0.70:
        return ReviewPriority.LOW.value

    # 3. Default priority
    return p_cfg.get("default_priority", ReviewPriority.MEDIUM.value)


@dataclass
class ReviewRequest:
    """
    Structured, auditable human review request linking to an underlying ExperienceRecord.
    Contains prediction context, confidence signals, priority, and state machine tracking.
    """
    review_id: str
    experience_id: str
    speaker_id: str
    session_id: str
    predicted_label: str
    confidence: float
    confidence_type: str = "SOFTMAX_PROBABILITY"
    decision: str = ActionDecision.CONFIRM.value
    decision_reason: str = "UNSPECIFIED"

    priority: str = ReviewPriority.MEDIUM.value
    prioritization_tags: List[str] = field(default_factory=list)

    requested_at: str = ""
    requested_by: str = "system"
    review_policy_version: str = "v1.0.0_research_draft"
    status: str = ReviewState.PENDING_REVIEW.value

    response_id: Optional[str] = None
    review_action: Optional[str] = None
    verified_label: Optional[str] = None
    resolved_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.requested_at:
            self.requested_at = datetime.now(timezone.utc).isoformat()
        if not self.review_id:
            self.review_id = f"rev_{uuid.uuid4().hex[:12]}"

    @property
    def is_pending(self) -> bool:
        return self.status == ReviewState.PENDING_REVIEW.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewRequest":
        return cls(**data)


@dataclass
class ReviewResponse:
    """
    Auditable record of a human verifier's review decision.
    Preserves original prediction while capturing the human-verified ground-truth intent.
    """
    response_id: str
    review_id: str
    experience_id: str
    review_action: str  # ACCEPT, CORRECT, REJECT, INVALID

    verified_label: Optional[str] = None
    verification_status: str = VerificationStatus.USER_VERIFIED.value
    verifier_type: str = VerifierType.USER.value
    verifier_id: Optional[str] = None

    submitted_at: str = ""
    policy_version: str = "v1.0.0_research_draft"
    reason: Optional[str] = None
    comment: Optional[str] = None
    validation_status: str = "VALID"

    def __post_init__(self):
        if not self.submitted_at:
            self.submitted_at = datetime.now(timezone.utc).isoformat()
        if not self.response_id:
            self.response_id = f"resp_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewResponse":
        return cls(**data)


def validate_review_request(request: ReviewRequest) -> List[str]:
    """Validate structural constraints and taxonomy compliance of a ReviewRequest."""
    errors = []
    if not request.review_id or not request.review_id.strip():
        errors.append("review_id must be a non-empty string")
    if not request.experience_id or not request.experience_id.strip():
        errors.append("experience_id must be a non-empty string")
    if not request.speaker_id or not request.speaker_id.strip():
        errors.append("speaker_id must be a non-empty string")
    if not request.predicted_label or not is_valid_intent(request.predicted_label):
        errors.append(f"predicted_label '{request.predicted_label}' must be a valid canonical taxonomy class")
    if request.status not in [s.value for s in ReviewState]:
        errors.append(f"status '{request.status}' is not a valid ReviewState")
    if request.priority not in [p.value for p in ReviewPriority]:
        errors.append(f"priority '{request.priority}' is not a valid ReviewPriority")
    return errors


def validate_review_response(request: ReviewRequest, response: ReviewResponse) -> List[str]:
    """
    Validate ReviewResponse integrity, checking action validity and canonical taxonomy compliance.
    """
    errors = []
    if response.review_id != request.review_id:
        errors.append(f"review_id mismatch: response '{response.review_id}' != request '{request.review_id}'")
    if response.experience_id != request.experience_id:
        errors.append(f"experience_id mismatch: response '{response.experience_id}' != request '{request.experience_id}'")

    if response.review_action not in [a.value for a in ReviewAction]:
        errors.append(f"Invalid review_action: '{response.review_action}'")
        return errors

    if response.review_action == ReviewAction.ACCEPT.value:
        # Upon ACCEPT, verified_label must equal original prediction
        if response.verified_label and response.verified_label != request.predicted_label:
            errors.append(
                f"ACCEPT action verified_label '{response.verified_label}' must match predicted_label '{request.predicted_label}'"
            )

    elif response.review_action == ReviewAction.CORRECT.value:
        # Upon CORRECT, verified_label must be explicitly supplied and canonical
        if not response.verified_label:
            errors.append("CORRECT action requires an explicit verified_label")
        elif not is_valid_intent(response.verified_label):
            errors.append(
                f"CORRECT action label '{response.verified_label}' is not in canonical taxonomy {CANONICAL_CLASSES}"
            )

    elif response.review_action in (ReviewAction.REJECT.value, ReviewAction.INVALID.value):
        # Upon REJECT or INVALID, verified_label should be None or ignored
        pass

    return errors
