"""
Active-Learning Review Queue & Human Review Service (Phase 7.3).
Provides:
- In-memory and persistent review queue abstractions for scheduling human review.
- Deterministic priority ordering (HIGH -> MEDIUM -> LOW).
- Speaker-isolated and session-scoped review access.
- Safe integration with ExperienceStore and EligibilityEngine.
- Idempotent submission handling and conflict prevention.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.store import (
    ExperienceStore,
    RecordNotFoundError,
    SpeakerIsolationError,
)
from src.continual.review import (
    ReviewRequest,
    ReviewResponse,
    ReviewState,
    ReviewAction,
    ReviewPriority,
    VerifierType,
    ReviewValidationError,
    ReviewConflictError,
    InvalidReviewStateTransitionError,
    ReviewNotFoundError,
    determine_review_priority,
    validate_review_request,
    validate_review_response,
    load_review_policy,
)
from src.continual.decision import ConfidenceDecision, ActionDecision
from src.continual.eligibility import evaluate_learning_eligibility


class ReviewQueue(ABC):
    """Abstract interface for active learning review queue."""

    @abstractmethod
    def enqueue(self, request: ReviewRequest) -> ReviewRequest:
        """Enqueue a new review request."""
        pass

    @abstractmethod
    def get_review(self, review_id: str) -> Optional[ReviewRequest]:
        """Retrieve review request by ID."""
        pass

    @abstractmethod
    def get_next_review(
        self,
        speaker_id: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> Optional[ReviewRequest]:
        """
        Get the next pending review request, ordered deterministically
        by priority (HIGH -> MEDIUM -> LOW) and chronological timestamp.
        """
        pass

    @abstractmethod
    def get_by_priority(
        self, priority: str, speaker_id: Optional[str] = None
    ) -> List[ReviewRequest]:
        """List reviews matching priority tier."""
        pass

    @abstractmethod
    def get_by_speaker(
        self, speaker_id: str, pending_only: bool = False
    ) -> List[ReviewRequest]:
        """List reviews scoped to a specific speaker."""
        pass

    @abstractmethod
    def get_by_session(
        self, session_id: str, pending_only: bool = False
    ) -> List[ReviewRequest]:
        """List reviews for a given session."""
        pass

    @abstractmethod
    def get_pending(self, speaker_id: Optional[str] = None) -> List[ReviewRequest]:
        """List all pending reviews."""
        pass

    @abstractmethod
    def mark_reviewed(
        self, review_id: str, response: ReviewResponse
    ) -> ReviewRequest:
        """Mark review as resolved with response, ensuring idempotency."""
        pass

    @abstractmethod
    def count_pending(self, speaker_id: Optional[str] = None) -> int:
        """Count pending reviews."""
        pass


class InMemoryReviewQueue(ReviewQueue):
    """
    In-memory deterministic ReviewQueue implementation.
    Maintains index by review_id, experience_id, speaker_id, and priority.
    """

    _PRIORITY_ORDER = {
        ReviewPriority.HIGH.value: 0,
        ReviewPriority.MEDIUM.value: 1,
        ReviewPriority.LOW.value: 2,
    }

    def __init__(self):
        self._reviews: Dict[str, ReviewRequest] = {}
        self._by_experience: Dict[str, str] = {}  # experience_id -> review_id
        self._by_speaker: Dict[str, List[str]] = defaultdict(list)
        self._by_session: Dict[str, List[str]] = defaultdict(list)
        self._responses: Dict[str, ReviewResponse] = {}

    def enqueue(self, request: ReviewRequest) -> ReviewRequest:
        errs = validate_review_request(request)
        if errs:
            raise ReviewValidationError(f"Invalid review request: {'; '.join(errs)}")

        # Check if experience already has a pending or active review
        if request.experience_id in self._by_experience:
            existing_id = self._by_experience[request.experience_id]
            existing = self._reviews[existing_id]
            if existing.is_pending:
                # Return existing pending review (idempotent enqueue)
                return existing

        self._reviews[request.review_id] = request
        self._by_experience[request.experience_id] = request.review_id
        self._by_speaker[request.speaker_id].append(request.review_id)
        self._by_session[request.session_id].append(request.review_id)
        return request

    def get_review(self, review_id: str) -> Optional[ReviewRequest]:
        return self._reviews.get(review_id)

    def get_next_review(
        self,
        speaker_id: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> Optional[ReviewRequest]:
        pending = self.get_pending(speaker_id=speaker_id)
        if not pending:
            return None

        if priority is not None:
            pending = [r for r in pending if r.priority == priority]
            if not pending:
                return None

        # Sort by priority rank ascending (HIGH=0, MED=1, LOW=2), then requested_at ascending
        def sort_key(r: ReviewRequest):
            rank = self._PRIORITY_ORDER.get(r.priority, 1)
            return (rank, r.requested_at)

        pending.sort(key=sort_key)
        return pending[0]

    def get_by_priority(
        self, priority: str, speaker_id: Optional[str] = None
    ) -> List[ReviewRequest]:
        res = [
            r for r in self._reviews.values()
            if r.priority == priority
            and (speaker_id is None or r.speaker_id == speaker_id)
        ]
        res.sort(key=lambda r: r.requested_at)
        return res

    def get_by_speaker(
        self, speaker_id: str, pending_only: bool = False
    ) -> List[ReviewRequest]:
        rev_ids = self._by_speaker.get(speaker_id, [])
        res = [self._reviews[rid] for rid in rev_ids if rid in self._reviews]
        if pending_only:
            res = [r for r in res if r.is_pending]
        res.sort(key=lambda r: r.requested_at)
        return res

    def get_by_session(
        self, session_id: str, pending_only: bool = False
    ) -> List[ReviewRequest]:
        rev_ids = self._by_session.get(session_id, [])
        res = [self._reviews[rid] for rid in rev_ids if rid in self._reviews]
        if pending_only:
            res = [r for r in res if r.is_pending]
        res.sort(key=lambda r: r.requested_at)
        return res

    def get_pending(self, speaker_id: Optional[str] = None) -> List[ReviewRequest]:
        res = [
            r for r in self._reviews.values()
            if r.is_pending and (speaker_id is None or r.speaker_id == speaker_id)
        ]
        res.sort(key=lambda r: (self._PRIORITY_ORDER.get(r.priority, 1), r.requested_at))
        return res

    def mark_reviewed(
        self, review_id: str, response: ReviewResponse
    ) -> ReviewRequest:
        req = self.get_review(review_id)
        if not req:
            raise ReviewNotFoundError(f"Review request '{review_id}' not found")

        errs = validate_review_response(req, response)
        if errs:
            raise ReviewValidationError(f"Invalid review response: {'; '.join(errs)}")

        # Check idempotency if already finalized
        if not req.is_pending:
            existing_resp = self._responses.get(review_id)
            if existing_resp:
                if (
                    existing_resp.review_action == response.review_action
                    and existing_resp.verified_label == response.verified_label
                ):
                    return req  # Idempotent match
            raise ReviewConflictError(
                f"Review '{review_id}' already finalized in state '{req.status}' with label '{req.verified_label}'. "
                f"Cannot overwrite with action '{response.review_action}'."
            )

        # Transition state
        if response.review_action == ReviewAction.ACCEPT.value:
            req.status = ReviewState.ACCEPTED.value
            req.verified_label = req.predicted_label
        elif response.review_action == ReviewAction.CORRECT.value:
            req.status = ReviewState.CORRECTED.value
            req.verified_label = response.verified_label
        elif response.review_action in (ReviewAction.REJECT.value, ReviewAction.INVALID.value):
            req.status = ReviewState.REJECTED.value
            req.verified_label = None

        req.response_id = response.response_id
        req.review_action = response.review_action
        req.resolved_at = response.submitted_at or datetime.now(timezone.utc).isoformat()

        self._responses[review_id] = response
        self._reviews[review_id] = req
        return req

    def count_pending(self, speaker_id: Optional[str] = None) -> int:
        return len(self.get_pending(speaker_id=speaker_id))

    def count_by_priority(self, speaker_id: Optional[str] = None) -> Dict[str, int]:
        counts = {p.value: 0 for p in ReviewPriority}
        for r in self.get_pending(speaker_id=speaker_id):
            if r.priority in counts:
                counts[r.priority] += 1
        return counts


class PersistentReviewQueue(InMemoryReviewQueue):
    """
    JSON Lines persistent review queue.
    Appends review requests and responses to disk for durability and auditability.
    """

    def __init__(self, storage_dir: str):
        super().__init__()
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.requests_file = os.path.join(storage_dir, "review_requests.jsonl")
        self.responses_file = os.path.join(storage_dir, "review_responses.jsonl")
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(self.requests_file):
            with open(self.requests_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        req_dict = json.loads(line)
                        req = ReviewRequest.from_dict(req_dict)
                        self._reviews[req.review_id] = req
                        self._by_experience[req.experience_id] = req.review_id
                        self._by_speaker[req.speaker_id].append(req.review_id)
                        self._by_session[req.session_id].append(req.review_id)

        if os.path.exists(self.responses_file):
            with open(self.responses_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        resp_dict = json.loads(line)
                        resp = ReviewResponse.from_dict(resp_dict)
                        self._responses[resp.review_id] = resp
                        if resp.review_id in self._reviews:
                            r = self._reviews[resp.review_id]
                            r.response_id = resp.response_id
                            r.review_action = resp.review_action
                            if resp.review_action == ReviewAction.ACCEPT.value:
                                r.status = ReviewState.ACCEPTED.value
                                r.verified_label = r.predicted_label
                            elif resp.review_action == ReviewAction.CORRECT.value:
                                r.status = ReviewState.CORRECTED.value
                                r.verified_label = resp.verified_label
                            elif resp.review_action in (ReviewAction.REJECT.value, ReviewAction.INVALID.value):
                                r.status = ReviewState.REJECTED.value
                                r.verified_label = None
                            r.resolved_at = resp.submitted_at

    def enqueue(self, request: ReviewRequest) -> ReviewRequest:
        res = super().enqueue(request)
        with open(self.requests_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(res.to_dict()) + "\n")
        return res

    def mark_reviewed(
        self, review_id: str, response: ReviewResponse
    ) -> ReviewRequest:
        req = super().mark_reviewed(review_id, response)
        with open(self.responses_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(response.to_dict()) + "\n")
        return req


class HumanReviewService:
    """
    Coordinating service bridging ExperienceStore, ReviewQueue, and EligibilityEngine.
    Enforces that:
    1. Model confidence NEVER verifies ground truth.
    2. Only explicit human review actions transition verification states.
    3. Speaker isolation is strictly preserved.
    4. Canonical taxonomy is rigorously enforced.
    5. Learning eligibility is evaluated only after human verification.
    """

    def __init__(self, queue: ReviewQueue, store: ExperienceStore, policy: Optional[Dict[str, Any]] = None):
        self.queue = queue
        self.store = store
        self.policy = policy or load_review_policy()

    def create_review_request(
        self,
        record: ExperienceRecord,
        decision: Optional[ConfidenceDecision] = None,
        priority: Optional[str] = None,
        requested_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReviewRequest:
        """
        Create and enqueue a review request for an ExperienceRecord.
        Computes deterministic priority using decision and confidence signals.
        """
        if not record.sample_id:
            raise ReviewValidationError("ExperienceRecord must have a valid sample_id")
        if not record.speaker_id:
            raise ReviewValidationError("ExperienceRecord must have a valid speaker_id")

        p_tags = decision.prioritization_tags if decision else []
        pred_label = record.predicted_intent
        conf = record.prediction_confidence
        dec_val = decision.decision if decision else ActionDecision.CONFIRM.value
        dec_reason = decision.decision_reason if decision else "CONFIRMATION_REQUIRED"

        computed_priority = priority or determine_review_priority(
            predicted_label=pred_label,
            confidence=conf,
            decision=dec_val,
            prioritization_tags=p_tags,
            policy=self.policy,
        )

        req = ReviewRequest(
            review_id=f"rev_{record.sample_id}",
            experience_id=record.sample_id,
            speaker_id=record.speaker_id,
            session_id=record.session_id,
            predicted_label=pred_label,
            confidence=conf,
            confidence_type=decision.confidence_type if decision else "SOFTMAX_PROBABILITY",
            decision=dec_val,
            decision_reason=dec_reason,
            priority=computed_priority,
            prioritization_tags=p_tags,
            requested_by=requested_by,
            metadata=metadata or {},
        )
        return self.queue.enqueue(req)

    def process_review(
        self, response: ReviewResponse
    ) -> Tuple[ReviewRequest, ExperienceRecord]:
        """
        Process human review response:
        1. Validate review response and canonical taxonomy.
        2. Handle idempotency and check for conflicts.
        3. Enforce speaker isolation against the ExperienceRecord.
        4. Mutate ExperienceRecord verification state accordingly.
        5. Re-evaluate learning eligibility via Phase 7.2 rules.
        6. Update review request in the queue.
        """
        req = self.queue.get_review(response.review_id)
        if not req:
            raise ReviewNotFoundError(f"Review request '{response.review_id}' not found")

        record = self.store.read(response.experience_id)
        if not record:
            raise RecordNotFoundError(f"ExperienceRecord '{response.experience_id}' not found")

        # Speaker isolation assertion
        if record.speaker_id != req.speaker_id:
            raise SpeakerIsolationError(
                f"Speaker mismatch: Record speaker '{record.speaker_id}' != Request speaker '{req.speaker_id}'"
            )

        # Mark reviewed in queue (handles idempotency & conflicts)
        updated_req = self.queue.mark_reviewed(response.review_id, response)

        # Normalize verifier_type to protocol canonical confirmation_source
        raw_source = response.verifier_type or VerifierType.USER.value
        if raw_source in ("user", "user_explicit", "researcher", "researcher_verified"):
            verif_source = "user_explicit"
        elif raw_source in ("clinician", "clinician_verified", "caregiver", "caregiver_verified"):
            verif_source = "clinician_verified"
        elif raw_source in ("oracle", "oracle_ground_truth", "oracle_eval"):
            verif_source = "oracle_ground_truth"
        else:
            verif_source = "user_explicit"

        # Mutate ExperienceRecord based on explicit human review action
        if response.review_action == ReviewAction.ACCEPT.value:
            # Human explicitly confirmed the prediction
            updated_record = self.store.mark_verified(
                sample_id=record.sample_id,
                verified_label=req.predicted_label,
                verification_source=verif_source,
                verifier_id=response.verifier_id,
                note=response.comment or "Human reviewer confirmed predicted label as ground truth",
            )

        elif response.review_action == ReviewAction.CORRECT.value:
            # Human supplied a corrected canonical label
            if not response.verified_label:
                raise ReviewValidationError("CORRECT action requires an explicit verified_label")
            updated_record = self.store.mark_verified(
                sample_id=record.sample_id,
                verified_label=response.verified_label,
                verification_source=verif_source,
                verifier_id=response.verifier_id,
                note=response.comment or f"Human reviewer corrected prediction from {req.predicted_label} to {response.verified_label}",
            )

        elif response.review_action in (ReviewAction.REJECT.value, ReviewAction.INVALID.value):
            # Human rejected interaction or marked invalid
            updated_record = self.store.mark_rejected(
                sample_id=record.sample_id,
                reason=response.reason or f"Human reviewer {response.review_action.lower()}ed record",
                verifier_id=response.verifier_id,
            )
            if response.review_action == ReviewAction.INVALID.value:
                updated_record.verification_status = VerificationStatus.INVALID.value
                updated_record.eligibility_reason = EligibilityReason.INELIGIBLE_INVALID_SCHEMA.value
                self.store.update(updated_record)

        else:
            raise ReviewValidationError(f"Unsupported review action '{response.review_action}'")

        return updated_req, updated_record
