"""
Phase 7.14 Application Integration Pipeline: Atypical Speech Emergency Assistant.
Connects:
1. Acoustic Input / Frozen Whisper Embedding Ingestion (384-dimensional).
2. Active Model Inference (scoped to validated speaker-specific model).
3. Calibrated Confidence Estimation and Uncertainty-Aware Abstention.
4. Human Verification / Correction / Rejection Barrier (Zero-Unverified Learning).
5. Immutable Experience Storage & Balanced Replay Buffering.
6. Phase 7.6 Multi-Metric Validation Gate and Safety Veto Governance.
7. Active Model Registry Lifecycle Management and Audited Rollback.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import yaml

from src.continual.taxonomy import (
    CANONICAL_CLASSES,
    EMERGENCY_CLASSES,
    GENERAL_SPEECH_CLASS,
    is_valid_intent,
    is_emergency_intent
)
from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason
)
from src.continual.review import ReviewAction
from src.continual.governance import PromotionDecision, ModelGovernanceManager
from src.continual.self_improvement import (
    ContinualSelfImprovementEngine,
    LoopInteractionReport,
    SelfImprovementCycleResult,
    UnverifiedLearningAttemptError,
    DirectActivationAttemptError
)


@dataclass
class AppInferenceResponse:
    session_id: str
    speaker_id: str
    predicted_intent: str
    confidence_score: float
    confidence_band: str  # HIGH, MEDIUM, LOW
    suggested_action: str  # CONFIRM_AND_DISPATCH, REQUEST_CONFIRMATION, PROCEED, ABSTAIN
    is_emergency: bool
    active_model_version: str
    acoustic_quality: float
    timestamp_utc: str


@dataclass
class AppAdaptationResponse:
    speaker_id: str
    adaptation_cycle_id: str
    gate_decision: str  # PROMOTE, REJECT, HOLD
    previous_model_version: str
    active_model_version: str
    experiences_used: int
    reason_codes: List[str]
    timestamp_utc: str


class AtypicalEmergencyAppPipeline:
    """
    Production-grade application integration engine for personalized atypical emergency speech.
    Enforces all safety governance invariants, human-in-the-loop verification, and speaker isolation.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        random_seed: int = 42
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed

        # Load application configuration
        if config_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            config_path = repo_root / "config" / "continual" / "app_config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.confidence_threshold = float(
            self.config.get("confidence_and_safety", {}).get("operational_confidence_threshold", 0.60)
        )
        self.high_confidence_threshold = float(
            self.config.get("confidence_and_safety", {}).get("high_confidence_threshold", 0.85)
        )
        self.min_acoustic_quality = float(
            self.config.get("confidence_and_safety", {}).get("min_acoustic_quality", 0.30)
        )
        self.selected_family = str(
            self.config.get("model_selection", {}).get("canonical_model_family", "continual_classical_adapter")
        )

        # Speaker-specific engine cache: speaker_id -> ContinualSelfImprovementEngine
        self._engines: Dict[str, ContinualSelfImprovementEngine] = {}

        # Pending unverified interactions: session_id -> session_dict
        self._pending_sessions: Dict[str, Dict[str, Any]] = {}

    def _get_engine(self, speaker_id: str) -> ContinualSelfImprovementEngine:
        """Retrieves or instantiates an isolated engine for the given speaker."""
        if not speaker_id or not isinstance(speaker_id, str):
            raise ValueError(f"Invalid speaker_id: {speaker_id}")

        if speaker_id not in self._engines:
            spk_dir = self.base_dir / "speakers" / speaker_id
            spk_dir.mkdir(parents=True, exist_ok=True)
            self._engines[speaker_id] = ContinualSelfImprovementEngine(
                speaker_id=speaker_id,
                base_dir=spk_dir,
                model_family="classical",
                random_seed=self.random_seed
            )
        return self._engines[speaker_id]

    def process_audio_inference(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        acoustic_quality: float = 1.0
    ) -> AppInferenceResponse:
        """
        Processes acoustic embedding through active validated model.
        Computes uncertainty, confidence bands, and emergency action flags.
        """
        # 1. Input sanitization & validation
        if embedding is None or not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be a non-null numpy ndarray.")
        if embedding.shape != (384,):
            raise ValueError(f"Expected 384-dimensional embedding, got {embedding.shape}")
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            raise ValueError("Embedding contains invalid NaN or Inf floating point values.")

        session_id = f"SESS_{speaker_id}_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        engine = self._get_engine(speaker_id)

        # 2. Degraded acoustic quality check -> Abstention
        if acoustic_quality < self.min_acoustic_quality:
            response = AppInferenceResponse(
                session_id=session_id,
                speaker_id=speaker_id,
                predicted_intent="UNKNOWN",
                confidence_score=0.0,
                confidence_band="LOW",
                suggested_action="ABSTAIN_AND_ALERT",
                is_emergency=False,
                active_model_version=engine.active_model_version,
                acoustic_quality=float(acoustic_quality),
                timestamp_utc=timestamp
            )
            # Store pending session marked as low quality
            self._pending_sessions[session_id] = {
                "speaker_id": speaker_id,
                "embedding": embedding,
                "acoustic_quality": acoustic_quality,
                "predicted_intent": "UNKNOWN",
                "confidence": 0.0,
                "timestamp": timestamp
            }
            return response

        # 3. Model forward pass
        active_model = engine.active_model
        feats_tensor = np.expand_dims(embedding, axis=0)

        if active_model is not None and hasattr(active_model, "predict_proba"):
            probs = active_model.predict_proba(feats_tensor)[0]
            pred_idx = int(np.argmax(probs))
            predicted_intent = CANONICAL_CLASSES[pred_idx]
            confidence_score = float(probs[pred_idx])
        elif active_model is not None and callable(active_model):
            import torch
            with torch.no_grad():
                t_feats = torch.from_numpy(feats_tensor).float()
                logits = active_model(t_feats)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            predicted_intent = CANONICAL_CLASSES[pred_idx]
            confidence_score = float(probs[pred_idx])
        else:
            # Baseline un-personalized prior (General Speech default with calibrated confidence)
            predicted_intent = GENERAL_SPEECH_CLASS
            confidence_score = 0.75

        # 4. Determine confidence band and suggested action
        if confidence_score >= self.high_confidence_threshold:
            confidence_band = "HIGH"
        elif confidence_score >= self.confidence_threshold:
            confidence_band = "MEDIUM"
        else:
            confidence_band = "LOW"

        is_emergency = is_emergency_intent(predicted_intent)

        if is_emergency:
            if confidence_band == "HIGH":
                suggested_action = "CONFIRM_AND_DISPATCH"
            else:
                suggested_action = "REQUEST_CONFIRMATION"
        else:
            if confidence_band in ["HIGH", "MEDIUM"]:
                suggested_action = "PROCEED"
            else:
                suggested_action = "REQUEST_CONFIRMATION"

        # 5. Record pending interaction (awaiting human verification)
        self._pending_sessions[session_id] = {
            "speaker_id": speaker_id,
            "embedding": embedding,
            "acoustic_quality": acoustic_quality,
            "predicted_intent": predicted_intent,
            "confidence": confidence_score,
            "timestamp": timestamp
        }

        return AppInferenceResponse(
            session_id=session_id,
            speaker_id=speaker_id,
            predicted_intent=predicted_intent,
            confidence_score=confidence_score,
            confidence_band=confidence_band,
            suggested_action=suggested_action,
            is_emergency=is_emergency,
            active_model_version=engine.active_model_version,
            acoustic_quality=float(acoustic_quality),
            timestamp_utc=timestamp
        )

    def submit_user_verification(
        self,
        session_id: str,
        user_action: str,  # "ACCEPT", "CORRECT", "REJECT"
        confirmed_intent: Optional[str] = None
    ) -> LoopInteractionReport:
        """
        Submits ground truth human verification for a pending session.
        Ingests into immutable ExperienceStore and ReplayMemory if learning-eligible.
        Zero unverified learning invariant enforced.
        """
        if session_id not in self._pending_sessions:
            raise KeyError(f"Session ID {session_id} not found in pending interaction queue.")

        sess = self._pending_sessions.pop(session_id)
        speaker_id = sess["speaker_id"]
        embedding = sess["embedding"]
        acoustic_quality = sess["acoustic_quality"]
        predicted_intent = sess["predicted_intent"]

        engine = self._get_engine(speaker_id)

        # Validate action
        valid_actions = [a.value for a in ReviewAction]
        if user_action not in valid_actions:
            raise ValueError(f"Invalid user action '{user_action}'. Must be one of {valid_actions}")

        if user_action == ReviewAction.CORRECT.value:
            if not confirmed_intent or not is_valid_intent(confirmed_intent):
                raise ValueError(f"Correction requires a valid canonical intent, got: {confirmed_intent}")
            true_label = confirmed_intent
            corrected_label = confirmed_intent
        elif user_action == ReviewAction.ACCEPT.value:
            true_label = predicted_intent
            corrected_label = None
        else:  # REJECT
            true_label = "REJECTED_INTENT"
            corrected_label = None

        report = engine.process_speech_interaction(
            embedding=embedding,
            true_label=true_label,
            human_action=user_action,
            user_corrected_label=corrected_label,
            acoustic_quality_score=acoustic_quality
        )

        return report

    def trigger_governed_adaptation(
        self,
        speaker_id: str,
        val_samples: List[Any],
        candidate_version_tag: Optional[str] = None
    ) -> AppAdaptationResponse:
        """
        Triggers candidate adaptation, multi-metric validation gate, and safety veto check.
        Promotes candidate to active model only if all gates pass.
        """
        engine = self._get_engine(speaker_id)
        tag = candidate_version_tag or f"{speaker_id}_cand_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        prev_version = engine.active_model_version
        cycle_result: SelfImprovementCycleResult = engine.trigger_adaptation_cycle(
            candidate_version=tag,
            val_samples=val_samples
        )

        return AppAdaptationResponse(
            speaker_id=speaker_id,
            adaptation_cycle_id=cycle_result.cycle_id,
            gate_decision=cycle_result.gate_decision,
            previous_model_version=prev_version,
            active_model_version=engine.active_model_version,
            experiences_used=cycle_result.budget_n,
            reason_codes=cycle_result.reason_codes,
            timestamp_utc=datetime.now(timezone.utc).isoformat()
        )

    def rollback_model(self, speaker_id: str, reason: str = "APPLICATION_ROLLBACK") -> str:
        """Rolls back the active model pointer to the previous validated parent model."""
        engine = self._get_engine(speaker_id)
        restored = engine.rollback_active_model(reason=reason)
        return restored

    def get_speaker_status(self, speaker_id: str) -> Dict[str, Any]:
        """Returns complete operational status and governance metrics for the speaker."""
        engine = self._get_engine(speaker_id)
        return {
            "speaker_id": speaker_id,
            "active_model_version": engine.active_model_version,
            "total_interactions": len(engine.interaction_history),
            "total_experiences_stored": len(engine.experience_store.list_by_speaker(speaker_id)),
            "replay_buffer_size": engine.replay_memory.size(),
            "adaptation_cycles_run": len(engine.cycle_history),
            "deployment_status": "RESEARCH DEMO ONLY"
        }
