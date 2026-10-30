"""
Phase 7.4 Continual Classical Personalization Baseline.
Implements the verified, speaker-specific continual adaptation loop:
- Ingestion of verified, learning-eligible ExperienceRecords from ExperienceStore.
- Strict anti-self-training and data eligibility enforcement.
- Chronological ordering of accumulated experiences (no future data leakage).
- Deterministic training of canonical Classical Adapter (25,223 parameters).
- Model versioning, artifact serialization, and ModelRegistry candidate indexing.
- Protected, zero-leakage evaluation against held-out validation/test sets.
- Emergency-sensitive metric computation matching canonical registry standards.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.taxonomy import (
    CANONICAL_CLASSES,
    EMERGENCY_CLASSES,
    GENERAL_SPEECH_CLASS,
    is_valid_intent,
    is_emergency_intent,
)
from src.continual.store import (
    ExperienceStore,
    SpeakerIsolationError,
)
from src.continual.model_version import (
    ModelLifecycleState,
    ModelVersionRecord,
    ModelRegistry,
    compute_dataset_hash,
    compute_configuration_hash,
)
from src.models.classical_adapter import (
    ClassicalAdapterNet,
    ClassicalAdapterClassifier,
)
from src.embeddings.cache import (
    has_cached_embedding,
    load_embedding_file,
)
from src.utils.seeding import set_seed


class ContinualTrainingError(Exception):
    """Base exception for continual training errors."""
    pass


class IneligibleExperienceError(ContinualTrainingError):
    """Raised when an unverified or ineligible experience enters training."""
    pass


class DataLeakageError(ContinualTrainingError):
    """Raised when evaluation data leaks into the training set."""
    pass


class EmbeddingIntegrityError(ContinualTrainingError):
    """Raised when an embedding is missing, corrupt, or has NaN values."""
    pass


@dataclass
class ContinualEvaluationMetrics:
    """Standardized emergency-sensitive evaluation metrics for Phase 7.4."""
    accuracy: float
    macro_f1: float
    emergency_macro_f1: float
    emergency_recall: float
    emergency_precision: float
    false_emergency_rate: float
    iper: float
    general_speech_recall: float
    total_test_samples: int
    emergency_test_samples: int
    general_test_samples: int
    predicted_emergency_count: int
    predicted_general_count: int
    per_class_f1: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContinualTrainingResult:
    """Comprehensive output container for a continual adaptation run."""
    speaker_id: str
    budget: int
    model_version: str
    parent_model_version: Optional[str]
    seed: int
    train_sample_ids: List[str]
    training_dataset_hash: str
    configuration_hash: str
    adaptation_time_sec: float
    metrics: ContinualEvaluationMetrics
    model_checkpoint_path: Optional[str] = None
    status: str = "COMPLETED"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metrics"] = self.metrics.to_dict()
        return d


def validate_training_experiences(
    records: List[ExperienceRecord],
    target_speaker: str,
) -> None:
    """
    Enforce strict data eligibility and speaker isolation on candidate training experiences.
    Raises IneligibleExperienceError or SpeakerIsolationError if violated.
    """
    if not records:
        raise ContinualTrainingError("Cannot train on an empty list of experience records.")

    for rec in records:
        # Speaker isolation
        if rec.speaker_id != target_speaker:
            raise SpeakerIsolationError(
                f"Speaker mismatch in training batch: record '{rec.sample_id}' belongs to "
                f"speaker '{rec.speaker_id}', but target speaker is '{target_speaker}'."
            )

        # Verification status check
        if rec.verification_status not in (
            VerificationStatus.USER_VERIFIED.value,
            VerificationStatus.SYSTEM_VERIFIED.value,
        ):
            raise IneligibleExperienceError(
                f"Record '{rec.sample_id}' has unverified status '{rec.verification_status}'. "
                "Only USER_VERIFIED or SYSTEM_VERIFIED records may enter training."
            )

        # Learning eligibility flag
        if not rec.learning_eligible:
            raise IneligibleExperienceError(
                f"Record '{rec.sample_id}' is marked learning_eligible=False "
                f"(reason: {rec.eligibility_reason}). Training admission denied."
            )

        # Confirmed intent validity
        if not rec.confirmed_intent or not is_valid_intent(rec.confirmed_intent):
            raise IneligibleExperienceError(
                f"Record '{rec.sample_id}' has invalid confirmed intent '{rec.confirmed_intent}'."
            )

        # Acoustic quality gate
        if rec.data_quality_status == "QUARANTINED" or (rec.quality_score is not None and rec.quality_score < 0.5):
            raise IneligibleExperienceError(
                f"Record '{rec.sample_id}' failed quality gate (status: {rec.data_quality_status}, score: {rec.quality_score})."
            )


def compute_comprehensive_continual_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> ContinualEvaluationMetrics:
    """
    Calculate canonical Phase 6.5 / Phase 7 emergency-sensitive metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=0))
    emerg_macro_f1 = float(f1_score(y_true, y_pred, labels=EMERGENCY_CLASSES, average="macro", zero_division=0))

    # Per-class F1
    per_class_f1_scores = f1_score(y_true, y_pred, labels=CANONICAL_CLASSES, average=None, zero_division=0)
    per_class_dict = {cls_name: float(score) for cls_name, score in zip(CANONICAL_CLASSES, per_class_f1_scores)}

    # Emergency counts
    true_emerg_count = sum(1 for yt in y_true if yt in EMERGENCY_CLASSES)
    pred_emerg_count = sum(1 for yp in y_pred if yp in EMERGENCY_CLASSES)
    correct_emerg_count = sum(
        1 for yt, yp in zip(y_true, y_pred) if yt in EMERGENCY_CLASSES and yp in EMERGENCY_CLASSES and yt == yp
    )

    emerg_recall = float(correct_emerg_count / true_emerg_count) if true_emerg_count > 0 else 0.0
    emerg_precision = float(correct_emerg_count / pred_emerg_count) if pred_emerg_count > 0 else 0.0

    # IPER (Intent Preservation Error Rate / Emergency Preservation)
    if true_emerg_count > 0:
        iper = float(correct_emerg_count / true_emerg_count)
    else:
        iper = 0.0

    # False Emergency Rate (FER) on general speech
    gen_total = sum(1 for yt in y_true if yt == GENERAL_SPEECH_CLASS)
    false_emerg_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == GENERAL_SPEECH_CLASS and yp in EMERGENCY_CLASSES)
    fer = float(false_emerg_count / gen_total) if gen_total > 0 else 0.0

    # General speech recall
    correct_gen_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == GENERAL_SPEECH_CLASS and yp == GENERAL_SPEECH_CLASS)
    gen_recall = float(correct_gen_count / gen_total) if gen_total > 0 else 0.0

    pred_gen_count = sum(1 for yp in y_pred if yp == GENERAL_SPEECH_CLASS)

    return ContinualEvaluationMetrics(
        accuracy=acc,
        macro_f1=macro_f1,
        emergency_macro_f1=emerg_macro_f1,
        emergency_recall=emerg_recall,
        emergency_precision=emerg_precision,
        false_emergency_rate=fer,
        iper=iper,
        general_speech_recall=gen_recall,
        total_test_samples=len(y_true),
        emergency_test_samples=true_emerg_count,
        general_test_samples=gen_total,
        predicted_emergency_count=pred_emerg_count,
        predicted_general_count=pred_gen_count,
        per_class_f1=per_class_dict,
    )


class ContinualClassicalAdapterTrainer:
    """
    Manages the continual classical adaptation pipeline for a target speaker.
    """

    def __init__(
        self,
        target_speaker: str,
        model_registry: Optional[ModelRegistry] = None,
        base_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.target_speaker = target_speaker
        self.model_registry = model_registry or ModelRegistry()
        self.base_dir = base_dir or "models/continual"

        # Canonical Phase 3 / Phase 6.5 hyperparameters
        default_config = {
            "input_dim": 384,
            "hidden_dim": 64,
            "num_classes": 7,
            "dropout": 0.1,
            "learning_rate": 0.005,
            "weight_decay": 1e-4,
            "epochs": 60,
            "batch_size": 8,
            "class_weight": "balanced",
            "seed": 42,
            "device": "cpu",
        }
        if config:
            default_config.update(config)
        self.config = default_config

    def train_step(
        self,
        experiences: List[ExperienceRecord],
        test_records: List[Tuple[str, str, np.ndarray]],  # list of (sample_id, true_intent, embedding)
        parent_model_version: Optional[str] = None,
        custom_seed: Optional[int] = None,
    ) -> ContinualTrainingResult:
        """
        Execute one continual personalization update.

        Args:
            experiences: List of verified ExperienceRecords available up to this accumulation step.
            test_records: Held-out protected test instances [(sample_id, true_intent, embedding), ...].
            parent_model_version: Version of predecessor model.
            custom_seed: Optional seed override.

        Returns:
            ContinualTrainingResult containing model metadata, metrics, and artifact paths.
        """
        seed = custom_seed if custom_seed is not None else self.config["seed"]
        set_seed(seed)

        # 1. Validate experiences and enforce eligibility firewall
        validate_training_experiences(experiences, self.target_speaker)

        # 2. Chronological ordering (prevent future lookahead)
        sorted_experiences = sorted(experiences, key=lambda r: r.timestamp)
        train_sample_ids = [r.sample_id for r in sorted_experiences]

        # 3. Check for data leakage against test records
        test_sample_ids = set(t[0] for t in test_records)
        overlap = set(train_sample_ids).intersection(test_sample_ids)
        if overlap:
            raise DataLeakageError(
                f"CRITICAL: Data leakage detected! {len(overlap)} samples exist in both "
                f"training and protected test sets: {list(overlap)[:5]}"
            )

        # 4. Extract training embeddings and verified labels
        train_embeddings_list = []
        train_labels_list = []

        for rec in sorted_experiences:
            # Check cached embedding path
            emb = None
            if hasattr(rec, "embedding_array") and rec.embedding_array is not None:
                emb = rec.embedding_array
            elif rec.embedding_reference and os.path.isfile(rec.embedding_reference):
                emb = np.load(rec.embedding_reference)
            elif rec.sample_id:
                # Try standard Whisper cache
                rid = rec.sample_id
                if has_cached_embedding(rid, feature_type="whisper", base_dir="embeddings/torgo"):
                    emb = load_embedding_file(rid, feature_type="whisper", base_dir="embeddings/torgo")

            if emb is None:
                raise EmbeddingIntegrityError(f"Failed to load embedding for record '{rec.sample_id}'")

            if np.isnan(emb).any() or np.isinf(emb).any():
                raise EmbeddingIntegrityError(f"Embedding for record '{rec.sample_id}' contains NaN or Inf")

            if emb.shape != (384,):
                raise EmbeddingIntegrityError(f"Embedding shape {emb.shape} does not match expected (384,)")

            train_embeddings_list.append(emb)
            train_labels_list.append(rec.confirmed_intent)

        X_train = np.stack(train_embeddings_list).astype(np.float32)
        y_train = train_labels_list

        # Compute dataset fingerprint and config hash
        dataset_hash = compute_dataset_hash(X_train, y_train, train_sample_ids)
        config_hash = compute_configuration_hash(self.config)

        # 5. Build and train canonical ClassicalAdapterClassifier
        classifier = ClassicalAdapterClassifier(
            input_dim=self.config["input_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_classes=self.config["num_classes"],
            dropout=self.config["dropout"],
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            class_weight=self.config["class_weight"],
            seed=seed,
            device=self.config["device"],
            all_classes=CANONICAL_CLASSES,
        )

        t0 = time.perf_counter()
        classifier.fit(X_train, y_train)
        adaptation_time = time.perf_counter() - t0

        # Verify model parameter count is canonical 25,223
        if classifier.model is not None:
            param_count = classifier.model.count_parameters()
            if param_count != 25223:
                raise ContinualTrainingError(
                    f"Model parameter count {param_count} does not match canonical 25,223"
                )

        # 6. Evaluate on protected test set
        X_test = np.stack([t[2] for t in test_records]).astype(np.float32)
        y_test = [t[1] for t in test_records]

        y_pred = classifier.predict(X_test)
        metrics = compute_comprehensive_continual_metrics(y_test, y_pred)

        # 7. Model versioning & artifact serialization
        budget = len(experiences)
        version_str = f"v{self.target_speaker}_budget{budget}_seed{seed}"

        # Serialize checkpoint
        version_dir = os.path.join(self.base_dir, self.target_speaker, "classical_adapter", version_str)
        os.makedirs(version_dir, exist_ok=True)
        checkpoint_path = os.path.join(version_dir, "model_checkpoint.pt")

        if classifier.model is not None:
            torch.save(classifier.model.state_dict(), checkpoint_path)

        # Register candidate in ModelRegistry
        version_record = ModelVersionRecord(
            model_id="classical_adapter",
            version=version_str,
            speaker_id=self.target_speaker,
            parent_version=parent_model_version,
            training_sample_ids=train_sample_ids,
            training_dataset_hash=dataset_hash,
            configuration_hash=config_hash,
            representation_version="whisper_tiny_384_unit_L2",
            hyperparameters=self.config,
            metrics=metrics.to_dict(),
            status=ModelLifecycleState.CANDIDATE.value,
            artifact_path=checkpoint_path,
        )
        self.model_registry.register_candidate(version_record)

        return ContinualTrainingResult(
            speaker_id=self.target_speaker,
            budget=budget,
            model_version=version_str,
            parent_model_version=parent_model_version,
            seed=seed,
            train_sample_ids=train_sample_ids,
            training_dataset_hash=dataset_hash,
            configuration_hash=config_hash,
            adaptation_time_sec=adaptation_time,
            metrics=metrics,
            model_checkpoint_path=checkpoint_path,
            status="COMPLETED",
        )
