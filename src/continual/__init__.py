"""
Continual Learning & Personalization Package (Phase 7).
Provides data abstractions, schema contracts, confidence representation,
decision logic, eligibility validation, session management, and experience storage.
"""

from src.continual.taxonomy import (
    CANONICAL_CLASSES,
    EMERGENCY_CLASSES,
    GENERAL_SPEECH_CLASS,
    is_valid_intent,
    is_emergency_intent,
    get_stage1_label,
)
from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
    AuditEntry,
)
from src.continual.validation import (
    validate_experience_record,
    assert_valid_experience_record,
    ValidationError,
)
from src.continual.confidence import (
    ConfidenceType,
    CalibrationStatus,
    ConfidenceBand,
    ConfidenceRepresentation,
    ConfidenceValidator,
    UncertaintyCalculator,
    load_confidence_policy,
)
from src.continual.decision import (
    ActionDecision,
    SuggestedAction,
    PrioritizationTag,
    ConfidenceDecision,
    DecisionEngine,
)
from src.continual.eligibility import (
    EligibilityEngine,
    evaluate_learning_eligibility,
)
from src.continual.session import (
    PersonalizationSession,
    SessionSpeakerMismatchError,
)
from src.continual.duplicate import (
    DuplicateDetector,
    DuplicatePolicy,
    DuplicateRecordError,
)
from src.continual.store import (
    ExperienceStore,
    InMemoryExperienceStore,
    PersistentExperienceStore,
    RecordNotFoundError,
    SpeakerIsolationError,
)
from src.continual.review import (
    ReviewState,
    ReviewAction,
    ReviewPriority,
    VerifierType,
    ReviewRequest,
    ReviewResponse,
    ReviewValidationError,
    ReviewConflictError,
    InvalidReviewStateTransitionError,
    ReviewNotFoundError,
    determine_review_priority,
    validate_review_request,
    validate_review_response,
    load_review_policy,
)
from src.continual.queue import (
    ReviewQueue,
    InMemoryReviewQueue,
    PersistentReviewQueue,
    HumanReviewService,
)

from src.continual.model_version import (
    ModelLifecycleState,
    ModelVersionRecord,
    ModelRegistry,
    ModelVersionError,
    ModelOverwriteError,
    ModelLineageError,
    compute_dataset_hash,
    compute_configuration_hash,
)
from src.continual.experiment_registry import (
    ExperimentStatus,
    ExperimentRunRecord,
    ExperimentRegistry,
)
from src.continual.continual_classical import (
    ContinualTrainingError,
    IneligibleExperienceError,
    DataLeakageError,
    EmbeddingIntegrityError,
    ContinualEvaluationMetrics,
    ContinualTrainingResult,
    ContinualClassicalAdapterTrainer,
    validate_training_experiences,
    compute_comprehensive_continual_metrics,
)
from src.continual.replay import (
    ReplayError,
    IneligibleReplayError,
    TemporalCausalityError,
    ReplayCapacityError,
    ReplayPolicy,
    ReplayMemoryItem,
    ReplayMemory,
    compute_emergency_forgetting,
)
from src.continual.governance import (
    PromotionDecision,
    GovernanceError,
    ContaminationError,
    SafetyVetoError,
    RollbackError,
    ValidationReasonCode,
    ValidationDecisionRecord,
    load_governance_policy,
    ValidationGate,
    ModelGovernanceManager,
    compute_cohort_governance_summary,
)

from src.continual.drift import (
    DriftType,
    DriftStatus,
    RecalibrationTriggerState,
    DriftError,
    SpeakerIsolationError as DriftSpeakerIsolationError,
    TemporalSafetyError,
    DriftDataError,
    DriftWindow,
    DriftEvent,
    RecalibrationCandidate,
    compute_mmd_rbf,
    compute_centroid_cosine_distance,
    compute_prediction_tvd,
    compute_confidence_ks,
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    compute_brier_score,
    TemperatureScalingRecalibrator,
    SpeakerDriftDetector,
    ControlledRecalibrationPipeline,
    load_drift_policy,
)
from src.continual.continual_quantum import (
    CANONICAL_QUANTUM_CONFIG,
    QuantumContinualError,
    QuantumConfigurationError,
    KernelIntegrityError,
    QuantumLeakageError,
    QuantumKernelDiagnostics,
    ContinualQuantumResult,
    compute_quantum_cache_key,
    ContinualQuantumKernelCache,
    ContinualQSVCClassifier,
    ContinualQuantumTrainer,
)
from src.continual.trainable_quantum import (
    CANONICAL_TRAINABLE_QUANTUM_CONFIG,
    TrainableQuantumError,
    TrainableQuantumArchitectureError,
    QuantumOptimizationError,
    TrainableKernelIntegrityError,
    TrainableKernelDiagnostics,
    ContinualTrainableQuantumResult,
    compute_param_fingerprint,
    ParameterizedQuantumFeatureMap,
    QuantumKernelAlignmentOptimizer,
    ContinualTrainableQSVCClassifier,
    ContinualTrainableQuantumTrainer,
)

__all__ = [
    # Taxonomy
    "CANONICAL_CLASSES",
    "EMERGENCY_CLASSES",
    "GENERAL_SPEECH_CLASS",
    "is_valid_intent",
    "is_emergency_intent",
    "get_stage1_label",
    # Schema
    "ExperienceRecord",
    "VerificationStatus",
    "LearningStatus",
    "EligibilityReason",
    "AuditEntry",
    # Validation
    "validate_experience_record",
    "assert_valid_experience_record",
    "ValidationError",
    # Confidence
    "ConfidenceType",
    "CalibrationStatus",
    "ConfidenceBand",
    "ConfidenceRepresentation",
    "ConfidenceValidator",
    "UncertaintyCalculator",
    "load_confidence_policy",
    # Decision
    "ActionDecision",
    "SuggestedAction",
    "PrioritizationTag",
    "ConfidenceDecision",
    "DecisionEngine",
    # Eligibility
    "EligibilityEngine",
    "evaluate_learning_eligibility",
    # Session
    "PersonalizationSession",
    "SessionSpeakerMismatchError",
    # Duplicate
    "DuplicateDetector",
    "DuplicatePolicy",
    "DuplicateRecordError",
    # Store
    "ExperienceStore",
    "InMemoryExperienceStore",
    "PersistentExperienceStore",
    "RecordNotFoundError",
    "SpeakerIsolationError",
    # Review (Phase 7.3)
    "ReviewState",
    "ReviewAction",
    "ReviewPriority",
    "VerifierType",
    "ReviewRequest",
    "ReviewResponse",
    "ReviewValidationError",
    "ReviewConflictError",
    "InvalidReviewStateTransitionError",
    "ReviewNotFoundError",
    "determine_review_priority",
    "validate_review_request",
    "validate_review_response",
    "load_review_policy",
    # Queue (Phase 7.3)
    "ReviewQueue",
    "InMemoryReviewQueue",
    "PersistentReviewQueue",
    "HumanReviewService",
    # Model Versioning (Phase 7.4)
    "ModelLifecycleState",
    "ModelVersionRecord",
    "ModelRegistry",
    "ModelVersionError",
    "ModelOverwriteError",
    "ModelLineageError",
    "compute_dataset_hash",
    "compute_configuration_hash",
    # Experiment Registry (Phase 7.4)
    "ExperimentStatus",
    "ExperimentRunRecord",
    "ExperimentRegistry",
    # Continual Classical (Phase 7.4)
    "ContinualTrainingError",
    "IneligibleExperienceError",
    "DataLeakageError",
    "EmbeddingIntegrityError",
    "ContinualEvaluationMetrics",
    "ContinualTrainingResult",
    "ContinualClassicalAdapterTrainer",
    "validate_training_experiences",
    "compute_comprehensive_continual_metrics",
    # Replay Memory (Phase 7.5)
    "ReplayError",
    "IneligibleReplayError",
    "TemporalCausalityError",
    "ReplayCapacityError",
    "ReplayPolicy",
    "ReplayMemoryItem",
    "ReplayMemory",
    "compute_emergency_forgetting",
    # Governance & Validation (Phase 7.6)
    "PromotionDecision",
    "GovernanceError",
    "ContaminationError",
    "SafetyVetoError",
    "RollbackError",
    "ValidationReasonCode",
    "ValidationDecisionRecord",
    "load_governance_policy",
    "ValidationGate",
    "ModelGovernanceManager",
    "compute_cohort_governance_summary",
    # Drift & Recalibration (Phase 7.7)
    "DriftType",
    "DriftStatus",
    "RecalibrationTriggerState",
    "DriftError",
    "DriftSpeakerIsolationError",
    "TemporalSafetyError",
    "DriftDataError",
    "DriftWindow",
    "DriftEvent",
    "RecalibrationCandidate",
    "compute_mmd_rbf",
    "compute_centroid_cosine_distance",
    "compute_prediction_tvd",
    "compute_confidence_ks",
    "compute_expected_calibration_error",
    "compute_maximum_calibration_error",
    "compute_brier_score",
    "TemperatureScalingRecalibrator",
    "SpeakerDriftDetector",
    "ControlledRecalibrationPipeline",
    "load_drift_policy",
    # Continual Quantum (Phase 7.8)
    "CANONICAL_QUANTUM_CONFIG",
    "QuantumContinualError",
    "QuantumConfigurationError",
    "KernelIntegrityError",
    "QuantumLeakageError",
    "QuantumKernelDiagnostics",
    "ContinualQuantumResult",
    "compute_quantum_cache_key",
    "ContinualQuantumKernelCache",
    "ContinualQSVCClassifier",
    "ContinualQuantumTrainer",
    # Trainable Quantum (Phase 7.9)
    "CANONICAL_TRAINABLE_QUANTUM_CONFIG",
    "TrainableQuantumError",
    "TrainableQuantumArchitectureError",
    "QuantumOptimizationError",
    "TrainableKernelIntegrityError",
    "TrainableKernelDiagnostics",
    "ContinualTrainableQuantumResult",
    "compute_param_fingerprint",
    "ParameterizedQuantumFeatureMap",
    "QuantumKernelAlignmentOptimizer",
    "ContinualTrainableQSVCClassifier",
    "ContinualTrainableQuantumTrainer",
    # Hybrid Classical + Quantum (Phase 7.10)
    "CANONICAL_HYBRID_CONFIG",
    "ContinualHybridResult",
    "HybridClassicalQuantumClassifier",
    "ContinualHybridTrainer",
    # Self-Improvement Closed Loop (Phase 7.11)
    "SelfImprovementError",
    "UnverifiedLearningAttemptError",
    "DirectActivationAttemptError",
    "LoopInteractionReport",
    "SelfImprovementCycleResult",
    "LearningEfficiencyMetrics",
    "ContinualSelfImprovementEngine",
    # Safety & Robustness Audit (Phase 7.13)
    "StressCondition",
    "FailureInjectionType",
    "StressTestResult",
    "FailureInjectionResult",
    "BypassAuditReport",
    "SafetyRobustnessAuditor",
    # Application Integration (Phase 7.14)
    "AppInferenceResponse",
    "AppAdaptationResponse",
    "AtypicalEmergencyAppPipeline",
]

from src.continual.hybrid import (
    CANONICAL_HYBRID_CONFIG,
    ContinualHybridResult,
    HybridClassicalQuantumClassifier,
    ContinualHybridTrainer,
)

from src.continual.self_improvement import (
    SelfImprovementError,
    UnverifiedLearningAttemptError,
    DirectActivationAttemptError,
    LoopInteractionReport,
    SelfImprovementCycleResult,
    LearningEfficiencyMetrics,
    ContinualSelfImprovementEngine,
)

from src.continual.safety_audit import (
    StressCondition,
    FailureInjectionType,
    StressTestResult,
    FailureInjectionResult,
    BypassAuditReport,
    SafetyRobustnessAuditor,
)

from src.continual.app_pipeline import (
    AppInferenceResponse,
    AppAdaptationResponse,
    AtypicalEmergencyAppPipeline,
)





