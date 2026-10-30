"""Typed Pydantic request and response schemas for the Gate B Prototype API."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.app.config import (
    DISCLAIMER_NOTICE,
    SCIENTIFIC_VERDICT_NOTICE,
    CANONICAL_CLASSES,
    DEFAULT_DECISION_TAU,
    FER_SAFETY_BOUND,
    CANONICAL_QUANTUM_MODEL_ID,
)


def current_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Health & Readiness Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="System operational health status (e.g. 'healthy')")
    service_name: str = "Atypical Speech Emergency Assistant — Gate B Engineering Prototype"
    version: str = "1.0.0-gate-b"
    timestamp_utc: str = Field(default_factory=current_utc_timestamp)
    disclaimer: str = DISCLAIMER_NOTICE


class ReadinessResponse(BaseModel):
    ready: bool = Field(..., description="Overall readiness for research demonstration inference")
    whisper_encoder_ready: bool = Field(..., description="Local Whisper 384-D feature extraction ready")
    whisper_asr_ready: bool = Field(..., description="Local Whisper speech transcription ready")
    quantum_model_ready: bool = Field(default=False, description="Operational quantum model readiness")
    quantum_readiness_status: str = Field(default="QUANTUM_MODEL_NOT_READY", description="Quantum model status")
    quantum_blocker_reason: Optional[str] = Field(default=None, description="Quantum model blocker reason")
    checkpoints_summary: Dict[str, int] = Field(..., description="Count of materialized vs unready model checkpoints")
    active_default_speaker: str
    active_default_model: str
    unmaterialized_blockers: List[str] = Field(default_factory=list, description="List of unmaterialized models that cannot perform inference")
    scientific_verdict: str = SCIENTIFIC_VERDICT_NOTICE
    disclaimer: str = DISCLAIMER_NOTICE
    timestamp_utc: str = Field(default_factory=current_utc_timestamp)


class ModelInfoResponse(BaseModel):
    model_id: str
    model_name: str
    speaker_id: str
    architecture: Optional[str] = None
    parameter_count: Optional[str] = None
    operational_role: Optional[str] = None
    qubit_count: Optional[int] = None
    feature_map: Optional[str] = None
    entanglement: Optional[str] = None
    reps: Optional[int] = None
    kernel_type: Optional[str] = None
    classifier: Optional[str] = None
    regularization_C: Optional[float] = None
    backend: Optional[str] = None
    simulation_framework: Optional[str] = None
    input_shape: List[int] = [384]
    pca_dimension: Optional[int] = None
    pca_feature_range: Optional[List[float]] = None
    taxonomy: List[str] = CANONICAL_CLASSES
    decision_policy: Dict[str, Any] = Field(default_factory=dict)
    safety_constraints: Optional[Dict[str, Any]] = None
    checkpoint_provenance: Dict[str, Any] = Field(default_factory=dict)
    status: str  # OPERATIONAL, QUANTUM_MODEL_NOT_READY, RESEARCH_BASELINE_ONLY, etc.
    readiness_reason: Optional[str] = None
    notice: Optional[str] = None
    error: Optional[str] = None
    quantum_verdict: str = SCIENTIFIC_VERDICT_NOTICE
    clinical_safety_verdict: str = "SAFETY-INFEASIBLE: Non-clinical research demonstration only."
    disclaimer: str = DISCLAIMER_NOTICE
    timestamp_utc: str = Field(default_factory=current_utc_timestamp)


# ---------------------------------------------------------------------------
# Audio Processing Schemas
# ---------------------------------------------------------------------------

class Base64AudioRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded audio payload (WAV, MP3, etc.)")


class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transcribed speech text from local Whisper ASR")
    language: str = "en"
    audio_duration_seconds: float
    acoustic_quality: float = 1.0
    timestamp_utc: str = Field(default_factory=current_utc_timestamp)
    disclaimer: str = DISCLAIMER_NOTICE


class AudioExtractionResponse(BaseModel):
    embedding_dimension: int = 384
    l2_norm: float
    audio_duration_seconds: float
    vector_preview: Optional[List[float]] = None
    full_vector: Optional[List[float]] = None
    timestamp_utc: str = Field(default_factory=current_utc_timestamp)


# ---------------------------------------------------------------------------
# Prediction & Inference Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    speaker_id: str = Field(default="F01", description="Target dysarthric/control speaker ID (e.g. 'F01', 'M02', 'MC04')")
    model_id: str = Field(default=CANONICAL_QUANTUM_MODEL_ID, description="Operational model ID (restricted to quantum model 'qsvc_q8_r1')")
    audio_base64: Optional[str] = Field(default=None, description="Base64-encoded raw audio (WAV, MP3, etc.)")
    embedding: Optional[List[float]] = Field(default=None, description="Pre-extracted 384-dimensional Whisper embedding")
    acoustic_quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Estimated acoustic recording quality [0.0 - 1.0]")
    operating_tau: Optional[float] = Field(default=None, description="Decision threshold override (must not be conflated with FER bound)")


class PredictResponse(BaseModel):
    session_id: str
    speaker_id: str
    model_id: str
    model_version: str
    predicted_intent: str
    is_emergency: bool
    confidence_score: float
    confidence_band: str  # HIGH, MEDIUM, LOW
    suggested_action: str  # CONFIRM_AND_DISPATCH, REQUEST_CONFIRMATION, PROCEED, ABSTAIN
    class_probabilities: Dict[str, float]
    decision_cutoff_tau: float
    fer_safety_bound: float = FER_SAFETY_BOUND
    transcription: Optional[str] = None
    processing_latency_ms: float
    timestamp_utc: str = Field(default_factory=current_utc_timestamp)
    disclaimer: str = DISCLAIMER_NOTICE


# ---------------------------------------------------------------------------
# Error Schemas
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    remediation: Optional[str] = None
    disclaimer: str = DISCLAIMER_NOTICE
    timestamp_utc: str = Field(default_factory=current_utc_timestamp)
