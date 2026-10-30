"""Health, readiness, and model metadata endpoints for Quantum-Only Speech Application."""

from typing import Any, Dict, List
from fastapi import APIRouter, Query

from src.app.audio_service import AudioService
from src.app.config import (
    DEFAULT_SPEAKER,
    CANONICAL_QUANTUM_MODEL_ID,
    DISCLAIMER_NOTICE,
    SCIENTIFIC_VERDICT_NOTICE,
)
from src.app.quantum_model_manager import QuantumModelManager
from src.app.schemas import (
    HealthResponse,
    ReadinessResponse,
    ModelInfoResponse,
    current_utc_timestamp,
)

router = APIRouter(tags=["System Health & Metadata"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Returns basic process health and operational status."""
    return HealthResponse(
        status="healthy",
        service_name="Atypical Speech Emergency Assistant — Quantum-Only Research Demonstration",
        version="1.0.0-gate-b-quantum",
        timestamp_utc=current_utc_timestamp(),
        disclaimer=DISCLAIMER_NOTICE,
    )


@router.get("/ready", response_model=ReadinessResponse)
def get_ready() -> ReadinessResponse:
    """
    Evaluates readiness of speech-processing components and model checkpoints.
    Accurately reports separate component readiness:
    - Whisper encoder (384-D feature extraction): Operational
    - Whisper ASR (speech transcription): Operational
    - Operational Quantum Model (qsvc_q8_r1): QUANTUM_MODEL_NOT_READY (missing X_train_pca reference vectors)
    """
    audio_service = AudioService.get_instance()
    quantum_mgr = QuantumModelManager.get_instance()

    # Check Whisper components readiness
    encoder_ready = True
    asr_ready = True
    try:
        _ = audio_service.encoder
    except Exception:
        encoder_ready = False

    try:
        _ = audio_service.asr_model
    except Exception:
        asr_ready = False

    # Check Quantum Model readiness
    q_readiness = quantum_mgr.get_readiness_status()
    quantum_ready = q_readiness.get("quantum_model_ready", False)

    # Checkpoint summary from quantum inventory
    inventory = quantum_mgr.get_inventory()
    materialized_count = sum(1 for item in inventory if item.get("status") in ("MATERIALIZED_AND_VERIFIED", "OPERATIONAL"))
    unready_count = sum(1 for item in inventory if item.get("status") == "QUANTUM_MODEL_NOT_READY")

    blockers = []
    if not quantum_ready:
        blockers = [
            f"{CANONICAL_QUANTUM_MODEL_ID} (missing X_train_pca reference vectors on disk)"
        ]

    # End-to-end readiness requires operational quantum model
    overall_ready = encoder_ready and asr_ready and quantum_ready

    return ReadinessResponse(
        ready=overall_ready,
        whisper_encoder_ready=encoder_ready,
        whisper_asr_ready=asr_ready,
        quantum_model_ready=quantum_ready,
        quantum_readiness_status=q_readiness.get("quantum_readiness_status", "QUANTUM_MODEL_NOT_READY"),
        quantum_blocker_reason=q_readiness.get("quantum_blocker_reason"),
        checkpoints_summary={
            "materialized_and_verified": materialized_count,
            "model_not_ready": unready_count,
            "total_catalogued": len(inventory),
        },
        active_default_speaker=DEFAULT_SPEAKER,
        active_default_model=CANONICAL_QUANTUM_MODEL_ID,
        unmaterialized_blockers=blockers,
        scientific_verdict=SCIENTIFIC_VERDICT_NOTICE,
        disclaimer=DISCLAIMER_NOTICE,
        timestamp_utc=current_utc_timestamp(),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info(
    model_id: str = Query(CANONICAL_QUANTUM_MODEL_ID, description="Target model candidate ID"),
    speaker_id: str = Query(DEFAULT_SPEAKER, description="Target speaker ID (e.g. 'F01', 'M02')"),
) -> ModelInfoResponse:
    """
    Returns architecture, input shape, canonical taxonomy, provenance hash,
    and decoupled threshold configuration for the specified model and speaker.
    """
    quantum_mgr = QuantumModelManager.get_instance()
    info = quantum_mgr.get_model_info(model_id=model_id, speaker_id=speaker_id)
    info["timestamp_utc"] = current_utc_timestamp()
    return ModelInfoResponse(**info)


@router.get("/api/v1/models")
def list_models() -> Dict[str, Any]:
    """Lists all catalogued models and their current materialization and readiness status."""
    quantum_mgr = QuantumModelManager.get_instance()
    inventory = quantum_mgr.get_inventory()
    return {
        "total": len(inventory),
        "models": inventory,
        "disclaimer": DISCLAIMER_NOTICE,
        "timestamp_utc": current_utc_timestamp(),
    }


@router.get("/api/v1/speakers")
def list_speakers() -> Dict[str, Any]:
    """Lists all speakers with verified quantum checkpoints catalogued."""
    quantum_mgr = QuantumModelManager.get_instance()
    speakers = quantum_mgr.get_available_speakers()
    return {
        "count": len(speakers),
        "speakers": speakers,
        "default_speaker": DEFAULT_SPEAKER,
        "disclaimer": DISCLAIMER_NOTICE,
        "timestamp_utc": current_utc_timestamp(),
    }
