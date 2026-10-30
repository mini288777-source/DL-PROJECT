"""Inference endpoints for Quantum-Only Speech Intent Application.

Under the quantum-only application mandate:
1. Operational prediction is restricted EXCLUSIVELY to the canonical quantum model (qsvc_q8_r1).
2. Classical models are strictly research-only and rejected with HTTP 400 MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE.
3. Zero automatic fallback to classical models under any circumstances.
4. If quantum model state is unmaterialized/incomplete, returns HTTP 503 QUANTUM_MODEL_NOT_READY.
5. Preserves independent Whisper transcription and 384-D feature extraction.
"""

import time
from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status
import numpy as np

from src.app.audio_service import AudioService, AudioValidationError
from src.app.config import (
    DEFAULT_SPEAKER,
    CANONICAL_QUANTUM_MODEL_ID,
    DISCLAIMER_NOTICE,
    FER_SAFETY_BOUND,
)
from src.app.quantum_model_manager import (
    QuantumModelManager,
    QuantumModelNotReadyException,
    ModelNotAllowedException,
    ModelNotFoundError,
    ModelServiceError,
)
from src.app.schemas import (
    PredictRequest,
    PredictResponse,
    ErrorResponse,
    current_utc_timestamp,
)

router = APIRouter(prefix="/api/v1", tags=["Model Inference"])


@router.post("/predict", response_model=PredictResponse)
def predict_json(request: PredictRequest) -> PredictResponse:
    """
    Executes operational speech-intent inference using ONLY the canonical quantum model.

    Accepts either precomputed 384-D Whisper embeddings or base64-encoded audio.
    Rejects classical models with HTTP 400 MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE.
    Returns HTTP 503 QUANTUM_MODEL_NOT_READY if quantum checkpoint state is unmaterialized.
    """
    audio_service = AudioService.get_instance()
    quantum_mgr = QuantumModelManager.get_instance()

    transcription = None
    embedding = None
    quality = request.acoustic_quality

    try:
        # 1. Resolve embedding from either base64 audio or pre-extracted vector
        if request.audio_base64:
            waveform, _ = audio_service.decode_base64_audio(request.audio_base64)
            quality = audio_service.estimate_acoustic_quality(waveform)
            embedding = audio_service.extract_whisper_embedding(waveform)
            # Optionally obtain transcription if audio provided
            try:
                transcription = audio_service.transcribe_audio(waveform)
            except Exception:
                transcription = None
        elif request.embedding:
            embedding = np.array(request.embedding, dtype=np.float32)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "MISSING_INPUT",
                    "message": "Either 'audio_base64' or 'embedding' must be provided in the request payload.",
                },
            )

        # 2. Execute operational inference via QuantumModelManager ONLY
        result = quantum_mgr.predict(
            embedding=embedding,
            speaker_id=request.speaker_id,
            model_id=request.model_id or CANONICAL_QUANTUM_MODEL_ID,
            operating_tau=request.operating_tau,
            acoustic_quality=quality,
        )

        if transcription is not None and result.get("transcription") is None:
            result["transcription"] = transcription

        return PredictResponse(**result)

    except ModelNotAllowedException as e:
        # Classical model requested on live prediction route: strictly prohibited
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": e.error_code,
                "model_id": e.model_id,
                "message": e.message,
                "disclaimer": DISCLAIMER_NOTICE,
            },
        )
    except QuantumModelNotReadyException as e:
        # Quantum model unmaterialized or missing reference vectors: fail closed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": e.error_code,
                "model_id": e.model_id,
                "message": e.message,
                "remediation": e.remediation,
                "artifact_path": e.artifact_path,
                "disclaimer": DISCLAIMER_NOTICE,
            },
        )
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": e.error_code, "message": str(e), "disclaimer": DISCLAIMER_NOTICE},
        )
    except AudioValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "AUDIO_VALIDATION_ERROR", "message": str(e)},
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "INFERENCE_ERROR", "message": f"Inference pipeline failure: {str(e)}"},
        )


@router.post("/predict/file", response_model=PredictResponse)
async def predict_file(
    file: UploadFile = File(...),
    speaker_id: str = Form(DEFAULT_SPEAKER),
    model_id: str = Form(CANONICAL_QUANTUM_MODEL_ID),
    operating_tau: Optional[float] = Form(None),
) -> PredictResponse:
    """
    Multipart file upload endpoint: Ingests raw audio, performs local Whisper
    feature extraction and transcription, and executes operational quantum classification.
    Rejects classical models with HTTP 400 and fails closed with HTTP 503 if quantum unready.
    """
    audio_service = AudioService.get_instance()
    quantum_mgr = QuantumModelManager.get_instance()

    try:
        content = await file.read()
        waveform, _ = audio_service.validate_and_load_audio_bytes(content)
        quality = audio_service.estimate_acoustic_quality(waveform)
        embedding = audio_service.extract_whisper_embedding(waveform)
        transcription = audio_service.transcribe_audio(waveform)

        result = quantum_mgr.predict(
            embedding=embedding,
            speaker_id=speaker_id,
            model_id=model_id,
            operating_tau=operating_tau,
            acoustic_quality=quality,
        )
        result["transcription"] = transcription

        return PredictResponse(**result)

    except ModelNotAllowedException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": e.error_code,
                "model_id": e.model_id,
                "message": e.message,
                "disclaimer": DISCLAIMER_NOTICE,
            },
        )
    except QuantumModelNotReadyException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": e.error_code,
                "model_id": e.model_id,
                "message": e.message,
                "remediation": e.remediation,
                "artifact_path": e.artifact_path,
                "disclaimer": DISCLAIMER_NOTICE,
            },
        )
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": e.error_code, "message": str(e), "disclaimer": DISCLAIMER_NOTICE},
        )
    except AudioValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "AUDIO_VALIDATION_ERROR", "message": str(e)},
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "INFERENCE_ERROR", "message": f"Inference pipeline failure: {str(e)}"},
        )
