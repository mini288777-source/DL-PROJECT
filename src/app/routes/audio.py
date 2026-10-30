"""Audio ingestion, Whisper transcription, and feature extraction endpoints."""

from typing import Any, Dict
from fastapi import APIRouter, File, UploadFile, HTTPException, status
import numpy as np

from src.app.audio_service import AudioService, AudioValidationError, AudioProcessingError
from src.app.config import DISCLAIMER_NOTICE
from src.app.schemas import (
    Base64AudioRequest,
    TranscriptionResponse,
    AudioExtractionResponse,
    current_utc_timestamp,
)

router = APIRouter(prefix="/api/v1/audio", tags=["Audio & Speech Processing"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(file: UploadFile = File(...)) -> TranscriptionResponse:
    """
    Ingests an uploaded audio file (WAV, MP3, FLAC, OGG, WebM), resamples to 16 kHz mono,
    and performs speech-to-text transcription using local offline Whisper.
    """
    audio_service = AudioService.get_instance()
    try:
        content = await file.read()
        waveform, duration = audio_service.validate_and_load_audio_bytes(content)
        quality = audio_service.estimate_acoustic_quality(waveform)
        text = audio_service.transcribe_audio(waveform)

        return TranscriptionResponse(
            text=text,
            language="en",
            audio_duration_seconds=round(duration, 3),
            acoustic_quality=round(quality, 4),
            timestamp_utc=current_utc_timestamp(),
            disclaimer=DISCLAIMER_NOTICE,
        )
    except AudioValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "AUDIO_VALIDATION_ERROR", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "TRANSCRIPTION_FAILED", "message": f"Transcription error: {str(e)}"},
        )


@router.post("/transcribe-base64", response_model=TranscriptionResponse)
def transcribe_base64(payload: Base64AudioRequest) -> TranscriptionResponse:
    """Ingests base64-encoded audio data and performs speech transcription using local Whisper."""
    audio_service = AudioService.get_instance()
    try:
        waveform, duration = audio_service.decode_base64_audio(payload.audio_base64)
        quality = audio_service.estimate_acoustic_quality(waveform)
        text = audio_service.transcribe_audio(waveform)

        return TranscriptionResponse(
            text=text,
            language="en",
            audio_duration_seconds=round(duration, 3),
            acoustic_quality=round(quality, 4),
            timestamp_utc=current_utc_timestamp(),
            disclaimer=DISCLAIMER_NOTICE,
        )
    except AudioValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "AUDIO_VALIDATION_ERROR", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "TRANSCRIPTION_FAILED", "message": f"Transcription error: {str(e)}"},
        )


@router.post("/extract-embedding", response_model=AudioExtractionResponse)
async def extract_embedding_file(
    file: UploadFile = File(...),
    include_vector: bool = False,
) -> AudioExtractionResponse:
    """
    Ingests an uploaded audio file and extracts the exact 384-dimensional
    Whisper-tiny frozen embedding used by the research pipeline.
    """
    audio_service = AudioService.get_instance()
    try:
        content = await file.read()
        waveform, duration = audio_service.validate_and_load_audio_bytes(content)
        embedding = audio_service.extract_whisper_embedding(waveform)
        norm = float(np.linalg.norm(embedding))

        vector_preview = [round(float(v), 5) for v in embedding[:16]] if include_vector else None
        full_vector = [round(float(v), 6) for v in embedding] if include_vector else None

        return AudioExtractionResponse(
            embedding_dimension=len(embedding),
            l2_norm=round(norm, 6),
            audio_duration_seconds=round(duration, 3),
            vector_preview=vector_preview,
            full_vector=full_vector,
            timestamp_utc=current_utc_timestamp(),
        )
    except AudioValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "AUDIO_VALIDATION_ERROR", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "EXTRACTION_FAILED", "message": f"Feature extraction error: {str(e)}"},
        )


@router.post("/extract-embedding-base64", response_model=AudioExtractionResponse)
def extract_embedding_base64(
    payload: Base64AudioRequest,
    include_vector: bool = False,
) -> AudioExtractionResponse:
    """Ingests base64-encoded audio and extracts the 384-dimensional Whisper-tiny frozen embedding."""
    audio_service = AudioService.get_instance()
    try:
        waveform, duration = audio_service.decode_base64_audio(payload.audio_base64)
        embedding = audio_service.extract_whisper_embedding(waveform)
        norm = float(np.linalg.norm(embedding))

        vector_preview = [round(float(v), 5) for v in embedding[:16]] if include_vector else None
        full_vector = [round(float(v), 6) for v in embedding] if include_vector else None

        return AudioExtractionResponse(
            embedding_dimension=len(embedding),
            l2_norm=round(norm, 6),
            audio_duration_seconds=round(duration, 3),
            vector_preview=vector_preview,
            full_vector=full_vector,
            timestamp_utc=current_utc_timestamp(),
        )
    except AudioValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "AUDIO_VALIDATION_ERROR", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "EXTRACTION_FAILED", "message": f"Feature extraction error: {str(e)}"},
        )

