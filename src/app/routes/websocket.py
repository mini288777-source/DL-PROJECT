"""WebSocket streaming audio processing endpoint for Quantum-Only Speech Application."""

import base64
import json
import logging
import uuid
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.app.audio_service import AudioService, AudioValidationError
from src.app.config import (
    DEFAULT_SPEAKER,
    CANONICAL_QUANTUM_MODEL_ID,
    DISCLAIMER_NOTICE,
    MAX_UPLOAD_BYTES,
    SAMPLE_RATE,
)
from src.app.quantum_model_manager import (
    QuantumModelManager,
    QuantumModelNotReadyException,
    ModelNotAllowedException,
    ModelNotFoundError,
)
from src.app.schemas import current_utc_timestamp

logger = logging.getLogger("gate_b_app.websocket")
router = APIRouter(tags=["Streaming Audio"])


@router.websocket("/ws/audio")
@router.websocket("/ws/inference")
async def websocket_audio_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for chunked streaming audio ingestion.

    Accepts base64 audio chunks, aggregates them in memory, and performs final
    offline Whisper transcription and quantum-only intent classification upon 'finalize'.
    Strictly rejects classical models and strictly fails closed with QUANTUM_MODEL_NOT_READY.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    buffer = bytearray()
    speaker_id = DEFAULT_SPEAKER
    model_id = CANONICAL_QUANTUM_MODEL_ID
    operating_tau = None

    audio_service = AudioService.get_instance()
    quantum_mgr = QuantumModelManager.get_instance()

    try:
        # Send initial connection greeting
        await websocket.send_json({
            "type": "connection_established",
            "session_id": session_id,
            "status": "ready",
            "operational_model": CANONICAL_QUANTUM_MODEL_ID,
            "disclaimer": DISCLAIMER_NOTICE,
            "timestamp_utc": current_utc_timestamp(),
        })

        while True:
            raw_text = await websocket.receive_text()
            try:
                msg = json.loads(raw_text)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error_code": "INVALID_JSON",
                    "message": "Message payload must be valid JSON.",
                })
                continue

            msg_type = msg.get("type", "").lower()

            if msg_type == "init":
                speaker_id = msg.get("speaker_id", DEFAULT_SPEAKER)
                req_model = msg.get("model_id", CANONICAL_QUANTUM_MODEL_ID)
                operating_tau = msg.get("operating_tau")

                # If classical model requested, reject immediately
                if "classical" in req_model.lower():
                    await websocket.send_json({
                        "type": "error",
                        "error_code": "MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE",
                        "model_id": req_model,
                        "message": (
                            f"Model '{req_model}' is a classical baseline and is NOT ALLOWED for operational prediction. "
                            f"Operational prediction is restricted exclusively to the quantum model ('{CANONICAL_QUANTUM_MODEL_ID}')."
                        ),
                    })
                    continue

                model_id = req_model or CANONICAL_QUANTUM_MODEL_ID
                buffer.clear()
                await websocket.send_json({
                    "type": "initialized",
                    "session_id": session_id,
                    "speaker_id": speaker_id,
                    "model_id": model_id,
                    "operating_tau": operating_tau,
                    "status": "session_active",
                })

            elif msg_type == "chunk":
                chunk_b64 = msg.get("data", "")
                if not chunk_b64:
                    await websocket.send_json({"type": "warning", "message": "Empty chunk data received."})
                    continue

                if "," in chunk_b64:
                    chunk_b64 = chunk_b64.split(",", 1)[1]

                try:
                    chunk_bytes = base64.b64decode(chunk_b64)
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error_code": "MALFORMED_CHUNK",
                        "message": f"Failed to decode base64 chunk: {e}",
                    })
                    continue

                if len(buffer) + len(chunk_bytes) > MAX_UPLOAD_BYTES:
                    await websocket.send_json({
                        "type": "error",
                        "error_code": "BUFFER_LIMIT_EXCEEDED",
                        "message": f"Aggregated audio exceeds maximum buffer limit of {MAX_UPLOAD_BYTES} bytes.",
                    })
                    buffer.clear()
                    continue

                buffer.extend(chunk_bytes)
                # Approximate duration calculation
                est_duration = len(buffer) / (SAMPLE_RATE * 2) if len(buffer) > 0 else 0.0

                await websocket.send_json({
                    "type": "chunk_received",
                    "bytes_buffered": len(buffer),
                    "estimated_duration_sec": round(est_duration, 2),
                    "status": "buffering",
                })

            elif msg_type == "finalize":
                if len(buffer) == 0:
                    await websocket.send_json({
                        "type": "error",
                        "error_code": "EMPTY_BUFFER",
                        "message": "No audio chunks have been received to finalize.",
                    })
                    continue

                try:
                    waveform, duration = audio_service.validate_and_load_audio_bytes(bytes(buffer))
                    quality = audio_service.estimate_acoustic_quality(waveform)
                    embedding = audio_service.extract_whisper_embedding(waveform)
                    transcription = audio_service.transcribe_audio(waveform)

                    prediction = quantum_mgr.predict(
                        embedding=embedding,
                        speaker_id=speaker_id,
                        model_id=model_id,
                        operating_tau=operating_tau,
                        acoustic_quality=quality,
                    )
                    prediction["transcription"] = transcription

                    await websocket.send_json({
                        "type": "final_result",
                        "session_id": session_id,
                        "audio_duration_seconds": round(duration, 3),
                        "acoustic_quality": round(quality, 4),
                        "transcription": transcription,
                        "prediction": prediction,
                        "disclaimer": DISCLAIMER_NOTICE,
                        "timestamp_utc": current_utc_timestamp(),
                    })
                except ModelNotAllowedException as mae:
                    await websocket.send_json({
                        "type": "error",
                        "error_code": mae.error_code,
                        "model_id": mae.model_id,
                        "message": mae.message,
                        "disclaimer": DISCLAIMER_NOTICE,
                    })
                except QuantumModelNotReadyException as qne:
                    await websocket.send_json({
                        "type": "error",
                        "error_code": qne.error_code,
                        "model_id": qne.model_id,
                        "message": qne.message,
                        "remediation": qne.remediation,
                        "disclaimer": DISCLAIMER_NOTICE,
                    })
                except ModelNotFoundError as mnf:
                    await websocket.send_json({
                        "type": "error",
                        "error_code": "SPEAKER_NOT_FOUND",
                        "message": str(mnf),
                    })
                except AudioValidationError as ave:
                    await websocket.send_json({
                        "type": "error",
                        "error_code": "AUDIO_VALIDATION_ERROR",
                        "message": str(ave),
                    })
                except Exception as ex:
                    await websocket.send_json({
                        "type": "error",
                        "error_code": "PROCESSING_FAILED",
                        "message": f"Processing error: {str(ex)}",
                    })
                finally:
                    # Clear in-memory buffer immediately after finalization
                    buffer.clear()

            elif msg_type == "reset":
                buffer.clear()
                await websocket.send_json({
                    "type": "buffer_reset",
                    "session_id": session_id,
                    "status": "ready",
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "error_code": "UNKNOWN_ACTION",
                    "message": f"Unknown action type: '{msg_type}'. Expected 'init', 'chunk', 'finalize', or 'reset'.",
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for session {session_id}")
    finally:
        buffer.clear()
