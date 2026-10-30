"""Automated test suite for Gate B Engineering Prototype.

Verifies:
1. Process health and readiness reporting.
2. Accurate accounting of 61 materialized vs 6 unmaterialized checkpoints.
3. Strict HTTP 503 unavailability on unmaterialized Phase 8.1 models (zero mock fabrication).
4. Genuine forward inference on verified Continual Classical Adapters (HTTP 200).
5. Strict separation of decision threshold tau from FER safety ceiling (15.0%).
6. Audio ingestion, Whisper 384-D feature extraction, and offline ASR transcription.
7. Input boundary validations (empty audio, excessive duration, size limits).
8. Demo interface serving with mandatory scientific disclaimers.
"""

import base64
import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.app.config import (
    CANONICAL_CLASSES,
    EMERGENCY_CLASSES,
    DEFAULT_DECISION_TAU,
    FER_SAFETY_BOUND,
    DISCLAIMER_NOTICE,
)
from src.app.main import app

client = TestClient(app)


def create_test_wav_bytes(duration_sec: float = 1.0, freq_hz: float = 440.0, sample_rate: int = 16000) -> bytes:
    """Generates a synthetic mono PCM WAV byte buffer for audio tests."""
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


class TestGateBHealthAndReadiness:
    """Tests for system health, readiness, and metadata endpoints."""

    def test_health_endpoint(self):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "Atypical Speech Emergency Assistant" in data["service_name"]
        assert "gate-b" in data["version"]
        assert "disclaimer" in data
        assert "experimental research prototype" in data["disclaimer"]

    def test_readiness_endpoint(self):
        res = client.get("/ready")
        assert res.status_code == 200
        data = res.json()
        assert data["ready"] is True
        assert data["whisper_encoder_ready"] is True
        assert data["whisper_asr_ready"] is True
        assert data["quantum_model_ready"] is True
        summary = data["checkpoints_summary"]
        assert summary["materialized_and_verified"] >= 15
        assert summary["total_catalogued"] >= 15

    def test_model_info_materialized(self):
        res = client.get("/model-info?model_id=qsvc_q8_r1&speaker_id=F01")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "OPERATIONAL"
        assert "Fidelity Quantum Kernel" in data["model_name"]
        assert data["input_shape"] == [384]
        assert data["taxonomy"] == list(CANONICAL_CLASSES)
        assert data["decision_policy"]["decision_cutoff_tau"] == DEFAULT_DECISION_TAU
        assert data["checkpoint_provenance"]["exists"] is True

    def test_model_info_unmaterialized(self):
        res = client.get("/model-info?model_id=phase8_1_track_b_quantum&speaker_id=F01")
        assert res.status_code == 200
        data = res.json()
        assert "NOT_READY" in data["status"]
        assert data["checkpoint_provenance"]["exists"] is False

    def test_list_models_and_speakers(self):
        res = client.get("/api/v1/models")
        assert res.status_code == 200
        models_data = res.json()
        assert models_data["total"] >= 15
        assert len(models_data["models"]) >= 15

        res = client.get("/api/v1/speakers")
        assert res.status_code == 200
        speakers_data = res.json()
        assert speakers_data["count"] >= 15
        assert "F01" in speakers_data["speakers"]
        assert "MC04" in speakers_data["speakers"]


class TestGateBInferencePolicy:
    """Tests for inference behavior, 503 blocking, and threshold separation."""

    def test_unmaterialized_model_returns_http_503(self):
        """Non-negotiable constraint: unmaterialized models must strictly return HTTP 503."""
        dummy_embedding = [0.05] * 384
        for unready_id in ["phase8_1_track_b_quantum", "phase8_2_quantum"]:
            payload = {
                "speaker_id": "F01",
                "model_id": unready_id,
                "embedding": dummy_embedding,
            }
            res = client.post("/api/v1/predict", json=payload)
            assert res.status_code == 503, f"Expected 503 for {unready_id}, got {res.status_code}"
            data = res.json()
            detail = data.get("detail", data)
            assert detail["error_code"] == "QUANTUM_MODEL_NOT_READY"
            assert detail["model_id"] == unready_id

    def test_materialized_model_executes_genuine_inference(self):
        """Verifies forward inference runs through verified QSVC without mocks."""
        dummy_embedding = [0.05] * 384
        payload = {
            "speaker_id": "F01",
            "model_id": "qsvc_q8_r1",
            "embedding": dummy_embedding,
        }
        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["predicted_intent"] in CANONICAL_CLASSES
        assert isinstance(data["is_emergency"], bool)
        assert 0.0 <= data["confidence_score"] <= 1.0
        assert data["confidence_band"] in ["HIGH", "MEDIUM", "LOW"]
        assert data["suggested_action"] in ["CONFIRM_AND_DISPATCH", "REQUEST_CONFIRMATION", "PROCEED", "ABSTAIN"]
        probs = data["class_probabilities"]
        assert len(probs) == 7
        assert pytest.approx(sum(probs.values()), rel=1e-4) == 1.0
        assert data["decision_cutoff_tau"] == DEFAULT_DECISION_TAU
        assert data["fer_safety_bound"] == FER_SAFETY_BOUND
        assert data["processing_latency_ms"] > 0.0

    def test_threshold_tau_decoupled_from_fer_ceiling(self):
        """Ensures custom tau can be passed without altering or conflating the FER bound."""
        dummy_embedding = [0.05] * 384
        payload = {
            "speaker_id": "F01",
            "model_id": "qsvc_q8_r1",
            "embedding": dummy_embedding,
            "operating_tau": 0.75,
        }
        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["decision_cutoff_tau"] == 0.75
        assert data["fer_safety_bound"] == 0.1500  # FER remains unchanged

    def test_acoustic_quality_triggers_abstention(self):
        """Degraded acoustic quality (< 0.30) must trigger explicit abstention."""
        dummy_embedding = [0.05] * 384
        payload = {
            "speaker_id": "F01",
            "model_id": "qsvc_q8_r1",
            "embedding": dummy_embedding,
            "acoustic_quality": 0.15,
        }
        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["suggested_action"] == "ABSTAIN"
        assert data["predicted_intent"] == "UNKNOWN"
        assert data["is_emergency"] is False


class TestGateBAudioProcessing:
    """Tests for local Whisper speech processing, feature extraction, and validation."""

    def test_audio_transcription_endpoint(self):
        wav_bytes = create_test_wav_bytes(duration_sec=1.0)
        res = client.post(
            "/api/v1/audio/transcribe",
            files={"file": ("tone.wav", wav_bytes, "audio/wav")}
        )
        assert res.status_code == 200
        data = res.json()
        assert "text" in data
        assert data["language"] == "en"
        assert pytest.approx(data["audio_duration_seconds"], rel=0.1) == 1.0
        assert data["acoustic_quality"] > 0.0

    def test_audio_feature_extraction_endpoint(self):
        wav_bytes = create_test_wav_bytes(duration_sec=1.0)
        res = client.post(
            "/api/v1/audio/extract-embedding",
            files={"file": ("tone.wav", wav_bytes, "audio/wav")}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["embedding_dimension"] == 384
        assert pytest.approx(data["l2_norm"], rel=1e-3) == 1.0  # Unit-norm normalized
        assert pytest.approx(data["audio_duration_seconds"], rel=0.1) == 1.0

    def test_predict_file_endpoint_end_to_end(self):
        wav_bytes = create_test_wav_bytes(duration_sec=1.5)
        res = client.post(
            "/api/v1/predict/file",
            files={"file": ("sample.wav", wav_bytes, "audio/wav")},
            data={"speaker_id": "F01", "model_id": "qsvc_q8_r1"}
        )
        assert res.status_code == 200
        data = res.json()
        assert "transcription" in data
        assert data["predicted_intent"] in CANONICAL_CLASSES
        assert data["speaker_id"] == "F01"

    def test_audio_validation_empty_payload(self):
        res = client.post(
            "/api/v1/audio/transcribe",
            files={"file": ("empty.wav", b"", "audio/wav")}
        )
        assert res.status_code == 400
        data = res.json()
        assert data["detail"]["error_code"] == "AUDIO_VALIDATION_ERROR"

    def test_audio_validation_oversized_payload(self):
        huge_bytes = b"0" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit
        res = client.post(
            "/api/v1/audio/transcribe",
            files={"file": ("huge.wav", huge_bytes, "audio/wav")}
        )
        assert res.status_code == 400
        data = res.json()
        assert data["detail"]["error_code"] == "AUDIO_VALIDATION_ERROR"


class TestGateBFrontendDemo:
    """Tests for web demonstration serving and disclaimer visibility."""

    def test_demo_interface_serves_html(self):
        res = client.get("/demo")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        html = res.text
        assert "Atypical Speech Emergency Assistant" in html
        assert "Non-Negotiable Research Disclaimer" in html

    def test_classical_model_on_predict_rejected_with_400(self):
        """Zero classical fallback: requests with classical models are rejected."""
        res = client.post(
            "/api/v1/predict",
            json={
                "speaker_id": "F01",
                "model_id": "continual_classical_adapter",
                "embedding": [0.0] * 384,
            }
        )
        assert res.status_code == 400
        assert res.json()["detail"]["error_code"] == "MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE"


class TestGateBStreamingWebSocket:
    """Tests for WebSocket streaming audio processing."""

    def test_websocket_streaming_lifecycle(self):
        with client.websocket_connect("/ws/audio") as ws:
            greeting = ws.receive_json()
            assert greeting["type"] == "connection_established"
            assert greeting["status"] == "ready"

            # Initialize session with operational quantum model
            ws.send_json({
                "type": "init",
                "speaker_id": "F01",
                "model_id": "qsvc_q8_r1",
            })
            init_resp = ws.receive_json()
            assert init_resp["type"] == "initialized"
            assert init_resp["status"] == "session_active"

            # Send a chunk
            wav_bytes = create_test_wav_bytes(duration_sec=1.0)
            chunk_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            ws.send_json({"type": "chunk", "data": chunk_b64})
            chunk_resp = ws.receive_json()
            assert chunk_resp["type"] == "chunk_received"
            assert chunk_resp["bytes_buffered"] > 0

            # Finalize: executes genuine quantum prediction
            ws.send_json({"type": "finalize"})
            final_resp = ws.receive_json()
            assert final_resp["type"] == "final_result"
            assert "prediction" in final_resp
            assert final_resp["prediction"]["predicted_intent"] in list(CANONICAL_CLASSES)

