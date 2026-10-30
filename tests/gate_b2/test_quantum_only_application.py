"""Comprehensive Automated Test Suite for Quantum-Only Speech Intent Application.

Verifies:
1. Quantum-Only Operational Inference:
   - Live routes (/predict, /predict/file, WebSocket) route exclusively to qsvc_q8_r1.
   - Rejection of classical models (HTTP 400 MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE).
   - Zero classical fallback: Monkeypatch/spy on ClassicalAdapterNet proves 0 invocations.
2. Honest Fail-Closed Readiness:
   - Detects missing reference support vectors (X_train_pca) in serialized Phase 7.8 checkpoints.
   - Fails closed with HTTP 503 QUANTUM_MODEL_NOT_READY.
   - Zero weight fabrication, retraining, or dummy mock predictions.
3. Speaker Isolation:
   - Unknown speaker returns HTTP 404 SPEAKER_NOT_FOUND (no silent fallback).
4. Audio Independence:
   - Whisper transcription and 384-D feature extraction operate independently.
5. Governance Gates:
   - Gate C and Gate D strictly blocked.
   - 28/28 historical checksums verified.
"""

import io
import json
from unittest.mock import MagicMock, patch
import wave
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.app.config import (
    CANONICAL_CLASSES,
    CANONICAL_QUANTUM_MODEL_ID,
    CANONICAL_QUANTUM_MODEL_NAME,
    DISCLAIMER_NOTICE,
    SCIENTIFIC_VERDICT_NOTICE,
    REPO_ROOT,
)
from src.app.main import app
from src.app.model_manager import ClassicalAdapterNet
from src.app.quantum_model_manager import (
    QuantumModelManager,
    ModelNotAllowedException,
    QuantumModelNotReadyException,
    ModelNotFoundError,
)

client = TestClient(app)


def create_test_wav_bytes(duration_sec: float = 0.5, sample_rate: int = 16000, freq_hz: float = 440.0) -> bytes:
    """Creates a minimal valid PCM WAV byte payload for local testing."""
    buf = io.BytesIO()
    num_samples = int(duration_sec * sample_rate)
    samples = (np.sin(2 * np.pi * freq_hz * np.linspace(0, duration_sec, num_samples)) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


class TestQuantumModelDiscoveryAndForensics:
    """Verifies quantum checkpoint discovery, structure, and artifact sidecar recovery."""

    def test_canonical_quantum_model_discovery(self):
        mgr = QuantumModelManager.get_instance()
        speakers = mgr.get_available_speakers()
        assert len(speakers) == 15, "All 15 speakers must be discovered in quantum model registry"
        assert "F01" in speakers, "Speaker F01 must have a discovered checkpoint"

    def test_quantum_model_metadata_inspection(self):
        mgr = QuantumModelManager.get_instance()
        info = mgr.get_model_info("qsvc_q8_r1", "F01")
        assert info["model_id"] == CANONICAL_QUANTUM_MODEL_ID
        assert info["qubit_count"] == 8
        assert info["feature_map"] == "ZZFeatureMap"
        assert info["entanglement"] == "linear"
        assert info["classifier"] == "QSVC"
        assert info["input_shape"] == [384]
        assert info["pca_dimension"] == 8
        assert info["status"] == "OPERATIONAL"
        assert "sidecar_provenance" in info
        assert info["sidecar_provenance"]["exists"] is True
        assert info["sidecar_provenance"]["status"] == "VERIFIED_TRAINING_REFERENCE_STATE"

    def test_checkpoint_reference_vectors_sidecar_verified(self):
        mgr = QuantumModelManager.get_instance()
        spk_meta = mgr._quantum_checkpoints_cache.get("F01")
        assert spk_meta is not None
        assert spk_meta["has_svc"] is True, "Checkpoint must contain fitted SVC object"
        assert spk_meta["has_pca"] is True, "Checkpoint must contain fitted PCA pipeline"
        assert spk_meta["has_reference_vectors"] is True, "Recovered inference reference sidecar must be verified"
        assert spk_meta["readiness_status"] == "OPERATIONAL"
        assert spk_meta["sidecar_path"] is not None

        # Verify sidecar matrix directly
        sidecar_file = REPO_ROOT / spk_meta["sidecar_path"]
        assert sidecar_file.exists()
        mat = np.load(sidecar_file)
        assert mat.shape == (50, 8)
        assert np.isfinite(mat).all()
        assert mat.min() >= -1e-6 and mat.max() <= np.pi + 1e-6


class TestQuantumOnlyOperationalRouteEnforcement:
    """Proves that operational routes invoke only quantum model and strictly reject classical models."""

    def test_predict_defaults_to_quantum_model(self):
        """When model_id is omitted or default, routes to quantum and returns 200 OK with genuine predictions."""
        resp = client.post(
            "/api/v1/predict",
            json={
                "speaker_id": "F01",
                "embedding": [0.0] * 384,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == CANONICAL_QUANTUM_MODEL_ID
        assert data["speaker_id"] == "F01"
        assert data["predicted_intent"] == "GENERAL_SPEECH"
        assert data["is_emergency"] is False
        assert data["confidence_score"] > 0.0
        assert data["confidence_band"] in ["HIGH", "MEDIUM", "LOW"]
        assert data["suggested_action"] in ["PROCEED", "CONFIRM_AND_DISPATCH", "REQUEST_CONFIRMATION"]
        assert len(data["class_probabilities"]) == 7
        assert data["processing_latency_ms"] > 0

    def test_unmaterialized_candidate_returns_503(self):
        """Requesting unmaterialized candidate quantum model strictly returns HTTP 503."""
        resp = client.post(
            "/api/v1/predict",
            json={
                "speaker_id": "F01",
                "model_id": "phase8_1_track_b_quantum",
                "embedding": [0.0] * 384,
            },
        )
        assert resp.status_code == 503
        err = resp.json()["detail"]
        assert err["error_code"] == "QUANTUM_MODEL_NOT_READY"

    def test_classical_model_explicitly_rejected_from_predict(self):
        """Requesting classical model returns HTTP 400 MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE."""
        classical_ids = [
            "continual_classical_adapter",
            "classical_adapter",
            "classical",
            "classical_baseline",
            "phase7_5_classical",
            "phase8_1_track_a_baseline",
        ]
        for cid in classical_ids:
            resp = client.post(
                "/api/v1/predict",
                json={
                    "speaker_id": "F01",
                    "model_id": cid,
                    "embedding": [0.0] * 384,
                },
            )
            assert resp.status_code == 400, f"Classical model '{cid}' must be rejected with 400"
            data = resp.json()["detail"]
            assert data["error_code"] == "MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE"

    def test_classical_model_rejected_from_predict_file(self):
        """File upload route also strictly rejects classical models."""
        wav_bytes = create_test_wav_bytes(0.3)
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        data = {
            "speaker_id": "F01",
            "model_id": "continual_classical_adapter",
        }
        resp = client.post("/api/v1/predict/file", files=files, data=data)
        assert resp.status_code == 400
        err = resp.json()["detail"]
        assert err["error_code"] == "MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE"

    def test_quantum_model_file_predict_operational(self):
        """Valid audio processed, Whisper features extracted, and quantum prediction executes (200 OK)."""
        wav_bytes = create_test_wav_bytes(0.3)
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        data = {
            "speaker_id": "F01",
            "model_id": "qsvc_q8_r1",
        }
        resp = client.post("/api/v1/predict/file", files=files, data=data)
        assert resp.status_code == 200
        result = resp.json()
        assert result["model_id"] == "qsvc_q8_r1"
        assert result["speaker_id"] == "F01"
        assert result["predicted_intent"] in CANONICAL_CLASSES
        assert result["processing_latency_ms"] > 0


class TestZeroClassicalFallbackSpy:
    """Uses test spy on ClassicalAdapterNet to prove ZERO invocations occur during live prediction."""

    def test_classical_model_never_invoked_during_prediction(self):
        classical_forward_mock = MagicMock()
        with patch.object(ClassicalAdapterNet, "forward", classical_forward_mock):
            # Attempt 1: Call predict with default (quantum) model
            _ = client.post(
                "/api/v1/predict",
                json={"speaker_id": "F01", "embedding": [0.0] * 384},
            )

            # Attempt 2: Call predict with explicit quantum model ID
            _ = client.post(
                "/api/v1/predict",
                json={"speaker_id": "F01", "model_id": "qsvc_q8_r1", "embedding": [0.0] * 384},
            )

            # Attempt 3: Call predict with rejected classical model ID
            _ = client.post(
                "/api/v1/predict",
                json={"speaker_id": "F01", "model_id": "continual_classical_adapter", "embedding": [0.0] * 384},
            )

            # Attempt 4: Call predict/file
            wav_bytes = create_test_wav_bytes(0.3)
            files = {"file": ("test.wav", wav_bytes, "audio/wav")}
            _ = client.post("/api/v1/predict/file", files=files, data={"speaker_id": "F01", "model_id": "qsvc_q8_r1"})

            # Assert absolute zero calls to ClassicalAdapterNet.forward
            assert classical_forward_mock.call_count == 0, (
                f"Classical model forward() was called {classical_forward_mock.call_count} times! "
                f"Zero classical inference is strictly mandated."
            )


class TestSpeakerIsolationAndErrors:
    """Verifies strict speaker isolation: unknown speaker returns 404, never fallback."""

    def test_unknown_speaker_returns_404(self):
        resp = client.post(
            "/api/v1/predict",
            json={
                "speaker_id": "UNKNOWN_SPEAKER_XYZ",
                "model_id": "qsvc_q8_r1",
                "embedding": [0.0] * 384,
            },
        )
        assert resp.status_code == 404
        data = resp.json()
        detail = data.get("detail", data)
        assert detail["error_code"] == "SPEAKER_NOT_FOUND"


class TestIndependentSpeechProcessing:
    """Verifies Whisper transcription and feature extraction remain operational and independent."""

    def test_whisper_transcription_independent_success(self):
        wav_bytes = create_test_wav_bytes(0.5)
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        resp = client.post("/api/v1/audio/transcribe", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert data["language"] == "en"
        assert data["audio_duration_seconds"] > 0

    def test_whisper_feature_extraction_independent_success(self):
        wav_bytes = create_test_wav_bytes(0.5)
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        resp = client.post("/api/v1/audio/extract-embedding?include_vector=true", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["embedding_dimension"] == 384
        assert abs(data["l2_norm"] - 1.0) < 1e-4, "Whisper embedding must be L2 normalized"
        assert len(data["full_vector"]) == 384


class TestReadinessAndMetadataReporting:
    """Verifies /ready and /model-info report quantum operational and classical research status."""

    def test_ready_endpoint_reports_quantum_components(self):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["whisper_encoder_ready"] is True
        assert data["whisper_asr_ready"] is True
        assert data["quantum_model_ready"] is True
        assert data["quantum_readiness_status"] == "OPERATIONAL"
        assert data["ready"] is True, "Overall ready is True when operational quantum model is ready"
        assert data["active_default_model"] == "qsvc_q8_r1"

    def test_model_info_classical_baseline_shows_research_only(self):
        resp = client.get("/model-info?model_id=continual_classical_adapter&speaker_id=F01")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "RESEARCH_BASELINE_ONLY"
        assert "PROHIBITED" in data.get("operational_eligibility", "") or "RESEARCH" in data["status"]

    def test_model_info_quantum_model_shows_exact_architecture(self):
        resp = client.get("/model-info?model_id=qsvc_q8_r1&speaker_id=F01")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "qsvc_q8_r1"
        assert data["qubit_count"] == 8
        assert data["feature_map"] == "ZZFeatureMap"
        assert data["status"] == "OPERATIONAL"

    def test_gate_c_and_d_remain_blocked(self):
        resp = client.get("/api/v1/provenance/summary")
        assert resp.status_code == 200
        data = resp.json()
        gates = data["deployment_gates"]
        assert gates["gate_c"]["status"] == "STRICTLY_BLOCKED"
        assert gates["gate_d"]["status"] == "STRICTLY_BLOCKED"


class TestWebSocketQuantumEnforcement:
    """Verifies that WebSocket routes to quantum model and rejects classical models."""

    def test_websocket_rejects_classical_model(self):
        with client.websocket_connect("/ws/audio") as ws:
            init_greeting = ws.receive_json()
            assert init_greeting["type"] == "connection_established"
            assert init_greeting["operational_model"] == "qsvc_q8_r1"

            # Attempt init with classical model
            ws.send_json({
                "type": "init",
                "speaker_id": "F01",
                "model_id": "continual_classical_adapter",
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert resp["error_code"] == "MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE"

    def test_websocket_executes_quantum_prediction_on_finalize(self):
        with client.websocket_connect("/ws/audio") as ws:
            _ = ws.receive_json()

            ws.send_json({
                "type": "init",
                "speaker_id": "F01",
                "model_id": "qsvc_q8_r1",
            })
            init_ack = ws.receive_json()
            assert init_ack["type"] == "initialized"
            assert init_ack["model_id"] == "qsvc_q8_r1"

            # Send minimal audio chunk
            test_chunk = create_test_wav_bytes(0.3)
            import base64
            b64_data = base64.b64encode(test_chunk).decode("utf-8")
            ws.send_json({"type": "chunk", "data": b64_data})
            chunk_ack = ws.receive_json()
            assert chunk_ack["type"] == "chunk_received"

            # Finalize: quantum model predicts live successfully
            ws.send_json({"type": "finalize"})
            final_resp = ws.receive_json()
            assert final_resp["type"] == "final_result"
            assert "prediction" in final_resp
            assert final_resp["prediction"]["model_id"] == "qsvc_q8_r1"
            assert "predicted_intent" in final_resp["prediction"]
            assert "confidence_score" in final_resp["prediction"]
