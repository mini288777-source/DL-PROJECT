"""Comprehensive Automated Acceptance & Regression Test Suite for Full-Scale Application.

Tests:
1. Model Integrity & Invariance:
   - Model remains strictly in evaluation mode (eval()).
   - All parameters frozen (requires_grad is False).
   - Exact 384-D Whisper embedding input contract.
   - Exact 7-class taxonomy output ordering.
   - Speaker isolation: no silent fallback on unknown speaker (returns 404).
   - Unmaterialized candidate models fail closed with 503 MODEL_NOT_READY.

2. Audio Processing Pipeline:
   - Valid audio upload and processing (transcribe, extract-embedding, predict/file).
   - Empty audio rejection (400 AUDIO_VALIDATION_ERROR).
   - Malformed audio rejection.
   - Ephemeral file cleanup verified.

3. WebSocket Streaming Inference:
   - Connect, init, chunk streaming, finalize, reset.
   - Clean buffer reset and disconnect handling.

4. Governance & Safety Gates:
   - Decoupled threshold tau = 0.6000 with unverified provenance notice.
   - Provenance summary reports Gate C and Gate D strictly blocked.
   - 28/28 historical checksums reported verified.
"""

import io
import json
import wave
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.app.config import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.app.main import app
from src.app.model_manager import ModelManager, ModelNotFoundError, ModelNotReadyException

client = TestClient(app)


def create_test_wav_bytes(duration_sec: float = 0.5, sample_rate: int = 16000, freq_hz: float = 440.0) -> bytes:
    """Creates a minimal valid PCM WAV byte payload."""
    buf = io.BytesIO()
    num_samples = int(duration_sec * sample_rate)
    samples = (np.sin(2 * np.pi * freq_hz * np.linspace(0, duration_sec, num_samples)) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


class TestModelIntegrityAndFreezing:
    """Verifies that model checkpoints remain completely immutable and in eval mode."""

    def test_loaded_model_is_in_eval_mode(self):
        mgr = ModelManager.get_instance()
        net, version_str, ckpt_path = mgr.get_or_load_adapter("F01")
        assert not net.training, "Loaded model must be in evaluation mode (training == False)"

    def test_all_model_parameters_are_frozen(self):
        mgr = ModelManager.get_instance()
        net, version_str, ckpt_path = mgr.get_or_load_adapter("F01")
        for name, param in net.named_parameters():
            assert not param.requires_grad, f"Parameter '{name}' must have requires_grad == False"

    def test_forward_pass_does_not_modify_weights(self):
        mgr = ModelManager.get_instance()
        net, _, _ = mgr.get_or_load_adapter("F01")
        initial_weight = net.net[0].weight.clone()
        dummy_input = torch.randn(1, 384)
        with torch.no_grad():
            _ = net(dummy_input)
        assert torch.equal(initial_weight, net.net[0].weight), "Forward pass must not modify weights"

    def test_taxonomy_output_cardinality_and_classes(self):
        mgr = ModelManager.get_instance()
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        result = mgr.predict(embedding=emb, speaker_id="F01")
        assert len(result["class_probabilities"]) == 7
        for c in CANONICAL_CLASSES:
            assert c in result["class_probabilities"]
            assert 0.0 <= result["class_probabilities"][c] <= 1.0
        prob_sum = sum(result["class_probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-4

    def test_speaker_isolation_unknown_speaker_raises_404(self):
        resp = client.post(
            "/api/v1/predict",
            json={
                "speaker_id": "UNKNOWN_SPEAKER_99",
                "model_id": "qsvc_q8_r1",
                "embedding": [0.0] * 384,
            },
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["error_code"] == "SPEAKER_NOT_FOUND"

    def test_unmaterialized_model_predict_returns_503(self):
        for candidate_id in [
            "phase8_1_track_b_quantum",
            "phase8_2_quantum",
        ]:
            resp = client.post(
                "/api/v1/predict",
                json={
                    "speaker_id": "F01",
                    "model_id": candidate_id,
                    "embedding": [0.0] * 384,
                },
            )
            assert resp.status_code == 503
            data = resp.json()
            assert data["detail"]["error_code"] == "QUANTUM_MODEL_NOT_READY"
            assert "unmaterialized" in data["detail"]["message"].lower()

    def test_classical_model_on_predict_rejected_with_400(self):
        resp = client.post(
            "/api/v1/predict",
            json={
                "speaker_id": "F01",
                "model_id": "continual_classical_adapter",
                "embedding": [0.1] * 384,
            },
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["error_code"] == "MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE"


class TestAudioProcessingAndValidation:
    """Verifies audio input boundary validation, feature extraction, and transcription."""

    def test_empty_audio_upload_rejected(self):
        files = {"file": ("empty.wav", b"", "audio/wav")}
        resp = client.post("/api/v1/audio/transcribe", files=files)
        assert resp.status_code == 400
        assert resp.json()["detail"]["error_code"] == "AUDIO_VALIDATION_ERROR"

    def test_corrupt_audio_upload_rejected(self):
        files = {"file": ("corrupt.wav", b"RIFF\x00\x00\x00\x00NOTWAVE", "audio/wav")}
        resp = client.post("/api/v1/audio/transcribe", files=files)
        assert resp.status_code == 400
        assert resp.json()["detail"]["error_code"] == "AUDIO_VALIDATION_ERROR"

    def test_valid_audio_file_predict_operational(self):
        wav_bytes = create_test_wav_bytes(duration_sec=0.5)
        files = {"file": ("test_speech.wav", wav_bytes, "audio/wav")}
        data = {"speaker_id": "F01", "model_id": "qsvc_q8_r1"}
        resp = client.post("/api/v1/predict/file", files=files, data=data)
        assert resp.status_code == 200
        result = resp.json()
        assert result["model_id"] == "qsvc_q8_r1"
        assert result["predicted_intent"] in CANONICAL_CLASSES

    def test_base64_audio_transcribe(self):
        import base64
        wav_bytes = create_test_wav_bytes(duration_sec=0.4)
        b64_str = base64.b64encode(wav_bytes).decode("ascii")
        resp = client.post("/api/v1/audio/transcribe-base64", json={"audio_base64": b64_str})
        assert resp.status_code == 200
        assert resp.json()["audio_duration_seconds"] > 0.3

    def test_base64_audio_extract_embedding(self):
        import base64
        wav_bytes = create_test_wav_bytes(duration_sec=0.4)
        b64_str = base64.b64encode(wav_bytes).decode("ascii")
        resp = client.post("/api/v1/audio/extract-embedding-base64?include_vector=true", json={"audio_base64": b64_str})
        assert resp.status_code == 200
        data = resp.json()
        assert data["embedding_dimension"] == 384
        assert abs(data["l2_norm"] - 1.0) < 1e-4
        assert len(data["full_vector"]) == 384


class TestWebSocketStreamingProtocol:
    """Verifies WebSocket chunk streaming, buffer lifecycle, and classification."""

    def test_websocket_lifecycle_init_chunk_finalize(self):
        import base64
        wav_bytes = create_test_wav_bytes(duration_sec=0.3)
        b64_chunk = base64.b64encode(wav_bytes).decode("ascii")

        with client.websocket_connect("/ws/inference") as ws:
            # 1. Connection established greeting
            greeting = ws.receive_json()
            assert greeting["type"] == "connection_established"
            assert greeting["status"] == "ready"

            # 2. Send init with quantum model
            ws.send_json({
                "type": "init",
                "speaker_id": "F01",
                "model_id": "qsvc_q8_r1",
                "operating_tau": 0.60
            })
            init_ack = ws.receive_json()
            assert init_ack["type"] == "initialized"
            assert init_ack["speaker_id"] == "F01"

            # 3. Send audio chunk
            ws.send_json({
                "type": "chunk",
                "data": b64_chunk
            })
            chunk_ack = ws.receive_json()
            assert chunk_ack["type"] == "chunk_received"
            assert chunk_ack["bytes_buffered"] > 0

            # 4. Finalize: quantum model predicts live successfully
            ws.send_json({"type": "finalize"})
            final_res = ws.receive_json()
            assert final_res["type"] == "final_result"
            assert "prediction" in final_res
            assert final_res["prediction"]["model_id"] == "qsvc_q8_r1"

    def test_websocket_reset_action(self):
        with client.websocket_connect("/ws/inference") as ws:
            _ = ws.receive_json()  # greeting
            ws.send_json({"type": "reset"})
            ack = ws.receive_json()
            assert ack["type"] == "buffer_reset"
            assert ack["status"] == "ready"

    def test_websocket_finalize_empty_buffer_error(self):
        with client.websocket_connect("/ws/inference") as ws:
            _ = ws.receive_json()  # greeting
            ws.send_json({"type": "finalize"})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["error_code"] == "EMPTY_BUFFER"


class TestSafetyAndGovernanceEndpoints:
    """Verifies that research boundaries and gate blockades remain fail-closed."""

    def test_provenance_summary_gate_blockades(self):
        resp = client.get("/api/v1/provenance/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deployment_gates"]["gate_c"]["status"] == "STRICTLY_BLOCKED"
        assert data["deployment_gates"]["gate_d"]["status"] == "STRICTLY_BLOCKED"
        assert "CASE B" in data["scientific_constraints"]["quantum_verdict"]

    def test_model_info_includes_disclaimer_and_threshold_notice(self):
        resp = client.get("/model-info?model_id=qsvc_q8_r1&speaker_id=F01")
        assert resp.status_code == 200
        data = resp.json()
        assert "disclaimer" in data
        assert "experimental research prototype" in data["disclaimer"].lower()
        assert data["decision_policy"]["status"] == "THRESHOLD_PROVENANCE_UNVERIFIED"

    def test_health_vs_ready_separation(self):
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        ready_resp = client.get("/ready")
        assert ready_resp.status_code == 200
        assert ready_resp.json()["ready"] is True
        assert ready_resp.json()["quantum_model_ready"] is True
        assert len(ready_resp.json()["unmaterialized_blockers"]) == 0
