"""Gate B.2 Research Demonstration Feature & Route Acceptance Tests.

Verifies:
- All research routes (/phases, /metrics, /operating-points, /figures)
- Provenance and reproducibility summary (/provenance/summary)
- Static figure mounting (/figures/readiness/*, /figures/phase8_1/*)
- Optional embedding vector inspection in audio extraction
- Unmaterialized model 503 enforcement
- Threshold decoupling and gate boundary restrictions
"""

import io
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app.main import app

client = TestClient(app)


def create_mock_wav_bytes(duration_sec: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Creates a minimal valid WAV file in memory."""
    buf = io.BytesIO()
    num_samples = int(duration_sec * sample_rate)
    samples = (np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, num_samples)) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


class TestGateB2ResearchRoutes:
    """Tests research data routes parsing frozen historical results."""

    def test_research_phases_endpoint(self):
        resp = client.get("/api/v1/research/phases")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 9
        phases = {p["phase_id"] for p in data["phases"]}
        assert "phase1" in phases
        assert "phase6" in phases
        assert "phase8_1" in phases
        assert "phase8_2_1" in phases
        assert "CASE B" in data["quantum_verdict"]

    def test_research_metrics_endpoint(self):
        resp = client.get("/api/v1/research/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "phase8_1_comparison" in data
        assert "phase6_5_6_comparison" in data
        assert len(data["phase8_1_comparison"]["rows"]) > 0
        assert len(data["phase6_5_6_comparison"]["rows"]) > 0

    def test_safety_operating_points_endpoint(self):
        resp = client.get("/api/v1/research/operating-points")
        assert resp.status_code == 200
        data = resp.json()
        assert data["canonical_fer_ceiling"] == 0.1500
        assert data["demonstration_default_tau"] == 0.60
        assert data["total_points"] > 0
        first_pt = data["operating_points"][0]
        assert "threshold_tau" in first_pt
        assert "emergency_recall" in first_pt
        assert "FER" in first_pt
        assert "safety_bound_passed" in first_pt

    def test_research_figures_endpoint(self):
        resp = client.get("/api/v1/research/figures")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 7
        for fig in data["figures"]:
            assert fig["exists_on_disk"] is True
            assert fig["status"] == "VERIFIED_ON_DISK"
            assert fig["url"].startswith("/figures/")

    def test_static_figures_serving(self):
        # Test readiness figure mount
        resp = client.get("/figures/readiness/fig3_fer_reduction_flat_vs_hierarchical.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] in ["image/png", "application/octet-stream"]

        # Test phase8_1 figure mount
        resp2 = client.get("/figures/phase8_1/emergency_precision_recall.png")
        assert resp2.status_code == 200
        assert resp2.headers["content-type"] in ["image/png", "application/octet-stream"]

    def test_provenance_summary_endpoint(self):
        resp = client.get("/api/v1/provenance/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset_provenance"]["total_utterances"] == 4852
        assert data["dataset_provenance"]["speakers_count"] == 15
        assert data["cryptographic_integrity"]["total_tracked_files"] == 28
        assert "100% REGRESSION INTEGRITY" in data["cryptographic_integrity"]["audit_result"]
        assert data["test_suite_status"]["total_tests"] >= 577
        assert data["deployment_gates"]["gate_c"]["status"] == "STRICTLY_BLOCKED"
        assert data["deployment_gates"]["gate_d"]["status"] == "STRICTLY_BLOCKED"


class TestGateB2AudioInspectionFeatures:
    """Tests audio vector inspection opt-in and extraction metadata."""

    def test_embedding_extraction_summary_only_by_default(self):
        wav_bytes = create_mock_wav_bytes(0.5)
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        resp = client.post("/api/v1/audio/extract-embedding", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["embedding_dimension"] == 384
        assert abs(data["l2_norm"] - 1.0) < 1e-4
        assert data["vector_preview"] is None
        assert data["full_vector"] is None

    def test_embedding_extraction_with_vector_opt_in(self):
        wav_bytes = create_mock_wav_bytes(0.5)
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        resp = client.post("/api/v1/audio/extract-embedding?include_vector=true", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["embedding_dimension"] == 384
        assert data["vector_preview"] is not None
        assert len(data["vector_preview"]) == 16
        assert data["full_vector"] is not None
        assert len(data["full_vector"]) == 384


class TestGateB2SafetyAndGateConstraints:
    """Tests non-negotiable threshold and model availability policies."""

    def test_unmaterialized_model_predict_returns_503(self):
        # Operational predict for unready quantum candidate returns 503 QUANTUM_MODEL_NOT_READY
        resp = client.post(
            "/api/v1/predict",
            json={
                "speaker_id": "F01",
                "model_id": "phase8_1_track_b_quantum",
                "embedding": [0.0] * 384,
            },
        )
        assert resp.status_code == 503
        data = resp.json()
        assert data["detail"]["error_code"] == "QUANTUM_MODEL_NOT_READY"

        # Operational predict requesting classical baseline returns 400 MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE
        resp_class = client.post(
            "/api/v1/predict",
            json={
                "speaker_id": "F01",
                "model_id": "phase8_1_track_a_baseline",
                "embedding": [0.0] * 384,
            },
        )
        assert resp_class.status_code == 400
        assert resp_class.json()["detail"]["error_code"] == "MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE"

    def test_model_info_threshold_provenance_warning(self):
        resp = client.get("/model-info?model_id=continual_classical_adapter&speaker_id=F01")
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision_policy"]["decision_cutoff_tau"] == 0.60
        assert data["decision_policy"]["status"] == "THRESHOLD_PROVENANCE_UNVERIFIED"
        assert data["safety_constraints"]["fer_safety_ceiling"] == 0.1500

    def test_demo_interface_serving(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers["content-type"]
        assert "Atypical Speech Emergency Assistant" in resp.text
