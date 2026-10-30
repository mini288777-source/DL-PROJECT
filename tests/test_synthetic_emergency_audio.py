"""Integration test suite for the 10 Synthetic Emergency Atypical Audio Benchmark Samples.

Verifies:
1. Presence of all 10 synthetic WAV files and manifest.
2. 16 kHz mono format compliance and valid durations.
3. AudioService feature extraction and transcription.
4. QuantumModelManager prediction without quality-gate aborts.
5. FastAPI /samples static route availability.
"""

from pathlib import Path
import wave
import pytest
from fastapi.testclient import TestClient

from src.app.audio_service import AudioService
from src.app.main import app
from src.app.quantum_model_manager import QuantumModelManager

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "data" / "synthetic_emergency_audio"
client = TestClient(app)

class TestSyntheticEmergencyAudio:
    """Verifies synthetic emergency audio benchmark set integrity."""

    def test_samples_directory_and_manifest_exist(self):
        assert SAMPLES_DIR.exists()
        manifest_path = SAMPLES_DIR / "synthetic_audio_manifest.json"
        assert manifest_path.exists()
        readme_path = SAMPLES_DIR / "README.md"
        assert readme_path.exists()

    def test_exactly_ten_wav_files_present(self):
        wav_files = sorted(list(SAMPLES_DIR.glob("*.wav")))
        assert len(wav_files) == 10, f"Expected 10 WAV files, found {len(wav_files)}"

    def test_wav_format_specification(self):
        wav_files = sorted(list(SAMPLES_DIR.glob("*.wav")))
        for wf in wav_files:
            with wave.open(str(wf), "rb") as w:
                assert w.getframerate() == 16000, f"{wf.name}: Expected 16 kHz, got {w.getframerate()}"
                assert w.getnchannels() == 1, f"{wf.name}: Expected mono, got {w.getnchannels()} channels"
                assert w.getsampwidth() == 2, f"{wf.name}: Expected 16-bit PCM, got {w.getsampwidth()} bytes"
                duration = w.getnframes() / w.getframerate()
                assert 2.0 <= duration <= 15.0, f"{wf.name}: Duration {duration}s outside expected range"

    def test_end_to_end_audio_pipeline_execution(self):
        audio_service = AudioService.get_instance()
        quantum_mgr = QuantumModelManager.get_instance()
        wav_files = sorted(list(SAMPLES_DIR.glob("*.wav")))

        for wf in wav_files:
            content = wf.read_bytes()
            waveform, duration = audio_service.validate_and_load_audio_bytes(content)
            quality = audio_service.estimate_acoustic_quality(waveform)
            assert quality >= 0.35, f"{wf.name}: Quality {quality} should exceed 0.35"

            emb = audio_service.extract_whisper_embedding(waveform)
            assert emb.shape == (384,)

            res = quantum_mgr.predict(
                embedding=emb,
                speaker_id="F01",
                model_id="qsvc_q8_r1",
                operating_tau=0.60,
                acoustic_quality=quality,
            )
            assert res.get("predicted_intent") != "UNKNOWN", f"{wf.name} was aborted with UNKNOWN"
            assert res.get("confidence_score") > 0.0, f"{wf.name} has zero confidence"
            assert "class_probabilities" in res

    def test_fastapi_samples_route_serves_files(self):
        resp = client.get("/samples/01_emergency_sos_help_atypical.wav")
        assert resp.status_code == 200
        assert resp.headers.get("content-type") in ("audio/wav", "audio/x-wav")
