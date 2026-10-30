"""Verifies that all 10 synthetic emergency audio samples pass through AudioService and QuantumModelManager."""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.app.audio_service import AudioService
from src.app.quantum_model_manager import QuantumModelManager

def main():
    audio_service = AudioService.get_instance()
    quantum_mgr = QuantumModelManager.get_instance()

    samples_dir = Path("data/synthetic_emergency_audio")
    wav_files = sorted(list(samples_dir.glob("*.wav")))

    print("=" * 105)
    print(f"VERIFYING {len(wav_files)} SYNTHETIC EMERGENCY ATYPICAL AUDIO SAMPLES")
    print("=" * 105)

    all_passed = True
    for idx, wf in enumerate(wav_files, start=1):
        content = wf.read_bytes()
        waveform, duration = audio_service.validate_and_load_audio_bytes(content)
        quality = audio_service.estimate_acoustic_quality(waveform)
        text = audio_service.transcribe_audio(waveform)
        emb = audio_service.extract_whisper_embedding(waveform)
        res = quantum_mgr.predict(
            embedding=emb,
            speaker_id="F01",
            model_id="qsvc_q8_r1",
            operating_tau=0.60,
            acoustic_quality=quality
        )
        intent = res.get("predicted_intent")
        conf = res.get("confidence_score", 0.0)
        action = res.get("suggested_action")
        latency = res.get("processing_latency_ms", 0.0)

        # Check that it did NOT abort with UNKNOWN (quality gate failure)
        is_aborted = (intent == "UNKNOWN" and conf == 0.0)
        if is_aborted:
            all_passed = False
            status_str = "FAILED (ABORTED)"
        else:
            status_str = "SUCCESS"

        print(f"[{idx:02d}/10] {wf.name:<42} | Quality: {quality:.2f} | Intent: {intent:<17} | Conf: {conf*100:5.1f}% | Lat: {latency:5.1f}ms | {status_str}")
        print(f"     Transcript: \"{text}\"")

    print("=" * 105)
    if all_passed:
        print("ALL 10 SAMPLES PASSED INFERENCE VERIFICATION WITH ZERO ABORTS OR GATE REJECTIONS!")
    else:
        print("SOME SAMPLES ENCOUNTERED ABORTS.")

if __name__ == "__main__":
    main()
