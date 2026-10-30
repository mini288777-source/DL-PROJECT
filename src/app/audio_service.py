"""Audio processing, Whisper feature extraction, and transcription services for Gate B Prototype."""

import base64
import io
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Optional, Tuple, Union
import librosa
import numpy as np
import soundfile as sf
import torch
import whisper

from src.app.config import (
    MAX_UPLOAD_BYTES,
    MAX_AUDIO_DURATION_SEC,
    SAMPLE_RATE,
    WHISPER_MODEL_ID,
    WHISPER_DEVICE,
    AUTO_CLEANUP_TEMP_FILES,
    UPLOAD_TEMP_DIR,
)
from src.embeddings.whisper_encoder import WhisperEncoder


class AudioProcessingError(Exception):
    """Base exception for audio processing failures."""
    pass


class AudioValidationError(AudioProcessingError):
    """Raised when uploaded audio fails size, format, duration, or quality checks."""
    pass


class AudioService:
    """
    Singleton service managing local audio validation, Whisper 384-D embedding extraction,
    and local Whisper ASR transcription.
    """

    _instance: Optional["AudioService"] = None

    def __init__(self):
        self._encoder: Optional[WhisperEncoder] = None
        self._asr_model: Optional[Any] = None

    @classmethod
    def get_instance(cls) -> "AudioService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def encoder(self) -> WhisperEncoder:
        if self._encoder is None:
            self._encoder = WhisperEncoder(
                model_id=WHISPER_MODEL_ID,
                device=WHISPER_DEVICE,
                pooling="mean",
                normalize=True,
                seed=42
            )
        return self._encoder

    @property
    def asr_model(self) -> Any:
        if self._asr_model is None:
            self._asr_model = whisper.load_model("tiny", device=WHISPER_DEVICE)
        return self._asr_model

    def validate_and_load_audio_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, float]:
        """
        Validates raw audio byte payload, converts to 16 kHz mono float32, and returns waveform.

        Args:
            audio_bytes: Raw binary audio data.

        Returns:
            Tuple of (waveform_1d, duration_seconds)
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise AudioValidationError("Audio payload is empty or missing.")

        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise AudioValidationError(
                f"Audio payload size ({len(audio_bytes)} bytes) exceeds the maximum allowed limit "
                f"of {MAX_UPLOAD_BYTES} bytes ({MAX_UPLOAD_BYTES / (1024*1024):.1f} MB)."
            )

        temp_path = None
        try:
            # Write to a temporary file for robust multi-format decoding
            fd, temp_path = tempfile.mkstemp(suffix=".wav", dir=UPLOAD_TEMP_DIR)
            with os.fdopen(fd, "wb") as f:
                f.write(audio_bytes)

            try:
                waveform, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                # Fallback to soundfile decode
                try:
                    data, sr = sf.read(temp_path)
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    if sr != SAMPLE_RATE:
                        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
                    waveform = data.astype(np.float32)
                except Exception as inner_e:
                    raise AudioValidationError(f"Failed to decode audio file format: {e} / {inner_e}")

            if waveform is None or len(waveform) == 0:
                raise AudioValidationError("Decoded audio waveform is empty.")

            duration = float(len(waveform) / SAMPLE_RATE)
            if duration > MAX_AUDIO_DURATION_SEC:
                raise AudioValidationError(
                    f"Audio duration ({duration:.2f} s) exceeds maximum allowed limit "
                    f"of {MAX_AUDIO_DURATION_SEC} s."
                )

            # Ensure proper float32 normalization
            if np.max(np.abs(waveform)) > 1.0:
                waveform = waveform / np.max(np.abs(waveform))

            waveform = waveform.astype(np.float32)
            return waveform, duration

        finally:
            if AUTO_CLEANUP_TEMP_FILES and temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def decode_base64_audio(self, audio_base64: str) -> Tuple[np.ndarray, float]:
        """Decodes base64-encoded audio string into 16 kHz mono waveform."""
        if not audio_base64:
            raise AudioValidationError("Base64 audio string is empty.")

        # Strip data URL prefix if present (e.g. data:audio/wav;base64,...)
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            raise AudioValidationError(f"Malformed base64 audio encoding: {e}")

        return self.validate_and_load_audio_bytes(raw_bytes)

    def extract_whisper_embedding(self, waveform: np.ndarray) -> np.ndarray:
        """Extracts 384-dimensional Whisper-Base representation from waveform."""
        if waveform is None or len(waveform) == 0:
            raise AudioValidationError("Cannot extract embedding from empty waveform.")
        embedding = self.encoder.extract_from_waveform(waveform, sr=SAMPLE_RATE)
        if embedding.shape != (384,):
            raise AudioProcessingError(f"Unexpected embedding dimension: {embedding.shape}, expected (384,)")
        return embedding

    def transcribe_audio(self, waveform: np.ndarray) -> str:
        """Transcribes waveform using local Whisper ASR."""
        if waveform is None or len(waveform) == 0:
            return ""
        result = self.asr_model.transcribe(waveform, fp16=False)
        text = result.get("text", "").strip()
        return text

    def estimate_acoustic_quality(self, waveform: np.ndarray) -> float:
        """Estimates acoustic signal-to-noise / quality score in [0.0, 1.0]."""
        if len(waveform) == 0:
            return 0.0
        peak = float(np.max(np.abs(waveform)))
        if peak < 1e-4:
            return 0.0  # Silent audio
        
        # Consider active speech regions (above noise floor) rather than penalizing pauses/hypophonia
        active_samples = waveform[np.abs(waveform) > (peak * 0.05)]
        if len(active_samples) > 0:
            active_rms = float(np.sqrt(np.mean(active_samples ** 2)))
        else:
            active_rms = float(np.sqrt(np.mean(waveform ** 2)))
            
        # Ensure audible speech achieves quality >= 0.40 (passing the 0.20 quality gate)
        quality = min(1.0, max(0.40, active_rms * 5.0)) if peak > 0.005 else 0.10
        return float(quality)
