"""Configuration settings and environment variable definitions for the Gate B Prototype."""

import os
from pathlib import Path
from typing import List

# Base repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Server Network Settings
APP_HOST: str = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT: int = int(os.environ.get("APP_PORT", "8000"))
DEBUG: bool = os.environ.get("APP_DEBUG", "false").lower() in ("true", "1", "yes")

# Audio Upload Limits & Controls
MAX_UPLOAD_BYTES: int = int(os.environ.get("APP_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB limit
MAX_AUDIO_DURATION_SEC: float = float(os.environ.get("APP_MAX_AUDIO_DURATION", "30.0"))  # 30 seconds
SAMPLE_RATE: int = 16000  # Canonical 16 kHz mono
ALLOWED_AUDIO_EXTENSIONS: List[str] = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]

# Storage & Privacy
UPLOAD_TEMP_DIR: Path = REPO_ROOT / "cache" / "app_uploads"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
AUTO_CLEANUP_TEMP_FILES: bool = True  # Always remove uploaded raw audio after processing

# Local Models & Feature Extraction
WHISPER_MODEL_ID: str = os.environ.get("APP_WHISPER_MODEL", "openai/whisper-tiny")
WHISPER_DEVICE: str = os.environ.get("APP_DEVICE", "cpu")
WHISPER_EMBEDDING_DIM: int = 384

# Canonical Operational Quantum Model Identifier
CANONICAL_QUANTUM_MODEL_ID: str = "qsvc_q8_r1"
CANONICAL_QUANTUM_MODEL_NAME: str = "Fidelity Quantum Kernel QSVC (8-Qubit ZZFeatureMap)"

# Default Operational Setup (Operational prediction restricted to quantum model)
DEFAULT_SPEAKER_ID: str = "F01"
DEFAULT_SPEAKER: str = "F01"
DEFAULT_MODEL_FAMILY: str = "quantum_kernel_qsvc"
DEFAULT_MODEL_ID: str = CANONICAL_QUANTUM_MODEL_ID

# Decision Threshold & Safety Policy (Decoupled from FER Bound)
DEFAULT_DECISION_TAU: float = 0.6000  # Demonstration default; THRESHOLD_PROVENANCE_UNVERIFIED for safety claims
FER_SAFETY_BOUND: float = 0.1500  # Internal research constraint (FER <= 15.0%); NOT a clinically accepted standard
FER_SAFETY_BOUND_PERCENT: float = 15.0  # Equivalent percent (internal constraint)

# Canonical Taxonomy (7 Classes)
CANONICAL_CLASSES: List[str] = [
    "GENERAL_SPEECH",
    "CRITICAL_RESPONSE",
    "FALL",
    "BREATHING_MEDICAL",
    "PAIN",
    "SOS_HELP",
    "LOCATION"
]

EMERGENCY_CLASSES: List[str] = [
    "CRITICAL_RESPONSE",
    "FALL",
    "BREATHING_MEDICAL",
    "PAIN",
    "SOS_HELP",
    "LOCATION"
]

GENERAL_SPEECH_CLASS: str = "GENERAL_SPEECH"

# Mandatory Research Prototype Disclaimer
DISCLAIMER_NOTICE: str = (
    "This application is an experimental research prototype. It is not validated for "
    "clinical use or real-world emergency response. Predictions can be incorrect, "
    "including failure to detect emergencies (False Negatives) and false emergency alerts "
    "(False Positives). Do not rely on this application for emergency decisions."
)

# Scientific Verdict Notice
SCIENTIFIC_VERDICT_NOTICE: str = (
    "Research Findings Status: CASE B AFFIRMED — No statistically supported quantum "
    "advantage was demonstrated over matched classical models under evaluated conditions. "
    "Emergency deployment status: SAFETY-INFEASIBLE due to severe class imbalance "
    "and dysarthric acoustic distortion."
)

# CORS Policy (Localhost only)
CORS_ORIGINS: List[str] = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]
