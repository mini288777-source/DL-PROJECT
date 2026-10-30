# Local Operations Runbook

## Project: Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication
**Document Reference:** `docs/application/LOCAL_OPERATIONS_RUNBOOK.md`  
**Classification:** Standard Operating Procedure (SOP) & Engineering Runbook  
**Effective Date:** 2026-10-11  

---

## 1. Quick Start Guide (One-Command Reference)

### 1.1 Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

### 1.2 Start the Research Application
```powershell
# Launch FastAPI backend with Uvicorn on local loopback
uvicorn src.app.main:app --host 127.0.0.1 --port 8000
```
*Application URL:* Open your web browser to `http://127.0.0.1:8000`

### 1.3 Run Automated Regression Tests
```powershell
# Run the entire repository test suite (629 tests)
python -m pytest tests/

# Run the focused full-scale application test suite (18 tests)
python -m pytest tests/gate_b2/test_full_scale_application.py -v
```

---

## 2. Configuration & Environment Setup

### 2.1 Configuration File (`.env`)
Copy the provided [`.env.example`](file:///d:/DL/.env.example) to `.env` in the project root:
```powershell
Copy-Item .env.example .env
```

Key environment variables:
| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `APP_HOST` | `127.0.0.1` | Local loopback interface binding. Do not change to `0.0.0.0`. |
| `APP_PORT` | `8000` | Local port for FastAPI web server. |
| `DEFAULT_THRESHOLD` | `0.6000` | Uncalibrated demonstration default cutoff ($\tau$). |
| `LOG_LEVEL` | `INFO` | Structured logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum file upload size in megabytes. |
| `MAX_AUDIO_DURATION_SEC`| `30.0` | Maximum audio duration in seconds. |

---

## 3. Operational Health & Readiness Verification

### 3.1 Health Check (Process Liveness)
Validates that the ASGI event loop is active and responding:
```powershell
curl.exe http://127.0.0.1:8000/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "service": "atypical-speech-research-app",
  "version": "1.0.0"
}
```

### 3.2 Readiness Check (Model & Dependency Availability)
Validates that PyTorch, Whisper-tiny, and materialized speaker checkpoints are loaded and operational:
```powershell
curl.exe http://127.0.0.1:8000/ready
```
**Expected Response:**
```json
{
  "status": "ready",
  "whisper_loaded": true,
  "default_speaker": "F01",
  "model_manager": "ready",
  "verified_speakers_count": 15
}
```

---

## 4. Troubleshooting Guide

### 4.1 Port 8000 Already in Use
**Symptom:** `ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`  
**Resolution:**
```powershell
# Identify process holding port 8000
netstat -ano | findstr :8000

# Terminate process by PID (replace <PID>)
Stop-Process -Id <PID> -Force

# Or run application on alternative local port
uvicorn src.app.main:app --host 127.0.0.1 --port 8080
```

### 4.2 Microphone Permission Blocked in Browser
**Symptom:** UI displays error: `Microphone access denied or audio recording failed.`  
**Resolution:**
- Modern browsers restrict `navigator.mediaDevices.getUserMedia()` to secure contexts (`https://` or `http://localhost`, `http://127.0.0.1`).
- In Chrome / Edge: Click the "Lock" / "Site settings" icon in the URL bar and ensure **Microphone** is toggled to **Allow**.
- Ensure Windows privacy settings allow desktop apps to access the microphone (*Settings ➔ Privacy & Security ➔ Microphone*).

### 4.3 `503 MODEL_NOT_READY` for Candidate Models
**Symptom:** Inference request with `model_id="phase8_1_quantum"` returns HTTP 503.  
**Explanation:** This is **by design and expected**. The six Phase 8.1 / Phase 8.2 candidate models are unmaterialized research specifications. Under the immutable model governance policy, the backend strictly fails closed with HTTP 503 rather than fabricating weights.  
**Resolution:** Use verified materialized models such as `phase7_5_classical` with valid speakers (`F01`, `M01`, `FC01`, etc.).

### 4.4 Soundfile / PySoundFile Architecture Warnings
**Symptom:** `UserWarning: PySoundFile failed. Trying audioread instead.`  
**Explanation:** On some Windows platforms, 24-bit PCM or specific MP3 headers may fall back to `audioread` without affecting transcription or embedding accuracy.  
**Resolution:** Standard 16 kHz 16-bit mono WAV files avoid this warning completely.

---

## 5. Graceful Teardown & Cleanup Procedures

### 5.1 Stopping the Server
Press `Ctrl + C` in the PowerShell terminal running Uvicorn.
- The server initiates graceful shutdown:
  1. Stops accepting new HTTP and WebSocket connections.
  2. Completes pending in-flight inference requests.
  3. Disconnects active WebSockets and deallocates streaming buffers.
  4. Releases PyTorch model instances from RAM.

### 5.2 Ephemeral File Disposal
The server automatically deletes all temporary audio files during runtime via `try...finally` unlinking. If an unexpected hard crash occurs (e.g., power loss), Windows operating-system temp files can be cleared:
```powershell
Remove-Item -Path "$env:TEMP\tmp*.wav" -Force -ErrorAction SilentlyContinue
```
