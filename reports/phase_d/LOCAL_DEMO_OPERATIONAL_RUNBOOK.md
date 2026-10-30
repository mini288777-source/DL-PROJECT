# Gate D: Local Research Demonstration Operational Runbook

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Document Identifier:** RUNBOOK-PHASE-D-2026-V1  
**Operational Scope:** **LOCAL WORKSTATION RESEARCH DEMONSTRATION ONLY (AIR-GAPPED / NO LIVE DISPATCH)**  
**Target Environment:** Local Developer / Investigator Workstation (`127.0.0.1:8000`)  
**Audit Date:** 2026-10-10  
**Lead Authors:** Systems Reliability Lead, Security Engineer, Clinical ML Safety Auditor  

---

> [!CAUTION]
> **CRITICAL OPERATOR SAFETY NOTICE:**  
> This software is an engineering research prototype. It is **NOT CONNECTED TO 911, EMERGENCY SERVICES, OR HOSPITAL DISPATCH**. It misses life-threatening emergencies (>80% missed in outer-fold evaluation) and triggers frequent false alarms. **DO NOT RELY ON THIS APPLICATION FOR MEDICAL OR EMERGENCY PURPOSES.** Operators and caregivers must maintain standard physical call buttons or telephone access at all times.

---

## 1. System Architecture & Network Isolation Boundaries

The demonstration software operates strictly within an isolated local host boundary:
- **Binding Address:** Loopback interface only (`127.0.0.1` / `localhost`). Non-local IP binding (`0.0.0.0`) is prohibited in demo configuration.
- **Port:** Default `8000` (configurable via `APP_PORT` environment variable).
- **External Network Sockets:** **ZERO (0).** The application makes no outbound HTTP, WebSocket, SMS, or telephony calls.
- **Data Persistence:** **RAM ONLY.** Uploaded audio files are processed in-memory and automatically purged from temporary buffers upon inference completion (`AUTO_CLEANUP_TEMP_FILES = True`).

---

## 2. Safe Startup & Health Readiness Verification

### 2.1 Starting the Demonstration Daemon
To start the local research demonstration server securely:
```powershell
# From the project root directory (d:\DL)
$env:APP_HOST = "127.0.0.1"
$env:APP_PORT = "8000"
$env:APP_DEBUG = "false"

# Launch daemon using the designated Python environment
& "C:\Users\lucky\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --log-level info
```

### 2.2 System Health & Readiness Verification
Verify operational health via the dedicated REST endpoints:
```powershell
# Check basic process liveness
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"

# Expected JSON Output:
# {"status": "healthy", "service_name": "Gate B Engineering Prototype", "version": "1.0.0-gate-b", ...}

# Check model inventory readiness
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ready"

# Expected JSON Output:
# {"status": "ready", "materialized_checkpoints": 61, "unmaterialized_checkpoints": 6, ...}
```

---

## 3. Expected Fail-Closed Behaviors

The system implements strict fail-closed responses to prevent fabricated predictions or ungrounded outputs:

| Scenario | Trigger / User Action | System Response | Operator Remediation |
| :--- | :--- | :--- | :--- |
| **Unmaterialized Model Requested** | Selecting Phase 8.1 or Phase 8.2 candidate model in dropdown | **HTTP 503 `MODEL_NOT_READY`** | Expected behavior. Candidate weights are unmaterialized. Switch back to `continual_classical_adapter`. |
| **Unknown Speaker ID** | Requesting an unenrolled speaker identifier | **HTTP 404 `MODEL_NOT_FOUND`** | Select an enrolled speaker (e.g., `F01`, `M01`, `MC01`) from `/api/v1/speakers`. |
| **Oversized Audio File** | Uploading file $> 10$ MB | **HTTP 400 `AUDIO_VALIDATION_ERROR`** | Ensure recording is under 10 MB and under 30 seconds in duration. |
| **Severe Acoustic Noise** | Audio SNR $< 10$ dB or silent input | **HTTP 400 / Low SNR Warning** | Adjust microphone positioning (3–5 cm from mouth, 45-degree angle). |

---

## 4. Audit Logging & Privacy Hygiene

1. **Log Location:** Standard application logs are directed to stdout / console. No raw audio waveforms or transcription text are written to persistent system logs.
2. **Sanitization:** Log messages contain only high-level inference timestamps, speaker IDs, inferred intent classes, uncalibrated scores, and processing latencies.
3. **Log Rotation:** If forwarding console logs to file, configure local log rotation capping log files at 50 MB with 3-day retention.

---

## 5. Emergency Incident Containment & Disabling the Application

If an operator observes unexpected behavior, UI misinterpretation, or caregiver distress:

### Immediate Application Shutdown:
```powershell
# Terminate the uvicorn server process immediately
Get-Process python | Where-Object { $_.CommandLine -like "*uvicorn*" } | Stop-Process -Force
```

### Clean Up Temporary Cache:
```powershell
# Securely purge any temporary audio cache files
Remove-Item -Path "d:\DL\cache\app_uploads\*" -Force -Recurse -ErrorAction SilentlyContinue
```

### Re-Verification of Host Boundaries:
```powershell
# Verify port 8000 is clean and closed
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
```
