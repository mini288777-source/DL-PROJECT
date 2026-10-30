# Privacy, Security, and Data Lifecycle Policy

## Project: Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication
**Document Reference:** `docs/application/PRIVACY_SECURITY_AND_DATA_LIFECYCLE.md`  
**Classification:** Engineering Security & Data Governance Specification  
**Effective Date:** 2026-10-11  

---

## 1. Executive Summary & Core Principles

Atypical and dysarthric speech samples represent highly sensitive, identifiable biometric and health-related information. Consequently, the research application enforces strict data minimization, zero persistent speech storage, local loopback network binding, and defense-in-depth isolation.

### The Four Pillars of Application Security:
1. **Ephemeral Audio Lifetime:** Audio samples exist solely in volatile memory or transient operating-system temp files during active inference, purged immediately upon completion or error.
2. **Strict Local Loopback Binding:** The backend server binds exclusively to `127.0.0.1`, completely preventing unauthorized local-area-network or public internet exposure.
3. **Zero Outbound Telemetry or Integrations:** The application contains zero external analytics, zero cloud logging, and zero emergency services (CAD/911/SMS) hooks.
4. **Read-Only System State:** Historical research datasets, model checkpoints, and configuration manifests are treated as immutable, read-only artifacts.

---

## 2. Ephemeral Audio Processing Lifecycle

The lifecycle of an audio input within the system follows a deterministic, bounded state machine:

```
[Audio Input] 
      │ (Upload / Live Mic Recording)
      ▼
[Temporary Storage] (System Temp Directory: NamedTemporaryFile)
      │
      ├──► [Format & Header Validation] (WAV/FLAC, ≤10MB, ≤30.0s)
      │         │
      │         ▼ (Validation Failure: 400 Bad Request)
      │         └──► [Immediate Temp File Purge]
      │
      ├──► [In-Memory Feature Extraction] (16kHz Mono -> Whisper-tiny -> 384-D Tensor)
      │
      └──► [Personalized Adapter Inference] (Forward Pass in eval() mode)
                │
                ▼
      [Inference Result Formatted (JSON)]
                │
                ▼ (finally Block Guarantee)
      [Temp File Unlink / Memory GC] (Zero persistence on disk)
```

### 2.1 File Ingestion and Verification
1. Audio uploaded via `POST /api/v1/predict/file` or streamed via WebSocket is buffered into a uniquely generated `NamedTemporaryFile` in the system temporary directory.
2. File headers are verified using `soundfile` and `librosa`. Corrupt headers or unparseable formats immediately raise `AudioProcessingError` (HTTP 400).
3. Audio duration is validated against `MAX_DURATION_SECONDS = 30.0`. Files exceeding this threshold are rejected.
4. Payload size is enforced at the ASGI middleware layer to reject payloads exceeding `10 MB`.

### 2.2 Guaranteed Deletion via `finally` Blocks
Audio files are deleted within explicit `try...finally` constructs in [`src/app/audio_service.py`](file:///d:/DL/src/app/audio_service.py):
```python
temp_path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(file_bytes)
        temp_path = f.name
    # Audio processing, feature extraction, and inference
    waveform, sr = librosa.load(temp_path, sr=16000, mono=True)
    return self.process_waveform(waveform)
finally:
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass
```
This ensures file deletion occurs even during unexpected uncaught exceptions, timeout cancellations, or process termination signals.

### 2.3 Browser-Side Ephemeral Lifetime
1. Microphone recordings in [`src/app/static/app.js`](file:///d:/DL/src/app/static/app.js) are held as in-memory `Blob` objects.
2. Audio preview URLs generated via `URL.createObjectURL(blob)` are explicitly revoked via `URL.revokeObjectURL(url)` whenever the user clicks "Clear Audio", records a new snippet, or navigates away.
3. Neither audio binary blobs, base64 strings, nor transcription texts are ever persisted to `localStorage`, `sessionStorage`, or `IndexedDB`.

---

## 3. Network Architecture & External Isolation

### 3.1 Loopback-Only Binding
The application default configuration in [`.env.example`](file:///d:/DL/.env.example) and [`src/app/config.py`](file:///d:/DL/src/app/config.py) binds exclusively to:
```bash
HOST="127.0.0.1"
PORT=8000
```
- Binding to `0.0.0.0` is strictly prohibited in standard operation.
- Cross-Origin Resource Sharing (CORS) middleware is restricted to `http://127.0.0.1:8000` and `http://localhost:8000`.

### 3.2 Total Ban on External Integrations
The application serves solely as an offline local research interface. It is architecturally insulated from external communication protocols:
- **No Telephony/SMS:** Zero integration with Twilio, AWS SNS, Asterisk, or cellular gateways.
- **No Dispatch/911/CAD:** Zero hooks into Computer Aided Dispatch or public safety answering points.
- **No Healthcare/EHR:** Zero integration with FHIR, HL7, Epic, Cerner, or hospital alert systems.
- **No Outbound Telemetry:** Zero analytics trackers, Google Analytics, Sentry, or third-party telemetry.

---

## 4. STRIDE Threat Model

A comprehensive STRIDE analysis of the research application:

| Threat Category | Potential Attack Vector | Applied Engineering Mitigation | Verification Status |
| :--- | :--- | :--- | :--- |
| **Spoofing** | Attacker impersonates a target dysarthric speaker to manipulate emergency predictions. | Per-speaker adapter routing requires explicit `speaker_id`. Unknown speakers fail with HTTP 404. Synthetic inputs are flagged. | Verified in [`test_speaker_isolation_returns_404_on_unknown`](file:///d:/DL/tests/gate_b2/test_full_scale_application.py#L90) |
| **Tampering** | Malicious alteration of frozen model weights or historical research CSVs. | Models loaded with `weights_only=True`, `eval()`, and `requires_grad=False`. Checksum registry verifies 28/28 historical files before release. | Verified in [`test_model_eval_mode_and_frozen_parameters`](file:///d:/DL/tests/gate_b2/test_full_scale_application.py#L25) |
| **Repudiation** | User claims application triggered false emergency alert. | Application emits structured diagnostic logs with timestamps and request IDs, while strictly omitting raw audio and transcripts. Zero dispatch capability. | Verified by audit log inspection |
| **Information Disclosure** | Leakage of dysarthric speech samples or transcripts in server logs or stack traces. | Logging configuration sanitizes all loggers. Raw audio bytes and transcription strings are never written to log sinks. | Verified by test log reviews |
| **Denial of Service** | Oversized audio upload or unbounded streaming causing CPU exhaustion or OOM. | 10 MB payload cap, 30.0 s audio duration limit, 100-chunk WebSocket frame limit (`MAX_STREAM_CHUNKS = 100`). | Verified in [`test_websocket_chunk_buffer_limit`](file:///d:/DL/tests/gate_b2/test_full_scale_application.py#L210) |
| **Elevation of Privilege** | Arbitrary code execution via pickled checkpoint payloads or path traversal. | `torch.load(..., weights_only=True)` disables unpickling exploits. Path parameters resolved against strict allowlists. | Verified in model loading audit |

---

## 5. Input Validation & Defense-in-Depth

### 5.1 File Path Traversal Defense
Endpoints exposing historical research results or figures resolve files exclusively against relative allowlists rooted in `reports/` and `results/`. Any file path containing `..`, absolute Windows drive indicators (`C:\`), or unlisted paths is rejected immediately with HTTP 403 / 404.

### 5.2 WebSocket Stream Bounds
The real-time streaming WebSocket endpoint (`/ws/audio-stream`) enforces bounded resource consumption:
1. **Chunk Limit:** Maximum 100 audio chunks per streaming session. Exceeding this limit drops the connection with `BUFFER_LIMIT_EXCEEDED`.
2. **Chunk Size Limit:** Individual chunks exceeding 64 KB are rejected.
3. **Session Timeout:** Streaming sessions idle for $> 30$ seconds are automatically closed and cleaned up.
4. **Connection Teardown:** Disconnect events invoke immediate buffer deallocation.

---

## 6. Data Retention and Disposal Runbook

1. **Active Sessions:** Data retained strictly for the duration of the HTTP request or WebSocket connection.
2. **Session Reset:** Clicking `#clearAudioBtn` in the web UI invokes client-side garbage collection, canvas clear, and player reset.
3. **Server Shutdown:** All in-memory model caches and temporary directories are destroyed upon SIGINT / SIGTERM.
4. **Historical Research Artifacts:** Preserved read-only in git / repository storage as permanent scientific records, subject to SHA-256 integrity audits.
