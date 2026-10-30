# Full-Scale Research Application — API Reference Manual

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Document ID:** `DOC-APP-API-001`  
**Base URL:** `http://127.0.0.1:8000`  
**Protocol Version:** `v1`  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. System Health & Metadata Endpoints

### 1.1 Process Health Check
- **Route:** `GET /health`
- **Description:** Verifies that the FastAPI process is running and responding to local HTTP requests.
- **Response Code:** `200 OK`
- **Response Schema:**
```json
{
  "status": "healthy",
  "service_name": "Atypical Speech Emergency Assistant — Gate B Engineering Prototype",
  "version": "1.0.0-gate-b",
  "timestamp_utc": "2026-10-10T19:00:00Z",
  "disclaimer": "This application is an experimental research prototype..."
}
```

### 1.2 Subsystem Readiness & Blocker Audit
- **Route:** `GET /ready`
- **Description:** Assesses the operational readiness of speech components, model checkpoints, and active gate blockers.
- **Response Code:** `200 OK`
- **Response Schema:**
```json
{
  "ready": true,
  "whisper_encoder_ready": true,
  "whisper_asr_ready": true,
  "checkpoints_summary": {
    "materialized_and_verified": 61,
    "model_not_ready": 6,
    "total_catalogued": 67
  },
  "active_default_speaker": "F01",
  "active_default_model": "continual_classical_adapter",
  "unmaterialized_blockers": [
    "phase8_1_track_a_baseline (checkpoints/phase8_1/track_a_fold_models.pt)",
    "phase8_1_int1_calib (checkpoints/phase8_1/int1_calib_models.pt)",
    "phase8_1_int2_cost (checkpoints/phase8_1/int2_cost_models.pt)",
    "phase8_1_int3_samp (checkpoints/phase8_1/int3_samp_models.pt)",
    "phase8_1_track_b_quantum (checkpoints/phase8_1/track_b_fold_models.pt)",
    "phase8_1_track_c_shadow (checkpoints/phase8_1/track_c_fold_models.pt)"
  ],
  "scientific_verdict": "Research Findings Status: CASE B AFFIRMED...",
  "disclaimer": "This application is an experimental research prototype...",
  "timestamp_utc": "2026-10-10T19:00:00Z"
}
```

### 1.3 Detailed Model Architecture & Policy Info
- **Route:** `GET /model-info`
- **Query Parameters:**
  - `model_id` (string, optional, default: `"qsvc_q8_r1"`)
  - `speaker_id` (string, optional, default: `"F01"`)
- **Response Code:** `200 OK`
- **Quantum Model Response Schema (`model_id=qsvc_q8_r1`):**
```json
{
  "model_id": "qsvc_q8_r1",
  "model_name": "Fidelity Quantum Kernel QSVC (8-Qubit ZZFeatureMap) (F01)",
  "speaker_id": "F01",
  "operational_role": "PRIMARY_OPERATIONAL_PREDICTOR",
  "qubit_count": 8,
  "feature_map": "ZZFeatureMap",
  "entanglement": "linear",
  "reps": 1,
  "kernel_type": "quantum_fidelity_kernel",
  "classifier": "sklearn.svm.SVC",
  "regularization_C": 1.0,
  "backend": "statevector_simulator (Aer / Pennylane / Qiskit)",
  "input_shape": [384],
  "pca_dimension": 8,
  "pca_feature_range": [0.0, 3.141592653589793],
  "status": "QUANTUM_MODEL_NOT_READY",
  "readiness_reason": "Serialized checkpoint contains fitted SVC and PCA pipeline but is missing reference adaptation embeddings (X_train_pca) required for precomputed fidelity kernel evaluation."
}
```

---

## 2. Speech & Audio Processing Endpoints

### 2.1 File Transcription (Offline Whisper-Tiny)
- **Route:** `POST /api/v1/audio/transcribe`
- **Content-Type:** `multipart/form-data`
- **Form Fields:** `file` (binary audio file, max 10 MB, $\le 30$ seconds).
- **Response Code:** `200 OK`
- **Response Schema:**
```json
{
  "text": "help me I cannot get up",
  "language": "en",
  "audio_duration_seconds": 2.45,
  "acoustic_quality": 0.8521,
  "timestamp_utc": "2026-10-10T19:00:00Z",
  "disclaimer": "This application is an experimental research prototype..."
}
```

### 2.2 Feature Extraction (384-Dimensional Representation)
- **Route:** `POST /api/v1/audio/extract-embedding`
- **Content-Type:** `multipart/form-data`
- **Query Parameters:** `include_vector` (boolean, default: `false`)
- **Form Fields:** `file` (binary audio file).
- **Response Code:** `200 OK`
- **Response Schema:**
```json
{
  "embedding_dimension": 384,
  "l2_norm": 1.000000,
  "audio_duration_seconds": 2.45,
  "vector_preview": [0.0412, -0.0152, 0.0891, "..."],
  "full_vector": null,
  "timestamp_utc": "2026-10-10T19:00:00Z"
}
```

---

## 3. Forward Inference Endpoints (Quantum-Only)

### 3.1 Audio File Quantum Intent Prediction Pipeline
- **Route:** `POST /api/v1/predict/file`
- **Content-Type:** `multipart/form-data`
- **Form Fields:**
  - `file`: Binary audio file
  - `speaker_id`: Target speaker ID (e.g. `"F01"`)
  - `model_id`: Operational model (defaults to `"qsvc_q8_r1"`)
  - `operating_tau`: Optional cutoff (float, default: `0.60`)
- **Response Codes:**
  - `400 BAD REQUEST`: If a classical model ID is specified (`MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE`)
  - `404 NOT FOUND`: Unknown speaker ID (`SPEAKER_NOT_FOUND`)
  - `422 UNPROCESSABLE ENTITY`: Malformed or unprocessable audio payload
  - `503 SERVICE UNAVAILABLE`: If quantum checkpoint is missing reference support vectors (`QUANTUM_MODEL_NOT_READY`)
- **Error Response Schema (503 QUANTUM_MODEL_NOT_READY):**
```json
{
  "detail": {
    "error_code": "QUANTUM_MODEL_NOT_READY",
    "message": "Quantum model candidate 'qsvc_q8_r1' for speaker 'F01' cannot execute live inference: missing reference adaptation embeddings (X_train_pca) required for precomputed fidelity kernel evaluation.",
    "model_id": "qsvc_q8_r1",
    "speaker_id": "F01",
    "readiness_status": "QUANTUM_MODEL_NOT_READY",
    "remediation": "Re-exporting model checkpoint with reference support vectors requires explicit dataset owner authorization. Retraining or weight fabrication is prohibited under the immutable checkpoint policy."
  }
}
```
- **Response Schema (200 OK):**
```json
{
  "session_id": "a1b2c3d4",
  "speaker_id": "F01",
  "model_id": "continual_classical_adapter",
  "transcription": "help me I cannot get up",
  "predicted_intent": "FALL",
  "is_emergency": true,
  "confidence_score": 0.9124,
  "confidence_band": "HIGH",
  "suggested_action": "CONFIRM_AND_DISPATCH",
  "class_probabilities": {
    "GENERAL_SPEECH": 0.0211,
    "CRITICAL_RESPONSE": 0.0125,
    "FALL": 0.9124,
    "BREATHING_MEDICAL": 0.0142,
    "PAIN": 0.0183,
    "SOS_HELP": 0.0154,
    "LOCATION": 0.0061
  },
  "decision_cutoff_tau": 0.60,
  "model_version": "vF01_b50_replay_seed2209",
  "processing_latency_ms": 14.2,
  "timestamp_utc": "2026-10-10T19:00:00Z"
}
```

### 3.2 Unmaterialized Model Error Contract (503 Service Unavailable)
- **Response Schema (503):**
```json
{
  "error_code": "MODEL_NOT_READY",
  "model_id": "phase8_1_track_a_baseline",
  "message": "Candidate model 'phase8_1_track_a_baseline' cannot be loaded because its checkpoint ('checkpoints/phase8_1/track_a_fold_models.pt') is unmaterialized on disk...",
  "remediation": "Use the verified 'continual_classical_adapter' model for demonstration, or obtain explicit owner authorization to materialize candidate checkpoints under an authorized development protocol.",
  "disclaimer": "This application is an experimental research prototype...",
  "timestamp_utc": "2026-10-10T19:00:00Z"
}
```

---

## 4. WebSocket Streaming Inference (`/ws/inference`)

- **Protocol:** JSON message frames over WebSocket.
- **Message Types:**
  1. `{"type": "init", "speaker_id": "F01", "model_id": "continual_classical_adapter", "operating_tau": 0.60}`
  2. `{"type": "chunk", "data": "<base64_audio_chunk>"}`
  3. `{"type": "finalize"}`
  4. `{"type": "reset"}`
- **Server Responses:**
  - `{"type": "connection_established", "session_id": "...", "status": "ready"}`
  - `{"type": "initialized", "session_id": "...", "speaker_id": "F01"}`
  - `{"type": "chunk_received", "bytes_buffered": 16000, "estimated_duration_sec": 0.5}`
  - `{"type": "final_result", "session_id": "...", "transcription": "...", "prediction": {...}}`
  - `{"type": "buffer_reset", "status": "ready"}`
  - `{"type": "error", "error_code": "...", "message": "..."}`
