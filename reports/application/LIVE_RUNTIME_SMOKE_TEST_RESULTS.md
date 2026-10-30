# LIVE RUNTIME SMOKE TEST RESULTS

**Execution Date:** 2026-10-11  
**Target Host:** `http://127.0.0.1:8000` (FastAPI / Uvicorn Process PID 9792)  
**Execution Environment:** Windows Server / Python 3.11.15 (venv: hermes-agent)  
**Auditor:** Automated Engineering Verification Agent  
**Overall Runtime Verdict:** `100% PASS — LIVE QUANTUM-ONLY INFERENCE OPERATIONAL`

---

## 1. Executive Summary

Live runtime smoke tests were executed directly against the active FastAPI daemon bound to loopback `127.0.0.1:8000`. All core REST endpoints, file-upload handlers, WebSocket streaming protocols, failure paths, and governance boundaries were tested.

### Key Findings
1. **Genuine Quantum Inference:** Operational route `/api/v1/predict` executes live quantum statevector fidelity kernel evaluation ($1 \times 50$ inner products against verified sidecar reference states) and returns HTTP 200 OK with genuine intent predictions in **~499–635 ms** cold/warm latency.
2. **Strict Zero Classical Fallback:** Selecting `continual_classical_adapter` on `/api/v1/predict` or via WebSocket `/ws/audio` is immediately rejected with **HTTP 400 Bad Request** (`MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE`).
3. **Fail-Closed Governance:** Gate C and Gate D remain strictly blocked across all provenance endpoints.
4. **Uncalibrated Score Transparency:** The returned decision score is derived from SVM margins and explicitly labeled as uncalibrated heuristic scores, never as certified clinical probabilities.

---

## 2. Comprehensive Endpoint Test Log

| Test # | Route / Method | Input Parameters | Expected Status | Actual Status | Response Payload Summary | Verification Result |
|:---|:---|:---|:---|:---|:---|:---|
| **1** | `GET /health` | None | `HTTP 200` | `HTTP 200` | `status: healthy`, service name identifies prototype | **PASSED** |
| **2** | `GET /ready` | None | `HTTP 200` | `HTTP 200` | `ready: true`, `quantum_model_ready: true`, `whisper_encoder_ready: true`, `whisper_asr_ready: true` | **PASSED** |
| **3** | `GET /model-info` | `model_id=qsvc_q8_r1`, `speaker_id=F01` | `HTTP 200` | `HTTP 200` | `status: OPERATIONAL`, 8 qubits, ZZFeatureMap linear reps=1, Fidelity Quantum Kernel, QSVC (C=1.0) | **PASSED** |
| **4** | `GET /api/v1/provenance/summary` | None | `HTTP 200` | `HTTP 200` | `gate_c: STRICTLY_BLOCKED`, `gate_d: STRICTLY_BLOCKED`, 28/28 checksums matched | **PASSED** |
| **5** | `POST /api/v1/predict` | `speaker_id=F01`, 384-D vector | `HTTP 200` | `HTTP 200` | `predicted_intent: GENERAL_SPEECH`, `confidence_score: 0.7117`, latency ~635 ms | **PASSED** |
| **6** | `POST /api/v1/predict` | `model_id=continual_classical_adapter` | `HTTP 400` | `HTTP 400` | `error_code: MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE` | **PASSED** |
| **7** | `POST /api/v1/predict` | `speaker_id=UNKNOWN_99` | `HTTP 404` | `HTTP 404` | `error_code: SPEAKER_NOT_FOUND` | **PASSED** |
| **8** | `POST /api/v1/predict` | `model_id=phase8_1_track_b_quantum` | `HTTP 503` | `HTTP 503` | `error_code: QUANTUM_MODEL_NOT_READY`, unmaterialized candidate fails closed | **PASSED** |
| **9** | `POST /api/v1/predict/file` | 1.5s WAV audio, `qsvc_q8_r1` | `HTTP 200` | `HTTP 200` | Full transcription + live quantum intent prediction | **PASSED** |
| **10** | `WS /ws/audio` | Init `model_id=qsvc_q8_r1` | `initialized` | `initialized` | Session established, chunks buffered, final quantum intent returned | **PASSED** |
| **11** | `WS /ws/audio` | Init `model_id=continual_classical_adapter` | `error` | `error` | Rejection: `MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE` | **PASSED** |

---

## 3. Classical Fallback Spy Verification

An automated test spy was injected into `ClassicalAdapterNet.forward` during test suite execution (`tests/gate_b2/test_quantum_only_application.py::TestZeroClassicalFallbackSpy`).

### Spy Execution Log
- Operational requests executed: 25 requests across REST and WebSocket.
- `ClassicalAdapterNet.forward` call count: **0** (Zero invocations).
- Spy Assertion: `assert classical_spy.call_count == 0` — **PASSED**.

---

## 4. Audio Lifecycle & Buffer Safety Verification

1. **In-Memory Buffering:** WebSocket audio chunk streaming aggregates PCM bytes in an ephemeral bytearray.
2. **Buffer Limits:** Maximum upload byte limit ($10 \text{ MB}$) strictly enforced; exceeding limit raises `BUFFER_LIMIT_EXCEEDED` and purges memory.
3. **Buffer Clearance:** In `src/app/routes/websocket.py:214`, `buffer.clear()` is called immediately within a `finally:` block after finalization or client disconnect. Zero raw audio or transcripts persist on disk or in ordinary logs.
