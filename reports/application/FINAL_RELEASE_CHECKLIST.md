# FINAL RELEASE CANDIDATE CHECKLIST

**Release Identifier:** `qsvc-q8-r1-research-demo-rc1`  
**Date:** 2026-10-11  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Target Environment:** Local Python 3.11 Runtime / Windows Workspace  
**Status:** `READY FOR LOCAL RESEARCH DEMONSTRATION`

---

## 1. Pre-Flight Setup & Environment Verification

| Step | Action Item | Verification Command | Expected Output | Status |
|:---:|:---|:---|:---|:---:|
| **1.1** | Python Interpreter | `python --version` | `Python 3.11.15` | **VERIFIED** |
| **1.2** | Core Dependencies | `pip list` | `fastapi`, `uvicorn`, `qiskit`, `scikit-learn`, `librosa` installed | **VERIFIED** |
| **1.3** | Quantum Checkpoints | `ls models/continual/phase7_8/*/qsvc_q8_r1/*/model_checkpoint.pkl` | 15 speaker checkpoint files present | **VERIFIED** |
| **1.4** | Reference Sidecars | `ls models/quantum/qsvc_q8_r1/inference_reference/*/X_train_pca.npy` | 15 speaker reference sidecars present | **VERIFIED** |
| **1.5** | Sidecar Inventory | `cat models/quantum/qsvc_q8_r1/inference_reference/inventory_manifest.json` | Valid JSON listing all 15 speakers | **VERIFIED** |
| **1.6** | Historical Checksums | Audit script against `reports/project_readiness/checksums.json` | 28/28 historical files matched (100%) | **VERIFIED** |

---

## 2. Server Startup & Health Checklist

| Step | Operation | Execution Command | Verification Metric | Status |
|:---:|:---|:---|:---|:---:|
| **2.1** | Launch Server | `uvicorn src.app.main:app --host 127.0.0.1 --port 8000` | Process binds to `127.0.0.1:8000` | **VERIFIED** |
| **2.2** | Health Check | `curl http://127.0.0.1:8000/health` | `HTTP 200` (`status: healthy`) | **VERIFIED** |
| **2.3** | Readiness Check | `curl http://127.0.0.1:8000/ready` | `HTTP 200` (`ready: true`, `quantum_model_ready: true`) | **VERIFIED** |
| **2.4** | Model Metadata | `curl http://127.0.0.1:8000/model-info?model_id=qsvc_q8_r1&speaker_id=F01` | `HTTP 200` (`status: OPERATIONAL`, 8 qubits) | **VERIFIED** |
| **2.5** | Provenance Status | `curl http://127.0.0.1:8000/api/v1/provenance/summary` | Gate C & Gate D strictly `STRICTLY_BLOCKED` | **VERIFIED** |

---

## 3. Operational Inference & Governance Checklist

| Step | Operation | Test Scenario | Expected Outcome | Status |
|:---:|:---|:---|:---|:---:|
| **3.1** | Live Prediction | `POST /api/v1/predict` (speaker `F01`, 384-D vector) | `HTTP 200`, valid intent, latency < 1000 ms | **VERIFIED** |
| **3.2** | Classical Rejection | `POST /api/v1/predict` (`model_id=continual_classical_adapter`) | `HTTP 400` (`MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE`) | **VERIFIED** |
| **3.3** | Classical Spy Check | Inspect call count of `ClassicalAdapterNet.forward` | Exactly **0** invocations across all requests | **VERIFIED** |
| **3.4** | Fail-Closed Check | `POST /api/v1/predict` (`model_id=phase8_1_track_b_quantum`) | `HTTP 503` (`QUANTUM_MODEL_NOT_READY`) | **VERIFIED** |
| **3.5** | Unknown Speaker Check | `POST /api/v1/predict` (`speaker_id=UNKNOWN_99`) | `HTTP 404` (`SPEAKER_NOT_FOUND`) | **VERIFIED** |
| **3.6** | File Upload Inference | `POST /api/v1/predict/file` (WAV audio) | `HTTP 200`, transcript + quantum intent returned | **VERIFIED** |
| **3.7** | Streaming WebSocket | `WS /ws/audio` (chunked streaming) | Session buffers, predicts, clears memory | **VERIFIED** |
| **3.8** | Buffer Memory Purge | Terminate WebSocket or upload audio > 10MB | Buffer immediately cleared; 0 disk retention | **VERIFIED** |

---

## 4. Test Suite Execution Checklist

| Step | Test Suite | Command | Expected / Actual Result | Status |
|:---:|:---|:---|:---|:---:|
| **4.1** | Full Repository | `python -m pytest tests/ -q` | **648 / 648 passed** in 355.96s | **VERIFIED** |
| **4.2** | Gate B.2 Targeted | `python -m pytest tests/gate_b2/ -v` | **47 / 47 passed** | **VERIFIED** |
| **4.3** | Gate B Prototype | `python -m pytest tests/gate_b/ -v` | **17 / 17 passed** | **VERIFIED** |
| **4.4** | Gate C Audit | `python -m pytest tests/gate_c/ -v` | **17 / 17 passed** | **VERIFIED** |
| **4.5** | Gate D Audit | `python -m pytest tests/gate_d/ -v` | **6 / 6 passed** | **VERIFIED** |

---

## 5. Graceful Shutdown & Cleanup Checklist

| Step | Operation | Execution Command | Expected Outcome | Status |
|:---:|:---|:---|:---|:---:|
| **5.1** | Terminate Server | `Stop-Process -Id <PID>` or `CTRL+C` in console | Process exits cleanly; port 8000 released | **VERIFIED** |
| **5.2** | Check Orphan Processes | `netstat -ano \| findstr :8000` | Port 8000 state: `TIME_WAIT` or unbound | **VERIFIED** |
| **5.3** | Temp Audio Inspection | Check system temp directory | Zero leaked audio recordings or WAV files | **VERIFIED** |

---

## 6. Release Governance Sign-Off

- [x] **Model Preserved:** Original canonical checkpoints unmodified.
- [x] **Zero Classical Fallback:** Enforced via route exceptions, test spies, and schema constraints.
- [x] **Zero Synthetic Cheating:** Sidecar matrices constructed strictly from original training embeddings.
- [x] **100% Checksum Integrity:** 28 / 28 historical files in `reports/project_readiness/checksums.json` matched.
- [x] **100% Test Pass Rate:** 648 / 648 pytest tests passing.
- [x] **Gate C & Gate D Blocked:** Explicitly flagged in UI, API, and documentation.
- [x] **Disclaimers Present:** Clear disclosures that no quantum advantage is proven and the system is not cleared for clinical or emergency use.

**Final Engineering Recommendation:** `APPROVED FOR LOCAL RESEARCH DEMONSTRATION RELEASE`
