# RELEASE NOTES — CANDIDATE `qsvc-q8-r1-research-demo-rc1`

**Release Candidate:** `qsvc-q8-r1-research-demo-rc1`  
**Date:** 2026-10-11  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Target Environment:** Local Python 3.11 Runtime / FastAPI Prototype  
**Auditor:** Automated Engineering Verification Agent  
**Release Readiness Verdict:** `RELEASE_READY_FOR_LOCAL_RESEARCH_DEMONSTRATION`

---

## 1. Overview and Release Intent

This release candidate hardens and verifies the **quantum-only** speech intent recognition prototype based on the canonical pre-trained `qsvc_q8_r1` model architecture. 

All engineering blockers previously preventing live quantum inference have been forensically resolved by materializing the exact training-reference sidecar matrices (`X_train_pca.npy`) across all 15 target speakers (`F01` to `MC04`). The live system now executes full quantum statevector fidelity kernel evaluations against verified reference states in real time, with **zero reliance on classical fallback**.

---

## 2. Key Highlights and Capabilities

### 2.1 Genuine Quantum Inference Restored
- **Model Preserved:** The original canonical checkpoint `model_checkpoint.pkl` across all 15 speakers remains 100% byte-for-byte identical to the pre-recovery state. No weights, support vectors, PCA pipelines, or scalers were modified, refit, or retrained.
- **Reference Sidecars Materialized:** Sidecars (`models/quantum/qsvc_q8_r1/inference_reference/{speaker}/X_train_pca.npy`) were materialized strictly from the training partition of the continual learning dataset.
- **Mathematical Exactness:** Recomputed Gram matrices evaluated against historical kernel caches (`cache/continual/quantum_kernels/`) demonstrated a maximum absolute difference of **0.00000000** across all 15 speakers.
- **Runtime Latency:** Operational queries execute in ~499–635 ms under Uvicorn/FastAPI.

### 2.2 Strict Zero Classical Fallback
- **Route Policy:** Any attempt to invoke classical baselines (`continual_classical_adapter`, `classical_svm`, etc.) on operational routes (`/api/v1/predict`, `/api/v1/predict/file`, `/ws/audio`) is rejected with **HTTP 400 Bad Request** (`MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE`).
- **Fail-Closed Design:** Unmaterialized quantum candidates return **HTTP 503** (`QUANTUM_MODEL_NOT_READY`), and unknown speakers return **HTTP 404** (`SPEAKER_NOT_FOUND`).
- **Test Spy Verification:** Injected test spy into `ClassicalAdapterNet.forward` confirmed **0 forward invocations** across all operational tests.

### 2.3 Audio Lifecycle & Buffer Security
- **Ephemeral Processing:** WebSocket streaming processes PCM audio in memory with strict 10 MB caps (`BUFFER_LIMIT_EXCEEDED`).
- **Memory Purge:** Buffers are immediately cleared within `finally:` blocks upon session termination or error.
- **Zero Disk Retention:** Audio data and transcription tokens are never persisted to disk or emitted into production logs.

---

## 3. Verification & Test Metrics

1. **Full Repository Regression:**
   - **Command:** `python -m pytest tests/ -q`
   - **Result:** **648 / 648 passed (100% pass rate)** in 355.96 seconds.
   - **Errors / Failures:** 0.

2. **Gate B.2 Targeted Suite:**
   - **Command:** `python -m pytest tests/gate_b2/ -v`
   - **Result:** **47 / 47 passed**.

3. **Historical Checksum Verification:**
   - **Target:** `reports/project_readiness/checksums.json`
   - **Result:** **28 / 28 historical files matched (100%)**.

4. **Live Smoke Tests:**
   - **Target:** `http://127.0.0.1:8000`
   - **Result:** 11/11 operational endpoints verified live.

---

## 4. Governance & Safety Barriers (Gate C & Gate D)

### Gate C: Controlled Clinical Pilot
- **Status:** `BLOCKED_EXTERNAL_PREREQUISITES`
- **Blockers:** 10 prerequisites remain strictly unresolved, including independent prospective atypical speech cohorts, formal data use agreements (DUAs), institutional review board (IRB) approvals, and clinical oversight protocols.

### Gate D: Emergency Clinical Deployment
- **Status:** `BLOCKED_SAFETY_EVIDENCE`
- **Blockers:** The prototype is **strictly experimental**. It contains zero connections to emergency dispatch (911/CAD), telephony, or clinical alerting systems.
- **Stop Rule & Thresholds:** The operating threshold ($\tau = 0.60$) is an unverified demonstration parameter. The internal 15% FER constraint and 20% recall stop-rule boundary are internal scientific checkpoints and do not represent clinical clearance standards.
- **No Quantum Advantage:** Statistical audits confirm that no statistically significant quantum advantage has been established over classical baselines.

---

## 5. Known Limitations & Usage Constraints

1. **Demonstration Only:** This release candidate is strictly intended for local research demonstration, code auditing, and architectural evaluation.
2. **Uncalibrated Heuristics:** Model scores represent raw SVM decision margins scaled via heuristic sigmoids and must not be interpreted as calibrated probabilities or clinical confidence.
3. **Hardware Backends:** Quantum kernel evaluations are currently executed via Qiskit Statevector simulation; execution on physical QPUs is not yet implemented.
