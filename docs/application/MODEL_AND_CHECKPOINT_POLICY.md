# Model and Checkpoint Policy

## Project: Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication
**Document Reference:** `docs/application/MODEL_AND_CHECKPOINT_POLICY.md`  
**Classification:** Research Prototype Software Policy  
**Effective Date:** 2026-10-11  

---

## 1. Executive Summary & Policy Scope

This document establishes the authoritative lifecycle, verification, and governance rules for all machine learning models and serialized checkpoints served or referenced by the local research application.

### Central Invariant: Immutable Existing Models
The existing model weights, serialized checkpoints, and historical research artifacts are **strictly frozen and immutable**. Under no circumstances will application runtime operations, test suites, or user workflows:
1. Retrain, fine-tune, adapt, re-initialize, prune, quantize, convert, or overwrite existing checkpoints.
2. Alter parameter tensors or modify `state_dict` structures in memory.
3. Silently substitute weights, fabricate random parameters, or route inference to arbitrary speaker adapters.
4. Materialize unverified models without verified training code, frozen manifests, and signed research approval.

---

## 2. Served Model Architecture & Operational Mandate

### 2.1 Canonical Operational Predictor: Quantum Kernel QSVC (`qsvc_q8_r1`)
Under the **Quantum-Only Operational Mandate**, the only model permitted to produce live speech intent predictions is the canonical quantum model:
- **Identifier**: `qsvc_q8_r1`
- **Family**: `quantum_kernel_qsvc`
- **Architecture**: 8-Qubit Fidelity Quantum Kernel QSVC
- **Quantum Feature Map**: `ZZFeatureMap` (qubits=8, reps=1, entanglement='linear')
- **Upstream Acoustic Features**: OpenAI Whisper-tiny encoder (`openai/whisper-tiny`), extracting 384-dimensional log-mel acoustic embeddings.
- **Dimensionality Reduction**: Train-only PCA projection ($384 \to 8$), followed by `MinMaxScaler` scaling to $[0, \pi]$.
- **Classifier**: Support Vector Classifier `SVC(kernel="precomputed", C=1.0)`.
- **Operational Status**: `QUANTUM_MODEL_NOT_READY` (HTTP 503).
  - *State Omission*: The serialized checkpoint file `model_checkpoint.pkl` contains the fitted PCA transformer, MinMaxScaler, and precomputed SVC, but omits the reference training sample embeddings ($X_{\text{train\_pca}}$) required to evaluate test-time Gram vectors.
  - *Non-Negotiable Remediation Policy*: Under the immutable artifact policy, retraining or synthesizing artificial support vectors is strictly prohibited. The system strictly fails closed with transparent error diagnostics.

### 2.2 Relegation of Classical Models to Read-Only Research Baselines
All classical models (such as `continual_classical_adapter`, `classical_adapter`, and `phase8_1_track_a_baseline`) are strictly **prohibited from live operational inference**:
- Any live inference request (`/api/v1/predict`, `/api/v1/predict/file`, WebSocket) attempting to invoke a classical model is rejected immediately with **HTTP 400 Bad Request** (`MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE`).
- Zero silent or automatic fallback to classical models is permitted under any circumstance.
- Classical models remain catalogued strictly for historical research comparisons, benchmark verification, and auditing via `/model-info` and `/api/v1/models`.

### 2.3 Storage & Checkpoint Hierarchy
- **Canonical Quantum Checkpoints:**
  ```
  models/continual/phase7_8/{SPEAKER_ID}/model_checkpoint.pkl
  ```
  Verified across 15 speakers (`F01`, `F03`, `F04`, `F05`, `M01`, `M02`, `M03`, `M04`, `M05`, `FC01`, `FC02`, `FC03`, `MC01`, `MC02`, `MC04`).
- **Historical Classical Baselines (Read-Only):**
  ```
  models/continual/phase7_5/{SPEAKER_ID}/classical_adapter/v{SPEAKER_ID}_b{BUDGET}_replay_seed{SEED}/model_checkpoint.pt
  ```

---

## 3. Mandatory Model Loading Protocol

Every model instance managed by `QuantumModelManager` ([`src/app/quantum_model_manager.py`](file:///d:/DL/src/app/quantum_model_manager.py)) must adhere to the four-step loading contract:

### 3.1 Safe Deserialization
Checkpoints must be loaded exclusively using PyTorch's secure weights-only mode to prevent arbitrary code execution:
```python
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
```

### 3.2 Strict Evaluation Mode (`eval()`)
Immediately upon initialization, the model must be placed in evaluation mode:
```python
model.eval()
```
This disables dropout sampling (`Dropout(0.20)`) and fixes batch normalization statistics (`BatchNorm1d(64)`) to inference mode.

### 3.3 Inference Parameter Freezing (`requires_grad = False`)
Every parameter tensor in the model hierarchy must have gradient accumulation explicitly disabled:
```python
for param in model.parameters():
    param.requires_grad = False
```
This guarantees that accidental gradient passes or backward triggers cannot modify parameter values in memory.

### 3.4 Verification of Architectural Compatibility
The checkpoint's `state_dict` keys must match the `ClassicalAdapterNet` definition exactly (`net.0.weight`, `net.0.bias`, `net.1.weight`, `net.1.bias`, `net.1.running_mean`, `net.1.running_var`, `net.4.weight`, `net.4.bias`). Any tensor size mismatch triggers an immediate fatal initialization error.

---

## 4. Speaker Isolation and Routing Policy

1. **Strict 1-to-1 Mapping:** Inference requests specify `speaker_id`. The application routes exclusively to the checkpoint matching that exact identifier.
2. **Zero Silent Fallback:** If a request targets an unregistered speaker (e.g., `UNKNOWN_01`, `F99`), the application **must fail immediately with HTTP 404 `SPEAKER_NOT_FOUND`**. It is strictly forbidden to silently substitute another speaker's adapter or fall back to an unpersonalized global model without explicit caller awareness.
3. **Cross-Speaker Incompatibility Warning:** Personalization adapters are highly specialized to individual dysarthric acoustic patterns. Cross-speaker execution degrades error rates severely and is rejected at the API boundary.

---

## 5. Policy on Unmaterialized Models (Phase 8.1 & Phase 8.2 Candidates)

### 5.1 Candidate Model Inventory
The following six candidate model families are documented in research protocols but **remain unmaterialized on disk**:
- `phase8_1_classical` (Phase 8.1 Safety Recovery Classical Baseline)
- `phase8_1_quantum` (Phase 8.1 Safety Recovery Hybrid Quantum-Classical)
- `phase8_2_classical` (Phase 8.2 Independent Confirmation Classical)
- `phase8_2_quantum` (Phase 8.2 Independent Confirmation Quantum)
- `phase8_2_1_classical` (Phase 8.2.1 Threshold-Re-audited Classical)
- `phase8_2_1_quantum` (Phase 8.2.1 Threshold-Re-audited Quantum)

### 5.2 Mandatory Fail-Closed Response
When an inference or status request references any of these six unmaterialized candidates:
1. The application **must return HTTP 503 `MODEL_NOT_READY`** with error code `MODEL_NOT_READY`.
2. The response body must explicitly state:
   - `status`: `"UNAVAILABLE"`
   - `reason`: `"Model checkpoint is unmaterialized on disk. Training, weight fabrication, and mock fallbacks are prohibited under the immutable model governance policy."`
3. Under no circumstances may application code generate pseudo-random probabilities, mock emergency predictions, or claim operational status for these candidates.

---

## 6. Operating Cutoff & Threshold Policy

### 6.1 Demonstration Default ($\tau = 0.6000$)
The application serves inference scores with a default confidence threshold of $\tau = 0.6000$.
- **Provenance Notice:** $\tau = 0.6000$ is an **uncalibrated demonstration default**. Its statistical provenance is unverified for the served Phase 7.5 model.
- **Non-Clinical Designation:** This value does **not** represent a calibrated clinical operating point, a safety-certified risk boundary, or an FDA-cleared decision rule.
- **Abstention Behavior:** When the top class probability is below $\tau = 0.6000$, the system issues an `ABSTAIN` decision with low confidence notice.

### 6.2 Distinctness from Internal Research Ceiling
The demonstration cutoff $\tau = 0.6000$ is completely decoupled from the internal research constraint of 15% False Escalation Rate (FER). Research FER constraints govern offline offline-of-fold optimization, whereas $\tau = 0.6000$ is a static, runtime prototype parameter.

---

## 7. Input and Output Contracts

| Stage | Input Contract | Output Contract | Verification Mechanism |
| :--- | :--- | :--- | :--- |
| **Acoustic Input** | 16,000 Hz, 1-channel Mono WAV / FLAC, $\le 30.0$ s, $\le 10$ MB | Normalised float32 array in $[-1.0, 1.0]$ | `AudioService.validate_and_load()` |
| **Feature Extraction** | Preprocessed waveform chunk ($16$ kHz) | 384-dimensional log-mel embedding tensor | `AudioService.extract_embedding()` via Whisper-tiny |
| **Adapter Inference** | Float32 tensor shape `(1, 384)` | Logits shape `(1, 7)`, Softmax probabilities summing to $1.0$ | `ClassicalAdapterNet.forward()` |
| **Taxonomy Output** | Softmax vector | 7 named class scores in canonical order | `config/taxonomy.yaml` validation |

---

## 8. Summary of Compliance Verification

Every test run verifies:
1. `model.training == False` across all instantiated adapters.
2. `all(not p.requires_grad for p in model.parameters()) == True`.
3. Model weight hash before and after forward pass remains identical (Bit-for-bit invariance).
4. Unmaterialized candidates return HTTP 503 without executing forward calculations.
5. All 28 registered historical artifacts match their recorded SHA-256 signatures in [`reports/project_readiness/checksums.json`](file:///d:/DL/reports/project_readiness/checksums.json).
