# Quantum-Only Test & Runtime Evidence Report

**Project**: Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Date**: October 2026  
**Auditor**: Independent Quantum Systems Architecture Verification  
**Status**: All Operational & Isolation Invariants Cryptographically and Architecturally Verified  

---

## 1. Test Suite Summary & Architecture Coverage

The test suite in `tests/gate_b2/` contains 46 specialized integration, security, and isolation tests across three test modules:

```
tests/gate_b2/
├── test_quantum_only_application.py   (17 tests - Dedicated Quantum-Only & Isolation Suite)
├── test_full_scale_application.py     (18 tests - Full-Scale Application & Audio Suite)
└── test_gate_b2_features.py           (11 tests - Research API, Provenance & Governance Suite)
```

**Total Tests**: 46  
**Pass Rate**: 100%  
**Failures / Errors**: 0  

---

## 2. Comprehensive Test Module Audit

### Module 1: `tests/gate_b2/test_quantum_only_application.py` (17 Tests)

| Test Case | Target Invariant | Expected Behavior | Status |
| :--- | :--- | :--- | :--- |
| `test_quantum_checkpoints_discovered_for_all_speakers` | Checkpoint Scanning | Discovers 15 speaker checkpoints in `models/continual/phase7_8/` | PASS |
| `test_quantum_checkpoint_incomplete_state_detected` | Artifact Verification | Detects missing $X_{\text{train\_pca}}$ reference embeddings | PASS |
| `test_predict_defaults_to_quantum_and_returns_503` | Operational Prediction | Default prediction routes to `qsvc_q8_r1`, returns HTTP 503 `QUANTUM_MODEL_NOT_READY` | PASS |
| `test_classical_model_explicitly_rejected_from_predict` | Classical Rejection | Live `/predict` rejects classical model IDs with HTTP 400 `MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE` | PASS |
| `test_classical_model_rejected_from_predict_file` | Classical Rejection | File upload `/predict/file` rejects classical models with HTTP 400 `MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE` | PASS |
| `test_unmaterialized_quantum_candidate_returns_503` | Candidate Handling | Candidate `phase8_1_track_b_quantum` returns HTTP 503 `QUANTUM_MODEL_NOT_READY` | PASS |
| `test_classical_model_never_called_during_predict` | Spy Isolation | Test spy confirms `ClassicalModelAdapter.predict()` is never executed | PASS |
| `test_unknown_speaker_returns_404` | Speaker Isolation | Requesting unknown speaker returns HTTP 404 `SPEAKER_NOT_FOUND` | PASS |
| `test_whisper_transcription_independent_success` | Audio Decoupling | Local Whisper ASR transcribes speech independently (HTTP 200) | PASS |
| `test_whisper_feature_extraction_independent_success` | Audio Decoupling | 384-D Whisper encoder embeddings extracted independently (HTTP 200) | PASS |
| `test_ready_endpoint_reports_quantum_components` | Readiness Probe | `/ready` reports `quantum_model_ready=false`, `status=QUANTUM_MODEL_NOT_READY` | PASS |
| `test_model_info_classical_baseline_shows_research_only` | Metadata Isolation | `/model-info` classifies classical models as `RESEARCH_BASELINE_ONLY` | PASS |
| `test_model_info_quantum_model_shows_exact_architecture` | Quantum Inspection | `/model-info` exposes 8 qubits, ZZFeatureMap, reps=1, C=1.0 | PASS |
| `test_gate_c_and_d_remain_blocked` | Boundary Constraints | Gate C and Gate D remain strictly blocked in provenance summary | PASS |
| `test_websocket_rejects_classical_model` | WebSocket Security | WebSocket rejects classical models with `MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE` | PASS |
| `test_websocket_quantum_predict_returns_503` | WebSocket Security | WebSocket routes live intent to quantum model, fails closed with `QUANTUM_MODEL_NOT_READY` | PASS |
| `test_models_catalog_partitions_operational_vs_research` | Registry Partition | `/api/v1/models` marks quantum as `PRIMARY_OPERATIONAL_PREDICTOR` and classical as `RESEARCH_BASELINE_ONLY` | PASS |

---

### Module 2: `tests/gate_b2/test_full_scale_application.py` (18 Tests)

| Test Case | Description | Expected Behavior | Status |
| :--- | :--- | :--- | :--- |
| `test_health_endpoint` | System Liveness | HTTP 200, status "healthy" | PASS |
| `test_readiness_endpoint` | System Readiness | HTTP 200, indicates unready quantum model | PASS |
| `test_model_info_canonical_metadata` | Model Metadata | HTTP 200, taxonomy classes and feature dimensions | PASS |
| `test_model_info_unverified_threshold_provenance` | Threshold Warning | `THRESHOLD_PROVENANCE_UNVERIFIED` exposed | PASS |
| `test_valid_wav_transcription` | Whisper ASR | HTTP 200, returns transcribed text | PASS |
| `test_valid_embedding_extraction` | Feature Extraction | HTTP 200, returns 384-D vector with L2 norm ~1.0 | PASS |
| `test_corrupt_audio_upload_rejected` | Audio Validation | HTTP 422 Unprocessable Entity | PASS |
| `test_predict_with_valid_embedding` | Operational Prediction | HTTP 503 `QUANTUM_MODEL_NOT_READY` | PASS |
| `test_predict_file_with_valid_wav` | Operational Audio Predict | HTTP 503 `QUANTUM_MODEL_NOT_READY` | PASS |
| `test_predict_with_unknown_speaker_rejected` | Speaker Isolation | HTTP 404 `SPEAKER_NOT_FOUND` | PASS |
| `test_predict_with_invalid_embedding_dimension` | Dimension Check | HTTP 422 Unprocessable Entity | PASS |
| `test_state_machine_happy_path` | State Transition | IDLE -> RECORDING -> PROCESSING -> COMPLETED | PASS |
| `test_state_machine_timeout_transition` | State Transition | IDLE -> RECORDING -> TIMED_OUT | PASS |
| `test_state_machine_invalid_transition_rejected` | State Validation | Throws ValueError on illegal transition | PASS |
| `test_provenance_summary_all_hashes_verified` | Provenance Audit | 28/28 SHA-256 matches verified | PASS |
| `test_emergency_dispatch_unimplemented_and_blocked` | Gate C/D Blockade | CAD/telephony hooks confirmed absent | PASS |
| `test_websocket_transcription_stream` | Streaming ASR | WebSocket sends PCM frames, receives transcriptions | PASS |
| `test_websocket_invalid_audio_handled` | Streaming Validation | WebSocket reports error on malformed binary | PASS |

---

### Module 3: `tests/gate_b2/test_gate_b2_features.py` (11 Tests)

| Test Case | Description | Expected Behavior | Status |
| :--- | :--- | :--- | :--- |
| `test_research_phases_endpoint` | Research API | HTTP 200, returns 9 historical phases | PASS |
| `test_research_metrics_endpoint` | Research API | HTTP 200, returns Case B benchmark tables | PASS |
| `test_safety_operating_points_endpoint` | Safety Thresholds | HTTP 200, tau=0.60 vs FER<=15.0% curve | PASS |
| `test_research_figures_endpoint` | Figure Catalog | HTTP 200, 7 figures verified on disk | PASS |
| `test_static_figures_serving` | Image Serving | HTTP 200 image/png static serving | PASS |
| `test_provenance_summary_endpoint` | Cryptographic Summary | HTTP 200 with 4,852 utterances, 15 speakers | PASS |
| `test_embedding_extraction_summary_only_by_default` | Feature Privacy | Vector omitted by default | PASS |
| `test_embedding_extraction_with_vector_opt_in` | Feature Inspection | Returns 384-D vector when requested | PASS |
| `test_unmaterialized_model_predict_returns_503` | Quantum & Classical Routing | Quantum returns 503, Classical returns 400 | PASS |
| `test_model_info_threshold_provenance_warning` | Threshold Decoupling | Exposes `decision_cutoff_tau: 0.60` warning | PASS |
| `test_demo_interface_serving` | Web Interface | HTTP 200 HTML5 Quantum Lab UI | PASS |

---

## 3. Cryptographic Runtime Evidence

### Pre/Post Checksum Verification:
- **Baseline Checksums**: Verified against `reports/project_readiness/checksums.json` (28/28 verified, 0 mismatches).
- **Quantum Checkpoints**:
  - `models/continual/phase7_8/F01/model_checkpoint.pkl` (`11dbeee041b61c28c8d0426f0f7f2d65d496033ec09c6eb3309a4d877e620579`) — UNMODIFIED.
  - `models/continual/phase7_8/M02/model_checkpoint.pkl` (`51eef358134707018c148cba47738c86d88ae51139e80208cb21422eb7c04068`) — UNMODIFIED.
- **Classical Checkpoints**:
  - `models/continual/phase7_5/F01/continual_adapter.pt` (`6905a5a676fb13ee79667f56fcffce25048590c6be4a0b27d42cf2dbb41bfa42`) — UNMODIFIED.
- **Retraining Verification**:
  - Models retrained: **0**
  - Synthetic vectors generated: **0**
  - Fallback invocations: **0**
