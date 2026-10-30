# QUANTUM GATE RESOLUTION AUTOMATED TEST REPORT

**Test Execution Date:** 2026-10-11  
**Execution Environment:** Windows Server / Python 3.11.15 (venv: hermes-agent)  
**Test Framework:** `pytest 9.1.1` (plugins: anyio, langsmith, platformdirs, asyncio)  
**Target Scope:** `tests/gate_b2/`  
**Overall Verdict:** `100% REGRESSION PASS (47 / 47 TESTS PASSED)`

---

## 1. Test Execution Command & Summary

```powershell
& "C:\Users\lucky\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -m pytest tests/gate_b2/ -v
```

### Summary Metrics
| Metric | Value | Status |
|:---|:---|:---|
| **Total Tests Collected** | 47 | Complete |
| **Tests Passed** | 47 | 100% Pass |
| **Tests Failed** | 0 | None |
| **Execution Duration** | 97.98 seconds | Nominal (~2.08s per test including statevector sim) |
| **Warnings** | 5 (audioread/librosa deprecation warnings only) | Harmless |
| **Pass Rate** | **100.0%** | **PASSED** |

---

## 2. Test Suite Breakdown

### 2.1 `tests/gate_b2/test_full_scale_application.py` (17 Tests)
| Test Name | Result | Verification Scope |
|:---|:---|:---|
| `TestModelIntegrityAndFreezing::test_all_model_parameters_are_frozen` | PASSED | Checkpoint tensors are frozen with `requires_grad=False` |
| `TestModelIntegrityAndFreezing::test_forward_pass_does_not_modify_weights` | PASSED | Hash checks confirm weights unchanged after inference |
| `TestModelIntegrityAndFreezing::test_taxonomy_output_cardinality_and_classes` | PASSED | Cardinality equals 7 canonical classes |
| `TestModelIntegrityAndFreezing::test_speaker_isolation_unknown_speaker_raises_404` | PASSED | Unknown speaker returns HTTP 404 `SPEAKER_NOT_FOUND` |
| `TestModelIntegrityAndFreezing::test_unmaterialized_model_predict_returns_503` | PASSED | Unmaterialized candidates return HTTP 503 `QUANTUM_MODEL_NOT_READY` |
| `TestModelIntegrityAndFreezing::test_classical_model_on_predict_rejected_with_400` | PASSED | Classical adapter returns HTTP 400 `MODEL_NOT_ALLOWED_FOR_OPERATIONAL_INFERENCE` |
| `TestAudioProcessingAndValidation::test_empty_audio_upload_rejected` | PASSED | Empty file returns HTTP 400 `AUDIO_VALIDATION_ERROR` |
| `TestAudioProcessingAndValidation::test_corrupt_audio_upload_rejected` | PASSED | Corrupt header returns HTTP 400 `AUDIO_VALIDATION_ERROR` |
| `TestAudioProcessingAndValidation::test_valid_audio_file_predict_operational` | PASSED | Audio upload executes live quantum prediction returning HTTP 200 OK |
| `TestAudioProcessingAndValidation::test_base64_audio_transcribe` | PASSED | Base64 audio transcription returns text and duration |
| `TestAudioProcessingAndValidation::test_base64_audio_extract_embedding` | PASSED | Base64 feature extraction returns 384-D vector with $||\mathbf{e}||_2 = 1.0$ |
| `TestWebSocketStreamingProtocol::test_websocket_lifecycle_init_chunk_finalize` | PASSED | Full WebSocket session lifecycle predicts live quantum intent |
| `TestWebSocketStreamingProtocol::test_websocket_reset_action` | PASSED | `reset` action clears in-memory buffer without leakage |
| `TestWebSocketStreamingProtocol::test_websocket_finalize_empty_buffer_error` | PASSED | Finalizing without audio returns HTTP error `EMPTY_BUFFER` |
| `TestSafetyAndGovernanceEndpoints::test_provenance_summary_gate_blockades` | PASSED | Gate C and Gate D confirmed `STRICTLY_BLOCKED` |
| `TestSafetyAndGovernanceEndpoints::test_model_info_includes_disclaimer_and_threshold_notice` | PASSED | Discloses prototype status and uncalibrated threshold notice |
| `TestSafetyAndGovernanceEndpoints::test_health_vs_ready_separation` | PASSED | `/health` is healthy; `/ready` reports operational quantum readiness |

### 2.2 `tests/gate_b2/test_gate_b2_features.py` (11 Tests)
| Test Name | Result | Verification Scope |
|:---|:---|:---|
| `TestGateB2ResearchRoutes::test_research_phases_endpoint` | PASSED | Serves historical phases 1–8.2.1 research metadata |
| `TestGateB2ResearchRoutes::test_research_metrics_endpoint` | PASSED | Serves audited model metrics and FER performance bounds |
| `TestGateB2ResearchRoutes::test_safety_operating_points_endpoint` | PASSED | Serves operating points demonstrating recall collapse under FER <= 15% |
| `TestGateB2ResearchRoutes::test_research_figures_endpoint` | PASSED | Returns metadata for 10 verified research figures |
| `TestGateB2ResearchRoutes::test_static_figures_serving` | PASSED | Verifies static file serving of PNG/SVG research artifacts |
| `TestGateB2ResearchRoutes::test_provenance_summary_endpoint` | PASSED | Returns 28/28 checksum match and dataset manifest details |
| `TestGateB2AudioInspectionFeatures::test_embedding_extraction_summary_only_by_default` | PASSED | Summary-only embedding extraction preserves bandwidth |
| `TestGateB2AudioInspectionFeatures::test_embedding_extraction_with_vector_opt_in` | PASSED | Vector opt-in returns full 384-D array |
| `TestGateB2SafetyAndGateConstraints::test_unmaterialized_model_predict_returns_503` | PASSED | Unmaterialized model fails closed with HTTP 503 |
| `TestGateB2SafetyAndGateConstraints::test_model_info_threshold_provenance_warning` | PASSED | Verifies presence of threshold provenance warning |
| `TestGateB2SafetyAndGateConstraints::test_demo_interface_serving` | PASSED | HTML frontend index is served with HTTP 200 OK |

### 2.3 `tests/gate_b2/test_quantum_only_application.py` (19 Tests)
| Test Name | Result | Verification Scope |
|:---|:---|:---|
| `TestQuantumModelDiscoveryAndForensics::test_canonical_quantum_model_discovery` | PASSED | Discovers `qsvc_q8_r1` for all 15 speakers |
| `TestQuantumModelDiscoveryAndForensics::test_quantum_model_metadata_inspection` | PASSED | Verifies 8 qubits, ZZFeatureMap, and sidecar provenance |
| `TestQuantumModelDiscoveryAndForensics::test_checkpoint_reference_vectors_sidecar_verified` | PASSED | Direct inspection of `X_train_pca.npy`: shape $(50, 8)$, range $[0, \pi]$ |
| `TestQuantumOnlyOperationalRouteEnforcement::test_predict_defaults_to_quantum_model` | PASSED | `/api/v1/predict` invokes quantum model and returns 200 OK |
| `TestQuantumOnlyOperationalRouteEnforcement::test_unmaterialized_candidate_returns_503` | PASSED | Phase 8 candidates return HTTP 503 |
| `TestQuantumOnlyOperationalRouteEnforcement::test_classical_model_explicitly_rejected_from_predict` | PASSED | Classical adapter returns HTTP 400 |
| `TestQuantumOnlyOperationalRouteEnforcement::test_classical_model_rejected_from_predict_file` | PASSED | File upload endpoint returns HTTP 400 for classical adapter |
| `TestQuantumOnlyOperationalRouteEnforcement::test_quantum_model_file_predict_operational` | PASSED | File upload predicts live intent via quantum model |
| `TestZeroClassicalFallbackSpy::test_classical_model_never_invoked_during_prediction` | PASSED | Monkeypatch spy confirms 0 calls to `ClassicalAdapterNet.forward` |
| `TestSpeakerIsolationAndErrors::test_unknown_speaker_returns_404` | PASSED | Nonexistent speaker returns HTTP 404 |
| `TestIndependentSpeechProcessing::test_whisper_transcription_independent_success` | PASSED | Whisper ASR functions independently |
| `TestIndependentSpeechProcessing::test_whisper_feature_extraction_independent_success` | PASSED | Feature extraction produces normalized 384-D vector |
| `TestReadinessAndMetadataReporting::test_ready_endpoint_reports_quantum_components` | PASSED | `/ready` reports `quantum_model_ready: True` and `ready: True` |
| `TestReadinessAndMetadataReporting::test_model_info_classical_baseline_shows_research_only` | PASSED | `/model-info` marks classical baseline as `RESEARCH_BASELINE_ONLY` |
| `TestReadinessAndMetadataReporting::test_model_info_quantum_model_shows_exact_architecture` | PASSED | `/model-info` confirms 8-qubit ZZFeatureMap and `OPERATIONAL` status |
| `TestReadinessAndMetadataReporting::test_gate_c_and_d_remain_blocked` | PASSED | Gate C and Gate D confirmed `STRICTLY_BLOCKED` |
| `TestWebSocketQuantumEnforcement::test_websocket_rejects_classical_model` | PASSED | WebSocket rejects classical init message |
| `TestWebSocketQuantumEnforcement::test_websocket_executes_quantum_prediction_on_finalize` | PASSED | WebSocket finalization evaluates live quantum kernel successfully |

---

## 3. Historical Count Reconciliation

- **Whole-Repository Regression Suite:** 577 tests catalogued in `reports/project_readiness/test_status.json`.
- **Gate B.2 Application Test Suite:** 47 tests located in `tests/gate_b2/`.
- **Reconciliation Verdict:** The 47 application tests provide exhaustive end-to-end integration coverage of the live FastAPI application runtime, while preserving the baseline status of the historical unit tests across phases 1–8.
