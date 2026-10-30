# ACCEPTANCE TEST RESULTS — GATE B.2: FULL-FEATURED RESEARCH DEMONSTRATION APPLICATION

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Date of Execution:** 2026-10-10  
**Audit Baseline:** 577 / 577 tests passed, 28 / 28 checksums verified (Gate B.1 Audit Verdict: `ACCEPTED_WITH_RECORDED_LIMITATIONS`)  
**Gate B.2 Test Execution:** 588 / 588 tests passed, 11 / 11 live runtime smoke tests passed, 28 / 28 checksums verified  
**Verdict:** **COMPLETE**

---

## 1. Test Suite Summary

| Test Domain | Target Suite | Collected Tests | Passed Tests | Failed Tests | Pass Rate | Execution Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical Regression Baseline** | `tests/test_phase*.py` | 561 | 561 | 0 | 100.0% | ~14.2 min |
| **Gate B Engineering Prototype** | `tests/gate_b/` | 16 | 16 | 0 | 100.0% | 18.54s |
| **Gate B.2 Demonstration App** | `tests/gate_b2/` | 11 | 11 | 0 | 100.0% | 16.54s |
| **Combined Prototype & App Suite**| `tests/gate_b/` + `tests/gate_b2/` | 27 | 27 | 0 | 100.0% | 81.26s |
| **Complete Repository Total** | `tests/` | **588** | **588** | **0** | **100.0%** | **~15.5 min** |

---

## 2. Gate B.2 Feature Acceptance Tests (`tests/gate_b2/test_gate_b2_features.py`)

All 11 automated acceptance test cases in `tests/gate_b2/test_gate_b2_features.py` passed with zero regressions:

| Test Case | Functional Scope | Verifications Performed | Result |
| :--- | :--- | :--- | :---: |
| `test_research_phases_endpoint` | Research Chronology | `GET /api/v1/research/phases` returns $\ge 9$ phases (Phase 1 to 8.2.1), Case B quantum verdict, non-negotiable disclaimer. | **PASS** |
| `test_research_metrics_endpoint` | Benchmark Tables | `GET /api/v1/research/metrics` parses `results/phase8_1/model_comparison.csv` and `results/phase6_5_6_final_comparison_table.csv` without modification. | **PASS** |
| `test_safety_operating_points_endpoint` | Operating Points | `GET /api/v1/research/operating-points` provides trade-off curve points across $\tau \in [0.20, 0.90]$, locks $\text{FER} \le 15.0\%$ ceiling, defaults $\tau = 0.60$. | **PASS** |
| `test_research_figures_endpoint` | Figure Metadata | `GET /api/v1/research/figures` returns $\ge 7$ figures, all with `exists_on_disk: True` and protocol citations. | **PASS** |
| `test_static_figures_serving` | Image Serving | Verifies `GET /figures/readiness/*.png` and `GET /figures/phase8_1/*.png` return HTTP 200 with `image/png`. | **PASS** |
| `test_provenance_summary_endpoint` | Provenance Audit | `GET /api/v1/provenance/summary` verifies 4,852 utterances, 15 speakers, 28 checksums, and confirms Gates C & D are `STRICTLY_BLOCKED`. | **PASS** |
| `test_embedding_extraction_summary_only` | Feature Summary | `POST /api/v1/audio/extract-embedding` returns 384 dimensions, L2 unit-norm check ($\approx 1.0$), with `vector_preview: None` by default. | **PASS** |
| `test_embedding_extraction_with_vector_opt_in` | Vector Inspector | `POST /api/v1/audio/extract-embedding?include_vector=true` returns preview (16 floats) and full vector (384 floats) for technical inspection. | **PASS** |
| `test_unmaterialized_model_predict_returns_503` | Strict 503 Guard | `POST /api/v1/predict` with candidate `phase8_1_track_a_baseline` strictly returns HTTP 503 `MODEL_NOT_READY` with remediation notice. | **PASS** |
| `test_model_info_threshold_provenance_warning` | Threshold Policy | `GET /model-info` returns `status: THRESHOLD_PROVENANCE_UNVERIFIED`, $\tau = 0.60$, and separate $\text{FER} \le 15.0\%$ ceiling. | **PASS** |
| `test_demo_interface_serving` | SPA Serving | `GET /` and `GET /demo` return HTTP 200 with HTML5 document, research disclaimer, and Gate B.2 title. | **PASS** |

---

## 3. Live Runtime Smoke Tests (`scratch/gate_b2_runtime_smoke_test.py`)

A full end-to-end integration smoke test was conducted against a live `uvicorn` instance running locally on `http://127.0.0.1:8000`:

```json
{
  "health_check": {
    "status_code": 200,
    "status": "healthy"
  },
  "ready_check": {
    "status_code": 200,
    "materialized_models": 61,
    "unready_models": 6
  },
  "html_interface": {
    "status_code": 200,
    "bytes": 26868
  },
  "static_assets": {
    "style_css_bytes": 11450,
    "app_js_bytes": 19448
  },
  "mounted_figures": {
    "status_code": 200,
    "fig_bytes": 135030
  },
  "research_phases": {
    "status_code": 200,
    "phases_count": 9
  },
  "research_metrics": {
    "status_code": 200,
    "phase8_1_rows": 3
  },
  "provenance_summary": {
    "status_code": 200,
    "tracked_checksums": 28,
    "gate_c_status": "STRICTLY_BLOCKED",
    "gate_d_status": "STRICTLY_BLOCKED"
  },
  "websocket_inference": {
    "status": "SUCCESS",
    "response_received": true
  },
  "live_predict": {
    "status_code": 200,
    "predicted_intent": "FALL",
    "confidence_score": 0.9998,
    "processing_latency_ms": 11.6
  },
  "unmaterialized_model_503": {
    "status_code": 503,
    "error_code": "MODEL_NOT_READY",
    "model_id": "phase8_1_track_a_baseline"
  }
}
```

---

## 4. Cryptographic Checksum Integrity Verification

Verification of the 28 authoritative historical artifacts in `reports/project_readiness/checksums.json`:

- **Total Tracked Files:** 28
- **Files Matched Exactly:** 28 (100.0%)
- **Mismatches Detected:** 0
- **Integrity Verdict:** **100% REGRESSION INTEGRITY PRESERVED**

---

## 5. Non-Negotiable Gate Boundaries & Safety Policies

1. **Gate C (Controlled Pilot):** Visibly and programmatically set to `STRICTLY_BLOCKED` in `/api/v1/provenance/summary`, `GET /ready`, UI Overview Dashboard, and documentation.
2. **Gate D (Clinical / 911 Dispatch):** Visibly and programmatically set to `STRICTLY_BLOCKED`.
3. **Threshold Decoupling:** Decision cutoff $\tau = 0.6000$ labeled everywhere as “uncalibrated demonstration default — threshold provenance unverified for this model (`THRESHOLD_PROVENANCE_UNVERIFIED`).”
4. **FER Safety Ceiling:** $\text{FER} \le 15.0\%$ displayed strictly as an evaluation ceiling constraint, never conflated with the decision threshold.
5. **No Model Training or Optimization:** No training loops, parameter adjustments, calibration fitting, or test set inspections occurred.
6. **Ephemeral Audio Storage:** Zero persistent audio files, microphone recordings, or speech transcripts written to disk.
