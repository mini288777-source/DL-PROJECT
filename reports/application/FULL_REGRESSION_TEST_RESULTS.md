# FULL REGRESSION TEST RESULTS

**Execution Date:** 2026-10-11  
**Execution Environment:** Windows Server / Python 3.11.15 (`C:\Users\lucky\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe`)  
**Pytest Version:** 9.1.1 (pluggy 1.6.0, asyncio 1.4.0)  
**Auditor:** Automated Engineering Verification Agent  
**Overall Regression Verdict:** `PASS — 100% SUITE PASS (648 / 648)`

---

## 1. Executive Summary

A full, independent regression execution was run across all test suites in the `tests/` tree. No tests were skipped, weakened, deleted, or mocked to force a passing result. Every phase suite, continual governance check, statistical test, baseline verification, engineering prototype test, Gate B.2 quantum-only suite, and Gate C/D audit suites executed to completion.

### High-Level Metrics
- **Total Tests Collected:** 648
- **Total Tests Passed:** 648
- **Total Tests Failed:** 0
- **Total Tests Skipped:** 0
- **Total Errors:** 0
- **Warnings:** 16 (non-fatal third-party deprecations: `librosa`, `audioread`, `scipy.pyprima`)
- **Total Wallclock Duration:** 355.96 seconds (5 minutes 55.96 seconds)
- **Exit Code:** 0
- **Output Log Location:** `file:///C:/Users/lucky/.gemini/antigravity-ide/brain/b6cd2788-e71b-4685-9df2-158868722057/.system_generated/tasks/task-1611.log`

---

## 2. Test Count Reconciliation

Historical project registry `reports/project_readiness/test_status.json` recorded 577 tests as of the initial audit. The current suite collects and passes **648 tests**. The exact delta of +71 tests is reconciled below:

| Test Suite / Phase | Baseline (`test_status.json`) | Current Suite | Delta | Justification |
|:---|:---:|:---:|:---:|:---|
| `tests/gate_b/` | 16 | 17 | +1 | Addition of explicit classical rejection & quantum-only route verification |
| `tests/gate_b2/` | 0 | 47 | +47 | Gate B.2 quantum-only application suite (37 full-scale + 10 quantum-only) |
| `tests/gate_c/` | 0 | 17 | +17 | Gate C external prerequisite evidence verification suite |
| `tests/gate_d/` | 0 | 6 | +6 | Gate D safety barrier & zero CAD/telephony verification suite |
| `tests/phase8_1/` | 23 | 23 | 0 | Phase 8.1 safety recovery protocol tests |
| `tests/phase8_2/` | 15 | 15 | 0 | Phase 8.2 independent confirmation protocol tests |
| `tests/phase8_2_1/` | 12 | 12 | 0 | Phase 8.2.1 threshold protocol v2 tests |
| `tests/phase8_80_20/` | 23 | 23 | 0 | Phase 8 80/20 train/test split protocol tests |
| `tests/phase8_separated/` | 9 | 9 | 0 | Phase 8 separated baseline models |
| `tests/phase7_continual_governance/` | 178 | 178 | 0 | Continual learning governance, drift detection, and budget tests |
| `tests/phase6_and_6_5_statistical/` | 128 | 128 | 0 | Permutation tests, bootstrap CIs, McNemar tests |
| `tests/phase5_validation/` | 28 | 28 | 0 | Cross-validation and partition integrity tests |
| `tests/phase4_quantum_baselines/` | 12 | 12 | 0 | Quantum kernel and QSVC baseline tests |
| `tests/phase3_classical_baselines/` | 12 | 12 | 0 | Classical baseline comparative tests |
| `tests/system_embeddings_cache_taxonomy/` | 121 | 121 | 0 | Whisper embeddings, taxonomy mapping, cache verification |
| **Total** | **577** | **648** | **+71** | **All 71 additional tests verify quantum-only & gate integrity** |

---

## 3. Targeted Gate Suite Results

### 3.1 Gate B.2 Targeted Suite (`tests/gate_b2/`)
- **Command:** `& "C:\Users\lucky\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -m pytest tests/gate_b2/ -v`
- **Collected:** 47
- **Passed:** 47
- **Failed:** 0
- **Skipped:** 0
- **Exit Code:** 0
- **Coverage Highlights:**
  - `TestZeroClassicalFallbackSpy`: Proves `ClassicalAdapterNet.forward` call count is 0.
  - `TestModelSelectionPolicy`: Proves classical selection returns HTTP 400.
  - `TestModelUnavailabilityFailure`: Proves unmaterialized candidate returns HTTP 503 (`QUANTUM_MODEL_NOT_READY`).
  - `TestProvenanceSummary`: Proves Gate C & D remain strictly blocked.
  - `TestQuantumInferenceDirect`: Proves mathematical output matches exact QSVC margins.

### 3.2 Gate B Engineering Prototype (`tests/gate_b/`)
- **Command:** `& "C:\Users\lucky\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -m pytest tests/gate_b/ -v`
- **Collected:** 17
- **Passed:** 17
- **Failed:** 0
- **Exit Code:** 0

### 3.3 Gate C Prerequisite Verification (`tests/gate_c/`)
- **Command:** `& "C:\Users\lucky\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -m pytest tests/gate_c/ -v`
- **Collected:** 17
- **Passed:** 17
- **Failed:** 0
- **Exit Code:** 0
- **Verdict:** Strictly verifies all 10 prerequisites remain unresolved without synthetic clearance.

### 3.4 Gate D Safety Evidence Verification (`tests/gate_d/`)
- **Command:** `& "C:\Users\lucky\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -m pytest tests/gate_d/ -v`
- **Collected:** 6
- **Passed:** 6
- **Failed:** 0
- **Exit Code:** 0
- **Verdict:** Verifies zero 911/CAD/telephony modules, and safety disclaimers across all endpoints.

---

## 4. Warnings and Non-Fatal Diagnostic Audit

Pytest logged 16 warnings during execution:
1. `DeprecationWarning: 'aifc' is deprecated and slated for removal in Python 3.13` (via audioread)
2. `DeprecationWarning: 'audioop' is deprecated and slated for removal in Python 3.13` (via audioread)
3. `DeprecationWarning: 'sunau' is deprecated and slated for removal in Python 3.13` (via audioread)
4. `UserWarning: PySoundFile failed. Trying audioread instead.` (librosa audio fallback on synthetic test buffer)
5. `FutureWarning: librosa.core.audio.__audioread_load is deprecated` (librosa 0.10.0 notice)
6. 11x `UserWarning: COBYLA: Invalid MAXFUN` (via scipy pyprima in phase 7/10 optimizer tests)

**Assessment:** None of these warnings affect model accuracy, runtime stability, or inference integrity. All tests passed assertions cleanly.
