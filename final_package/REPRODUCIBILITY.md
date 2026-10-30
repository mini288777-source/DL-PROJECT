# FINAL REPRODUCIBILITY & SCIENTIFIC AUDIT GUIDE

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Repository State:** Final Phase 7 Closure (Canonical & Reproducible)  
**Date:** October 2026  

---

## 1. System & Environment Specifications

- **Operating System:** Windows 11 Enterprise (x86_64)
- **Python Runtime:** Python 3.11.15 (`C:\Users\lucky\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe`)
- **Key Package Versions:**
  - `torch`: 2.6.0+cpu
  - `pennylane`: 0.40.0
  - `scikit-learn`: 1.6.1
  - `numpy`: 2.2.3
  - `scipy`: 1.15.2
  - `pandas`: 2.2.3
  - `pytest`: 9.1.1
  - `pyyaml`: 6.0.2

---

## 2. Seed & Reproducibility Policy

- **Global Master Seed:** `42`
- **Speaker Permutation / Split Seed:** `42` (Fixed stratified partitions)
- **Deterministic Flags:**
  - PyTorch CPU deterministic operations enabled
  - NumPy random state initialized with fixed seeds (`seed=42`)
  - PennyLane statevector simulator (`default.qubit`) analytical execution

---

## 3. Dataset Provenance & Invariants

- **Corpus:** TORGO Database of Dysarthric Speech
- **Cohort:** 15 Validated Speakers (8 Dysarthric, 7 Matched Control)
  - Dysarthric: `F01`, `F03`, `F04`, `M01`, `M02`, `M03`, `M04`, `M05`
  - Control: `FC01`, `FC02`, `FC03`, `MC01`, `MC02`, `MC03`, `MC04`
- **Audio Preprocessing:** 16 kHz Mono PCM, validated duration ($>0.5\text{ s}$), transcript cross-validated.
- **Acoustic Representation:** Frozen `openai/whisper-tiny` encoder, 384-dimensional pooled embeddings, unit-$L_2$ normalized.
- **Dataset Fingerprint / Manifest:** `data/manifests/torgo_intent_mapping.csv` (16,554 labeled records).
- **Split Hierarchy:** Nested budget splits $\mathcal{D}_{\text{adapt}}^{(20)} \subset \mathcal{D}_{\text{adapt}}^{(30)} \subset \mathcal{D}_{\text{adapt}}^{(40)} \subset \mathcal{D}_{\text{adapt}}^{(50)}$.

---

## 4. Experiment Registries & Results Directory Structure

| Subphase | Primary Result Directory | Canonical Report | Summary JSON |
| :--- | :--- | :--- | :--- |
| **Phase 7.4 (Continual Classical)** | `results/continual/phase7_4/` | `research/continual/PHASE7_4_CONTINUAL_CLASSICAL.md` | `reports/continual/phase7_4/continual_classical_summary.json` |
| **Phase 7.5 (Replay Memory)** | `results/continual/phase7_5/` | `research/continual/PHASE7_5_REPLAY_PROTECTION.md` | `reports/continual/phase7_5/continual_replay_summary.json` |
| **Phase 7.6 (Validation Gate)** | `results/continual/phase7_6/` | `research/continual/PHASE7_6_MODEL_VALIDATION_GATE.md` | `reports/continual/phase7_6/governance_summary.json` |
| **Phase 7.7 (Drift & Recalibration)**| `results/continual/phase7_7/` | `research/continual/PHASE7_7_DRIFT_RECALIBRATION.md` | `reports/continual/phase7_7/drift_analysis_summary.md` |
| **Phase 7.8 (Fixed Quantum Kernel)**| `results/continual/phase7_8/` | `research/continual/PHASE7_8_CONTINUAL_QUANTUM.md` | `reports/continual/phase7_8/continual_quantum_summary.json` |
| **Phase 7.9 (Trainable Quantum)** | `results/continual/phase7_9/` | `research/continual/PHASE7_9_TRAINABLE_QUANTUM_REPRESENTATION.md`| `reports/continual/phase7_9/continual_trainable_quantum_summary.json` |
| **Phase 7.10 (Hybrid Fusion)** | `results/continual/phase7_10/`| `research/continual/PHASE7_10_HYBRID_CLASSICAL_QUANTUM.md` | `reports/continual/phase7_10/continual_hybrid_summary.json` |
| **Phase 7.11 (Self-Improvement)** | `results/continual/phase7_11/`| `research/continual/PHASE7_11_SELF_IMPROVEMENT.md` | `reports/continual/phase7_11/self_improvement_summary.json` |
| **Phase 7.12 (Final Comparison)** | `results/continual/phase7_12/`| `research/continual/PHASE7_12_FINAL_COMPARISON.md` | `reports/continual/phase7_12/final_comparison_summary.json` |
| **Phase 7.13 (Safety Audit)** | `results/continual/phase7_13/`| `research/continual/PHASE7_13_SAFETY_ROBUSTNESS.md` | `reports/continual/phase7_13/safety_robustness_summary.json` |
| **Phase 7.14 (App Integration)** | `results/continual/phase7_14/`| `research/continual/PHASE7_14_APPLICATION_INTEGRATION.md` | Config: `config/continual/app_config.yaml` |
| **Phase 7.15 (Final Closure)** | `results/continual/final/` | `research/continual/PHASE7_FINAL_REPORT.md` | `reports/continual/final/phase7_final_closure_summary.json` |

---

## 5. Test Suite Reproduction

To reproduce and verify the complete 469-test regression suite:
```powershell
python -m pytest tests/ -v
```
**Expected Outcome:** Exactly `469 passed` in ~145 seconds.

---

## 6. Demo Launch Command

To reproduce and interact with the research prototype:
```powershell
python scripts/run_app_demo.py --port 8000
```
Open `http://127.0.0.1:8000` in any web browser.
