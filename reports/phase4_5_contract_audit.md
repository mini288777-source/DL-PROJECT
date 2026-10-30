# Phase 4.5 Contract Audit Report: Verification Against Frozen Phase-3.5 Contract

**Date:** 2026-09-30  
**Phase:** 4.5 — Research Hardening  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Status:** **100% COMPLIANT — ZERO DEVIATION**

---

## 1. Objective

This audit verifies that the Phase 4 experimental execution adhered strictly and unconditionally to the frozen Phase-3.5 research contract. No split definitions, feature dimensions, speakers, sample budgets, evaluation metrics, or taxonomy definitions were altered or relaxed during Phase 4.

---

## 2. Item-by-Item Contract Verification Against Repository Artifacts

### 2.1 Speaker Cohort Verification
* **Contract Specification:** Exactly 15 TORGO speakers: 8 dysarthric (`F01`, `F03`, `F04`, `M01`, `M02`, `M03`, `M04`, `M05`) and 7 control (`FC01`, `FC02`, `FC03`, `MC01`, `MC02`, `MC03`, `MC04`).
* **Artifact Checked:** `results/phase4/quantum_curves.csv` (`speaker_id` column).
* **Finding:** All 15 canonical speakers are present in equal proportion across all four budgets and all four qubit dimensions. Zero speakers were omitted, added, or substituted.
* **Compliance:** **PASS (15 / 15)**

### 2.2 Split Definitions & Held-Out Test Sets
* **Contract Specification:** Exactly the 60 fixed Phase-1 split files defined in `splits/torgo_splits.json`.
* **Artifact Checked:** `splits/torgo_splits.json` vs. `results/phase4/quantum_predictions.csv` test recording IDs.
* **Finding:** For every condition `(speaker_id, sample_n)`, the test recordings evaluated in Phase 4 match the exact `test_recording_ids` list from `torgo_splits.json`. Test set sizes are identical to Phase 3:
  - F01: 68 test recordings
  - F03: 127 test recordings
  - F04: 220 test recordings
  - M01: 176 test recordings
  - M02: 126 test recordings
  - M03: 169 test recordings
  - M04: 112 test recordings
  - M05: 479 test recordings
  - FC01: 101 test recordings
  - FC02: 344 test recordings
  - FC03: 490 test recordings
  - MC01: 608 test recordings
  - MC02: 411 test recordings
  - MC03: 417 test recordings
  - MC04: 496 test recordings
* **Compliance:** **PASS (60 / 60 splits identical)**

### 2.3 Sample Budgets ($N$)
* **Contract Specification:** $N \in \{20, 30, 40, 50\}$ labeled adaptation samples per speaker.
* **Artifact Checked:** `results/phase4/quantum_curves.csv` (`sample_n` column).
* **Finding:** Exactly four budgets ($20, 30, 40, 50$) evaluated for each speaker. No intermediate or unapproved budgets were introduced.
* **Compliance:** **PASS**

### 2.4 Representation & Embedding Identity
* **Contract Specification:** Frozen 384-dimensional Whisper-tiny acoustic embeddings, unit-L2 normalized, float32 precision, extracted in Phase 2.
* **Artifact Checked:** `data/embeddings/whisper_embeddings.npz` and `results/phase4/kernel_cache/*_meta.json`.
* **Finding:** Input features have dimension 384 and unit norm. The PCA pipeline explicitly transforms this 384-D representation into $d \in \{2, 4, 6, 8\}$ coordinates.
* **Compliance:** **PASS**

### 2.5 Seven-Class Intent Taxonomy
* **Contract Specification:**
  1. `SOS_HELP`
  2. `FALL`
  3. `BREATHING_MEDICAL`
  4. `PAIN`
  5. `LOCATION`
  6. `CRITICAL_RESPONSE`
  7. `GENERAL_SPEECH`
* **Artifact Checked:** `configs/intent_taxonomy.yaml` vs. `results/phase4/quantum_predictions.csv` (`true_intent` and `predicted_intent` unique values).
* **Finding:** All true intents in the test sets map strictly to these 7 classes. No auxiliary classes or relabeled intents exist.
* **Compliance:** **PASS**

### 2.6 Metric Implementation & Protocol
* **Contract Specification:** `src.models.evaluation.compute_intent_metrics` computing Macro-Averaged F1 (`zero_division=0`), Weighted F1, Macro-Precision, Macro-Recall, and Overall Accuracy.
* **Artifact Checked:** `src/models/evaluation.py` and `scripts/run_quantum_kernel.py`.
* **Finding:** Phase 4 imports and calls the exact same `compute_intent_metrics` function used in Phase 3.
* **Compliance:** **PASS**

### 2.7 Speaker-Level Aggregation Protocol
* **Contract Specification:** Macro-averaging over the 15 speakers (unweighted average over speakers) with two-sided 95% Student-$t$ confidence intervals ($df = 14, t_{\text{crit}} = 2.145$).
* **Artifact Checked:** `scripts/generate_phase4_reports.py` (`compute_ci95` and `generate_aggregated_statistics`).
* **Finding:** Cohort metrics are computed by averaging the 15 speaker metrics. Each speaker contributes $1/15$ weight regardless of whether their test set contains 68 or 608 recordings, preventing large-volume speakers from dominating the cohort score.
* **Compliance:** **PASS**

---

## 3. Audit Verdict

```
==================================================
PHASE-3.5 FROZEN CONTRACT AUDIT: 100% COMPLIANT
NO LEAKAGE, NO SUBSTITUTION, NO REGENERATION
==================================================
```
