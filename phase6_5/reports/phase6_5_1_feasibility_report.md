# Subphase 6.5.1 — Data Feasibility & Speaker-Stratified Splits Report

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Database (15 Speakers: 8 Dysarthric, 7 Control)  
**Track:** Phase 6.5 — Emergency Recognition Recovery  
**Subphase:** 6.5.1 — Data Feasibility & Speaker-Stratified Recovery Splits  
**Date:** October 2026  
**Status:** COMPLETE & VERIFIED — NO MODEL TRAINING PERFORMED  

---

## 1. Executive Summary & Purpose

Subphase 6.5.0 established that the complete collapse of kernel classifiers to `GENERAL_SPEECH` was fundamentally a **data support failure**: under unstratified random few-shot sampling ($N \in \{20, 30, 40, 50\}$), adaptation subsets contained on average only 1.1 to 3.1 emergency utterances across all 6 emergency classes combined (~94% majority class dominance), with `SOS_HELP` completely absent from adaptation in 59/60 experimental cells and `LOCATION` completely absent in 60/60 cells.

The objective of Subphase 6.5.1 is to construct a **scientifically rigorous, deterministic, speaker-stratified recovery partitioning protocol** that:
1. **Audits true corpus feasibility** across all 15 speakers, identifying physical class availability and structural limits.
2. **Eliminates acoustic and semantic prompt leakage** by enforcing whole prompt-group allocations (`prompt_group_id` / normalized transcript grouping).
3. **Guarantees strict nested few-shot budgets:** $S_{20} \subset S_{30} \subset S_{40} \subset S_{50}$ for every speaker.
4. **Preserves held-out test support** so that test sets retain unlearned emergency instances for honest evaluation.
5. **Maintains absolute isolation** from the frozen Phase 1–6 benchmarks.

---

## 2. Full Speaker Inventory & Corpus Audit

The complete physical TORGO manifest contains 16,552 audio files across 15 speakers. However, TORGO recorded each utterance simultaneously using two microphone channels (`headMic` and `arrayMic`). Including both channels in experimental subsets would cause severe multi-channel acoustic leakage. Therefore, the canonical experimental pool established in Phase 1 comprises **6,067 primary single-channel recordings** (one primary recording per physical prompt trial, with validated Whisper embeddings already extracted).

The table below reports the complete labeled inventory for all 15 speakers within this active experimental pool:

| Speaker ID | Cohort Group | Total Recs | General Speech | SOS Help | Fall | Breathing Medical | Pain | Location | Critical Response | Total Emergency | Emer Classes Present | Unique Prompts | Sessions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F01** | dysarthric | 118 | 108 | 1 | 3 | 3 | 0 | 0 | 3 | 10 | 4 | 97 | 1 |
| **F03** | dysarthric | 213 | 202 | 3 | 3 | 1 | 0 | 0 | 4 | 11 | 4 | 191 | 3 |
| **F04** | dysarthric | 250 | 240 | 1 | 1 | 2 | 1 | 0 | 5 | 10 | 5 | 219 | 2 |
| **FC01** | control | 151 | 138 | 1 | 4 | 4 | 0 | 0 | 4 | 13 | 4 | 131 | 1 |
| **FC02** | control | 975 | 936 | 4 | 10 | 4 | 12 | 2 | 7 | 39 | 6 | 764 | 2 |
| **FC03** | control | 405 | 393 | 3 | 1 | 1 | 1 | 1 | 5 | 12 | 6 | 334 | 3 |
| **M01** | dysarthric | 285 | 269 | 1 | 4 | 4 | 3 | 0 | 4 | 16 | 5 | 256 | 2 |
| **M02** | dysarthric | 232 | 217 | 3 | 2 | 3 | 1 | 0 | 6 | 15 | 5 | 207 | 2 |
| **M03** | dysarthric | 406 | 382 | 2 | 7 | 4 | 3 | 0 | 8 | 24 | 5 | 348 | 1 |
| **M04** | dysarthric | 286 | 261 | 3 | 7 | 5 | 3 | 0 | 7 | 25 | 5 | 228 | 2 |
| **M05** | dysarthric | 357 | 338 | 2 | 4 | 2 | 3 | 1 | 7 | 19 | 6 | 313 | 2 |
| **MC01** | control | 419 | 404 | 3 | 4 | 2 | 0 | 1 | 5 | 15 | 5 | 358 | 2 |
| **MC02** | control | 383 | 363 | 4 | 4 | 4 | 3 | 1 | 4 | 20 | 6 | 333 | 2 |
| **MC03** | control | 594 | 573 | 4 | 3 | 4 | 2 | 1 | 7 | 21 | 6 | 475 | 2 |
| **MC04** | control | 993 | 946 | 5 | 10 | 8 | 11 | 2 | 11 | 47 | 6 | 717 | 2 |
| **TOTAL** | — | **6,067** | **5,770** | **40** | **67** | **51** | **43** | **9** | **87** | **297** | — | **940** | — |

---

## 3. Common Emergency Vocabulary Analysis

A critical question for multi-speaker personalized ASR is: **Which emergency intent classes are universally available across speakers, and which are structurally absent?**

The audit reveals the following cross-speaker availability across all 15 speakers:

| Emergency Class | Speakers with $\ge 1$ Sample | Speakers with $\ge 2$ Samples | Speakers with $\ge 3$ Samples | Speakers with $\ge 5$ Samples | Total Corpus Recs | Min / Max per Speaker | Median Support | Feasibility Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CRITICAL_RESPONSE** | **15 / 15** | 15 / 15 | 15 / 15 | 10 / 15 | 87 | 3 / 11 | 5.0 | **Universal / Robust** |
| **FALL** | **15 / 15** | 13 / 15 | 12 / 15 | 4 / 15 | 67 | 1 / 10 | 4.0 | **Universal / Feasible** |
| **BREATHING_MEDICAL** | **15 / 15** | 13 / 15 | 10 / 15 | 2 / 15 | 51 | 1 / 8 | 4.0 | **Universal / Feasible** |
| **SOS_HELP** | **15 / 15** | 11 / 15 | 9 / 15 | 1 / 15 | 40 | 1 / 5 | 3.0 | **Universal / Low Budget** |
| **PAIN** | **11 / 15** | 8 / 15 | 7 / 15 | 2 / 15 | 43 | 0 / 12 | 2.0 | **Partially Absent (4 spks)** |
| **LOCATION** | **7 / 15** | 2 / 15 | 0 / 15 | 0 / 15 | 9 | 0 / 2 | 0.0 | **Severely Deficient (8 spks)** |

### Key Discoveries:
1. **The 4-Class Common Emergency Core:** Four emergency classes (`CRITICAL_RESPONSE`, `FALL`, `BREATHING_MEDICAL`, and `SOS_HELP`) exist in **100% of speakers (15/15)**.
2. **Missing Classes in Reading Stimuli:**
   - `PAIN` has 0 recordings in F01, F03, FC01, and MC01.
   - `LOCATION` has 0 recordings in F01, F03, F04, FC01, M01, M02, M03, and M04 (all female dysarthric and several male dysarthric speakers).
3. **Definitive Conclusion on Direct 7-Class Personalization:** No algorithmic intervention or few-shot resampling can train or test a 7-class personalized classifier for speakers where certain classes physically do not exist in the recorded corpus. A scientifically honest pipeline must evaluate speakers on their **feasible class set** or adopt a **hierarchical binary detector**.

---

## 4. Prompt Grouping & Semantic Leakage Control

### 4.1 Grouping Mechanism
In atypical speech databases, speakers frequently repeat the same sentence stimulus across trials or recording sessions. If one recording of a prompt is placed in adaptation and a second recording of the identical prompt is placed in test, the classifier benefits from **prompt-level acoustic and semantic leakage**.

To eliminate this:
1. Every recording's transcript is normalized to uppercase alphanumeric text (`norm_transcript`).
2. A unique identifier `prompt_group_id` is computed via deterministic cryptographic hash: `PGRP_<SHA256(norm_transcript)[:10]>`.
3. **Atomic Unit Rule:** The prompt group is treated as an indivisible atomic unit. All recordings belonging to a prompt group are assigned strictly to adaptation OR strictly to test.
4. **Zero Split Overlap:** Verified across all 15 speakers at all 4 budgets:
   $$\text{Adaptation Prompt Groups} \cap \text{Test Prompt Groups} = \emptyset$$
   $$\text{Adaptation Recording IDs} \cap \text{Test Recording IDs} = \emptyset$$

### 4.2 Exact Budget Matching via Singletons
Because prompt groups can contain multiple recordings (1 to 5), packing whole prompt groups could potentially cause budget overruns. However, our inventory confirmed that every speaker possesses between **79 and 635 singleton prompt groups** (size = 1) in `GENERAL_SPEECH`. By utilizing singletons to fill exact remaining deficits, our deterministic packing algorithm hits **EXACTLY $N \in \{20, 30, 40, 50\}$ recordings** for every speaker without ever splitting a prompt group.

---

## 5. Side-by-Side Comparison: Original vs Recovery Splits

The table below contrasts the data quality of the original unstratified splits against the recovery stratified splits (Track A) across all 15 speakers:

| Metric | Budget N | Original Random Splits | Recovery Stratified Splits | Net Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Emergency Samples in Adapt** | $N=20$ | 1.1 | **11.3** | **+10.2 samples (+927%)** |
| | $N=30$ | 1.6 | **12.7** | **+11.1 samples (+694%)** |
| | $N=40$ | 2.3 | **13.2** | **+10.9 samples (+474%)** |
| | $N=50$ | 3.1 | **13.2** | **+10.1 samples (+326%)** |
| **Mean Represented Emergency Classes** | $N=20$ | 1.1 / 6.0 | **5.2 / 6.0** | **+4.1 classes** |
| | $N=30$ | 1.4 / 6.0 | **5.2 / 6.0** | **+3.8 classes** |
| | $N=40$ | 1.8 / 6.0 | **5.2 / 6.0** | **+3.4 classes** |
| | $N=50$ | 2.3 / 6.0 | **5.2 / 6.0** | **+2.9 classes** |
| **Zero-Support Emergency Classes** | $N=20$ | 4.9 / 6.0 | **0.8 / 6.0** | **-4.1 missing classes** |
| | $N=30$ | 4.6 / 6.0 | **0.8 / 6.0** | **-3.8 missing classes** |
| | $N=40$ | 4.2 / 6.0 | **0.8 / 6.0** | **-3.4 missing classes** |
| | $N=50$ | 3.7 / 6.0 | **0.8 / 6.0** | **-2.9 missing classes** |
| **Mean Majority Fraction (GENERAL)** | $N=20$ | 94.3% | **43.7%** | Restored balanced few-shot regime |
| | $N=50$ | 93.7% | **73.5%** | Preserved general speech base |

---

## 6. Feasibility Tiers Assigned

For every speaker $\times$ budget condition, a feasibility tier was formally assigned based on empirical class support:
- **TIER A — FULL EMERGENCY COVERAGE (100% of conditions in Recovery Protocol):** All feasible emergency classes physically present in the speaker's corpus are represented in adaptation, with $\ge 1$ instance reserved for held-out test evaluation where corpus size permits.
- **Contrast with Original Benchmark:** In the original Phase 1–6 benchmark, 100% of conditions were classified as **TIER D (EMERGENCY-INFEASIBLE)** or **TIER C (EMERGENCY-SPARSE)**.

---

## 7. Delivery Artifacts Summary

The following formal artifacts were produced and validated:
- `phase6_5/splits/recovery_splits_stratified.json`: 60 records (15 speakers $\times$ 4 budgets) for Track A.
- `phase6_5/splits/recovery_splits_hierarchical_ready.json`: 60 records for Track B.
- `phase6_5/manifests/recovery_adaptation_manifest.csv`: 4,200 total sample assignments.
- `phase6_5/manifests/recovery_test_manifest.csv`: 44,336 total held-out test evaluations.
- `phase6_5/analysis/full_speaker_inventory.csv`: 15 speaker comprehensive profiles.
- `phase6_5/analysis/emergency_class_availability.csv`: Cross-speaker common vocabulary breakdown.
- `phase6_5/analysis/recovery_class_support.csv`: Exact class distribution across all 120 conditions.
- `phase6_5/analysis/speaker_feasibility_matrix.csv`: Feasibility tier assignments.
- `phase6_5/analysis/split_integrity_audit.csv`: Cryptographic hashes and zero-leakage verification.
- `phase6_5/analysis/original_vs_recovery_feasibility.csv`: Quantitative data-quality delta matrix.
- `tests/test_phase6_5_1_splits.py`: 13 automated verification tests (100% passing).
