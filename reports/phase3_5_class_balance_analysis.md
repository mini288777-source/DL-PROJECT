# Class Balance Analysis & Single-Class Conditions Audit

**Date:** 2026-09-30  
**Phase:** 3.5 — Research Validity, Consistency & Phase 4 Readiness Gate  
**Target File:** `reports/phase3_5_class_balance_analysis.md`  
**Companion Data File:** `reports/phase3_5_class_balance_analysis.csv` (254 rows)  

---

## 1. Executive Summary

This report analyzes the label distribution across the 15-speaker TORGO few-shot cohort. TORGO was recorded primarily as an acoustic and dysarthric speech corpus featuring reading passages and word lists, rather than an emergency-first dispatch dataset. Consequently, mapping TORGO transcripts to an emergency intent taxonomy results in extreme class imbalance dominated by `GENERAL_SPEECH`.

* **Overall Cohort Size:** 6,067 audio recordings across 15 speakers.
* **Dominant Majority Class (`GENERAL_SPEECH`):** 5,770 recordings (**95.10%** of all data).
* **Emergency Intent Classes (6 classes combined):** 297 recordings (**4.90%** of all data).
* **Single-Class Adaptation Conditions:** 7 out of 60 conditions (**11.7%**) contain only `GENERAL_SPEECH` in their adaptation training set.
* **Held-Out Test Sets:** All 15 speakers possess emergency intents in their held-out test sets, but the test prevalence of `GENERAL_SPEECH` remains between 92.5% and 97.4%.

---

## 2. Cohort-Wide Intent Distribution

The exact frequency breakdown across all 6,067 unique cohort recordings is:

| Intent Category | Semantic Role | Total Count | Cohort % |
|---|---|---|---|
| **GENERAL_SPEECH** | Standard reading vocabulary / non-emergency | 5,770 | 95.10% |
| **CRITICAL_RESPONSE** | Time-critical commands ("yes", "no", "stop", "go", "wait") | 87 | 1.43% |
| **FALL** | Slip, trip, fall, or collapse indicators | 67 | 1.10% |
| **BREATHING_MEDICAL**| Respiratory distress, medication, cough, air | 51 | 0.84% |
| **PAIN** | Acute pain, injury, ache, somatic distress | 43 | 0.71% |
| **SOS_HELP** | Explicit distress and emergency calls ("help", "trouble") | 40 | 0.66% |
| **LOCATION** | Room, bed, door location indicators | 9 | 0.15% |
| **TOTAL** | — | **6,067** | **100.00%** |

```
Distribution Visualization:
GENERAL_SPEECH      [==================================================] 95.10%
CRITICAL_RESPONSE   [=] 1.43%
FALL                [=] 1.10%
BREATHING_MEDICAL   [ ] 0.84%
PAIN                [ ] 0.71%
SOS_HELP            [ ] 0.66%
LOCATION            [ ] 0.15%
```

---

## 3. Analysis of Single-Class Adaptation Conditions

In a few-shot setting where $N \in \{20, 30, 40, 50\}$ samples are randomly drawn from a speaker's available recordings according to the Phase 1 protocol, the probability of drawing zero emergency intent samples is strictly non-zero for speakers with low overall prompt counts.

From `reports/phase3_5_sample_budget_audit.csv`, exactly **7 out of 60 adaptation conditions (11.7%)** exhibit single-class training (`unique_intent_count == 1`):

| Speaker | Sample Budget ($N$) | Adaptation Composition | Test Composition | `single_class_training` | Fallback Handler Executed |
|---|---|---|---|---|---|
| **F04** | $N=20$ | `GENERAL_SPEECH`: 20 (100%) | `GENERAL`: 211, `EMERGENCY`: 9 | **TRUE** | Single-class prior fallback |
| **FC03**| $N=20$ | `GENERAL_SPEECH`: 20 (100%) | `GENERAL`: 473, `EMERGENCY`: 17 | **TRUE** | Single-class prior fallback |
| **M05** | $N=20$ | `GENERAL_SPEECH`: 20 (100%) | `GENERAL`: 449, `EMERGENCY`: 30 | **TRUE** | Single-class prior fallback |
| **M05** | $N=30$ | `GENERAL_SPEECH`: 30 (100%) | `GENERAL`: 449, `EMERGENCY`: 30 | **TRUE** | Single-class prior fallback |
| **M05** | $N=40$ | `GENERAL_SPEECH`: 40 (100%) | `GENERAL`: 449, `EMERGENCY`: 30 | **TRUE** | Single-class prior fallback |
| **MC01**| $N=20$ | `GENERAL_SPEECH`: 20 (100%) | `GENERAL`: 589, `EMERGENCY`: 19 | **TRUE** | Single-class prior fallback |
| **MC01**| $N=30$ | `GENERAL_SPEECH`: 30 (100%) | `GENERAL`: 589, `EMERGENCY`: 19 | **TRUE** | Single-class prior fallback |

### Evolution with Sample Budget ($N$):
* At $N=20$: 4 speakers have single-class adaptation sets (26.7%).
* At $N=30$: 2 speakers have single-class adaptation sets (13.3%).
* At $N=40$: 1 speaker has single-class adaptation sets (6.7%).
* At $N=50$: **0 speakers have single-class adaptation sets (0.0%)**. At $N=50$, 100% of speakers have multi-class adaptation sets.

This behavior reflects the natural sample complexity of atypical speech data collection: under low sample budgets ($N=20$), small emergency classes are frequently unobserved.

---

## 4. Evaluation of Single-Class Conditions: Research Recommendation

The user prompt mandates an explicit research recommendation on how to treat these 7 single-class conditions:

### Options Considered:
1. **EXCLUDE WITH JUSTIFICATION:** Drop these 7 conditions from the dataset.
   * *Critique:* Artificially alters sample size across budgets ($N=20$ would have 11 speakers while $N=50$ would have 15), violating matched-cohort sample-efficiency design.
2. **ARTIFICIALLY OVERSAMPLE / REBALANCE:** Force emergency intents into $N=20$ adaptation sets.
   * *Critique:* Violates the Phase 1 split freeze and introduces synthetic sampling bias.
3. **RETAIN IN PRIMARY BENCHMARK WITH EXPLICIT MARKING (RECOMMENDED):**
   * Maintain the full 15-speaker cohort ($N=20, 30, 40, 50$, 60 conditions).
   * Mark `single_class_training = TRUE` in the benchmark metadata.
   * Include them in primary cohort reporting to reflect realistic cold-start few-shot personalization where the user may not have provided emergency utterances during initial enrollment.
   * Provide a secondary stratified ablation reporting metrics on the $53$ multi-class conditions.

### Official Research Recommendation: **PRIMARY (WITH EXPLICIT METADATA FLAG)**

* **Scientific Justification:** In realistic assistive speech deployment, a dysarthric patient enrolling in a system with only 20 speech samples cannot be guaranteed to have uttered an emergency command during calibration. The benchmark must test how adaptation algorithms behave under this exact real-world circumstance.
* **Downstream Phase 4 Requirement:** In Phase 4, the Quantum Kernel SVM must execute the exact same fallback or soft-margin formulation on these 7 conditions as the classical baselines, ensuring matched comparability.
