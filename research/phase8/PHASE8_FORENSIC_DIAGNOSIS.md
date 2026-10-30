# Phase 8 Forensic Diagnosis & Root-Cause Audit

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Document:** Phase 8 Forensic Diagnosis (`PHASE8_FORENSIC_DIAGNOSIS.md`)  
**Status:** Audit Verified & Programmatically Replicated  

---

## 1. Executive Summary of Forensic Finding

In the original Phase 8 confirmatory benchmark (`results/phase8/phase8_results.csv`), all three tracks:
- **Track A** (Hierarchical Classical Baseline)
- **Track B** (Quantum Residual Module)
- **Track C** (Matched Classical Shadow Control)

exhibited mathematically identical metrics across:
- **Emergency Recall:** 46.8347% (diff = 0.000000)
- **Emergency Precision:** 1.6041% (diff = 0.000000)
- **False Emergency Rate (FER):** 52.5252% (diff = 0.000000)
- **Dysarthric Emergency Recall:** 38.0903% (diff = 0.000000)
- **Control Emergency Recall:** 56.8282% (diff = 0.000000)

Yet **Macro-F1** differed across tracks (Track A = 8.17%, Track B = 8.10%, Track C = 8.23%).

Programmatic hashing and inspection of `src/phase8/phase8_models.py` definitively proves the root cause:
1. **Hard Shared Binary Gate:** All three models routed inferences through an identical Stage 1 binary gate (`gate_probs[:, 1] >= 0.5`).
2. **Subtype-Only Branching:** Track B (Quantum VQC) and Track C (Matched Shadow MLP) were invoked *only* to select which of the 6 emergency sub-classes (`CRITICAL_RESPONSE`, `FALL`, etc.) to predict.
3. **Metric Calculation Plumbing:** `compute_phase8_metrics()` defines `is_pred_emergency = (pred in EMERGENCY_CLASSES)`. Because any sample flagged by the Stage-1 gate was assigned one of the 6 emergency classes, `is_pred_emergency` was 100% identical to the Stage-1 gate output.
4. **Zero Quantum Emergency Participation:** The quantum residual circuit had **zero mathematical participation** in the primary emergency vs. non-emergency decision.

---

## 2. Programmatic Verification & Hash Analysis

Evaluation on TORGO speaker `F01` ($N=20$) confirmed:

```
Binary Emergency Prediction Hash A: 0489fbcfdbe9ca5c
Binary Emergency Prediction Hash B: 0489fbcfdbe9ca5c
Binary Emergency Prediction Hash C: 0489fbcfdbe9ca5c
Binary Emergency Equal (A == B):    True
Binary Emergency Equal (B == C):    True
Binary Emergency Equal (A == C):    True

7-Class Final Prediction Hash A:    071aa542b6d4c0c9
7-Class Final Prediction Hash B:    0eb9ef4b7301707c
7-Class Final Prediction Hash C:    00ae9a3335aed2ce
7-Class Equal (A == B):            False
7-Class Equal (B == C):            False

Gate Binary Equal (gateA == binA):  True
Gate Binary Equal (gateB == binB):  True
Gate Binary Equal (gateC == binC):  True
```

---

## 3. Explicit Answers to Forensic Questions

- **Q1: Which prediction vector is used for Emergency Recall?**  
  `is_pred_emergency = np.array([label in EMERGENCY_CLASSES for label in y_pred])`. In the original models, this was strictly the binary Stage-1 gate thresholding output (`gate_probs[:, 1] >= threshold`).
- **Q2: Which prediction vector is used for Emergency Precision?**  
  The same binary indicator `is_pred_emergency`.
- **Q3: Which prediction vector is used for False Emergency Rate?**  
  The same binary indicator `is_pred_emergency`.
- **Q4: Which prediction vector is used for dysarthric Emergency Recall?**  
  The same binary indicator `is_pred_emergency` restricted to dysarthric speaker rows.
- **Q5: Does Track B's quantum residual affect the FINAL emergency decision?**  
  **NO.** In the original Phase 8 architecture, the quantum circuit only mapped emergency candidates to 1 of 6 subtypes; it had zero influence on whether an utterance was classified as an emergency or general speech.
- **Q6: Does Track C's matched shadow affect the FINAL emergency decision?**  
  **NO.** Track C also operated solely as a subtype classifier downstream of the gate.
- **Q7: Are Track A/B/C using the same Stage-1 output?**  
  **YES.** All three used the identical `BinaryEmergencyDetector` with identical random seed and training parameters.
- **Q8: Are Track A/B/C using the same final emergency prediction?**  
  **YES.** Exactly identical binary predictions across all 60 cells.
- **Q9: Are the metric-generation functions receiving track-specific predictions?**  
  The metric functions received `y_pred` (the 7-class string array), but because binary emergency metrics collapse all 6 emergency subtypes into a single boolean `pred != 'GENERAL_SPEECH'`, the subtype differences were erased in binary metrics.
- **Q10: Are plots reading the correct track-specific metric columns?**  
  Yes, but the underlying CSV columns were identical because the models produced identical binary predictions.
- **Q11: Are per-budget trajectories using track-specific predictions?**  
  Yes, but all tracks plotted identical values due to the shared gate decision.
- **Q12: Are subgroup plots using track-specific predictions?**  
  Yes, but identical for the same reason.
- **Q13: Does the result CSV contain track-specific prediction identifiers?**  
  The CSV contains track labels (`Track_A_Classical`, `Track_B_Quantum`, `Track_C_Shadow`), but the values for emergency recall, precision, and FER were identical across tracks.

---

## 4. Preservation of Baseline Phase 8

The original Phase 8 results remain preserved in:
- `results/phase8/phase8_results.csv`
- `results/phase8/phase8_summary.json`
- `results/phase8/phase8_statistical_tests.json`
- `results/phase8/figures/*.png`
- `reports/phase8/PHASE8_RECOVERY_AND_CONFIRMATORY_REPORT.md`

They serve as the historical record of the forensic baseline. The repaired run will execute in the isolated namespace:
- `config/phase8_repaired/`
- `src/phase8_repaired/`
- `results/phase8_repaired/`
- `reports/phase8_repaired/`
- `tests/test_phase8_repaired.py`
