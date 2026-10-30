# PHASE 8.1 METHODS AND SCIENTIFIC LIMITATIONS

**Document:** `research/phase8_1/PHASE8_1_METHODS_AND_LIMITATIONS.md`  
**Phase:** Phase 8.1 Safety Operating-Point Recovery  
**Purpose:** Formal scientific documentation of methodology, experimental design, and empirical limitations.

---

## 1. Methodology Specification

### 1.1 Development Pool Boundary
- **Training Pool:** Strictly the 80% training partition from `results/phase8_80_20/split_manifest.csv` containing **4,852 active utterances** across 15 speakers.
- **Test Set Quarantine:** The 20% locked test set (1,215 utterances) was inspected during Phase 8 and cannot serve as an unbiased confirmatory test set for new model or threshold selection. It is quarantined as **diagnostic-only**.

### 1.2 Nested Grouped Stratified Cross-Validation Design
To ensure that threshold calibration and model selection are completely isolated from generalization assessment:
1. **Outer Loop (5 Folds):**
   - The 4,852 training samples are partitioned into 5 outer folds $O_1, \dots, O_5$, stratified by speaker and binary emergency intent.
   - For each outer fold $k \in \{1, \dots, 5\}$:
     - Outer Training Pool: $\mathcal{D}_{\text{outer-train}} = \bigcup_{j \ne k} O_j$ (~3,881 samples).
     - Outer Evaluation Pool: $\mathcal{D}_{\text{outer-eval}} = O_k$ (~971 samples).
2. **Inner Loop (5 Folds within Outer Training Pool):**
   - $\mathcal{D}_{\text{outer-train}}$ is split into 5 inner folds $I_1, \dots, I_5$.
   - Models are fit on $I_{-m}$, and predictions are generated on held-out inner fold $I_m$.
   - Aggregated inner-OOF predictions are used to select operating threshold $\tau^*$:
     $$\tau^* = \arg\max_{\tau} \left\{ \text{Recall}_{\text{inner}}(\tau) \;\middle|\; \text{FER}_{\text{inner}}(\tau) \le 0.15 \right\}$$
3. **Outer Fold Generalization Assessment:**
   - The selected threshold $\tau^*$ and model fitted on all $\mathcal{D}_{\text{outer-train}}$ are evaluated on the outer held-out fold $O_k$.
   - This measures the true generalization gap between inner-validation selection and unseen outer evaluation.

### 1.3 Strict Grouping Policy
All prompt-level recording IDs are verified. Since all 6,067 utterances represent distinct acoustic events, prompt grouping preserves sample independence.

---

## 2. Scientific Limitations

1. **Severe Imbalance in Dysarthric Speech:**
   - Only 237 of 4,852 training samples are emergencies (4.88%).
   - In individual outer folds, only ~47 emergency samples exist across all 15 speakers (~3.1 emergencies per speaker per fold).
2. **Acoustic Overlap in Severe Dysarthria:**
   - Dysarthric speech distortions (slurred articulation, reduced vocal intensity, breathiness) mimic the acoustic features of physical distress and emergency vocalizations.
   - This acoustic confusion limits the separability achievable with general speech embeddings (e.g. Whisper-Base).
3. **Absence of Independent Confirmation Data:**
   - The TORGO corpus is exhausted. No independent confirmatory dataset of atypical emergency communication exists in this benchmark.
   - Any safety claim remains provisional to internal cross-validation; formal deployment safety requires newly recorded, consented confirmatory data.
