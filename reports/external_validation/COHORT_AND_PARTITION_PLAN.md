# Independent Cohort Partition Plan and Anti-Leakage Protocol
## Disjoint Partition Architecture, Speaker Stratification, and Evaluation Safeguards

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Document Identifier:** SPEC-COHORT-PARTITION-2026-V1  
**Status:** **ACTIVE PLANNING PROTOCOL — NO CONFIRMATORY EVALUATION AUTHORIZED**  
**Date:** 2026-10-11  
**Authors:** Lead Biostatistician, Senior Speech Machine Learning Engineer, Research Reproducibility Auditor  

---

> [!CAUTION]
> **EVALUATION DISCIPLINE REMINDER:**  
> This partition plan specifies the mathematical architecture for splitting future acquired external atypical speech cohorts. **TORGO SPEECH DATA ARE CONTAMINATED AND EXCLUDED FROM THIS PARTITION PLAN.** The locked confirmatory partition must remain sealed until the evaluation protocol is publicly preregistered and explicit owner authorization is granted.

---

## 1. Study Population & Primary Estimands

### 1.1 Target Clinical Population
Adult individuals (aged 18+) with clinically documented motor speech disorders (dysarthria secondary to cerebral palsy, amyotrophic lateral sclerosis, stroke, Parkinson's disease, or traumatic brain injury) presenting with mild, moderate, or severe speech intelligibility impairment.

### 1.2 Target Estimands
1. **Primary Estimand (Population-Average Emergency Sensitivity $\theta_{recall}$):**
   The marginal probability that an emergency intent utterance produced by an individual with dysarthria is correctly recognized as an emergency:
   $$\theta_{recall} = E\left[ \frac{\sum_{j=1}^{M_i} Y_{ij} \cdot \hat{Y}_{ij}(\tau)}{\sum_{j=1}^{M_i} Y_{ij}} \right]$$
   where $Y_{ij} \in \{0, 1\}$ is the true binary emergency label for utterance $j$ of participant $i$, and $\hat{Y}_{ij}(\tau) = \mathbb{I}(\text{Score}_{emergency} \ge \tau)$ is the classification decision under the frozen threshold $\tau$.
2. **Secondary Estimand (Population-Average False Emergency Rate $\theta_{fer}$):**
   The marginal rate at which non-emergency conversational utterances are misclassified as emergencies:
   $$\theta_{fer} = E\left[ \frac{\sum_{j=1}^{M_i} (1 - Y_{ij}) \cdot \hat{Y}_{ij}(\tau)}{\sum_{j=1}^{M_i} (1 - Y_{ij})} \right]$$
3. **Tertiary Estimand (General-Speech Preservation Rate $\theta_{gspr}$):**
   $$\theta_{gspr} = 1.0 - \theta_{fer} = E\left[ \frac{\sum_{j=1}^{M_i} (1 - Y_{ij}) \cdot (1 - \hat{Y}_{ij}(\tau))}{\sum_{j=1}^{M_i} (1 - Y_{ij})} \right]$$
4. **Speaker-Level Worst-Case Estimand ($\theta_{fer, max}$):**
   $$\theta_{fer, max} = \max_{i \in \{1..K\}} \left( \frac{\sum_{j=1}^{M_i} (1 - Y_{ij}) \cdot \hat{Y}_{ij}(\tau)}{\sum_{j=1}^{M_i} (1 - Y_{ij})} \right)$$
   *Safety Rule:* The system fails the secondary safety endpoint if $\theta_{fer, max} > 20.0\%$, regardless of the population-average mean.

---

## 2. Disjoint Partition Architecture

To maintain strict methodological discipline and prevent the catastrophic threshold-tuning contamination that compromised earlier research phases, any acquired external cohort must be partitioned **exclusively at the participant/speaker level**:

```mermaid
flowchart TD
    ACQ[Acquired External Cohort: K Speakers, N Utterances] --> STRAT[Stratified Clustered Randomization<br>Stratified by Etiology, Gender, Intelligibility]
    STRAT --> DEV[Development / Calibration Split: 30% of Speakers<br>K_dev = ceil 0.30 * K]
    STRAT --> CONF[Locked Confirmatory Split: 70% of Speakers<br>K_conf = K - K_dev]
    
    subgraph Development Domain [Development Domain (Open to Research Team)]
        DEV --> DEV_TUNE[Derive & Calibrate tau on Dev Split Only]
        DEV_TUNE --> DEV_CALIB[Compute ECE & Brier Calibration Score]
        DEV_CALIB --> PROTO_LOCK[Cryptographically Freeze Protocol & tau]
    end
    
    subgraph Confirmatory Domain [Confirmatory Domain (Single-Pass Sealed Vault)]
        PROTO_LOCK --> RUNNER[Guarded Confirmatory Runner: src/evaluation/confirmatory_runner.py]
        CONF --> RUNNER
        RUNNER --> EVAL_LOCK[Generate Permanent Run Lock & Audit Trail]
        EVAL_LOCK --> FINAL_REP[Unredacted Final Statistical Report]
    end
```

### 2.1 Partition Allocation Rules
1. **Development / Calibration Split ($30\%$ of Speakers):**
   - **Purpose:** Derivation and empirical calibration of the operational decision threshold $\tau$; temperature scaling parameter optimization; feature pipeline stability verification.
   - **Access:** Accessible to ML research and engineering team for development and threshold derivation.
   - **Prohibition:** Candidate models must NOT be selected based on test-set metrics; threshold derivation must use nested out-of-fold cross-validation within this dev split.
2. **Confirmatory Split ($70\%$ of Speakers):**
   - **Purpose:** Definitive, single-pass evaluation of the primary and secondary statistical endpoints.
   - **Access:** Completely sealed and encrypted until:
     - Development calibration is finalized and frozen.
     - Preregistration is timestamped on Open Science Framework.
     - Explicit project owner token is supplied.
   - **Prohibition:** Zero exploratory evaluation, zero threshold re-tuning, zero post-hoc debugging.

---

## 3. Stratification and Subgroup Balance Protocol

Randomization of participants between the Development ($30\%$) and Confirmatory ($70\%$) partitions must use **Stratified Clustered Sampling** across three primary clinical strata:
1. **Dysarthria Severity Strata:**
   - Low Intelligibility ($< 40\%$ word intelligibility)
   - Moderate Intelligibility ($40\% - 70\%$)
   - High Intelligibility ($> 70\%$)
2. **Etiology Strata:**
   - Congenital / Non-Progressive (Cerebral Palsy)
   - Progressive / Neurodegenerative (ALS, Parkinson's Disease)
   - Acquired / Vascular (Stroke, TBI)
3. **Biological Sex Strata:**
   - Balanced representation across male and female participants (target: $45\% - 55\%$ female).

---

## 4. Acoustic, Identity, and Prompt Leakage Prevention

| Risk Domain | Specific Failure Mode | Preventive Engineering & Governance Control |
| :--- | :--- | :--- |
| **Speaker Leakage** | Same participant appears in both Dev and Confirmatory partitions. | Partition ledger strictly generated by unique participant ID hash; verified by `ExternalDatasetValidator.partition_leakage_detected`. |
| **TORGO Contamination** | Participant previously recorded in TORGO appears under new ID. | Automated acoustic fingerprinting (Chromaprint / Dejavu) and self-supervised embedding distance ($L_2 < 0.15$) against all 15 TORGO speakers. |
| **Prompt Group Leakage** | Identical phrase prompts over-represented in one split. | Stratified allocation of standard prompt lists across splits to prevent lexical memorization. |
| **Session / Temporal Leakage** | Earlier recordings of a speaker used for testing and later for tuning. | Chronological sorting enforced; adaptation samples must strictly precede evaluation samples. |
| **Repeated Confirmatory Runs** | Running multiple evaluation passes to pick best outcome. | Single-pass cryptographic run lock written to disk; re-evaluation blocked with `SINGLE_PASS_VIOLATION`. |

---

## 5. Pre-Specified Handling of Audio Quality, Missing Labels, and Abstention

1. **Acoustic Quality Degradation:**
   - Any audio utterance recording a Signal-to-Noise Ratio $\text{SNR} < 5.0\text{ dB}$ or Speech Quality Metric $< 0.30$ triggers the **Signal Quality Guardrail**.
   - The system outputs `suggested_action = "ABSTAIN"` and `predicted_intent = "UNKNOWN"`.
   - *Intention-to-Treat Rule:* Abstained emergency utterances are treated as **false negatives (missed emergencies)** in the primary sensitivity calculation, preventing artificial score inflation through selective filtering.
2. **Missing Transcripts / Disputed Intent Annotations:**
   - If two clinical SLP annotators disagree on an emergency label and cannot reach consensus, the utterance is classified as **ambiguous** and quarantined into an exploratory subgroup.
   - Primary confirmatory endpoints are evaluated strictly on the consensus adjudicated set ($\kappa \ge 0.80$).
3. **Short / Incomplete Vocalizations:**
   - Vocalizations shorter than 300ms or containing no detectable speech activity are flagged as invalid audio and reported in the participant attrition ledger.

---

## 6. Predefined Stopping Boundaries and Go / No-Go Criteria

Prior to unlocking the confirmatory partition, the project adopts immutable stopping boundaries:

| Evaluation Metric | Minimum Acceptable Threshold | Target Benchmark | Formal Stop Decision upon Breach |
| :--- | :---: | :---: | :--- |
| **Emergency Recall ($\theta_{recall}$)** | $\ge 60.0\%$ | $\ge 70.0\%$ | **NO-GO:** Emergency recognition infeasible; reject deployment. |
| **False Emergency Rate ($\theta_{fer}$)** | $\le 15.0\%$ | $\le 10.0\%$ | **NO-GO:** False alarm burden exceeds research safety ceiling. |
| **Max Speaker FER ($\theta_{fer, max}$)** | $\le 20.0\%$ | $\le 15.0\%$ | **SAFETY VETO:** Severe speaker disparity; adaptation failed. |
| **General Speech Preservation** | $\ge 85.0\%$ | $\ge 90.0\%$ | **NO-GO:** Excessive disruption to routine communication. |
| **Acoustic Abstention Rate** | $\le 10.0\%$ | $\le 5.0\%$ | **TECHNICAL HOLD:** Audio pipeline unsuited to clinical acoustics. |
