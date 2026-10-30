# Subphase 6.5.2-C — Effect Attribution Audit
## Critical Methodological Analysis: Disentangling Protocol Effects vs. Causal Attribution

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Database (15 Speakers: 8 Dysarthric, 7 Control)  
**Track:** Phase 6.5 — Emergency Recognition Recovery  
**Subphase:** 6.5.2-C — Classical Rescue Analytical Closure  
**Date:** October 2026  
**Status:** COMPLETE & AUDITED  

---

## 1. Executive Summary & Objective

In earlier documentation for Subphase 6.5.2, the comparison between the historical baseline (Phase 5) and the unweighted models trained on Track A recovery splits was colloquially referred to as the *"Split Recovery Effect"* or interpreted as the split "causing" an improvement in emergency recall.

The objective of this audit is to conduct a **formal methodological correction** regarding effect attribution:
1. **Explain why causal split attribution is confounded:** The historical baseline and the recovery experiment differ not only in their adaptation sets, but simultaneously in their held-out test sets.
2. **Correct scientific terminology:** Replace any claims that the "split caused improvement" with the scientifically accurate formulation: *"The recovery protocol produced higher observed emergency recognition under its corresponding held-out evaluation protocol."*
3. **Investigate the feasibility of a Common Evaluation Set:** Determine whether an untouched common held-out evaluation set can be constructed without data leakage or emergency support extinction.

---

## 2. Why Direct Causal Split Attribution is Confounded

In experimental design, isolating the causal effect of an intervention (here, the adaptation split $\mathcal{A}$) requires that all other experimental variables remain strictly constant:

$$\Delta = \mathbb{E}[f_{\mathcal{A}_2}(\mathcal{T})] - \mathbb{E}[f_{\mathcal{A}_1}(\mathcal{T})]$$

where the evaluation partition $\mathcal{T}$ is invariant.

However, in the comparison between the original Phase 3–5 protocol and the Phase 6.5.2 recovery protocol:

### 2.1 Dual Distribution Shift
1. **The Training / Calibration Distribution Changed ($\mathcal{A}_{\text{orig}} \to \mathcal{A}_{\text{rec}}$):**
   - Original: Unstratified random sampling, resulting in extreme class starvation (average 1.1 to 3.1 emergency samples per speaker).
   - Recovery: Deterministic speaker-stratified sampling with guaranteed prompt-group isolation, ensuring representation for every available emergency class.
2. **The Evaluation / Test Distribution Changed ($\mathcal{T}_{\text{orig}} \to \mathcal{T}_{\text{rec}}$):**
   - In the original protocol, prompt trials were split randomly, which inadvertently distributed utterances from identical prompt groups across both adaptation and test sets (prompt-group leakage).
   - In the recovery protocol, prompt groups were allocated atomically. Recordings containing emergency utterances that were previously part of the test set $\mathcal{T}_{\text{orig}}$ were moved into adaptation $\mathcal{A}_{\text{rec}}$ to ensure few-shot feasibility.
   - Consequently, the recovery test set $\mathcal{T}_{\text{rec}}$ has a different prompt composition, a different size, and different acoustic instances than $\mathcal{T}_{\text{orig}}$.

### 2.2 Confounding Consequence
Because $\mathcal{T}_{\text{orig}} \ne \mathcal{T}_{\text{rec}}$, the observed delta:

$$\Delta_{\text{observed}} = \text{Metric}(f_{\mathcal{A}_{\text{rec}}}, \mathcal{T}_{\text{rec}}) - \text{Metric}(f_{\mathcal{A}_{\text{orig}}}, \mathcal{T}_{\text{orig}})$$

reflects **both** the change in calibration data and the change in test set composition. Therefore, attributing the gain purely to "the split causing improvement" is an overreach. The scientifically valid claim is:

> *"The recovery protocol produced higher observed emergency recognition under its corresponding held-out evaluation protocol than the original baseline achieved under its original evaluation protocol."*

---

## 3. Common-Evaluation Set Feasibility Analysis

To evaluate whether a common evaluation set could resolve this confounding, we conducted an empirical audit on the physical recording IDs across all 15 speakers and all sample budgets ($N \in \{20, 30, 40, 50\}$).

### 3.1 Necessary Conditions for a Valid Common Evaluation Set
For a subset $\mathcal{C}$ to serve as an uncompromised common test set:
1. **Adaptation Disjointness:** $\mathcal{C}$ must contain no sample used during training of either model:
   $$\mathcal{C} \subseteq (\mathcal{T}_{\text{orig}} \cap \mathcal{T}_{\text{rec}}) \setminus (\mathcal{A}_{\text{orig}} \cup \mathcal{A}_{\text{rec}})$$
2. **Prompt-Group Isolation:** No recording in $\mathcal{C}$ may share a prompt group with any recording in $\mathcal{A}_{\text{orig}}$ or $\mathcal{A}_{\text{rec}}$:
   $$\forall r \in \mathcal{C}, \quad \text{PromptGroup}(r) \notin \{\text{PromptGroup}(a) \mid a \in \mathcal{A}_{\text{orig}} \cup \mathcal{A}_{\text{rec}}\}$$
3. **Emergency Support Preservation:** $\mathcal{C}$ must contain non-zero emergency utterances for all speakers to enable computing Emergency Recall and IPER.

### 3.2 Empirical Audit Results
We evaluated candidate common subsets across all 15 speakers for $N \in \{20, 50\}$:

| Speaker ID | Budget $N$ | Raw Overlap $(\mathcal{T}_{\text{orig}} \cap \mathcal{T}_{\text{rec}})$ | Leaked Prompt Recordings | Clean Common Recordings | Clean Emergency Samples | Feasibility Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **F01** | 20 | 55 | 5 | 50 | **1** | Severely Depleted |
| **F01** | 50 | 38 | 1 | 37 | **0** | **EXTINCT (Zero Emergency Samples)** |
| **F03** | 20 | 147 | 4 | 143 | 2 | Constrained |
| **F03** | 50 | 125 | 2 | 123 | 2 | Constrained |
| **F04** | 20 | 184 | 0 | 184 | **1** | Severely Depleted |
| **F04** | 50 | 154 | 2 | 152 | **0** | **EXTINCT (Zero Emergency Samples)** |
| **FC01** | 20 | 88 | 3 | 85 | 4 | Viable |
| **FC01** | 50 | 69 | 1 | 68 | 3 | Viable |
| **FC02** | 20 | 907 | 19 | 888 | 16 | Viable |
| **FC02** | 50 | 881 | 33 | 848 | 7 | Viable |
| **M01** | 50 | 200 | 1 | 199 | 4 | Viable |
| **M04** | 50 | 194 | 12 | 182 | 5 | Viable |
| **MC04** | 50 | 895 | 49 | 846 | 9 | Viable |

### 3.3 Audit Finding: Common Evaluation Set is NOT AVAILABLE
- For critical dysarthric speakers with low total emergency tokens (specifically `F01` and `F04` at $N=50$), enforcing prompt isolation against both $\mathcal{A}_{\text{orig}}$ and $\mathcal{A}_{\text{rec}}$ reduces the number of test emergency samples to **exactly zero**.
- An evaluation set with zero emergency samples cannot compute Emergency Recall, Emergency Precision, or IPER.
- Constructing such a subset would discard significant portions of the corpus and destroy statistical validity for atypical speakers.
- **Conclusion:** A clean, prompt-isolated common evaluation set **CANNOT BE CONSTRUCTED**. The two protocols must be evaluated under their respective held-out test distributions, with confounding explicitly documented.

---

## 4. Corrected Scientific Lexicon & Guidelines

To maintain scientific integrity in all future publications and reports:

| Prohibited / Imprecise Phrasing | Approved / Rigorous Phrasing |
| :--- | :--- |
| "The new split caused a 16% increase in IPER." | "Under the recovery protocol, observed IPER reached 17.35% at N=20, compared to near-zero observed IPER under the original protocol." |
| "Split recovery proves that data was the sole problem." | "Stratified adaptation ensures that models observe emergency classes during calibration, avoiding trivial non-representation." |
| "Class weighting solves the emergency recognition problem." | "Class-balanced optimization successfully reduced majority-class collapse, though false emergency alarms remained elevated." |
| "Reliable emergency recognition achieved." | "Non-zero emergency intent discrimination was recovered, but the high false emergency rate indicates that flat 7-class classification is not deployment-ready." |

---

## 5. Audit Sign-Off

- **Confounding Verified:** Dual distribution shift between training and test sets is formally documented.
- **Common Evaluation Set Status:** Formally audited and declared **NOT AVAILABLE** due to emergency support extinction for scarce speakers.
- **Reporting Nomenclature:** All documentation updated to use conservative, non-causal attribution language.

**Audit Status: APPROVED & CLOSED.**
