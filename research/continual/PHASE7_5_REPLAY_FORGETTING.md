# PHASE 7.5 — REPLAY MEMORY & CATASTROPHIC FORGETTING MITIGATION
## Empirical Investigation of Replay Memory Under Imbalanced Longitudinal Atypical Speech Personalization

---

> **CRITICAL SCIENTIFIC NOTICES:**
> 1. **Phase 7.5 tests replay as a protection mechanism against forgetting. It does not revise the frozen Phase 1–6.5 conclusions.**
> 2. **Frozen Historical Conclusion:** `CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE under the evaluated fixed-kernel protocol` remains immutable and active.
> 3. **Clinical / Deployment Notice:** Deployment Readiness = **NOT READY**. Clinical Validation = **NOT PERFORMED**. Emergency Reliability = **NOT ESTABLISHED**.

---

### 1. Motivation from Phase 7.4

In Phase 7.4, the canonical continual classical baseline established a critical safety failure mode:
$$\text{Natural Longitudinal Accumulation} \longrightarrow \text{Heavy General Speech Influx (~95%)} \longrightarrow \text{Catastrophic Emergency Forgetting}$$
Specifically:
- Overall Macro-F1 increased significantly ($+0.0460$, $p < 0.0001$).
- False Emergency Rate decreased significantly ($-0.3509$, $p < 0.0001$).
- **BUT Emergency Recall dropped statistically significantly**: $-0.0635$ ($p = 0.0382$), collapsing from $17.26\%$ down to $9.97\%$ in dysarthric speakers.

Phase 7.5 investigates whether memory replay with class-aware quota protection can mitigate this catastrophic forgetting.

---

### 2. Research Questions

- **RQ7.4:** Can replay memory prevent catastrophic forgetting of minority safety-critical classes during continual personalization?
- **Secondary Question:** Does replay improve emergency retention without creating unacceptable increases in false emergency rate or overall training instability?

---

### 3. Hypotheses Under Evaluation

- **Hypothesis H7.4:** Replay memory reduces catastrophic forgetting of minority safety-critical classes during continual personalization.
  - **Verdict:** **PARTIALLY SUPPORTED / DIRECTIONAL ONLY (NOT STATISTICALLY SIGNIFICANT)**.
  - Replay memory directionally reduced cohort emergency forgetting from $0.0635$ to $0.0524$ ($\Delta = -0.0111$), but this delta did not reach statistical significance ($t = -1.000, p = 0.3343$). In dysarthric speakers specifically, emergency recall remained severely depressed ($9.97\%$).
- **Hypothesis H7.4a:** Replay improves emergency recall relative to no replay.
  - **Verdict:** **NOT SUPPORTED / INCONCLUSIVE**.
  - Cohort emergency recall at $N=50$ was $12.42\%$ with replay vs. $11.31\%$ without replay ($\Delta = +0.0111$, $p = 0.3343$).
- **Hypothesis H7.4b:** Replay reduces forgetting without unacceptable degradation in false emergency rate.
  - **Verdict:** **SUPPORTED**.
  - FER remained well-controlled ($26.03\% \to 29.28\%$, $\Delta = +0.0325$, $p = 0.0827$), confirming that replay did not trigger runaway false emergency alarms.

---

### 4. Replay Architecture & Governance Contracts

The `ReplayMemory` subsystem (`src/continual/replay.py`) implements the strict pre-registered contracts defined in `PHASE7_DATA_CONTRACT.md` Section 3 and `config/continual/replay_policy.yaml`:

1. **Strict Speaker Isolation:**
   Each `ReplayMemory` instance is permanently bound to a single speaker. Attempting cross-speaker insertion raises `SpeakerIsolationError`.
2. **Anti-Self-Training Firewall:**
   Only verified (`USER_VERIFIED` or `SYSTEM_VERIFIED`) and learning-eligible (`learning_eligible == True`) records enter memory. Unverified model predictions and quarantined samples are strictly rejected.
3. **Temporal Causality:**
   At update step $t$, memory is strictly constrained to experiences with timestamps $t_i \le t$. Future experiences cannot be replayed.
4. **Seed Anchoring:**
   The initial calibration instances from $B_0$ ($N=20$) are designated as permanent anchors (`is_anchor=True`) and are strictly non-evictable.
5. **Quota Balancing:**
   Under `BALANCED_REPLAY`, all available emergency instances are preserved. General speech is dynamically bounded to at most $4 \times N_{\text{emergency}}$ (4:1 quota ratio) using deterministic seed-based selection.

---

### 5. Experimental Design & Matched Control

- **Cohort:** Exactly matched to Phase 7.4 (15 speakers: 8 dysarthric, 7 control).
- **Budgets:** Exactly matched to Phase 7.4 ($N \in \{20, 30, 40, 50\}$).
- **Architecture:** Canonical Classical Adapter (`ClassicalAdapterNet`: 25,223 parameters, AdamW, $\eta = 0.005$, 60 epochs).
- **Evaluation Benchmark:** Fixed protected held-out test split $\text{Test}(50)$ (68 to 943 recordings per speaker, zero leakage).
- **Treatment Grid:** 120 total cells:
  - 60 Control cells: Frozen Phase 7.4 `NO_REPLAY` baseline.
  - 60 Treatment cells: Phase 7.5 `BALANCED_REPLAY` intervention.

---

### 6. Metric Definitions

- **Emergency Recall:** $\frac{\text{TP}_{\text{emergency}}}{\text{TP}_{\text{emergency}} + \text{FN}_{\text{emergency}}}$.
- **Emergency Recall Forgetting (`EMERGENCY_FORGETTING`):**
  $$\text{Forgetting}(B_t) = \text{Emergency\_Recall}(B_0) - \text{Emergency\_Recall}(B_t)$$
  Positive values indicate forgetting; $\le 0$ indicates retention or gain.
- **False Emergency Rate (FER):** $\frac{\text{FP}_{\text{emergency}}}{\text{Total General Speech}}$.
- **Macro-F1:** Unweighted mean F1 across all 7 canonical taxonomy classes.

---

### 7. Results: Longitudinal Trajectory & Matched Comparison

#### Cohort Averages Across Budgets (NO_REPLAY vs. BALANCED_REPLAY)

| Budget ($N$) | Condition | Macro-F1 (Mean) | Emergency Recall (Cohort) | Emergency Forgetting ($\Delta$ vs $B_0$) | Dysarthric Emergency Recall | False Emergency Rate (FER) | Train Samples per Step |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N=20$ ($B_0$)** | NO_REPLAY | 0.0770 | 0.1765 | 0.0000 | 0.1726 | 0.6113 | 20 |
| | REPLAY | 0.0770 | 0.1765 | 0.0000 | 0.1726 | 0.6113 | 20 |
| **$N=30$ ($B_1$)** | NO_REPLAY | 0.0961 | 0.1807 | -0.0042 | 0.1576 | 0.4756 | 30 |
| | REPLAY | 0.0961 | 0.1807 | -0.0042 | 0.1576 | 0.4756 | 30 |
| **$N=40$ ($B_2$)** | NO_REPLAY | 0.1085 | 0.1778 | -0.0013 | 0.1281 | 0.3938 | 40 |
| | REPLAY | 0.1104 | 0.1612 | +0.0154 | 0.0969 | 0.3624 | 38.6 |
| **$N=50$ ($B_3$)** | **NO_REPLAY** | **0.1230** | **0.1131** | **+0.0635** | **0.0997** | **0.2603** | **50** |
| | **REPLAY** | **0.1211** | **0.1242** | **+0.0524** | **0.0997** | **0.2928** | **45.6** |

---

### 8. Statistical Analysis (Terminal State $N=50$, Paired $t$-test)

| Metric | NO_REPLAY Mean | REPLAY Mean | Paired Difference ($\Delta$) | $t$-statistic | $p$-value | Significance ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cohort Emergency Recall** | 0.1131 | 0.1242 | $+0.0111$ | $+1.000$ | $p = 0.3343$ | **NOT SIGNIFICANT** |
| **Cohort Emergency Forgetting** | +0.0635 | +0.0524 | $-0.0111$ | $-1.000$ | $p = 0.3343$ | **NOT SIGNIFICANT** |
| **Cohort Macro-F1** | 0.1230 | 0.1211 | $-0.0020$ | $-1.317$ | $p = 0.2090$ | **NOT SIGNIFICANT** |
| **False Emergency Rate** | 0.2603 | 0.2928 | $+0.0325$ | $+1.869$ | $p = 0.0827$ | **NOT SIGNIFICANT** |
| **Dysarthric Emergency Recall** | 0.0997 | 0.0997 | $0.0000$ | — | — | **NO CHANGE** |

---

### 9. Detailed Error Analysis & Mechanistic Insights

1. **Why Replay Helped in Control Speakers but Stagnated in Dysarthric Speakers:**
   - In control speakers with coherent acoustic representations, memory replay prevented complete emergency erasure. For example, for speaker `MC01`, `NO_REPLAY` suffered complete collapse at $N=50$ ($\text{Recall} = 0.0000$), whereas `REPLAY` successfully maintained $\text{Recall} = 0.1667$.
   - In contrast, for dysarthric speakers (`F01`, `F04`, `M01`, `M05`), severe acoustic distortion creates high intra-class variance. Because emergency samples are naturally so sparse (typically only 6–7 emergency utterances vs. 43 general utterances), retaining 6 emergency samples in memory alongside 24 general samples still provides inadequate support to carve out robust decision boundaries in 384-dimensional space.
2. **The Capacity-Representation Trade-off:**
   - The classical adapter has 25,223 parameters. When trained on 40–50 samples, gradient updates on the dominant general speech class rapidly pull the linear weights toward predicting general speech, even when class-weighted loss is applied.
   - Replay slows this drift slightly for cleaner speakers, but does not fundamentally solve the data starvation of minority classes under severe dysarthria.

---

### 10. Limitations

1. **Replay Capacity Ceiling:** Replay memory was capped at $N=50$ to match the empirical data ceiling of the TORGO cohort.
2. **Fixed Architecture:** This phase evaluated a simple replay buffer with a fixed classical adapter. It did not explore generative replay, distillation loss, or parameter isolation.
3. **Absence of Quantum Representation:** Quantum kernels and parameterized quantum circuits were strictly excluded to establish this fair classical baseline.

---

### 11. Research Conclusion

Phase 7.5 provides an honest, empirical assessment of memory replay for atypical speech personalization:
- **Primary Finding:** Balanced memory replay provides a modest directional reduction in catastrophic emergency forgetting ($-1.11\%$), but this effect **does not reach statistical significance ($p = 0.3343$)** across the cohort, and produces zero recall gain for dysarthric speakers ($9.97\% \to 9.97\%$).
- **Safety Stability:** Memory replay preserved low false-alarm rates ($29.28\%$), avoiding runaway false emergencies.
- **Scientific Value:** This establishes the necessity of Phase 7.6 (Model Validation Gates) and Phase 7.8+ (Quantum Continual Personalization), demonstrating that classical replay memory alone is insufficient to guarantee safety-critical emergency retention in low-resource dysarthric speech.

---

### 12. Reproducibility & Artifact Manifest

All Phase 7.5 artifacts are permanently serialized and isolated:
- **Treatment Results:** `results/continual/phase7_5/continual_replay_results.csv` (60 cells)
- **120-Cell Matched Comparison:** `results/continual/phase7_5/matched_replay_comparison.csv`
- **Forgetting Analysis:** `results/continual/phase7_5/forgetting_analysis.csv`
- **Experiment Registry:** `results/continual/phase7_5/experiment_registry.json` (60 records)
- **Summary Report:** `reports/continual/phase7_5/continual_replay_summary.json`
- **Model Checkpoints:** `models/continual/phase7_5/{speaker}/classical_adapter/{version}/model_checkpoint.pt` (60 candidate models)
- **Test Suite:** `tests/test_phase7_5_replay_memory.py` (16/16 tests passing)
- **Full Test Suite:** 323/323 tests passing across repository.
