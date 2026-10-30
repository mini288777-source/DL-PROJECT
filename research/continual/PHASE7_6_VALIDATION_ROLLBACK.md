# PHASE 7.6: MODEL VALIDATION GATE & SAFETY GOVERNANCE REPORT
## Strict Research / Safety-Governance Mode

**Project**: Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Phase**: Phase 7.6 — Model Validation Gate & Rollback Governance  
**Policy Contract**: `config/continual/governance_policy.yaml` (`v1.0.0_frozen`)  
**Status**: COMPLETE  
**Frozen Scientific Conclusion**: **CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE** (UNCHANGED)

---

## 1. Purpose

Phase 7.6 establishes a formal, conservative research governance layer that determines whether a newly personalized candidate model is permitted to become the active model for a given user.

In longitudinal continual learning, a candidate model trained on natural speech accumulation will routinely improve aggregate metrics (such as Macro-F1 or overall accuracy) and suppress false emergencies simply because background speech overwhelmingly dominates the longitudinal stream (~95% background speech). However, Phase 7.4 and Phase 7.5 empirical results demonstrated that this aggregate improvement frequently masks catastrophic safety-critical regressions:
- Cohort emergency recall regressed by $-0.0635$ ($p = 0.0382$) in Phase 7.4.
- Dysarthric speaker emergency recall collapsed from $17.26\%$ down to $9.97\%$.
- Treatment with balanced replay memory in Phase 7.5 directionally improved cohort recall by $+0.0111$, but failed to statistically restore dysarthric emergency recall ($9.97\%$, $p = \text{NaN}$).

Therefore, Phase 7.6 establishes the fundamental governing principles:
> **“Model training completion does not imply promotion eligibility.”**

And:
> **“Macro-F1 improvement alone is insufficient for promotion when safety-critical metrics regress.”**

And:
> **“Validation decisions are policy-versioned and auditable.”**

And:
> **“Phase 7.6 does not alter the frozen Phase 1–6.5 research conclusions.”**

---

## 2. Governance Architecture

The governance architecture consists of a deterministic, layered pipeline:

```
Candidate Model
      ↓
Protected Validation
      ↓
Multi-Metric Evaluation
      ↓
Safety / Quality Gate
      ↓
PROMOTE / REJECT / HOLD
```

And for active deployment:

```
Active Model
      ↓
Safety / Validation Failure
      ↓
ROLLBACK
      ↓
Previous Validated Active Model
```

The system is conservative: a candidate model cannot become `ACTIVE` merely because training converged.

---

## 3. Model Lifecycle

The model version lifecycle strictly adheres to `docs/continual/PHASE7_MODEL_VERSION_CONTRACT.md` and enforces explicit, finite-state transitions:

```
[TRAINED ARTIFACT]
       ↓
  CANDIDATE ────────── (Evidence Insufficient) ─────────→ HOLD
       ↓
PROTECTED VALIDATION
       ↓
SAFETY GATE EVALUATION
       ↓
   ┌───┴───────────────┐
   ↓                   ↓
PROMOTE             REJECT
   ↓                   ↓
 ACTIVE             REJECTED
   ↓
(Superseded) ───→ ARCHIVED
   ↓
(Safety Alert) ─→ ROLLED_BACK
```

1. **`CANDIDATE`**: Initial immutable state upon creation/registration. A candidate model can never directly serve predictions without passing the gate.
2. **`ACTIVE`**: Exactly one model version per speaker may hold this pointer. Promotion occurs only through verified `PromotionDecision.PROMOTE`.
3. **`HOLD`**: The candidate model remains in `CANDIDATE` state awaiting further safety-critical validation samples when sample sizes are below statistical sufficiency thresholds.
4. **`REJECTED`**: The candidate failed one or more safety or quality gates; metadata and failure reasons are permanently recorded.
5. **`ARCHIVED`**: A previously active model that was superseded by a newly validated and promoted candidate model.
6. **`ROLLED_BACK`**: An active model demoted due to post-deployment safety alerts or regression triggers; the active pointer reverts to the prior validated parent model.

---

## 4. Candidate Registration

Candidate models enter the system through `ModelRegistry.register_candidate()`:
- **Version Uniqueness**: Overwriting existing version identifiers is prohibited (`ModelOverwriteError`).
- **Lineage Integrity**: Parent version existence is verified before admission (`ModelLineageError`).
- **Initial Status**: Must enter strictly as `CANDIDATE`.
- **Audit Logging**: Emits `CANDIDATE_CREATED` audit event with timestamp and speaker scope.

---

## 5. Protected Validation

Validation datasets must be strictly separated from candidate training data, replay memory, and future experience:
- **Provenance Fingerprinting**: Every validation dataset is hashed using SHA-256 over all `(sample_id, true_label, embedding_bytes)`.
- **Contamination Firewall**:
  1. *Training Overlap*: If `len(training_ids ∩ validation_ids) > 0`, validation fails with `REJECTED_VALIDATION_CONTAMINATION`.
  2. *Replay Overlap*: If `len(replay_ids ∩ validation_ids) > 0`, validation fails with `REJECTED_VALIDATION_CONTAMINATION`.
  3. *Temporal Causality*: If validation data includes sample timestamps after candidate creation or cutoff time, validation fails with `REJECTED_VALIDATION_CONTAMINATION`.
- **Speaker Isolation**: Rejects cross-speaker validation datasets with `REJECTED_SPEAKER_MISMATCH`.
- **Dataset Integrity**: Mismatched fingerprints trigger `REJECTED_INVALID_MODEL_VERSION`.

---

## 6. Multi-Metric Validation

The gate does not rely on Macro-F1 alone. It computes and evaluates the complete canonical Phase 7 metric set via `compute_comprehensive_continual_metrics`:
1. **Emergency Recall**: Safety-critical proportion of true emergencies identified.
2. **False Emergency Rate (FER)**: Proportion of non-emergencies misclassified as emergencies.
3. **Emergency Precision**: Precision under imbalanced negative prevalence.
4. **Intent Preservation Rate (IPER)**: Emergency routing preservation.
5. **Macro-F1**: Balanced multi-class metric across all 7 canonical classes.
6. **General Speech Recall**: Baseline routine communication preservation ($\ge 0.70$).
7. **Per-Class Recall**: Evaluated for all 7 canonical classes. If class sample count is 0, reports `NOT_FEASIBLE` (never zero, imputed, or fabricated).

---

## 7. Safety Veto

The core of Phase 7.6 is the **Unconditional Safety Veto**:
If any mandatory emergency safety condition is violated, the candidate model **CANNOT BE PROMOTED**, regardless of improvements in Macro-F1, overall Accuracy, or False Emergency Rate.

Pre-registered safety gates in `config/continual/governance_policy.yaml`:
- **Emergency Recall Regression Tolerance**: `0.00` ($\Delta \text{Emergency Recall} \ge 0.00$ vs active parent). Any drop in emergency recall triggers `REJECTED_SAFETY_VETO_EMERGENCY_RECALL`.
- **Absolute False Emergency Rate Ceiling**: $0.50$ ($\text{FER} \le 0.50$). Spikes above 50% trigger `REJECTED_SAFETY_VETO_FER_EXCEEDED`.
- **FER Increase Tolerance**: $+0.05$ ($\Delta \text{FER} \le +0.05$ vs active parent). Spikes beyond $+0.05$ trigger `REJECTED_SAFETY_VETO_FER_EXCEEDED`.

---

## 8. Promote Decision

A candidate model receives `PromotionDecision.PROMOTE` if and only if all mandatory safety and quality gates pass:
- Contamination checks clean.
- Evidence sufficiency met ($\ge 2$ emergencies, $\ge 10$ total samples).
- Emergency recall delta $\ge 0.00$.
- False Emergency Rate $\le 0.50$ and delta $\le +0.05$.
- Macro-F1 delta $\ge -0.01$.
- Reason code emitted: `PROMOTED_ALL_GATES_PASS`.
- Validation status: `PROMOTABLE`.

---

## 9. Reject Decision

A candidate model receives `PromotionDecision.REJECT` if any safety or quality gate is violated:
- `REJECTED_SAFETY_VETO_EMERGENCY_RECALL`
- `REJECTED_SAFETY_VETO_FER_EXCEEDED`
- `REJECTED_MACRO_F1_REGRESSION`
- `REJECTED_VALIDATION_CONTAMINATION`
- `REJECTED_INVALID_PROVENANCE`
- `REJECTED_SPEAKER_MISMATCH`
- `REJECTED_PARENT_LINEAGE_MISMATCH`
- `REJECTED_INVALID_MODEL_VERSION`

---

## 10. HOLD Decision

When validation data is insufficient to compute safety metrics reliably:
- Total validation samples $< 10$ $\rightarrow$ `HOLD_INSUFFICIENT_VALIDATION_SAMPLES`.
- Emergency validation samples $< 2$ $\rightarrow$ `HOLD_INSUFFICIENT_EMERGENCY_EVIDENCE`.
- Candidate remains in `CANDIDATE` state; active model pointer remains untouched.

---

## 11. Rollback Rules & Mechanics

If an active model exhibits operational degradation or safety alerts:
1. `ModelGovernanceManager.rollback(speaker_id, reason)` is triggered.
2. Validates that previous active history exists for that speaker.
3. Validates target model existence and verifies `target_rec.speaker_id == speaker_id`.
4. Enforces that target model was not rejected (`target_rec.status != REJECTED`).
5. Demotes currently active model to `ROLLED_BACK`.
6. Restores previous validated model to `ACTIVE`.
7. Preserves the rolled-back model's metadata, training provenance, and failure logs immutably.
8. Appends `ROLLBACK` event to audit log.

---

## 12. Active Model Registry

The active model registry enforces strict controls:
- **Speaker Isolation**: Pointers are strictly mapped `speaker_id -> active_model_version`.
- **Single Active Invariant**: At most one model may be `ACTIVE` per speaker.
- **Gatekeeping Barrier**: Candidates cannot become active without traversing `ValidationGate` and receiving a verified `PROMOTE` record.

---

## 13. Model Immutability

- Checkpoint weights and metadata are immutable upon creation.
- Lifecycle transitions are tracked through registry records and append-only audit entries.
- `ValidationDecisionRecord` instances enforce immutability: attempting to mutate attributes in place raises `AttributeError`.

---

## 14. Policy Versioning

- Every validation decision record explicitly logs `validation_policy_version` (`v1.0.0_frozen`).
- If governance policy thresholds change, historical decisions remain untouched; new evaluations bind to the incremented policy version.

---

## 15. Provenance

Every decision produces a complete provenance trail:
- Unique `validation_id`
- `candidate_model_version` & `parent_model_version`
- `speaker_id` & `speaker_group`
- `validation_dataset_id` & SHA-256 fingerprint
- `validation_policy_version`
- Timestamp and full metric dictionaries

---

## 16. Audit Trail

`ModelGovernanceManager` logs all lifecycle events into an append-only JSON audit log:
- `CANDIDATE_CREATED`
- `VALIDATION_STARTED`
- `VALIDATION_COMPLETED`
- `PROMOTED`
- `REJECTED`
- `HELD`
- `ROLLBACK`
- `RETIRED`

Each entry preserves: `timestamp`, `model_version`, `speaker_id`, `event`, `reason`, `policy_version`.

---

## 17. Failure Scenarios (Empirical Regression from Phase 7.4 / 7.5)

To validate the governance gate without training new models (`Models Trained: 0`, `New Experiments: 0`), all 120 historical candidate models from Phase 7.4 (Continual Classical) and Phase 7.5 (Replay Memory) were evaluated through the frozen gate:

| Evaluation Track | Total Candidates | Promoted | Rejected | Safety Vetoes | Dysarthric Promotion Rate | Control Promotion Rate | Mean $\Delta$ Emerg Recall | Mean $\Delta$ Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| **Phase 7.4 (Classical)** | 60 | 27 (45.0%) | 33 (55.0%) | 33 | 46.9% (17 vetoes) | 42.9% (16 vetoes) | $-0.0243$ | $+0.0137$ |
| **Phase 7.5 (Replay)** | 60 | 26 (43.3%) | 34 (56.7%) | 34 | 40.6% (19 vetoes) | 46.4% (15 vetoes) | $-0.0143$ | $+0.0163$ |
| **Total Overall** | 120 | 53 (44.2%) | 67 (55.8%) | 67 | 43.8% (36 vetoes) | 44.6% (31 vetoes) | $-0.0193$ | $+0.0150$ |

### Key Regression Observations
- **Scenario 1 (Macro-F1 Gain with Emergency Recall Regression)**: Across Phase 7.4 candidates, Macro-F1 gained $+0.0137$, but emergency recall dropped $-0.0243$. The safety veto blocked 33 candidates from unsafe promotion.
- **Scenario 2 (FER Improvement Masking Dysarthric Regression)**: In Phase 7.5, despite replay memory buffering experiences, 19 dysarthric candidates were caught regressing emergency recall and vetoed.
- **Scenario 3 (Insufficient Evidence)**: Unit tests verified that sample sets with $< 2$ emergencies enter `HOLD` without false positive promotion claims.
- **Scenario 4 (Data Leakage)**: Training overlap, replay overlap, and future timestamps trigger immediate contamination rejection.

---

## 18. Test Coverage (348 / 348 Passing)

The full test suite executes in 27.10s with zero failures:
- Historical Regression Tests: **225 / 225**
- Phase 7 Boundary Tests: **10 / 10**
- Phase 7.1 Experience Store: **16 / 16**
- Phase 7.2 Confidence Engine: **20 / 20**
- Phase 7.3 Human-in-the-Loop: **20 / 20**
- Phase 7.4 Continual Classical: **16 / 16**
- Phase 7.5 Replay Memory: **16 / 16**
- Phase 7.6 Model Governance & Validation Gate: **25 / 25**
- Repository Integrity Audit: **PASS**

---

## 19. Limitations

1. **Validation Sample Scarcity**: Severely dysarthric speakers have limited emergency validation instances, requiring frequent reliance on the `HOLD` state until sufficient evidence is accumulated.
2. **Fixed Acoustic Representation**: Whisper-tiny 384-D bottleneck embeddings remain frozen; acoustic representation drift cannot be corrected at the encoder level in Phase 7.6.
3. **No Quantum Operations**: In accordance with the frozen Case B conclusion, quantum methods remain excluded from continual personalization governance.

---

## 20. Closure Status

Phase 7.6 is **COMPLETE**:
- Model validation gate, safety veto, multi-metric comparison, 3-way decision (`PROMOTE`, `REJECT`, `HOLD`), speaker isolation, contamination firewall, active model registry, auditable rollback, and immutable audit logging are fully implemented, tested, and audited.
- Historical Phase 1–6.5 baselines and Phase 7.4/7.5 experimental results remain untouched.
- `CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE` conclusion is preserved.
- Phase 7.7 has NOT been started.
