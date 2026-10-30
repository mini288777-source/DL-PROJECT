# PHASE 7 MODEL VERSIONING & PROMOTION CONTRACT
## Lifecycle State Machine, Lineage Provenance & Rollback Protocol

---

### 1. Model Version Schema Specification

Every personalization model instance evaluated in Phase 7 is cryptographically tracked to ensure complete reproducibility, auditability, and safety.

#### Model Version Record Fields

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `model_id` | String | Model architecture identifier (`classical_adapter`, `classical_rbf_svm`, `qsvc_q8_r1`, etc.). |
| `version` | String | Semantic version string formatted as `v{speaker}_{iteration}_{patch}` (e.g., `vF01_iter1_0`). |
| `parent_version` | String | Version string of the predecessor model from which this version was derived. |
| `training_sample_ids` | List[String] | Complete list of all sample UUIDs present in the calibration/adaptation set. |
| `training_dataset_hash` | String | SHA-256 fingerprint of the feature matrix and labels used in training. |
| `configuration_hash` | String | SHA-256 hash of all hyperparameters (learning rate, epochs, regularization $C$, etc.). |
| `representation_version` | String | Hash identifier of the Whisper encoder and PCA/scaler transformation pipeline. |
| `quantum_configuration` | Dict / None | Parameters for quantum circuit (qubits, reps, feature map, simulation backend) or `null`. |
| `metrics` | Dict | Comprehensive benchmark performance recorded during protected validation. |
| `created_at` | String | ISO 8601 UTC timestamp of model artifact serialization. |
| `status` | String | Current operational state within the version lifecycle state machine. |

---

### 2. Model Lifecycle State Machine

Models transition strictly through the following formal state graph:

```
[ CANDIDATE ] ──(Protected Validation Fails)──► [ REJECTED ]
      │
(Protected Validation Passes)
      │
      ▼
 [ VALIDATED ]
      │
(Promotion Approved)
      │
      ▼
   [ ACTIVE ] ──(Downstream Performance Anomaly)──► [ ROLLED_BACK ]
      │                                                   │
(Superseded by New Active)                                ▼
      │                                              [ ARCHIVED ]
      ▼
 [ ARCHIVED ]
```

#### State Definitions
1. `CANDIDATE`: Newly trained candidate model in isolated sandbox; strictly barred from serving live or simulated inference.
2. `VALIDATED`: Successfully evaluated against protected validation data; meets baseline parity but awaiting promotion approval.
3. `ACTIVE`: Officially approved model currently serving speaker-specific predictions. Only one version per speaker may be `ACTIVE` simultaneously.
4. `REJECTED`: Failed protected validation checks (e.g., higher False-Emergency Rate or degraded Emergency Recall). Permanently sealed.
5. `ROLLED_BACK`: A previously active model that exhibited operational degradation and was demoted in favor of its parent version.
6. `ARCHIVED`: Superseded active model or historical baseline preserved immutably for audit and ablation comparisons.

---

### 3. Model Immutability & Anti-Overwrite Rule

- **No In-Place Overwrites:** Under no circumstances shall any model checkpoint file (`.pt`, `.pkl`, `.joblib`) be overwritten.
- **Unique Version Path:** Every model artifact is stored under a version-stamped directory:
  `models/continual/{speaker_id}/{model_family}/{version}/`
- **Hash Verification:** Prior to loading any checkpoint, its `training_dataset_hash` and `configuration_hash` are validated against the model registry.

---

### 4. Promotion Criteria & Validation Contract

To earn `ACTIVE` status, a `CANDIDATE` model must undergo evaluation across all 6 core metrics:

1. **End-to-End Emergency Recall:** Must not degrade relative to the parent model.
2. **False-Emergency Rate (FER):** Must not increase by more than pre-specified margin $\epsilon$.
3. **End-to-End Emergency Precision:** Evaluated under natural ~95% negative prevalence.
4. **Intent Preservation Rate (IPER):** Fraction of emergency utterances correctly routed as an emergency.
5. **Stage-2 Balanced Accuracy:** Subtype routing accuracy across feasible emergency classes.
6. **End-to-End Macro-F1:** Overall multi-class performance across all 7 canonical classes.

> **Governance Notice:**
> The exact numerical thresholds and decision boundaries for promotion are explicitly pre-registered as:
> **PROMOTION POLICY = TO BE DEFINED IN PHASE 7.6**
> No automated promotion engine shall be implemented prior to Phase 7.6.
