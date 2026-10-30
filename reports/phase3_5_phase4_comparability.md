# Phase 4 Experimental Comparability Contract

**Date:** 2026-09-30  
**Phase:** 3.5 — Research Validity, Consistency & Phase 4 Readiness Gate  
**Target File:** `reports/phase3_5_phase4_comparability.md`  

---

## 1. Research Principle: Strict Experimental Control

To scientifically answer the core research question:
> *"Can a quantum kernel provide measurable few-shot personalization benefits for atypical speech when only a small number of labeled samples from an individual speaker are available?"*

the future Phase 4 Quantum Kernel experiment must be **strictly controlled**. The ONLY experimental independent variable must be:

$$\text{Classical Method (Adapter / RBF-SVM)} \quad \longleftrightarrow \quad \text{Quantum Kernel (ZZFeatureMap + FidelityQuantumKernel)}$$

Every other aspect of the pipeline—data, representations, labels, partitions, evaluation metrics, and random seeds—must be identical.

---

## 2. The Phase 4 Input Contract Specification

Any Phase 4 implementation MUST conform to the following specifications:

| Contract Parameter | Exact Required Value / Specification | Source of Truth |
|---|---|---|
| **Speaker Cohort** | Exactly 15 speakers (8 dysarthric: F01, F03, F04, M01, M02, M03, M04, M05; 7 control: FC01, FC02, FC03, MC01, MC02, MC03, MC04) | `data/manifests/torgo_speaker_shortlist.csv` |
| **Sample Budgets ($N$)** | $N \in \{20, 30, 40, 50\}$ per speaker | `configs/torgo_config.yaml` |
| **Split Definitions** | Exact matching `adaptation_recording_ids` and `test_recording_ids` | `splits/torgo_splits.json` |
| **Raw Speech Representation** | Exact 384-dimensional frozen Whisper embeddings loaded via `np.load()` | `embeddings/torgo/whisper/<recording_id>.npy` |
| **Label Space** | Exact 7-class taxonomy (`SOS_HELP`, `FALL`, `BREATHING_MEDICAL`, `PAIN`, `LOCATION`, `CRITICAL_RESPONSE`, `GENERAL_SPEECH`) | `configs/intent_taxonomy.yaml` |
| **Dimensionality Reduction** | `sklearn.decomposition.PCA(n_components=n_qubits)` fitted **ONLY on the adaptation set** ($N$ samples) | Standard ML protocol |
| **Quantum Feature Map** | `ZZFeatureMap(num_qubits=n_qubits, reps=2, entanglement='linear')` | Project Research Spec |
| **Quantum Kernel** | `FidelityQuantumKernel` using Qiskit statevector simulation | `tests/smoke_test_quantum_kernel.py` |
| **Classifier Formulation** | `sklearn.svm.SVC(kernel=kernel_matrix, C=1.0)` | Matched to RBF baseline |
| **Evaluation Function** | `src.models.evaluation.compute_intent_metrics()` | `src/models/evaluation.py` |
| **Random Seed** | Seed = 42 for all stochastic initialization | `src/utils/seeding.py` |

---

## 3. Matched Pipeline Architecture

```
                    +------------------------------------------+
                    | Frozen Whisper-tiny Acoustic Embeddings  |
                    | (D = 384, Unit-Sphere L2 Normalized)      |
                    +------------------------------------------+
                                         |
            +----------------------------+----------------------------+
            |                            |                            |
            v                            v                            v
   [Classical Adapter]            [Classical RBF-SVM]          [Quantum Kernel SVM]
   Linear(384->64)                SVC(kernel='rbf')            PCA(384 -> d qubits)
   LayerNorm + GELU               C = 1.0                      ZZFeatureMap(reps=2)
   Linear(64->7)                  gamma = 'scale'              FidelityQuantumKernel
   AdamW (lr=1e-3, 50 ep)                                      SVC(kernel=K_q, C=1.0)
            |                            |                            |
            +----------------------------+----------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    | Identical Evaluation: compute_intent_metrics |
                    | (Accuracy, Macro-F1, Weighted-F1, ConfMat)|
                    +------------------------------------------+
```

---

## 4. Prohibited Deviations in Phase 4

To protect the defensibility of the final paper, the following practices are **STRICTLY PROHIBITED** in Phase 4:
1. **NO Test-Set PCA Fitting:** PCA must NEVER be fitted on concatenated adaptation + test embeddings. It must be fitted strictly on the $N$ adaptation samples and transformed onto the test set.
2. **NO Synthetic Label Rebalancing for Quantum Only:** If class weights ($C_k$) are introduced, they must be ablated identically against the classical RBF baseline.
3. **NO Speaker Exclusions:** Quantum kernels must be evaluated on all 15 speakers, including the 7 single-class adaptation conditions.
4. **NO Metric Swapping:** The primary comparison must remain Macro-F1.

## 5. Audit Conclusion

**PHASE 4 COMPARABILITY STATUS: READY & LOCKED**  
The Phase 4 input contract is formally defined and frozen.
