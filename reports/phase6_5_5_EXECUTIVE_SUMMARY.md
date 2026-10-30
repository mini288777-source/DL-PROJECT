# Subphase 6.5.5 Executive Summary: Quantum Kernel Recovery & Optimization

## 1. Objective
Determine whether a fidelity quantum kernel Support Vector Classifier (QSVC) provides any measurable performance benefit over matched classical classifiers for personalized atypical emergency speech recognition, under the recovered speaker-stratified few-shot protocol (Track B, $N \in \{20, 30, 40, 50\}$, 15 TORGO speakers) and validated representation space.

## 2. Quantum Configuration
- **Primary Configuration (Q8-R1):** $d=8$, 8 qubits, `ZZFeatureMap` with linear entanglement, `reps=1`, `FidelityQuantumKernel`, `QSVC(C=1.0)`. Simulated Hilbert space dimension $2^8 = 256$.
- **Secondary Ablation 1 (Q4-R1):** $d=4$, 4 qubits, `ZZFeatureMap` with linear entanglement, `reps=1`, `FidelityQuantumKernel`, `QSVC(C=1.0)`. Hilbert space dimension $2^4 = 16$.
- **Secondary Ablation 2 (Q8-R2):** $d=8$, 8 qubits, `ZZFeatureMap` with linear entanglement, `reps=2`, `FidelityQuantumKernel`, `QSVC(C=1.0)`.
- **Backend:** Qiskit statevector exact simulation (noise-free ideal simulation).
- **Kernel Integrity:** 180 / 180 matrices strictly finite, zero NaNs/Infs, symmetry error $= 0.0$, diagonal error $= 0.0$, strictly positive semi-definite ($\lambda_{\min} \ge 0.000186$, mean $\lambda_{\min} = 0.5589$). Zero kernels required repair.

## 3. Representation Contract
- **Encoder:** `openai/whisper-tiny` (frozen, float32, unit-L2 normalized, 384 dimensions).
- **Dimensionality Reduction:** Strictly train-only PCA fitted on adaptation speech only. No test leakage.
- **Quantum Preprocessing:** Train-only `MinMaxScaler` mapping PCA coordinates to $[0, \pi]$.
- **Primary Dimension:** $d=8$ (retaining $89.73\% \pm 2.99\%$ variance).
- **Secondary Dimension:** $d=4$ (retaining $80.66\% \pm 4.37\%$ variance).

## 4. Primary Matched Comparison Results (Across All 60 Conditions)
| Metric | QSVC Q8-R1 | Classical RBF-SVM ($d=8$) | Classical Adapter ($d=8$) | QSVC Q4-R1 | QSVC Q8-R2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **End-to-End Macro-F1** | $0.1051 \pm 0.0163$ | **$0.1087 \pm 0.0182$** | $0.0949 \pm 0.0195$ | $0.1108 \pm 0.0152$ | $0.1070 \pm 0.0171$ |
| **Emergency Recall (%)** | $27.52\% \pm 23.36\%$ | $31.89\% \pm 25.10\%$ | **$45.81\% \pm 22.84\%$** | $32.48\% \pm 24.12\%$ | $27.81\% \pm 23.01\%$ |
| **False-Emergency Rate (%)** | **$28.12\% \pm 19.82\%$** | $29.32\% \pm 20.37\%$ | $49.82\% \pm 18.91\%$ | $28.91\% \pm 19.45\%$ | $27.95\% \pm 19.64\%$ |
| **Emergency Precision (%)** | $11.24\% \pm 9.45\%$ | $11.89\% \pm 9.87\%$ | $9.21\% \pm 7.15\%$ | **$12.04\% \pm 9.71\%$** | $11.45\% \pm 9.52\%$ |
| **IPER (%)** | $8.25\% \pm 9.60\%$ | $11.69\% \pm 12.39\%$ | **$14.21\% \pm 11.85\%$** | $11.81\% \pm 11.95\%$ | $8.72\% \pm 9.88\%$ |
| **Stage-2 Balanced Accuracy** | $32.53\% \pm 18.21\%$ | **$33.91\% \pm 17.65\%$** | $33.57\% \pm 16.92\%$ | $29.84\% \pm 17.15\%$ | $32.41\% \pm 18.05\%$ |

## 5. Statistical Conclusions
- **Primary Planned Comparison (QSVC Q8-R1 vs. RBF-SVM $d=8$):**
  - Macro-F1 Mean Difference: $-0.0036$ (95% CI: $[-0.0083, +0.0010]$, Wilcoxon $W=721.5$, raw $p=0.1714$, Holm-adjusted $p=0.3428$).
  - Emergency Recall Mean Difference: $-0.0437$ (95% CI: $[-0.0814, -0.0060]$, raw $p=0.0230$, Holm-adjusted $p=0.0689$).
  - False-Emergency Rate Mean Difference: $-0.0120$ (95% CI: $[-0.0448, +0.0208]$, raw $p=0.9505$, Holm-adjusted $p=0.9505$).
  - End-to-End IPER Mean Difference: $-0.0344$ (95% CI: $[-0.0611, -0.0077]$, raw $p=0.0145$, Holm-adjusted $p=0.0582$).
- **Secondary Comparison (QSVC Q8-R1 vs. Adapter $d=8$):**
  - QSVC yields higher Macro-F1 ($+0.0102$, $p=0.0067$) and lower false-emergency rate ($-21.71\%$, $p < 10^{-5}$), but substantially lower Emergency Recall ($-18.29\%$, $p < 10^{-4}$) and IPER ($-5.97\%$, $p=0.0030$).
- **Qubit & Circuit Depth Ablations:**
  - Q4-R1 vs. Q8-R1: No statistically significant difference ($p=0.2025$).
  - Q8-R1 vs. Q8-R2: Negligible difference in Macro-F1 ($\Delta = -0.0019$).

## 6. Quantum Advantage Decision
**CASE B: No Statistically Supported Difference.**
Under the pre-specified recovery protocol, the fidelity quantum kernel is statistically indistinguishable from the classical RBF kernel on primary end-to-end metrics, while achieving inferior emergency recall relative to the classical neural adapter. No quantum advantage is observed.

## 7. Computational Cost
- **Classical RBF-SVM ($d=8$):** $0.33$s total wall time across 60 conditions ($0.0014$s per test evaluation).
- **Classical Adapter ($d=8$):** $15.79$s total wall time.
- **QSVC Q8-R1 ($d=8, q=8$):** $70.22$s total wall time ($1.0816$s per test evaluation) — **$\sim 770\times$ slower** than classical RBF-SVM.
- **QSVC Q8-R2 ($d=8, q=8$):** $128.04$s total wall time — **$\sim 1,400\times$ slower**.

## 8. Limitations
1. **Low-Resource Acoustic Bottleneck:** Extreme class imbalance and small sample size ($N \le 50$) limit margin separation in both classical and quantum feature spaces.
2. **Statevector Simulation Bound:** Results reflect ideal noise-free simulation; physical quantum devices with shot noise and decoherence would face additional performance degradation.
3. **No Clinical / Deployment Readiness:** Exploratory research protocol only; NOT clinically certified or deployment-ready.

## 9. Phase 6.5.6 Handoff
- **Phase Status:** Complete.
- **Definitive Quantum Finding:** Fidelity quantum kernel QSVC provides zero measurable advantage over matched classical RBF-SVM and underperforms classical Adapter on safety-critical emergency recall.
- **Scientific Recommendation:** Freeze quantum exploration. For low-resource atypical speech recognition, prioritized investment should focus on cost-sensitive classical adapters, acoustic augmentation, and multi-speaker transfer rather than quantum feature maps.
- **Phase 6.5.6:** NOT STARTED.
