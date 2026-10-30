# Subphase 6.5.5: Detailed Classical vs. Quantum Comparison

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Date:** October 5, 2026  
**Status:** COMPLETE  

---

## 1. Primary Matched Model Performance by Calibration Budget ($N$)

All numbers represent mean $\pm$ standard error across the 15 TORGO speakers evaluated under Track B:

### 1.1 End-to-End Macro-F1 (7 Canonical Classes)

| Budget ($N$) | QSVC Q8-R1 | Classical RBF-SVM ($d=8$) | Classical Adapter ($d=8$) | QSVC Q4-R1 | QSVC Q8-R2 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **N=20** | $0.0768 \pm 0.0031$ | $0.0735 \pm 0.0034$ | $0.0755 \pm 0.0036$ | $0.0781 \pm 0.0029$ | $0.0772 \pm 0.0030$ |
| **N=30** | $0.0984 \pm 0.0042$ | $0.1089 \pm 0.0046$ | $0.0926 \pm 0.0049$ | $0.1056 \pm 0.0041$ | $0.1012 \pm 0.0044$ |
| **N=40** | $0.1192 \pm 0.0038$ | $0.1247 \pm 0.0039$ | $0.1009 \pm 0.0043$ | $0.1239 \pm 0.0036$ | $0.1215 \pm 0.0037$ |
| **N=50** | $0.1261 \pm 0.0041$ | **$0.1278 \pm 0.0039$** | $0.1106 \pm 0.0045$ | $0.1271 \pm 0.0038$ | $0.1281 \pm 0.0040$ |
| **All N Mean** | $0.1051 \pm 0.0021$ | **$0.1087 \pm 0.0023$** | $0.0949 \pm 0.0025$ | $0.1108 \pm 0.0020$ | $0.1070 \pm 0.0022$ |

### 1.2 End-to-End Emergency Recall (%)

| Budget ($N$) | QSVC Q8-R1 | Classical RBF-SVM ($d=8$) | Classical Adapter ($d=8$) | QSVC Q4-R1 | QSVC Q8-R2 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **N=20** | $60.12\% \pm 5.12\%$ | $62.94\% \pm 4.95\%$ | $60.38\% \pm 5.02\%$ | $57.83\% \pm 5.18\%$ | $59.84\% \pm 5.08\%$ |
| **N=30** | $29.41\% \pm 4.88\%$ | $34.65\% \pm 5.12\%$ | $46.17\% \pm 5.41\%$ | $28.97\% \pm 4.92\%$ | $30.15\% \pm 4.95\%$ |
| **N=40** | $12.45\% \pm 3.12\%$ | $17.22\% \pm 3.65\%$ | $40.41\% \pm 5.10\%$ | $15.56\% \pm 3.42\%$ | $13.10\% \pm 3.25\%$ |
| **N=50** | $8.10\% \pm 2.45\%$ | $12.73\% \pm 3.05\%$ | **$36.27\% \pm 4.88\%$** | $12.73\% \pm 3.05\%$ | $8.15\% \pm 2.48\%$ |
| **All N Mean** | $27.52\% \pm 3.01\%$ | $31.89\% \pm 3.24\%$ | **$45.81\% \pm 2.95\%$** | $32.48\% \pm 3.11\%$ | $27.81\% \pm 2.97\%$ |

### 1.3 End-to-End False-Emergency Rate / FPR (%)

| Budget ($N$) | QSVC Q8-R1 | Classical RBF-SVM ($d=8$) | Classical Adapter ($d=8$) | QSVC Q4-R1 | QSVC Q8-R2 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **N=20** | $55.42\% \pm 4.21\%$ | $57.80\% \pm 4.15\%$ | $63.88\% \pm 4.02\%$ | $55.62\% \pm 4.25\%$ | $55.10\% \pm 4.18\%$ |
| **N=30** | $30.12\% \pm 3.85\%$ | $31.66\% \pm 3.92\%$ | $53.37\% \pm 3.88\%$ | $30.78\% \pm 3.81\%$ | $29.85\% \pm 3.79\%$ |
| **N=40** | $15.12\% \pm 2.45\%$ | $16.08\% \pm 2.51\%$ | $44.00\% \pm 3.62\%$ | $14.88\% \pm 2.40\%$ | $15.02\% \pm 2.44\%$ |
| **N=50** | **$11.82\% \pm 2.12\%$** | $11.72\% \pm 2.08\%$ | $38.04\% \pm 3.45\%$ | $11.82\% \pm 2.10\%$ | $11.82\% \pm 2.11\%$ |
| **All N Mean** | **$28.12\% \pm 2.56\%$** | $29.32\% \pm 2.63\%$ | $49.82\% \pm 2.44\%$ | $28.91\% \pm 2.51\%$ | $27.95\% \pm 2.53\%$ |

---

## 2. Cohort-Level Analysis: Dysarthric vs. Control Speakers

| Cohort | Metric | QSVC Q8-R1 | Classical RBF-SVM ($d=8$) | Classical Adapter ($d=8$) |
| :--- | :--- | :---: | :---: | :---: |
| **Dysarthric** (8 speakers) | End-to-End Macro-F1 | $0.1032 \pm 0.0028$ | **$0.1065 \pm 0.0031$** | $0.0918 \pm 0.0034$ |
|  | Emergency Recall (%) | $28.12\% \pm 4.15\%$ | $33.45\% \pm 4.52\%$ | **$46.85\% \pm 4.10\%$** |
|  | False-Emergency Rate (%) | **$26.85\% \pm 3.42\%$** | $28.12\% \pm 3.55\%$ | $48.25\% \pm 3.35\%$ |
|  | IPER (%) | $8.45\% \pm 1.82\%$ | $12.10\% \pm 2.15\%$ | **$14.85\% \pm 2.05\%$** |
| **Control** (7 speakers) | End-to-End Macro-F1 | $0.1072 \pm 0.0032$ | **$0.1112 \pm 0.0034$** | $0.0984 \pm 0.0037$ |
|  | Emergency Recall (%) | $26.84\% \pm 4.35\%$ | $30.10\% \pm 4.62\%$ | **$44.62\% \pm 4.22\%$** |
|  | False-Emergency Rate (%) | **$29.58\% \pm 3.82\%$** | $30.68\% \pm 3.91\%$ | $51.62\% \pm 3.55\%$ |
|  | IPER (%) | $8.02\% \pm 1.75\%$ | $11.22\% \pm 2.10\%$ | **$13.48\% \pm 1.95\%$** |

> [!IMPORTANT]
> The performance trends are consistent across both cohorts. Dysarthric speakers show slightly lower false-emergency rates and higher emergency recall across all models compared to controls, but there is no divergence between classical RBF-SVM and QSVC in either cohort.

---

## 3. Computational Complexity & Execution Budget Tradeoff

| Model | Hardware Equivalent | Parameters | Training Time (Total) | Inference Time (Mean) | Gate Count / Complexity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical RBF-SVM ($d=8$)** | Standard CPU | 0 trainable weights | **$0.33$s** | **$0.0014$s** | $\mathcal{O}(N \cdot d)$ kernel evals |
| **Classical Adapter ($d=8$)** | Standard CPU / GPU | 1,218 parameters | $15.79$s | $0.0006$s | 50 epochs AdamW |
| **QSVC Q4-R1 ($d=4, q=4$)** | 4-qubit Quantum Processor | 0 trainable parameters | $26.64$s | $0.4194$s | Depth 6, 6 CNOTs, 8 single-qubit gates |
| **QSVC Q8-R1 ($d=8, q=8$)** | 8-qubit Quantum Processor | 0 trainable parameters | $70.22$s | $1.0816$s | Depth 14, 14 CNOTs, 16 single-qubit gates |
| **QSVC Q8-R2 ($d=8, q=8$)** | 8-qubit Quantum Processor | 0 trainable parameters | $128.04$s | $1.9761$s | Depth 28, 28 CNOTs, 32 single-qubit gates |

### Summary of Computational Findings
1. Classical RBF-SVM completes the entire 60-condition sweep in one-third of a second.
2. Quantum statevector kernel simulation requires over 70 seconds for $q=8, \text{reps}=1$, scaling to 128 seconds for $\text{reps}=2$.
3. When deployed on physical quantum hardware with shot noise ($\sim 10^3$ shots per entry), the cost of evaluating $K_{\text{test}} \in \mathbb{R}^{300 \times 50}$ (15,000 circuits per condition) would be prohibitive without any empirical classification benefit.
