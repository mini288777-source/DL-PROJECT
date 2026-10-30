# MASTER PROJECT REPORT RESEARCH AUDIT, RESULTS CONSOLIDATION, AND CONTENT BLUEPRINT

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Team Name:** APPY FIZZ  
**Institution:** Amrita Vishwa Vidyapeetham, Amrita School of Computing, Chennai Campus, Tamil Nadu, India  
**Target Document Path:** `docs/PROJECT_REPORT_CONTENT_BLUEPRINT.md`  
**Date of Audit & Blueprint Freeze:** October 2026  
**Auditor Role:** Senior Research Engineer, Machine Learning Experimental Auditor, Quantum Machine Learning Researcher, Technical Documentation Specialist, Academic Project Report Consultant  

---

## 1. Project Identity and Scope

### 1.1 Formal Project Metadata
- **Full Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication
- **Team Name:** APPY FIZZ
- **Degree Award:** Bachelor of Technology in Computer Science and Engineering (Artificial Intelligence) / Bachelor of Technology in Computer Science and Engineering
- **Institution:** Amrita Vishwa Vidyapeetham
- **School:** Amrita School of Computing
- **Campus:** Chennai Campus, Vengal, Thiruvallur District, Tamil Nadu – 601 103, India
- **Academic Year:** 2025–2026 / Final Year Project

### 1.2 Core Research Domain & Problem Statement
- **Domain:** Atypical/dysarthric speech recognition, acoustic feature extraction, few-shot speaker personalization, quantum machine learning (QML), classical machine learning baselines, environmental noise robustness, emergency-related intent classification, and safety-critical deployment governance.
- **Underlying Challenge:** Dysarthric speech—resulting from neuro-motor conditions such as cerebral palsy, amyotrophic lateral sclerosis (ALS), Parkinson's disease, or stroke—exhibits pronounced inter-speaker acoustic heterogeneity, imprecise consonant articulation, reduced vocal intensity, and hypernasality. Commercial automatic speech recognition (ASR) engines exhibit catastrophic word error rates (frequently exceeding $60\text{--}80\%$) on dysarthric speech, rendering generic emergency voice interfaces unusable.
- **Few-Shot Personalization Constraint:** Individuals with severe motor speech impairments experience vocal fatigue, making the collection of extensive speaker-specific calibration data impossible. Personalization must succeed with strictly low-resource sample budgets ($N \in \{20, 30, 40, 50\}$ calibration utterances per individual speaker).
- **Core Scientific Inquiry:** *Can a quantum Hilbert-space feature embedding generated via a parameter-free fidelity quantum kernel ($ZZ\text{FeatureMap}$) provide a measurable, statistically defensible advantage in accuracy, few-shot sample efficiency, or emergency recall over matched classical kernels (e.g., RBF-SVM) and lightweight neural adaptation baselines when adapting to an atypical speaker from frozen Whisper embeddings?*

### 1.3 Strict Operational vs. Research Separation Contract
To preserve complete academic and engineering integrity throughout the final project report, a non-negotiable boundary is enforced:

```
+----------------------------------------------------------------------------------------------------+
|                                    STRICT SYSTEM DUALITY CONTRACT                                  |
+----------------------------------------------------------------------------------------------------+
| 1. RESEARCH & SCIENTIFIC COMPARISON (Chapter 3, 4, 5)                                             |
|    - Evaluates Quantum Model vs. Matched Classical Baselines (RBF-SVM, Linear SVM, Weighted Adapter) |
|    - Spans Track 1 (Original Flat Benchmark), Track 2 (Hierarchical Recovery), and Continual Adaptation |
|    - Rigorously records positive, negative, and null results (Wilcoxon tests, Holm-Bonferroni correction) |
|    - Formal Scientific Finding: CASE B (No Statistically Supported Quantum Advantage)               |
+----------------------------------------------------------------------------------------------------+
| 2. DEPLOYED OPERATIONAL APPLICATION (Chapter 4, Section 4.7-4.10)                                  |
|    - Operational Speech Classifier: EXCLUSIVELY canonical 'qsvc_q8_r1' (8-Qubit Fidelity QSVC)    |
|    - Classical Models: RESEARCH-ONLY; strictly barred from live prediction (HTTP 400 Rejected)      |
|    - Classical Fallback: PROHIBITED (Zero classical fallback; fail-closed under all error states)    |
|    - Deployment Status: LOCAL RESEARCH DEMONSTRATION ONLY (Gate C & Gate D strictly Blocked)       |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Official Report Template Requirements

### 2.1 Authoritative Template Source
- **Inspected File:** `D:\DL\docs\Project Report Template (1).docx` (106,537 bytes, verified and audited directly).
- **Issuing Body:** Amrita Vishwa Vidyapeetham, Amrita School of Computing, Chennai.

### 2.2 Sequence of Arrangement
The report must strictly follow this physical sequence:
1. **Title Page** (Specimen 1)
2. **Cover Page** (Identical to Title Page on hardcover/binding)
3. **Bonafide Certificate** (Specimen 2, signed by Supervisor, Program Head, Internal & External Examiners)
4. **Declaration by Candidate** (Specimen 3, signed by all student team members)
5. **Abstract** (Specimen 4, concise overview of problem, methodology, findings, and keywords)
6. **Acknowledgement** (Specimen 5, acknowledging Chancellor, Director, Principal, Supervisor, Faculty)
7. **Table of Contents** (Specimen 6)
8. **List of Tables** (Specimen 7, chapter-based numbering)
9. **List of Figures** (Specimen 8, chapter-based numbering)
10. **List of Symbols, Abbreviations and Nomenclature** (Specimen 9)
11. **Chapter 1: Introduction**
12. **Chapter 2: Literature Review / Problem Background**
13. **Chapter 3: Problem Statement and Methodology**
14. **Chapter 4: System Design / Analysis / Experimental Work / Simulation**
15. **Chapter 5: Results and Discussions**
16. **Chapter 6: Conclusions and Scope for Further Work**
17. **References** (Alphabetical order by first author)
18. **Appendices** (Plagiarism Report, Published Papers/Manuscripts, Certificates)

### 2.3 Page Dimensions, Margins, and Typography Rules

| Formatting Element | Official Template Specification | Implementation Guidance for Final Document |
| :--- | :--- | :--- |
| **Paper Size** | Standard A4 ($297\text{ mm} \times 210\text{ mm}$) | Strict A4 page setup in Word/LaTeX |
| **Margins** | Top: $1.0''$ ($2.54\text{ cm}$), Bottom: $1.0''$ ($2.54\text{ cm}$), Left: $1.0''$ ($2.54\text{ cm}$), Right: $1.0''$ ($2.54\text{ cm}$) | Symmetric 1-inch margins on all 4 sides |
| **Font Family** | Times New Roman throughout | Strict Times New Roman; JetBrains Mono for code blocks in appendices |
| **Chapter Headings** | 14 pt, Bold, ALL CAPS, Centre-aligned | `CHAPTER 1`, `INTRODUCTION` |
| **Division Headings (X.Y)** | 12 pt, Bold, ALL CAPS, Left-aligned | `1.1. BACKGROUND AND MOTIVATION` |
| **Subdivision Headings (X.Y.Z)**| 12 pt, Bold, Capitalize Each Word (Title Case), Left-aligned | `1.1.1. Dysarthric Speech Characteristics` |
| **Body Paragraphs** | 12 pt, Sentence case, Fully Justified | Justified alignment, 1 cm first-line indentation |
| **Line Spacing** | 1.5 line spacing for general running text | Exactly 1.5 lines; single-spaced within table cells |
| **Page Numbering** | Arabic numerals ($1, 2, 3, \dots$) commencing from preliminary pages | Bottom-center or top-right consistent placement |
| **Figure Numbering** | Chapter-based: `Figure X.Y` (e.g., `Figure 3.4` for 4th figure in Ch 3) | Centered caption below figure, italicized explanatory text |
| **Table Numbering** | Chapter-based: `Table X.Y` (e.g., `Table 5.2` for 2nd table in Ch 5) | Centered caption above table, clean borders |
| **Equation Numbering** | Chapter-based: `(X.Y)` (e.g., `(3.1)` for 1st equation in Ch 3) | Right-aligned parenthesized numbering |
| **Reference Ordering** | Strictly ALPHABETICAL by first author surname | Author, Title (*italics*), Volume, Issue, Pages, Year |

---

## 3. Repository and Artifact Inventory

A comprehensive forensic audit of the `d:\DL` repository identified all authoritative assets, models, and scripts:

### 3.1 Primary Directories and File Asset Distribution
- **Root Directory (`d:\DL`):** Core documentation, master phase execution prompts, acceptance audits, environment configurations (`pyproject.toml`, `requirements.txt`).
- **`src/` (Core Application & Algorithmic Modules):**
  - `src/app/main.py`: FastAPI backend, route registration, static mounts, exception handlers.
  - `src/app/quantum_model_manager.py`: Fail-closed quantum model manager; enforces `qsvc_q8_r1` exclusively, rejects classical models with HTTP 400.
  - `src/app/audio_service.py`: 16 kHz mono resampling, peak normalization, offline Whisper feature extraction.
  - `src/app/config.py`: System constants, canonical 7-class taxonomy, safety thresholds ($\tau=0.6000$, $\text{FER}\le 15.0\%$).
  - `src/app/static/`: Production Quantum Research Command Center UI (`index.html`, `style.css`, `app.js`).
  - `src/quantum/fidelity_kernel.py`: Qiskit Aer statevector fidelity kernel engine, 8-qubit $ZZ\text{FeatureMap}$.
  - `src/embeddings/whisper_encoder.py`: Frozen Whisper-tiny acoustic embedding extraction (384-D).
- **`models/` (Serialized Checkpoints & Forensic Sidecars):**
  - `models/quantum/qsvc_q8_r1/model_checkpoint.pkl`: Fitted scikit-learn QSVC model artifact.
  - `models/quantum/qsvc_q8_r1/preprocessor.pkl`: Fitted StandardScaler and PCA ($384\to 8$) transformation.
  - `models/quantum/qsvc_q8_r1/inference_reference/`: Verified sidecar directory containing 15 speaker reference statevectors (`X_train_pca.npy`, `reference_manifest.json`) and `inventory_manifest.json`.
  - `models/continual/`: Checkpoints from Phase 7 longitudinal adaptation experiments.
- **`results/` (Master Experimental CSV & JSON Datasets):**
  - `results/phase3/`: Baseline flat classification performance tables across budgets $N\in\{5,10,20,50\}$.
  - `results/phase4/`: Flat 4-qubit and 8-qubit QSVC results documenting majority collapse.
  - `results/phase5/`: Acoustic noise perturbation evaluations ($15\text{ dB}, 10\text{ dB}, 5\text{ dB}, 0\text{ dB}$ SNR).
  - `results/phase6_5_4_representation_audit.csv`: PCA cumulative variance audit across dimensions $d\in\{2,4,8,16,32,64\}$.
  - `results/phase6_5_5_matched_comparison.csv`: Definitive 60-condition matched dataset (QSVC Q8-R1 vs. RBF-SVM $d=8$).
  - `results/phase6_5_5_statistical_tests.csv`: Wilcoxon signed-rank and Holm-Bonferroni test outputs.
  - `results/phase6_5_5_runtime.csv`: Statevector CPU execution latency and wall-clock audit.
  - `results/phase8/`, `results/phase8_1/`, `results/phase8_separated/`: Phase 8 safety operating points and decoupled evaluation data.
- **`reports/` (Audit Reports & Generated Figure Stores):**
  - `reports/COMPLETE_RESEARCH_RESULTS.md`: Authoritative 292-line empirical synthesis across all phases.
  - `reports/project_readiness/figures/`: High-resolution research comparison plots.
  - `reports/application/figures/`: Full-scale verified application UI screenshots (`01_overview_dashboard.png` through `07_help_and_limitations.png`).
  - `reports/phase_c/` & `reports/phase_d/`: Governance registries confirming Gate C and Gate D blockade status.

---

## 4. Project Status and Implemented Components

### 4.1 Chronological Development Phase Audit

```
+----------------------------------------------------------------------------------------------------+
|                                    PROJECT DEVELOPMENT TRAJECTORY                                  |
+----------------------------------------------------------------------------------------------------+
| Phase 0: Environment & Smoke Tests (Python 3.10/3.11, Qiskit 1.x, PyTorch, Librosa)   [COMPLETE]   |
| Phase 1: TORGO Data Foundation (15 Speakers, 16,552 Utterances, Clean Metadata)       [COMPLETE]   |
| Phase 2: Whisper-Tiny Embedding Extraction (384-D Bottleneck Embeddings Cached)       [COMPLETE]   |
| Phase 3: Classical Baselines Benchmark (Flat 7-Class: Linear SVM, RBF, MLP, KNN)       [COMPLETE]   |
| Phase 4: Quantum Kernel Personalization (Flat 4/8-Qubit QSVC: Majority Collapse Seen)  [COMPLETE]   |
| Phase 5: Environmental Acoustic Noise Perturbation (AWGN at 15, 10, 5, 0 dB SNR)       [COMPLETE]   |
| Phase 6: Statistical Analysis & Epistemic Verdict (Case B: No Quantum Advantage)       [COMPLETE]   |
| Phase 6.5: Forensic Recovery Track (Hierarchical Gate, PCA Audit, Matched Evaluation)  [COMPLETE]   |
| Phase 7: Continual Adaptation & Safety Governance (Experience Replay, Veto Policy)    [COMPLETE]   |
| Phase 8: Emergency Pipeline Gating & OOF-to-Test Generalization Audit                  [COMPLETE]   |
| Gate B.2: Quantum-Only Research Command Center Web Application                         [OPERATIONAL]|
| Gate C: External Pre-Registration & Independent Clinical Cohort Prerequisites         [BLOCKED]    |
| Gate D: Clinical Safety Evidence, Regulatory Compliance & Prospective Trial           [BLOCKED]    |
+----------------------------------------------------------------------------------------------------+
```

### 4.2 Gate Status Registry
1. **Gate A (Scientific Verification):** **APPROVED.** Mathematical and experimental verification complete. Null hypothesis affirmed; quantum-vs-classical parity confirmed without fabrication.
2. **Gate B.2 (Software & Prototype Demonstration):** **APPROVED / OPERATIONAL.** Full-stack FastAPI + Vanilla CSS/JS command center verified locally on `http://127.0.0.1:8000`. 47/47 Gate B.2 automated tests passing.
3. **Gate C (External Cohort Pre-Registration):** **BLOCKED (`BLOCKED_EXTERNAL_PREREQUISITES`).** No multi-site dysarthric validation data acquired; no pre-registered independent clinical trial conducted.
4. **Gate D (Clinical Safety Evidence):** **BLOCKED (`BLOCKED_SAFETY_EVIDENCE`).** False-emergency rates ($15\text{--}28\%$) and low emergency precision under natural prevalence ($0.81\%$) preclude autonomous emergency dispatch.

---

## 5. Dataset and Experimental Protocol Audit

### 5.1 The TORGO Database Profile
- **Source:** University of Toronto & Bloorview Kids Rehab (Rudzicz et al., 2012).
- **Physical Cohort Size:** 15 distinct speakers:
  - **Dysarthric Group:** 8 speakers (`F01`, `F03`, `F04`, `M01`, `M02`, `M03`, `M04`, `M05`).
  - **Control Group:** 7 speakers (`FC01`, `FC02`, `FC03`, `MC01`, `MC02`, `MC03`, `MC04`).
- **Physical Audio Holdings:** 16,552 raw WAV files; 16,550 verified valid PCM audio files (13.68 total hours; 49,238.30 seconds). Uniformly resampled to 16 kHz mono.
- **Transcript Holdings:** 16,552 aligned transcripts; 967 unique prompt strings.

### 5.2 Emergency Intent Taxonomy Mapping (Proxy Label Formulation)
TORGO is fundamentally a phonetic and clinical dysarthric speech corpus consisting of phonetically balanced sentences, non-words, and short phrases; it is **not** a corpus of real 911/emergency dispatch calls. The emergency taxonomy was established by mapping specific semantic prompt utterances into 6 life-critical emergency categories, with all remaining utterances designated as general non-emergency speech:

| Canonical Label | Semantic Definition | Representative Prompt Triggers | Cohort Count | Natural Prevalence |
| :--- | :--- | :--- | :---: | :---: |
| `GENERAL_SPEECH` | Everyday conversation, conversational prompts | Sentences, numbers, non-emergency phrases | 15,852 | **$95.77\%$** |
| `CRITICAL_RESPONSE`| Affirmation or negation during triage | "Yes", "No", "Confirm", "Stop" | 182 | $1.10\%$ |
| `FALL` | Physical collapse, slip, immobility | "I fell", "Help me up", "Cannot move" | 98 | $0.59\%$ |
| `BREATHING_MEDICAL`| Respiratory distress, cardiac events | "Cannot breathe", "Chest hurts" | 112 | $0.68\%$ |
| `PAIN` | Acute physical trauma or severe distress | "Severe pain", "It hurts", "Agony" | 104 | $0.63\%$ |
| `SOS_HELP` | Explicit urgent call for external assistance | "Help", "Emergency", "Call someone" | 126 | $0.76\%$ |
| `LOCATION` | Location status of victim | "Bedroom", "Bathroom", "Living room" | 78 | $0.47\%$ |
| **Total Emergency** | Aggregated emergency-class instances | 6 emergency subtypes combined | **700** | **$4.23\%$** |

> **Critical Reporting Caveat for Chapter 3 & 5:**
> The emergency intent categories represent **heuristic proxy labels** constructed from phonetic reading prompts. The model's classification accuracy reflects its capacity to discriminate these specific acoustic prompts, **not** genuine real-world distress audio captured during acute trauma or panic.

### 5.3 Acoustic Preprocessing and Frozen Feature Extraction
1. **Audio Resampling:** Librosa / soundfile pipeline resamples input audio to 16,000 Hz, 16-bit mono PCM. Peak normalization scales waveform amplitudes to $[-1.0, 1.0]$.
2. **Log-Mel Spectrogram:** 80 Mel frequency filterbanks computed using a 25 ms Hann window (400 samples) and 10 ms hop length (160 samples).
3. **Frozen Upstream Encoder:** Evaluated using `openai/whisper-tiny` (39 million parameters). The encoder weights are strictly frozen (`eval()`, `requires_grad=False`).
4. **Bottleneck Pooling:** Mean pooling across temporal frames produces a fixed-size 384-dimensional acoustic latent vector ($z \in \mathbb{R}^{384}$). L2 unit-sphere normalization is applied: $\|z\|_2 = 1.0$.

### 5.4 Personalization Sample Budgets & Leakage Controls
- **Few-Shot Budgets:** $N \in \{20, 30, 40, 50\}$ calibration utterances per speaker.
- **Split Strategy:** Speaker-stratified $K$-fold cross-validation. For each speaker $s$, calibration samples are drawn strictly from $s$'s training split. Test evaluation is conducted strictly on held-out utterances from $s$.
- **Leakage Prevention:** Zero channel overlap; audio from the head-worn microphone and array microphone corresponding to the same speech event are never split across train and test partitions.
- **Representation Scalers:** All StandardScaler and PCA transformation matrices are fit strictly on the $N$ calibration samples and applied forward to test sets.

---

## 6. Research Model Inventory

### 6.1 Classical Baseline Models (Research Comparison Only)
1. **Linear SVM (`linear_svm`):** Scikit-learn LinearSVC with $C=1.0$, balanced class weighting. Serves as the minimal linear baseline in the 384-D Whisper space.
2. **Classical RBF-SVM ($d=8$) (`rbf_svm_d8`):** Scikit-learn SVC with radial basis function kernel $K(x, z) = \exp(-\gamma \|x-z\|^2)$, $\gamma = 1/d$, $C=1.0$. Operates on the identical 8-D PCA representation as the quantum model; serves as the **primary matched classical baseline**.
3. **Classical Adapter (MLP) (`adapter_mlp`):** Lightweight 2-layer perceptron (Linear $384\to 64$, ReLU, BatchNorm, Dropout $0.2$, Linear $64\to 7$, CrossEntropyLoss with cost-sensitive class weights).
4. **$K$-Nearest Neighbors (`knn`):** $k=3$ and $k=5$ Euclidean distance classifier on normalized embeddings.

### 6.2 Quantum Models (Evaluated in Research)
1. **QSVC Q4-R1 (`qsvc_q4_r1`):** 4-qubit Quantum Support Vector Classifier using $ZZ\text{FeatureMap}$ with 1 repetition. Evaluates dimension $d=4$.
2. **QSVC Q8-R1 (`qsvc_q8_r1`):** 8-qubit Quantum Support Vector Classifier using $ZZ\text{FeatureMap}$ with 1 repetition (linear nearest-neighbor entanglement). Evaluates dimension $d=8$. **Primary Quantum Architecture.**
3. **QSVC Q8-R2 (`qsvc_q8_r2`):** 8-qubit QSVC with 2 variational feature map repetitions (circuit depth ablation).
4. **Trainable Quantum Variational Classifier (VQC) (Phase 7.9):** Variational circuit combining $ZZ\text{FeatureMap}$ with parameterized $R_y(\theta)$ and $CZ$ layers optimized via COBYLA / Adam.

### 6.3 Mathematical Formulation of the Fidelity Quantum Kernel
The 8-qubit quantum state is prepared via the canonical $ZZ\text{FeatureMap}$ operating on scaled input feature vector $x \in [0, 2\pi]^8$:

$$\Phi(x) = U_{\Phi}(x)|0\rangle^{\otimes 8}$$

$$U_{\Phi}(x) = \left( \exp\left( i \sum_{j=1}^8 x_j Z_j + \sum_{j=1}^7 \sum_{k=j+1}^8 (\pi - x_j)(\pi - x_k) Z_j Z_k \right) H^{\otimes 8} \right)^{\text{reps}}$$

The quantum kernel entry between two speech representations $x$ and $z$ is the transition probability (statevector fidelity) in the 256-dimensional complex Hilbert space:

$$K_{\text{Quantum}}(x, z) = |\langle \Phi(x) | \Phi(z) \rangle|^2 = \left| \langle 0|^{\otimes 8} U_{\Phi}^{\dagger}(x) U_{\Phi}(z) |0\rangle^{\otimes 8} \right|^2$$

---

## 7. Quantum-versus-Classical Comparison Plan

### 7.1 Methodological Comparability Criteria
To ensure an irreproachably fair comparison, all comparative evaluations between QSVC Q8-R1 and Classical RBF-SVM ($d=8$) satisfied 7 strict matching conditions:
1. **Identical Acoustic Features:** Both models receive identical 384-D Whisper-tiny embeddings.
2. **Identical Dimensionality Reduction:** Both models operate on the identical 8-dimensional PCA projection space ($d=8$).
3. **Identical Train/Test Splits:** Evaluated across the exact same 60 speaker $\times$ budget conditions (15 speakers $\times$ 4 sample budgets $N \in \{20, 30, 40, 50\}$).
4. **Identical Classifier Objective:** Both models solve the dual soft-margin Support Vector Machine problem with regularization parameter $C=1.0$:
   $$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
   differing solely in the choice of kernel matrix $K$: $K_{\text{Quantum}}$ vs. $K_{\text{RBF}}$.
5. **Identical Evaluation Metric Implementations:** Exact same scikit-learn macro-F1, recall, and false-alarm calculations.
6. **Paired Statistical Testing:** Evaluated using non-parametric Wilcoxon signed-rank tests across the 60 matched pairs.
7. **Multiplicity Control:** Family-wise error rate controlled via the Holm-Bonferroni step-down procedure across all tested metrics.

---

## 8. Master Experimental Results Inventory

### 8.1 Primary Matched Comparison: QSVC Q8-R1 vs. Classical RBF-SVM ($d=8$)
Conducted across **60 matched conditions** (15 speakers $\times$ 4 sample budgets $N \in \{20, 30, 40, 50\}$) under the recovery protocol (Phase 6.5.5 / 6.5.5-C).

| Evaluation Metric | QSVC Q8-R1 (Quantum) | Classical RBF-SVM ($d=8$) | Difference ($\Delta$) | 95% Confidence Interval | Wilcoxon $W$ | Raw $p$-value | Holm-Adjusted $p$ | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **End-to-End Macro-F1** | $0.1051 \pm 0.0575$ | $0.1087 \pm 0.0487$ | $-0.0036$ | $[-0.0083, +0.0010]$ | $475.0$ | $0.1714$ | **$0.3428$** | **NOT SIGNIFICANT** |
| **Emergency Recall** | $27.52\% \pm 22.84\%$ | $31.89\% \pm 16.42\%$ | $-4.37\text{ pp}$ | $[-9.12\text{ pp}, +0.38\text{ pp}]$ | $374.0$ | $0.0230$ | **$0.0689$** | **NOT SIGNIFICANT** |
| **False-Emergency Rate (FER)**| $28.12\% \pm 14.77\%$ | $29.32\% \pm 13.92\%$ | $-1.20\text{ pp}$ | $[-4.80\text{ pp}, +2.40\text{ pp}]$ | $681.5$ | $0.9505$ | **$0.9505$** | **NOT SIGNIFICANT** |
| **Intent Preservation Rate (IPER)** | $8.25\% \pm 4.09\%$ | $11.69\% \pm 8.21\%$ | $-3.44\text{ pp}$ | $[-6.15\text{ pp}, -0.73\text{ pp}]$ | $654.5$ | $0.0194$ | **$0.0582$** | **NOT SIGNIFICANT** |

> **Authoritative Scientific Decision:**
> **CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE.**  
> Across all four primary evaluation metrics, after family-wise error rate control via the Holm-Bonferroni method, no difference achieved statistical significance ($p \ge 0.05$). The empirical performance of the fidelity quantum kernel is statistically indistinguishable from the classical RBF kernel.

### 8.2 Secondary Model Comparison: QSVC Q8-R1 vs. Weighted Classical Adapter
Under the identical 60-condition protocol, comparing the quantum kernel to the non-kernel neural baseline:

| Evaluation Metric | QSVC Q8-R1 | Weighted Classical Adapter | Difference ($\Delta$) | Wilcoxon $W$ | Holm-Adjusted $p$ | Operational Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Emergency Recall** | $27.52\%$ | **$45.81\%$** | $-18.29\text{ pp}$ | $166.5$ | **$0.000874$** | **Adapter achieves higher emergency capture ($p < 0.001$)** |
| **False-Emergency Rate** | **$28.12\%$** | $49.82\%$ | $-21.70\text{ pp}$ | $241.0$ | **$0.004120$** | **QSVC maintains lower false-alarm rate ($p < 0.01$)** |
| **End-to-End Macro-F1** | **$0.1051$** | $0.0984$ | $+0.0067$ | $580.0$ | $0.1245$ | Statistically indistinguishable |

### 8.3 Quantum Architecture Ablation Studies

#### A. Qubit Scaling ($d=4$ vs. $d=8$)
- **4 Qubits (Q4-R1):** Macro-F1 $= 0.1108 \pm 0.0512$, Emergency Recall $= 27.25\% \pm 21.40\%$
- **8 Qubits (Q8-R1):** Macro-F1 $= 0.1051 \pm 0.0575$, Emergency Recall $= 27.52\% \pm 22.84\%$
- **Paired Test:** Wilcoxon $W=598.0$, Holm-adjusted $p=1.0000$ (No significant difference).

#### B. Variational Circuit Depth Scaling ($\text{reps}=1$ vs. $\text{reps}=2$)
- **1 Repetition (Q8-R1):** Macro-F1 $= 0.1051$, Emergency Recall $= 27.52\%$, Wall-Clock $= 70.22\text{ s}$
- **2 Repetitions (Q8-R2):** Macro-F1 $= 0.1070$, Emergency Recall $= 27.52\%$, Wall-Clock $= 128.04\text{ s}$
- **Finding:** Doubling circuit depth doubled simulation runtime ($+82.3\%$) with $0.00\text{ pp}$ recall change.

### 8.4 Computational Complexity & Statevector Runtime Audit

| Execution Metric | Classical RBF-SVM ($d=8$) | QSVC Q8-R1 (Quantum) | QSVC Q8-R2 (Quantum) | Overhead Ratio (Q8-R1 / Classical) |
| :--- | :---: | :---: | :---: | :---: |
| **Inference Latency per Test Sample** | **$0.0014\text{ s}$** ($1.4\text{ ms}$) | **$1.0816\text{ s}$** ($1081.6\text{ ms}$) | **$1.9723\text{ s}$** ($1972.3\text{ ms}$) | **$772.6\times$ slower** |
| **Total Experiment Wall-Clock (60 Cells)** | **$0.33\text{ s}$** | **$70.22\text{ s}$** | **$128.04\text{ s}$** | **$212.8\times$ higher** |

---

## 9. Result Figures and Visualization Inventory

### 9.1 Authoritative Research Figures Inventory (Reports & Results)

| Proposed Figure ID | Exact Repository File Path | Description and Research Finding | Suggested Chapter |
| :--- | :--- | :--- | :---: |
| **Figure 4.1** | `reports/phase6_5_6/figures/fig8_evidence_chain_workflow.png` | Complete longitudinal evidence architecture: Track 1 vs. Track 2. | Chapter 4 |
| **Figure 4.2** | `reports/phase6_5_4/figures/explained_variance_vs_dimension.png` | Cumulative explained variance curve across PCA dimensions $d \in [2, 64]$. | Chapter 4 |
| **Figure 4.3** | `reports/phase6_5_5/figures/quantum_kernel_eigenvalues.png` | Eigenvalue decay spectrum of the 8-qubit fidelity quantum kernel matrix. | Chapter 4 |
| **Figure 5.1** | `reports/phase6_5_6/figures/fig1_original_vs_recovery_progression.png` | Longitudinal trajectory across milestones: Benchmark collapse to recovery. | Chapter 5 |
| **Figure 5.2** | `reports/phase6_5_6/figures/fig5_matched_q8_vs_rbf.png` | Primary matched comparison: QSVC Q8-R1 vs. RBF-SVM across 4 metrics. | Chapter 5 |
| **Figure 5.3** | `reports/phase6_5_6/figures/fig6_emergency_recall_comparison.png` | Emergency recall vs. false-alarm trade-off (QSVC vs. Adapter vs. RBF). | Chapter 5 |
| **Figure 5.4** | `reports/phase6_5_6/figures/fig7_quantum_computational_cost.png` | Statevector CPU execution latency ($772.6\times$) and wall-clock breakdown. | Chapter 5 |
| **Figure 5.5** | `results/phase4/figures/classical_vs_quantum_macro_f1.png` | Historical flat 7-class benchmark comparison showing mutual collapse ($\approx 0.10$). | Chapter 5 |
| **Figure 5.6** | `phase5/figures/macro_f1_vs_snr.png` | Acoustic noise perturbation response: Macro-F1 across SNR $\{15, 10, 5, 0\}\text{ dB}$. | Chapter 5 |
| **Figure 5.7** | `phase5/figures/iper_vs_snr.png` | Intent Preservation Error Rate (IPER) response to acoustic degradation. | Chapter 5 |
| **Figure 5.8** | `phase6/figures/speaker_variation.png` | Speaker-wise heterogeneity across the 15 TORGO dysarthric and control speakers. | Chapter 5 |
| **Figure 5.9** | `phase6_5/hierarchical/figures/confusion_matrix_end_to_end.png` | Hierarchical confusion matrix showing Stage-1 triage and Stage-2 subtype classification. | Chapter 5 |
| **Figure 5.10** | `reports/project_readiness/figures/threshold_stability_by_fold.png` | Out-of-fold threshold calibration curve showing generalization gap to test FER. | Chapter 5 |

---

## 10. Quantum-Only Application Analysis

### 10.1 Application Purpose and Architecture
- **Application Identity:** Atypical Speech Emergency Assistant — Quantum Research Command Center.
- **Architectural Paradigm:** Pure quantum inference application. The live operational backend runs **exclusively** on the canonical quantum model: `qsvc_q8_r1` (Fidelity Quantum Kernel QSVC, 8 Qubits, Reps=1, linear nearest-neighbor $ZZ$ entanglement).
- **Backend Stack:** FastAPI, Uvicorn, Python 3.10/3.11, Qiskit Aer statevector simulator, NumPy, SciPy, Librosa.
- **Frontend Stack:** Modern Vanilla HTML5/CSS3/JavaScript (ES2022) with custom Web Audio visualizers and Canvas physics engines. Zero heavy frontend dependencies.
- **Fail-Closed Execution Policy:**
  - If a client requests any classical baseline model (e.g., `model_id="classical_adapter"` or `model_id="rbf_svm"`), the request is strictly **rejected with HTTP 400 Bad Request** (`MODEL_NOT_ALLOWED`).
  - If quantum checkpoint artifacts are missing or unmaterialized, the server strictly returns **HTTP 503 Service Unavailable** (`QUANTUM_MODEL_NOT_READY`).
  - **Zero Classical Fallback:** Under no circumstance does the application fail over to classical prediction.

### 10.2 End-to-End Application Workflow Pipeline
```
[User Audio Input (WAV/MP3/Mic/Tone)]
                 |
                 v
   [Audio Validation & 16 kHz Mono Resampling]
                 |
                 v
   [Offline Whisper-Tiny Feature Extraction (384-D)]
                 |
                 v
   [Speaker Selection (F01...MC04 Isolation)]
                 |
                 v
   [Fitted StandardScaler & PCA Projection (384-D -> 8-D)]
                 |
                 v
   [8-Qubit Statevector Fidelity Kernel Computation (ZZFeatureMap)]
                 |
                 v
   [QSVC Support Vector Classification (qsvc_q8_r1)]
                 |
                 v
   [Decision Margin Evaluation vs. Threshold tau (0.6000)]
                 |
                 v
   [JSON Response with Latency, Probabilities, & Safety Disclaimers]
```

---

## 11. Application Screenshot Inventory

The repository contains verified screenshots of the running Quantum Research Command Center application located in `reports/application/figures/` and `assets/second_review/`:

| Proposed Screenshot ID | File Path | Application View / Purpose | Visible Features & Academic Caption |
| :--- | :--- | :--- | :--- |
| **Figure 4.4** | `reports/application/figures/01_overview_dashboard.png` | Overview Command Center | *Figure 4.4: Quantum Research Command Center overview dashboard displaying system vitals, 8-qubit quantum kernel operational readiness, Gate C/D blockade governance, and architectural vitals.* |
| **Figure 4.5** | `reports/application/figures/02_quantum_audio_lab.png` | Quantum Audio Lab | *Figure 4.5: Quantum Audio Lab ingestion interface showing the 5-stage processing pipeline stepper, drag-and-drop acoustic input, benchmark audio generator, and operational model selector.* |
| **Figure 4.6** | `reports/application/figures/03_model_speaker_registry.png` | Model & Speaker Registry | *Figure 4.6: Model and speaker registry displaying verified TORGO speaker profiles (15 speakers) and model status, showing qsvc_q8_r1 as the sole operational model and classical models marked research-only.* |
| **Figure 4.7** | `reports/application/figures/04_research_results_explorer.png` | Research Results Explorer | *Figure 4.7: Interactive research results explorer displaying verified comparative empirical metrics, longitudinal research progression, and statistical test outputs.* |
| **Figure 4.8** | `reports/application/figures/05_provenance_workspace.png` | Provenance Audit Workspace | *Figure 4.8: Provenance audit workspace displaying SHA-256 cryptographic checksums for frozen dataset manifests, model artifacts, and evaluation sidecars.* |
| **Figure 4.9** | `reports/application/figures/06_runtime_diagnostics.png` | Runtime Diagnostics Page | *Figure 4.9: Runtime diagnostics page displaying memory usage, CPU telemetry, Whisper feature extraction readiness, and quantum statevector execution health.* |
| **Figure 4.10** | `reports/application/figures/07_help_and_limitations.png` | Governance & Limitations | *Figure 4.10: Governance, help, and safety limitations page displaying explicit warnings regarding Gate C and Gate D blockades, research disclaimers, and prohibited clinical usage.* |

---

## 12. Chapter-by-Chapter Report Content Blueprint

### PRELIMINARY PAGES BLUEPRINT

#### 1. Title Page & Cover Page (Specimen 1)
- **Title:** LOW-RESOURCE PERSONALIZED ATYPICAL SPEECH RECOGNITION USING HYBRID QUANTUM-CLASSICAL DEEP LEARNING FOR RELIABLE EMERGENCY COMMUNICATION
- **Degree:** A PROJECT REPORT Submitted by `[STUDENT_NAMES_AND_REG_NOS]` in partial fulfillment for the award of the degree of BACHELOR OF TECHNOLOGY IN COMPUTER SCIENCE AND ENGINEERING (ARTIFICIAL INTELLIGENCE)
- **Supervision:** Under the guidance of `[SUPERVISOR_NAME_AND_DESIGNATION]`
- **Institution:** AMRITA VISHWA VIDYAPEETHAM, AMRITA SCHOOL OF COMPUTING, CHENNAI – 601 103
- **Month & Year:** November 2026 / May 2027

#### 2. Bonafide Certificate (Specimen 2)
- Formal university certificate text certifying that the report entitled *"Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication"* is the bonafide work carried out under supervision.
- Signatures: Supervisor, Program Head (CSE-AIE), Internal Examiner, External Examiner.

#### 3. Declaration by the Candidate (Specimen 3)
- Formal university declaration by the student team confirming original work and that the report has not formed the basis for the award of any other degree.

#### 4. Abstract (Specimen 4)
- **Background:** Individuals with dysarthria experience severe communication barriers, particularly during critical medical emergencies. Commercial ASR systems fail catastrophically due to motor-speech distortion.
- **Methodology:** This project investigates few-shot personalization ($N \in \{20, 30, 40, 50\}$ calibration samples) using frozen 384-D Whisper-tiny acoustic embeddings and an 8-qubit statevector fidelity quantum kernel ($ZZ\text{FeatureMap}$) paired with a Support Vector Classifier (`qsvc_q8_r1`), benchmarked against matched classical baselines across the 15-speaker TORGO database.
- **Principal Findings:** Under a strictly matched 60-condition protocol, the quantum model achieved an End-to-End Macro-F1 of $0.1051 \pm 0.0575$ compared to $0.1087 \pm 0.0487$ for the classical RBF-SVM baseline ($p=0.3428$, Wilcoxon signed-rank test with Holm-Bonferroni correction). Emergency recall was $27.52\%$ vs. $31.89\%$ ($p=0.0689$). Exact classical statevector simulation required $772.6\times$ higher per-evaluation latency than the matched RBF kernel. Formal scientific verdict confirms **Case B: No statistically supported quantum advantage**.
- **Application & Governance:** A local research command center was deployed exposing the quantum model exclusively. Due to extreme natural class imbalance ($\approx 95\%$ non-emergency) and high false-emergency rates ($28.12\%$), the system is formally governed as a research demonstration; Gate C and Gate D safety gates remain strictly blocked from autonomous clinical dispatch.
- **Keywords:** Atypical Speech Recognition, Dysarthria, Quantum Machine Learning, Quantum Kernel Estimation, Few-Shot Personalization, Whisper Embeddings, Emergency Intent Classification.

---

### CHAPTER 1 — INTRODUCTION
- **1.1. Background and Motivation:** Neuromotor speech disorders (cerebral palsy, ALS, stroke). Vocal tract articulation constraints in dysarthric speech. Inadequacy of conventional acoustic models.
- **1.2. Challenges of Low-Resource Personalization:** Vocal fatigue preventing large-scale speaker adaptation. Mathematical instability of high-dimensional estimators under $N \le 50$.
- **1.3. Emergency Communication Imperative:** Life-critical consequences of unrecognized distress calls. The fundamental trade-off between emergency recall (detecting real events) and false-emergency rate (preventing dispatch exhaustion).
- **1.4. The Promise and Reality of Quantum Machine Learning:** High-dimensional Hilbert spaces, fidelity kernels, non-linear feature maps. The need for rigorous empirical benchmarking against classical counterparts rather than theoretical assumptions.
- **1.5. Aim and Objectives:**
  1. Extract and cache robust 384-D acoustic embeddings from dysarthric speech using a frozen Whisper-tiny foundation model.
  2. Implement and evaluate an 8-qubit $ZZ\text{FeatureMap}$ quantum fidelity kernel (`qsvc_q8_r1`) for few-shot speaker personalization.
  3. Benchmark the quantum kernel against strictly matched classical estimators (RBF-SVM, Linear SVM, Weighted Adapter) across $N \in \{20, 30, 40, 50\}$ calibration budgets and acoustic noise perturbations.
  4. Perform rigorous statistical hypothesis testing (Wilcoxon signed-rank, Holm-Bonferroni) to evaluate quantum advantage claims.
  5. Develop a production-grade, local research command center interface enforcing quantum-only operational inference and safety gating.
- **1.6. Scope and Limitations:** Confined to the 15-speaker TORGO corpus; proxy emergency intent taxonomy; CPU statevector simulation; non-clinical research prototype.
- **1.7. Organization of the Report:** Outline of Chapters 2 through 6.

---

### CHAPTER 2 — LITERATURE REVIEW / PROBLEM BACKGROUND
- **2.1. Atypical and Dysarthric Speech Processing:** Acoustic characteristics of dysarthria (Rudzicz et al., 2012; Christensen et al., 2012; Green et al., 2021). Acoustic variability, formants centralization, phoneme elongation.
- **2.2. Foundation Models and Speech Embeddings:** Evolution from MFCCs and filterbanks to self-supervised representations (Wav2Vec 2.0, HuBERT, Whisper). Radford et al. (2022) Whisper architecture and bottleneck feature robustness.
- **2.3. Few-Shot Learning and Adaptation Techniques:** Transfer learning, lightweight adapter modules (Houlsby et al., 2019), support vector machines in low-sample regimes (Vapnik, 1995).
- **2.4. Quantum Machine Learning and Quantum Kernels:** Quantum feature maps and quantum state fidelity (Havlicek et al., 2019; Schuld & Killoran, 2019). The $ZZ\text{FeatureMap}$ and its connection to kernel methods.
- **2.5. Critical Evaluation of Quantum Advantage Claims:** Theoretical bounds vs. empirical realities (Huang et al., 2021; Cerezo et al., 2021). Barren plateaus, inductive bias, and the necessity of matched classical baselines.
- **2.6. Critical Synthesis & Research Gap:** Identification of the literature gap: *Lack of rigorous, matched empirical evaluations comparing quantum fidelity kernels to classical kernels on real-world clinical bio-acoustic speech data under strict few-shot budgets.*

---

### CHAPTER 3 — PROBLEM STATEMENT AND METHODOLOGY
- **3.1. Problem Definition & Mathematical Formulation:** Few-shot multiclass classification under severe class imbalance and acoustic distortion.
- **3.2. Dataset Foundation (TORGO Corpus):** Detailed speaker breakdown (8 dysarthric, 7 control). Recording protocols, acoustic channels, prompt distribution.
- **3.3. Proxy Emergency Intent Taxonomy:** Derivation and validation of the 7-class taxonomy. Separation of the 95.8% general speech majority class from the 6 emergency subtypes.
- **3.4. Feature Extraction Pipeline:** Log-Mel spectrogram generation, Whisper-tiny encoder forward pass, mean temporal pooling, L2 unit-sphere normalization ($z \in \mathbb{R}^{384}$).
- **3.5. Dimensionality Reduction & PCA Representation Audit:** Empirical explained variance analysis ($d \in \{2, 4, 8, 16, 32, 64\}$). Selection justification for $d=8$ ($89.73\%$ variance retention).
- **3.6. Classical Baseline Formulations:** Linear SVM, RBF-SVM ($d=8$), Weighted MLP Adapter, $K$-NN.
- **3.7. Quantum Architecture Formulation:** 8-qubit $ZZ\text{FeatureMap}$ mathematical derivation, linear nearest-neighbor entanglement, statevector fidelity computation.
- **3.8. Personalization Protocol & Leakage Controls:** Deterministic speaker-stratified sampling, prompt-level channel isolation, train-only transformation fitting.
- **3.9. Evaluation Metrics:** Strict mathematical definitions:
  - Macro-Averaged F1-Score: $\text{Macro-F1} = \frac{1}{C}\sum_{c=1}^C \frac{2 P_c R_c}{P_c + R_c}$
  - Emergency Recall: $\text{Recall}_{\text{Emerg}} = \frac{\sum_{c \in \mathcal{E}} \text{TP}_c}{\sum_{c \in \mathcal{E}} (\text{TP}_c + \text{FN}_c)}$
  - False-Emergency Rate (FER): $\text{FER} = \frac{\sum_{c \in \mathcal{E}} \text{FP}_c}{N_{\text{General}}}$
  - Intent Preservation Error Rate (IPER): $\text{IPER} = 1 - \text{Accuracy}_{\text{Matched}}$
- **3.10. Statistical Hypothesis Testing Framework:** Paired Wilcoxon signed-rank tests across 60 conditions; Holm-Bonferroni family-wise error correction.

---

### CHAPTER 4 — SYSTEM DESIGN, EXPERIMENTAL WORK AND IMPLEMENTATION
- **4.1. Overall Research System Architecture:** Complete pipeline diagram from raw audio ingestion to classification output.
- **4.2. Longitudinal Experimental Tracks:**
  - Track 1 (Original Flat Benchmark, Phases 3–6): Flat 7-class formulation and discovery of mutual majority-class collapse.
  - Track 2 (Hierarchical Recovery Track, Phase 6.5): Two-stage routing, representation space auditing, and cost-sensitive weighting.
- **4.3. Acoustic Noise Robustness Experimental Framework (Phase 5):** Controlled injection of Additive White Gaussian Noise (AWGN) across SNR levels ($15\text{ dB}, 10\text{ dB}, 5\text{ dB}, 0\text{ dB}$).
- **4.4. Continual Learning & Longitudinal Adaptation (Phase 7):** Class-balanced experience replay, validation rollback, and safety veto mechanisms.
- **4.5. Quantum-Only Operational Application Design (Gate B.2):**
  - Clean separation between research baselines and operational deployment.
  - FastAPI backend architecture, endpoints (`/predict`, `/predict/file`, `/api/audio/transcribe`, `/api/audio/extract-embedding`).
  - Quantum model manager (`src/app/quantum_model_manager.py`): Fail-closed model loading, strict rejection of classical models (`HTTP 400`), zero classical fallback.
  - Interactive Quantum Command Center Frontend: HTML5 canvas quantum lattice engine, 8-qubit Bloch statevector precession HUD, multi-mode audio spectrogram visualizer.
- **4.6. Implementation Environment & Tooling:** Hardware environment, Python libraries (FastAPI, Qiskit 1.x, PyTorch, Librosa, Scikit-learn).
- **4.7. Testing, Verification, and Safety Constraints:** Automated regression test suite (47/47 Gate B.2 tests passing).

---

### CHAPTER 5 — RESULTS AND DISCUSSIONS
- **5.1. Longitudinal Research Progression & Benchmark Discovery:** Detailed analysis of the flat 7-class collapse (Table 5.1). Explanation of why both classical and quantum models defaulted to majority-class prediction under extreme imbalance and unweighted loss.
- **5.2. Definitive Matched Classical vs. Quantum Comparison:** Presentation and deep analysis of the 60-condition matched evaluation (Table 5.2, Figure 5.2). Statistical confirmation of Case B ($p=0.3428$ for Macro-F1, $p=0.0689$ for Emergency Recall).
- **5.3. Few-Shot Sample Efficiency Analysis:** Performance trajectories across calibration budgets $N \in \{20, 30, 40, 50\}$.
- **5.4. Environmental Acoustic Noise Robustness:** Degradation curves across Clean, 15 dB, 10 dB, 5 dB, and 0 dB SNR (Figures 5.6 and 5.7). Quantum vs. classical degradation slopes.
- **5.5. Secondary Architecture Comparison (Adapter vs. QSVC):** Analysis of the emergency capture vs. false-alarm trade-off (Table 5.3, Figure 5.3). Why the neural adapter achieves higher recall ($45.81\%$) at the cost of high false-alarm rates ($49.82\%$).
- **5.6. Quantum Ablations (Qubit Count & Circuit Depth):** Empirical verification that scaling from 4 to 8 qubits and 1 to 2 repetitions yielded no statistically significant improvement.
- **5.7. Computational Complexity & Statevector Runtime Audit:** In-depth discussion of simulation overhead: $772.6\times$ per-sample latency and $212.8\times$ experiment wall-clock cost.
- **5.8. Forensic Metric Reconciliation:** Complete audit resolving historical reporting artifacts (Emergency Precision reconciliation from exploratory $11.24\%$ to natural $0.81\%$; PCA variance reconciliation from $27.4\%$ to $80.66\%$).
- **5.9. Comprehensive Limitations Disclosure:** Detailed discussion of the 15 research limitations (corpus scale, proxy labels, statevector simulation, frozen features).
- **5.10. Canonical Research Interpretation:** Grounded academic conclusion explaining why the fidelity quantum kernel behaved functionally as a stationary distance metric without quantum advantage.

---

### CHAPTER 6 — CONCLUSIONS AND SCOPE FOR FURTHER WORK
- **6.1. Summary of Implemented Work:** Successful implementation of an end-to-end dysarthric speech feature extraction, personalization, quantum kernel evaluation, and interactive research command center.
- **6.2. Key Empirical Conclusions:**
  1. No statistically significant advantage was observed for the 8-qubit fidelity quantum kernel over the matched classical RBF-SVM across any evaluated metric.
  2. Quantum scaling (qubits and depth) provided no measurable benefit under the evaluated representation.
  3. Classical statevector simulation incurs substantial computational cost ($>770\times$).
  4. Few-shot personalization under naturalistic class imbalance remains bottlenecked by high false-alarm rates.
- **6.3. Engineering Contributions:**
  1. A rigorous, reproducible matched comparative protocol for clinical speech QML.
  2. An audited, frozen Whisper acoustic representation and PCA dimensional audit.
  3. A hardened, fail-closed quantum-only research command center interface.
- **6.4. Scope for Future Work:**
  1. Exploration of problem-tailored, parameterized variational circuits and trainable quantum embeddings.
  2. Evaluation on physical NISQ quantum hardware (QPUs) with error mitigation.
  3. Acquisition of genuine multi-site clinical emergency dispatch recordings.
  4. Multi-modal integration (combining acoustic audio with lip-reading video or wearable biosensors).
- **6.5. Concluding Statement:** Reaffirming honest, evidence-grounded scientific inquiry as the highest standard of engineering research.

---

### REFERENCES & BIBLIOGRAPHY PLAN
(See Section 15 for complete alphabetical citation register.)

---

### APPENDICES PLAN
- **Appendix A: Complete Mathematical Derivations:** $ZZ\text{FeatureMap}$ Hamiltonian, fidelity transition probability, dual SVM formulation.
- **Appendix B: Hyperparameters & Configuration Registry:** Complete table of Whisper, PCA, SVM, Adapter, and Quantum settings.
- **Appendix C: Speaker-Wise Breakdown Tables:** Exhaustive results across all 15 individual TORGO speakers.
- **Appendix D: Reproducibility & Environment Manifest:** Python environment packages, exact library versions, hardware specifications.
- **Appendix E: Plagiarism Clearance Report:** Placeholder for official institutional Turnitin/plagiarism certificate.

---

## 13. Proposed Tables and Figures Register

### 13.1 Master Tables Register

| Table ID | Proposed Table Title | Type | Chapter | Authoritative Source File | Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Table 2.1** | Comparison of Existing Literature in Dysarthric Speech and QML | Literature Synthesis | Ch 2 | `reports/phase6_5_6/phase6_5_6_final_evidence_synthesis.md` | VERIFIED |
| **Table 3.1** | Speaker Demographic and Acoustic Profile of the TORGO Cohort | Dataset Profile | Ch 3 | `reports/torgo_summary.md` | VERIFIED |
| **Table 3.2** | Canonical 7-Class Emergency Intent Taxonomy and Prevalence | Label Formulation | Ch 3 | `src/app/config.py` & `reports/torgo_summary.md` | VERIFIED |
| **Table 3.3** | Cumulative Explained Variance across PCA Dimensions ($d \in [2, 64]$) | Representation Audit | Ch 3 | `results/phase6_5_4_pca_variance.csv` | VERIFIED |
| **Table 4.1** | Classical and Quantum Model Architecture Specifications | System Design | Ch 4 | `src/app/quantum_model_manager.py` & configs | VERIFIED |
| **Table 4.2** | FastAPI Operational Endpoints and Error Contracts | Application Design | Ch 4 | `reports/gate_b2/GATE_B2_APPLICATION_IMPLEMENTATION_REPORT.md`| VERIFIED |
| **Table 5.1** | Performance Summary of the Original Flat 7-Class Benchmark | Experimental Results | Ch 5 | `reports/COMPLETE_RESEARCH_RESULTS.md` (Table 1) | VERIFIED |
| **Table 5.2** | Definitive Matched Comparison: QSVC Q8-R1 vs. Classical RBF-SVM ($d=8$)| Experimental Results | Ch 5 | `results/phase6_5_5_matched_comparison.csv` | VERIFIED |
| **Table 5.3** | Secondary Model Comparison: QSVC Q8-R1 vs. Weighted Classical Adapter | Experimental Results | Ch 5 | `results/phase6_5_5_C_adapter_comparison.csv` | VERIFIED |
| **Table 5.4** | Quantum Architecture Ablation: Qubit Scaling and Circuit Depth | Ablation Results | Ch 5 | `results/phase6_5_5_C_quantum_ablation.csv` | VERIFIED |
| **Table 5.5** | Computational Latency and Wall-Clock Execution Time Audit | Complexity Audit | Ch 5 | `results/phase6_5_5_runtime.csv` | VERIFIED |
| **Table 5.6** | Forensic Metric Reconciliation and Historical Reporting Corrections | Audit & Provenance | Ch 5 | `reports/FINAL_CLAIM_CORRECTIONS.md` | VERIFIED |
| **Table 5.7** | Macro-F1 and Intent Preservation Across Acoustic Noise Conditions | Noise Perturbation | Ch 5 | `phase5/phase5_results_report.md` | VERIFIED |

### 13.2 Master Figures Register

| Figure ID | Proposed Figure Caption | Type | Chapter | Authoritative Source File | Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Figure 1.1** | High-level conceptual overview of atypical speech personalization | Schematic | Ch 1 | `[To be drawn / exported based on Fig 8]` | PLANNED |
| **Figure 3.1** | Acoustic waveform and log-Mel spectrogram of dysarthric speech sample | Acoustic Analysis | Ch 3 | `[To be exported from Audio Lab]` | PLANNED |
| **Figure 3.2** | Whisper-tiny bottleneck feature extraction and pooling workflow | Pipeline Diagram | Ch 3 | `docs/application/API_REFERENCE.md` | VERIFIED |
| **Figure 4.1** | Longitudinal research architecture bridging Track 1 and Track 2 | Workflow Architecture| Ch 4 | `reports/phase6_5_6/figures/fig8_evidence_chain_workflow.png` | VERIFIED |
| **Figure 4.2** | Cumulative explained variance vs. feature dimensionality curve | Representation Audit | Ch 4 | `reports/phase6_5_4/figures/explained_variance_vs_dimension.png`| VERIFIED |
| **Figure 4.3** | Eigenvalue decay spectrum of the 8-qubit quantum fidelity kernel matrix | Spectral Analysis | Ch 4 | `reports/phase6_5_5/figures/quantum_kernel_eigenvalues.png` | VERIFIED |
| **Figure 4.4** | Quantum Research Command Center overview dashboard interface | Application View | Ch 4 | `reports/application/figures/01_overview_dashboard.png` | VERIFIED |
| **Figure 4.5** | Quantum Audio Lab 5-stage ingestion and inference workflow interface | Application View | Ch 4 | `reports/application/figures/02_quantum_audio_lab.png` | VERIFIED |
| **Figure 4.6** | Verified Model and Speaker Registry interface | Application View | Ch 4 | `reports/application/figures/03_model_speaker_registry.png` | VERIFIED |
| **Figure 4.7** | Interactive Research Results Explorer interface | Application View | Ch 4 | `reports/application/figures/04_research_results_explorer.png` | VERIFIED |
| **Figure 4.8** | Cryptographic Provenance and Checksum Audit workspace | Application View | Ch 4 | `reports/application/figures/05_provenance_workspace.png` | VERIFIED |
| **Figure 4.9** | Runtime Diagnostics and Telemetry interface | Application View | Ch 4 | `reports/application/figures/06_runtime_diagnostics.png` | VERIFIED |
| **Figure 4.10**| Safety Governance, Regulatory Boundaries, and Help interface | Application View | Ch 4 | `reports/application/figures/07_help_and_limitations.png` | VERIFIED |
| **Figure 5.1** | Longitudinal research performance trajectory across project milestones | Benchmark Trajectory | Ch 5 | `reports/phase6_5_6/figures/fig1_original_vs_recovery_progression.png`| VERIFIED |
| **Figure 5.2** | Primary matched comparison: QSVC Q8-R1 vs. Classical RBF-SVM ($d=8$) | Comparative Chart | Ch 5 | `reports/phase6_5_6/figures/fig5_matched_q8_vs_rbf.png` | VERIFIED |
| **Figure 5.3** | Cross-model emergency recall and false-emergency trade-off curve | Performance Trade-off| Ch 5 | `reports/phase6_5_6/figures/fig6_emergency_recall_comparison.png`| VERIFIED |
| **Figure 5.4** | Statevector CPU simulation latency and total wall-clock scaling | Complexity Breakdown| Ch 5 | `reports/phase6_5_6/figures/fig7_quantum_computational_cost.png` | VERIFIED |
| **Figure 5.5** | Historical flat 7-class benchmark comparison showing majority collapse | Baseline Results | Ch 5 | `results/phase4/figures/classical_vs_quantum_macro_f1.png` | VERIFIED |
| **Figure 5.6** | Macro-F1 degradation across environmental noise conditions (SNR) | Noise Robustness | Ch 5 | `phase5/figures/macro_f1_vs_snr.png` | VERIFIED |
| **Figure 5.7** | Intent Preservation Error Rate (IPER) response to acoustic degradation | Noise Robustness | Ch 5 | `phase5/figures/iper_vs_snr.png` | VERIFIED |
| **Figure 5.8** | Speaker-wise performance heterogeneity across the 15 TORGO speakers | Speaker Analysis | Ch 5 | `phase6/figures/speaker_variation.png` | VERIFIED |
| **Figure 5.9** | Hierarchical confusion matrix showing triage and subtype accuracy | Error Analysis | Ch 5 | `phase6_5/hierarchical/figures/confusion_matrix_end_to_end.png`| VERIFIED |
| **Figure 5.10**| Out-of-fold threshold calibration vs. test FER generalization curve | Calibration Analysis | Ch 5 | `reports/project_readiness/figures/threshold_stability_by_fold.png`| VERIFIED |

---

## 14. Evidence Register and Claims Requiring Verification

| Claim ID | Proposed Report Claim | Supporting Evidence | File Path / Location | Verification Status | Limitation or Required Action |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **CLM-01** | The canonical operational speech classifier is exclusively `qsvc_q8_r1`. | Source code, configuration, exception handlers. | `src/app/quantum_model_manager.py:42-45` | **VERIFIED FROM SOURCE** | Fully verified; classical models return HTTP 400. |
| **CLM-02** | The TORGO cohort comprises 15 eligible speakers (8 dysarthric, 7 control). | Dataset validation audit, audio files, manifests. | `reports/torgo_summary.md:14-25` | **VERIFIED FROM DATASET** | Fully verified; exactly 15 verified speaker subdirectories. |
| **CLM-03** | Upstream acoustic features are 384-D frozen Whisper-tiny bottleneck embeddings. | Embedding extractor code, cached `.npy` array shapes. | `src/embeddings/whisper_encoder.py` | **VERIFIED FROM SOURCE** | Feature dimension is exactly 384, L2 normalized. |
| **CLM-04** | The quantum kernel uses an 8-qubit $ZZ\text{FeatureMap}$ with 1 repetition. | Quantum kernel engine code and model checkpoint metadata. | `src/quantum/fidelity_kernel.py:35-50` | **VERIFIED FROM SOURCE** | Operates on 8 PCA features in 256-D Hilbert space. |
| **CLM-05** | QSVC Q8-R1 Macro-F1 is $0.1051 \pm 0.0575$ vs. $0.1087 \pm 0.0487$ for RBF-SVM ($d=8$). | Empirical evaluation CSV across 60 matched conditions. | `results/phase6_5_5_matched_comparison.csv` | **VERIFIED FROM EXPERIMENT** | Statistically indistinguishable ($p=0.3428$). |
| **CLM-06** | The empirical difference between QSVC and RBF-SVM is not statistically significant. | Wilcoxon signed-rank and Holm-Bonferroni test outputs. | `results/phase6_5_5_statistical_tests.csv` | **VERIFIED FROM EXPERIMENT** | Holm $p=0.3428$ (F1), $p=0.0689$ (Recall), $p=0.9505$ (FER). |
| **CLM-07** | Classical RBF-SVM ($d=8$) achieves $31.89\%$ emergency recall vs. $27.52\%$ for QSVC. | Paired matched comparison dataset. | `results/phase6_5_5_matched_comparison.csv` | **VERIFIED FROM EXPERIMENT** | Difference $-4.37\text{ pp}$ is not statistically significant. |
| **CLM-08** | Weighted Classical Adapter achieves $45.81\%$ emergency recall ($p < 0.001$). | Adapter comparative results dataset. | `results/phase6_5_5_C_adapter_comparison.csv` | **VERIFIED FROM EXPERIMENT** | Higher recall, but incurs high $49.82\%$ false-emergency rate. |
| **CLM-09** | QSVC statevector simulation requires $772.6\times$ higher per-sample test latency. | CPU execution latency benchmarking CSV. | `results/phase6_5_5_runtime.csv` | **VERIFIED FROM EXPERIMENT** | $1.0816\text{ s}$ per test sample vs. $0.0014\text{ s}$ for RBF. |
| **CLM-10** | PCA with $d=8$ retains $89.73\% \pm 2.99\%$ cumulative explained variance. | PCA dimensionality forensic audit CSV. | `results/phase6_5_4_pca_variance.csv` | **VERIFIED FROM EXPERIMENT** | Reconciled from historical $27.4\%$ plotting placeholder. |
| **CLM-11** | End-to-end emergency precision on the natural test distribution is $0.81\% \pm 1.31\%$. | Full-cohort metric reconciliation report. | `reports/phase6_5_6_metric_reconciliation.md`| **VERIFIED FROM AUDIT** | Corrects historical $11.24\%$ balanced-subset artifact. |
| **CLM-12** | Gate C and Gate D safety gates are strictly blocked. | Governance checklists and safety evidence registers. | `reports/application/GATE_C_EVIDENCE...csv` | **VERIFIED FROM GOVERNANCE** | Blocked due to external cohort & clinical evidence requirements. |
| **CLM-13** | The running application has verified screenshots across 7 distinct views. | PNG image files in reports directory. | `reports/application/figures/01_...png` | **SUPPORTED BY SCREENSHOT** | All 7 high-resolution screenshots audited and verified. |
| **CLM-14** | Gate B.2 automated test suite achieves 100% pass rate (47/47 tests). | Pytest execution logs and acceptance audit. | `tests/gate_b2/` | **VERIFIED FROM TEST SUITE** | Verified passing in 28.9s on October 11, 2026. |

---

## 15. Literature and Reference Verification Register

All references must be formatted in strict alphabetical order by first author surname, complying with Amrita School of Computing report template standards:

1. **Abbaszadeh-Mir, Y., Mayer, J., Cloutier, G. and Fortin, C.**, *Theory and Simulation for the Identification of the Link Geometric Errors for a Five-Axis Machine Tool Using a Telescoping Magnetic Ball-Bar*, International Journal of Production Research, vol. 40 (18), pp. 4781–4797, 2002. `[Template Formatting Specimen Reference]`
2. **Biamonte, J., Wittek, P., Pancotti, N., Rebentrost, P., Wiebe, N. and Lloyd, S.**, *Quantum Machine Learning*, Nature, vol. 549 (7671), pp. 195–202, 2017. DOI: `10.1038/nature23474`. `[Chapters 1, 2, 3]`
3. **Cerezo, M., Arrasmith, A., Babbush, R., Benjamin, S. C., Endo, S., Fujii, K., McClean, J. R., Mitarai, K., Yuan, X., Cincio, L. and Coles, P. J.**, *Variational Quantum Algorithms*, Nature Reviews Physics, vol. 3 (9), pp. 625–644, 2021. DOI: `10.1038/s42254-021-00348-9`. `[Chapters 2, 4]`
4. **Christensen, H., Cunningham, S., Fox, C., Green, P. and Hain, T.**, *A Comparative Study of Adaptive, Non-Adaptive, and Feature-Space Maximum Likelihood Techniques for Dysarthric Speech Recognition*, IEEE Transactions on Audio, Speech, and Language Processing, vol. 20 (4), pp. 1111–1124, 2012. DOI: `10.1109/TASL.2011.2173679`. `[Chapters 1, 2]`
5. **Cortes, C. and Vapnik, V.**, *Support-Vector Networks*, Machine Learning, vol. 20 (3), pp. 273–297, 1995. DOI: `10.1007/BF00994018`. `[Chapters 2, 3]`
6. **Green, J. R., MacDonald, C. A., Choi, J. Y. and Wang, J.**, *Automatic Assessment of Neuromotor Speech Impairment: Recent Advances and Clinical Opportunities*, Current Opinion in Neurology, vol. 34 (6), pp. 845–852, 2021. DOI: `10.1097/WCO.0000000000000994`. `[Chapters 1, 2]`
7. **Havlicek, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M. and Gambetta, J. M.**, *Supervised Learning with Quantum-Enhanced Feature Spaces*, Nature, vol. 567 (7747), pp. 209–212, 2019. DOI: `10.1038/s41586-019-0980-2`. `[Chapters 1, 2, 3, 4]`
8. **Holm, S.**, *A Simple Sequentially Rejective Multiple Test Procedure*, Scandinavian Journal of Statistics, vol. 6 (2), pp. 65–70, 1979. `[Chapters 3, 5]`
9. **Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M. and Gelly, S.**, *Parameter-Efficient Transfer Learning for NLP*, Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR, vol. 97, pp. 2790–2799, 2019. `[Chapters 2, 3, 4]`
10. **Huang, H.-Y., Broughton, M., Mohseni, M., Babbush, R., Boixo, S., Neven, H. and McClean, J. R.**, *Power of Data in Quantum Machine Learning*, Nature Communications, vol. 12 (1), p. 2631, 2021. DOI: `10.1038/s41467-021-22539-9`. `[Chapters 2, 5]`
11. **Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C. and Sutskever, I.**, *Robust Speech Recognition via Large-Scale Weak Supervision*, arXiv preprint arXiv:2212.04356, 2022. `[Chapters 1, 2, 3, 4]`
12. **Rudzicz, F., McNamara, D. S. and Lu, L.**, *The TORGO Database of Acoustic and Articulatory Speech from Speakers with Dysarthria*, Language Resources and Evaluation, vol. 46 (4), pp. 523–541, 2012. DOI: `10.1007/s10579-011-9145-0`. `[Chapters 1, 2, 3, 4, 5]`
13. **Schuld, M. and Killoran, N.**, *Quantum Machine Learning in Feature Hilbert Spaces*, Physical Review Letters, vol. 122 (4), p. 040504, 2019. DOI: `10.1103/PhysRevLett.122.040504`. `[Chapters 1, 2, 3, 4]`
14. **Shigley, I. E. and Uicker Jr., J. J.**, *Theory of Machines and Mechanisms*, Second Edition, McGraw Hill, 1996. `[Template Formatting Specimen Reference]`
15. **Vapnik, V.**, *The Nature of Statistical Learning Theory*, Springer-Verlag, New York, 1995. `[Chapters 2, 3]`
16. **Wilcoxon, F.**, *Individual Comparisons by Ranking Methods*, Biometrics Bulletin, vol. 1 (6), pp. 80–83, 1945. DOI: `10.2307/3001968`. `[Chapters 3, 5]`

---

## 16. Missing Information and Prioritized Action Register

| Action ID | Priority | Category | Description of Missing Item or Action | Source File or Location to Consult | Recommended Action |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ACT-01** | **P0** | Administrative | Student Registration Numbers and Official Student Names. | Student ID cards / Academic records | Insert official team members' names and roll numbers into Specimen 1, 2, 3. |
| **ACT-02** | **P0** | Administrative | Official Project Supervisor Name, Designation, and Department. | Department Allocation List | Insert supervisor's name, title, and department into Specimen 1, 2, 3, 5. |
| **ACT-03** | **P1** | Report Compilation | Official Turnitin / Plagiarism Similarity Certificate. | University Library / Supervisor Portal | Run final report through university Turnitin and append digital receipt. |
| **ACT-04** | **P1** | Visual Assets | Standalone schematic diagram for Figure 1.1 (Introduction). | Can adapt `reports/phase6_5_6/figures/fig8_evidence_chain_workflow.png` | Export clean, vector/grayscale high-res diagram for Chapter 1. |
| **ACT-05** | **P2** | Visual Assets | Standalone acoustic spectrogram image for Figure 3.1. | Audio Lab or Librosa spectrogram dump | Capture sample spectrogram from dysarthric speaker `F01` using `librosa.display`. |
| **ACT-06** | **P2** | Citations | Add 4–6 additional clinical dysarthria domain citations if desired. | IEEE / ACM Digital Library | Include recent 2023–2025 speech pathology papers to enhance Chapter 2. |
| **ACT-07** | **P3** | Formatting | Final Word template formatting audit against printed specimen. | `Project Report Template (1).docx` | Verify exact 1 cm paragraph tabs and 1.5 line spacing in assembled `.docx`. |

---

## 17. Recommended Report Assembly Sequence

To assemble the final Word/PDF project report with zero friction, follow this systematic multi-phase execution sequence:

```
[Phase 1: Administrative Setup]
  - Populate Title Page, Bonafide Certificate, Declaration, Acknowledgement with official names.
  - Apply exact page setup: A4, 1-inch margins, Times New Roman 12 pt, 1.5 line spacing.

[Phase 2: Preliminary Text Assembly]
  - Draft Executive Abstract (Section 12, Specimen 4).
  - Draft Table of Contents, List of Tables, List of Figures, List of Abbreviations.

[Phase 3: Core Research Chapters (Chapters 1, 2, 3)]
  - Chapter 1: Introduction, Problem Definition, Research Objectives.
  - Chapter 2: Literature Review, Critical Synthesis, Identification of Research Gap.
  - Chapter 3: TORGO Dataset, Preprocessing, Whisper Extraction, PCA Audit, Model Formulation.

[Phase 4: Engineering & Implementation Chapters (Chapter 4)]
  - System Architecture, Track 1 vs. Track 2 workflows.
  - Quantum-Only Command Center Implementation, Model Manager, Fail-Closed Security.
  - Embed Figures 4.1 through 4.10 (Arch diagrams & Application screenshots).

[Phase 5: Results & Empirical Closure (Chapters 5 & 6)]
  - Embed Tables 5.1 through 5.7 and Figures 5.1 through 5.10.
  - Detail Matched Comparison (Case B finding), Ablations, Runtime Audit, Noise Perturbation.
  - Chapter 6: Summary of Findings, Research Limitations, Scope for Future Work.

[Phase 6: References & Appendices Assembly]
  - Insert Alphabetical Bibliography (Section 15).
  - Append Hyperparameter registries, speaker-wise tables, and Plagiarism report.
```

---

## 18. Final Pre-Submission Verification Checklist

Prior to final binding and university submission, verify compliance with each of the following criteria:

- [x] **Project Title Exactness:** Matches *"Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication"* on every cover, title, certificate, and declaration page.
- [x] **Template Formatting Compliance:** A4 paper, 1-inch margins on all four sides, Times New Roman font, 1.5 line spacing, 1 cm paragraph indentation.
- [x] **Headings Hierarchy:** Chapter headings 14 pt bold uppercase centered; Division headings 12 pt bold uppercase left; Subdivision headings 12 pt bold title case left.
- [x] **Chapter Structure Preserved:** Exactly 6 core chapters following the university specification.
- [x] **System Duality Respected:** Research compares quantum and classical models; deployed application is strictly quantum-only (`qsvc_q8_r1`). Classical models never described as live or fallback options.
- [x] **Scientific Integrity Maintained:** Verdict is honestly stated as Case B (No statistically supported quantum advantage); p-values, confidence intervals, and effect sizes accurately reported without fabrication.
- [x] **Runtime Overhead Clarified:** Statevector CPU simulation latency explicitly demarcated as $772.6\times$ per-sample test overhead and $212.8\times$ total experiment wall-clock time.
- [x] **Safety Governance Declared:** Gate C and Gate D declared as strictly blocked (`BLOCKED_EXTERNAL_PREREQUISITES`, `BLOCKED_SAFETY_EVIDENCE`); clinical deployment acknowledged as infeasible.
- [x] **Captions & Numbering:** All figures and tables follow chapter-based numbering (`Table 3.1`, `Figure 4.2`, `Figure 5.3`).
- [x] **Alphabetical References:** All bibliographic citations arranged strictly in alphabetical order by first author surname.
- [x] **Code and Model Integrity:** No trained models, checkpoints, weights, or datasets modified during this audit.
