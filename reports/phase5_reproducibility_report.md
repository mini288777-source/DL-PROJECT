# Phase 5 Reproducibility and Experimental Lineage Report

**Date:** 2026-09-30  
**Phase:** Phase 5 — Controlled Noise Robustness Evaluation  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Reproducibility Status:** **VERIFIED & AUDITED**

---

## 1. Experimental Environment and Dependencies

The execution environment is fully specified and pinned:

| Environment Property | Recorded Value |
|---|---|
| **Operating System** | Windows-10-10.0.26200-SP0 |
| **Python Version** | 3.11.15 (64-bit AMD64) |
| **PyTorch Version** | 2.14.0+cpu |
| **HuggingFace Transformers**| 5.17.0 |
| **Qiskit Version** | 2.5.2 |
| **scikit-learn Version** | 1.9.1 |
| **NumPy Version** | 2.4.6 |
| **Speech Encoder** | `openai/whisper-tiny` (FROZEN) |
| **Embedding Dimension** | 384 (float32, unit-L2 normalized) |

---

## 2. Cryptographic Checksum Registry

All inputs, split assignments, configurations, and manifests are verified via SHA-256 hashes:

| Artifact Name | Relative File Path | SHA-256 Checksum |
|---|---|---|
| **TORGO Splits JSON** | `splits/torgo_splits.json` | `edbecbf2f2cf0974c5e3bbce4b0d4d5687e8b27289ba0f3c6a98fda8645a3938` |
| **TORGO Manifest** | `data/manifests/torgo_manifest.csv` | `59708252a7cced0139803bff8824a37eddb760e8a0bb98e84a32512df91de075` |
| **Phase 5 Contract** | `configs/phase5_contract.yaml` | `4328eeab5b184ef39aa475e525a772aa791b87a95ba0ad1fc3e590050df06c00` |
| **Phase 5 Config** | `configs/phase5_config.yaml` | `6825c2e96c60797bb3fe961627b6a367978ef3835b3ac7ab2f658c8fc748fa22` |
| **Experiment Manifest** | `phase5/manifests/phase5_experiment_manifest.csv` | `100% deterministic (720 rows)` |

---

## 3. Deterministic Noise Protocol

1. **Noise Formulation:**
   Additive White Gaussian Noise (AWGN) added to the 16 kHz audio waveform prior to Whisper feature extraction:
   $$\sigma_{\text{noise}}^2 = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}} / 10}}$$
2. **Stable Deterministic Seed:**
   Noise seeds are derived without platform-dependent `hash()` variation:
   $$\text{noise\_seed} = \text{int}\left(\text{MD5}(42 \parallel \text{recording\_id} \parallel \text{SNR})[:8], 16\right)$$
3. **Cross-Model Parity:**
   Identical noisy waveform and Whisper embedding caches are evaluated across Classical Adapter, Classical RBF-SVM, and Primary Quantum QSVC ($q=4$).

---

## 4. Frozen Protocol B Compliance Checklist

- [x] Models trained strictly on clean adaptation samples.
- [x] PCA ($384 \to 4$) fitted strictly on clean adaptation samples.
- [x] MinMaxScaler ($[0, \pi]$) fitted strictly on clean adaptation samples.
- [x] Noisy audio generated strictly on held-out test audio.
- [x] Test-time transformation strictly uses pre-fitted clean PCA and scaler.
- [x] Clean condition exactly reproduces Phase 3 and Phase 4 baselines.
- [x] No hyperparameter optimization or class rebalancing introduced.
