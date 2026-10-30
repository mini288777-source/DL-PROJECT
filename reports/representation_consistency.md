# Representation Dimensionality & Consistency Audit

**Date:** 2026-09-30  
**Phase:** 3.5 — Research Validity, Consistency & Phase 4 Readiness Gate  
**Target File:** `reports/representation_consistency.md`  

---

## 1. Dimensionality Verification Matrix

To ensure rigorous scientific validity before Phase 4 (Quantum Kernel), every artifact across the experimental pipeline was audited for dimensionality consistency.

| Pipeline Stage / Artifact | Source Code / File Path | Declared Dimension | Actual Verified Dimension | Agreement Status |
|---|---|---|---|---|
| **Raw Model Output** | `src/embeddings/whisper_encoder.py` (`model.config.d_model`) | 384 | `(batch_size, 384)` | **PERFECT MATCH** |
| **Acoustic Config** | `configs/embedding_config.yaml` (`model.embedding_dim`) | 384 | 384 | **PERFECT MATCH** |
| **Cache Binary Files** | `embeddings/torgo/whisper/*.npy` (6,067 files) | 384 | `shape == (384,)`, `dtype == float32` | **PERFECT MATCH** |
| **Cache Master Manifest** | `embeddings/metadata/embedding_manifest.csv` (`whisper_dim`) | 384.0 | 384.0 (100% of 6,067 rows) | **PERFECT MATCH** |
| **Phase 3 Config** | `configs/phase3_baselines.yaml` (`model.adapter.input_dim`) | 384 | 384 | **PERFECT MATCH** |
| **Classical Adapter Input** | `src/models/classical_adapter.py` (`ClassicalAdapterNet.input_dim`) | 384 | `in_features=384` | **PERFECT MATCH** |
| **Classical RBF-SVM Input** | `src/models/rbf_svm.py` (`fit(X, y)` input shape) | 384 | `X.shape[1] == 384` | **PERFECT MATCH** |
| **Phase 3 Results** | `results/phase3/baseline_curves.csv` (`embedding_dim`) | 384 | 384 (100% of 120 rows) | **PERFECT MATCH** |
| **Phase 4 Quantum Contract** | `reports/phase3_5_phase4_comparability.md` (PCA Input Dim) | 384 | 384 (exact input to PCA) | **PERFECT MATCH** |

$$\text{Model Output } (384) = \text{Cached Vector } (384) = \text{Adapter Input } (384) = \text{RBF Input } (384) = \text{Quantum PCA Input } (384)$$

---

## 2. Forensic Code Checks

### 2.1 Whisper Encoder Output Shape
```python
# src/embeddings/whisper_encoder.py
hidden_states = self.model(input_features).last_hidden_state  # (B, T, 384)
pooled = hidden_states.mean(dim=1)                             # (B, 384)
normalized = pooled / pooled.norm(dim=-1, keepdim=True)       # (B, 384)
```
No ad-hoc truncation, slicing, or zero-padding is performed. The vector is the full, unreduced mean-pooled transformer representation.

### 2.2 Cache Storage Integrity
All 6,067 audio files in the 15-speaker few-shot cohort were loaded directly via `np.load()`:
* **Total files:** 6,067
* **Dimensions:** Shape `(384,)` on 100.0% of files.
* **Data Type:** `numpy.float32` on 100.0% of files.
* **Vector Norm:** $\|\mathbf{e}\|_2 = 1.0 \pm 10^{-6}$ (strictly unit-sphere normalized).
* **Non-finite Values:** Zero NaNs, zero Infs across all 2,329,728 scalar elements ($6067 \times 384$).

### 2.3 Classical Baseline Inputs
In `src/models/classical_adapter.py`:
```python
class ClassicalAdapterNet(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, num_classes: int = 7, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 384 -> 64
        self.norm = nn.LayerNorm(hidden_dim)          # 64
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes) # 64 -> 7
```
Total trainable parameters:
$$\text{Parameters} = (384 \times 64 + 64) + (64 \times 2) + (64 \times 7 + 7) = 24,640 + 128 + 455 = 25,223$$
This matches the specification of 25,223 parameters.

In `src/models/rbf_svm.py`:
The SVM receives the exact $N \times 384$ numpy array without dimensionality reduction or feature filtering.

---

## 3. Future Quantum Input Specification (Phase 4 Contract)

To prevent dimensionality mismatch in Phase 4:
1. **Raw Acoustic Vector:** The input vector fed into the Phase 4 pipeline MUST be the cached 384-dimensional Whisper embedding from `embeddings/torgo/whisper/<recording_id>.npy`.
2. **Standardization / Projection:** A `sklearn.decomposition.PCA(n_components=n_qubits)` will be fitted **strictly on the adaptation set** ($N \in \{20, 30, 40, 50\}$).
3. **Qubit Mapping:** The PCA-projected vector of dimension $d \in \{2, 4, 6, 8\}$ will be scaled to $[0, 2\pi]$ and passed directly into the Qiskit `zz_feature_map` / `ZZFeatureMap`.
4. **No Artificial Reshaping:** No reshaping, truncation, or synthetic dimension padding will occur.

## 4. Conclusion

**BLOCKING ISSUE #2 STATUS: PASS**  
Dimensionality consistency across the entire pipeline is verified. All artifacts agree on $D = 384$.
