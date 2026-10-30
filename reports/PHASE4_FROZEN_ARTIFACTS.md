# Phase 4 Frozen Artifact Registry & Cryptographic Hash Fingerprints

**Date Frozen:** 2026-09-30  
**Audit Stage:** Phase 4.5 Closure  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Registry Purpose:** Establish cryptographic immutability and provenance for all canonical Phase 4 experimental artifacts prior to Phase 5.

---

## 1. Primary Empirical Data Artifacts

| Relative File Path | Size (Bytes) | SHA-256 Checksum Hash | Verification Description |
|---|---|---|---|
| `results/phase4/quantum_curves.csv` | 42,188 | `67a2a506a4fd624543470668a4b6ac33dc574165cebd681e43b043a6a3d3da65` | Full 240 condition sweep results ($15\text{ speakers} \times 4 N \times 4 q$) |
| `results/phase4/quantum_predictions.csv` | 9,775,910 | `5a4e497ac46d1bd41deb8691807f56e1afa30d91440e40bc091160c86d96da16` | All 85,072 per-recording prediction records with IPER metadata |
| `results/phase4/quantum_aggregated.csv` | 2,183 | `436c1695ce461c577ecc6696e65e160845bd0813a3a3e99521d2fb3e018ecf4e` | Speaker-macro aggregated means, std, and 95% Student-$t$ CIs |
| `results/phase4/ablation_table.csv` | 929 | `7ce2838c9b2481081d980478ae49e0ecad13de9abad64af0563c780c651c9bb9` | Parameter and compute transparency matching table |
| `results/phase4/classical_vs_quantum_comparison.csv` | 607 | `9fba72141f83933b151e6dae786f8a9c4df3273deb0c94209fad8b6673616c85` | Three-way matched comparison dataset across budgets |

---

## 2. Configuration & Parameter Artifacts

| Relative File Path | Size (Bytes) | SHA-256 Checksum Hash | Verification Description |
|---|---|---|---|
| `configs/phase4_quantum.yaml` | 1,356 | `7a08eed84bf17b0b8c29e129ba3587f5b15df62bf4f4e76e7bd838f9d567fd19` | Immutable Phase 4 quantum architecture specification |

---

## 3. High-Resolution Visualizations

| Relative File Path | Size (Bytes) | SHA-256 Checksum Hash | Verification Description |
|---|---|---|---|
| `results/phase4/figures/quantum_sample_efficiency.png` | 205,093 | `6bcf3a964ff6892fe76f6acf5fcf4c80fd3f5507e311b13df33876dfc62c069e` | Qubit ablation few-shot sample-efficiency curves |
| `results/phase4/figures/classical_vs_quantum_macro_f1.png`| 139,925 | `1551e4ff60a93b5d7654273095b64a314dd729e0d52783ee49df76351ceeb1b1` | Three-way Macro-F1 comparison (Adapter vs RBF vs QSVC $q=4$) |
| `results/phase4/figures/qubit_ablation_figure.png` | 196,080 | `658ff1174df335ffb5a82eee2a598975305b762fa34430a344eb1ab1faa217ba` | Dedicated qubit dimension vs PCA variance and train runtime |

---

## 4. Formal Research Reports

| Relative File Path | Size (Bytes) | SHA-256 Checksum Hash | Verification Description |
|---|---|---|---|
| `reports/phase4_complete_results.md` | 18,454 | `d38eb446e784a2ff4bba5380dd8b44aa93186d74ffc4a0ee390a7f25923769f6` | Master Phase 4 results report with complete findings |
| `reports/phase4_kernel_validation.md` | 5,229 | `b0d3fdade94a1ae690cfefb43758421a7dddd68de26c529e3b41a14728584bb7` | Comprehensive Mercer PSD and condition number audit |
| `reports/phase4_ablation_report.md` | 5,139 | `2e7f30e7b523c91cf2868a71a8b5e933669a05ca11b4b67001bafb9187085c6c` | Qubit ablation and circuit complexity analysis |
| `reports/phase4_environment.md` | 4,185 | `b9feb3ff16ad309f037a9021d55633f3687ba75a586f3bcb2586043bd6c2d029` | Software dependencies, Qiskit version, and simulator details |
| `reports/phase4_fairness_audit.md` | 3,691 | `cb8ac19d0849ec62db168b27ea3adf592500c636ead4f09ee1f89fbeca71c1f9` | Controlled evaluation fairness audit |

---

## 5. Pre-Audit Historical Archive

In accordance with project integrity rules, the unedited pre-audit prediction file was archived prior to enhancement:
* Archive Directory: `results/archive/phase4_pre_audit/`
* Archive File: `results/archive/phase4_pre_audit/quantum_predictions.csv`
* Archive SHA-256: `dde5af8f59368e366966f8b123a73d774c09a599f78c377449140f6cc0f86520`

---

## 6. Freeze Declaration

```
==================================================
PHASE 4 CANONICAL ARTIFACTS FROZEN
ANY MODIFICATION TO THESE FILES INVALIDATES PHASE 4.5
==================================================
```
