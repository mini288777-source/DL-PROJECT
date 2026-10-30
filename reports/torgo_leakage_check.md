# TORGO Experimental Split Leakage Audit Report

**Date:** 2026-09-30  
**Random Seed:** `42`  
**Split Protocol:** Nested Few-Shot (`20 ⊂ 30 ⊂ 40 ⊂ 50`) with Prompt-Level Grouping  
**Audit Status:** PASSED (ZERO LEAKAGE)  

---

## 1. Leakage Verification Criteria

| Audit Check | Status | Verification Rule |
|-------------|--------|-------------------|
| No identical recording ID in adaptation and test | **PASS** | Strict disjoint set verification across all speakers and all N. |
| No prompt ID overlap across splits | **PASS** | Simultaneous arrayMic and headMic recordings for prompt X quarantined to single split. |
| Strict nesting integrity (20 ⊂ 30 ⊂ 40 ⊂ 50) | **PASS** | Adaptation subsets at N=20, 30, 40 are strict subsets of N=50. |
| Test set invariant across N sweeps | **PASS** | Held-out test set remains identical across all N sweeps for each speaker. |
| Future embedding cache split isolation | **PASS** | Recording IDs include speaker and prompt metadata for unambiguous cache keys. |

---

## 2. Speaker-by-Speaker Leakage and Sample Audit

| Speaker ID | Total Prompts | Adapt Pool | Test Pool | N=20 Leakage | N=30 Leakage | N=40 Leakage | N=50 Leakage | Overall |
|------------|---------------|------------|-----------|--------------|--------------|--------------|--------------|---------|
| `F01` | 118 | 50 | 68 | 0 | 0 | 0 | 0 | **PASS** |
| `F03` | 213 | 50 | 163 | 0 | 0 | 0 | 0 | **PASS** |
| `F04` | 250 | 50 | 200 | 0 | 0 | 0 | 0 | **PASS** |
| `FC01` | 151 | 50 | 101 | 0 | 0 | 0 | 0 | **PASS** |
| `FC02` | 975 | 50 | 925 | 0 | 0 | 0 | 0 | **PASS** |
| `FC03` | 405 | 50 | 355 | 0 | 0 | 0 | 0 | **PASS** |
| `M01` | 285 | 50 | 235 | 0 | 0 | 0 | 0 | **PASS** |
| `M02` | 232 | 50 | 182 | 0 | 0 | 0 | 0 | **PASS** |
| `M03` | 406 | 50 | 356 | 0 | 0 | 0 | 0 | **PASS** |
| `M04` | 286 | 50 | 236 | 0 | 0 | 0 | 0 | **PASS** |
| `M05` | 357 | 50 | 307 | 0 | 0 | 0 | 0 | **PASS** |
| `MC01` | 419 | 50 | 369 | 0 | 0 | 0 | 0 | **PASS** |
| `MC02` | 383 | 50 | 333 | 0 | 0 | 0 | 0 | **PASS** |
| `MC03` | 594 | 50 | 544 | 0 | 0 | 0 | 0 | **PASS** |
| `MC04` | 993 | 50 | 943 | 0 | 0 | 0 | 0 | **PASS** |

---

## 3. Conclusion

All 15 speakers passed the leakage audit with zero prompt overlap, zero recording overlap, and strict nested set compliance.
