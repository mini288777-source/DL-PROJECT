# Phase 2 — Embedding Cache Audit Report

## 1. Executive Summary
- **Overall Status:** PASS
- **Audit Timestamp:** 2026-09-29 20:27:58 UTC
- **Target Split Cohort:** `adaptation,test`
- **Target Recordings:** 6067 across 15 speakers
- **Expected Dimensions:** Whisper = 384, MFCC = 78

---

## 2. Integrity Metrics Table

| Metric | Target | Actual | Status |
| :--- | :--- | :--- | :--- |
| **Whisper Embeddings Cached** | 6067 | 6067 | PASS |
| **Whisper Corruptions (NaN/Inf/Shape)** | 0 | 0 | PASS |
| **Whisper Missing Recordings** | 0 | 0 | PASS |
| **MFCC Features Cached** | 6067 | 6067 | PASS |
| **MFCC Corruptions (NaN/Inf/Shape)** | 0 | 0 | PASS |
| **MFCC Missing Recordings** | 0 | 0 | PASS |
| **Metadata Manifest Duplicate IDs** | 0 | 0 | PASS |
| **Orphan Whisper Files** | 0 | 0 | PASS |
| **Orphan MFCC Files** | 0 | 0 | PASS |

---

## 3. Dimension & Storage Audit
- **Whisper Dimension:** 384 float32 values per recording (1536 bytes raw array)
- **MFCC Dimension:** 78 float32 values per recording (312 bytes raw array)
- **Zero-Dependency Reload:** Verified via `numpy.load` without model weights or deep learning frameworks.
- **Traceability:** 100% of `.npy` filenames strictly match `recording_id`.

---

## 4. Issues and Corrective Actions
- No integrity violations detected.
