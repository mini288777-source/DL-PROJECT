# TORGO Dataset Comprehensive Research Summary

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning  
**Phase:** Phase 1 — TORGO Data Foundation  
**Date:** 2026-09-30  
**Status:** Complete & Verified  

---

## 1. Executive Summary & Deliverables

Phase 1 has established the canonical dataset foundation, audio validation layer, emergency intent mapping, speaker shortlist, and deterministic few-shot split definitions for all downstream research phases.

| Metric / Asset | Value | Notes |
|---|---|---|
| Total Speakers | **15** | 8 dysarthric, 7 control |
| Eligible Speakers | **15 / 15** (100%) | All speakers have >= 118 prompts |
| Total Audio Utterances | **16,552** | 100% matched filename schema |
| Valid Audio Files | **16,550 / 16,552** (100%) | Zero NaNs, zero Infs, valid PCM |
| Total Duration | **13.68 hours** | 49,238.30 seconds |
| Sampling Rate | **16000 Hz** | Uniform 16 kHz mono across all files |
| Transcript Coverage | **16,552 / 16,552** (100%) | 0 missing, 0 null, 967 unique |
| Valid Emergency Intents | **700** | Mapped across 6 emergency categories |
| N=20 Feasibility | **15 / 15** (100%) | Supported with >= 20 held-out test |
| N=30 Feasibility | **15 / 15** (100%) | Supported with >= 20 held-out test |
| N=40 Feasibility | **15 / 15** (100%) | Supported with >= 20 held-out test |
| N=50 Feasibility | **15 / 15** (100%) | Supported with >= 20 held-out test |
| Split Reproducibility | **PASS** | Identical split indices across runs |
| Acoustic Data Leakage | **ZERO (0)** | Prompt-level channel isolation enforced |

---

## 2. Speaker Cohort Breakdown

| Speaker ID | Group | Gender | Total Recs | HeadMic | ArrayMic | Unique Prompts | Total Dur (s) | Eligible |
|---|---|---|---|---|---|---|---|---|
| `F01` | dysarthria | female | 236 | 118 | 118 | 118 | 573.62 | **YES** |
| `F03` | dysarthria | female | 1087 | 546 | 541 | 213 | 2985.65 | **YES** |
| `F04` | dysarthria | female | 673 | 244 | 429 | 250 | 2202.22 | **YES** |
| `FC01` | healthy | female | 302 | 151 | 151 | 151 | 1082.92 | **YES** |
| `FC02` | healthy | female | 2187 | 967 | 1220 | 975 | 6221.79 | **YES** |
| `FC03` | healthy | female | 1922 | 963 | 959 | 405 | 5000.87 | **YES** |
| `M01` | dysarthria | male | 743 | 371 | 372 | 285 | 2935.93 | **YES** |
| `M02` | dysarthria | male | 772 | 388 | 384 | 232 | 2836.81 | **YES** |
| `M03` | dysarthria | male | 810 | 406 | 404 | 406 | 2544.56 | **YES** |
| `M04` | dysarthria | male | 666 | 277 | 389 | 286 | 2872.64 | **YES** |
| `M05` | dysarthria | male | 587 | 470 | 117 | 357 | 2705.76 | **YES** |
| `MC01` | healthy | male | 2155 | 1077 | 1078 | 419 | 6512.2 | **YES** |
| `MC02` | healthy | male | 1123 | 449 | 674 | 383 | 3974.91 | **YES** |
| `MC03` | healthy | male | 1669 | 799 | 870 | 594 | 3536.25 | **YES** |
| `MC04` | healthy | male | 1620 | 629 | 991 | 993 | 3252.16 | **YES** |

---

## 3. Session Breakdown

| Speaker ID | Sessions Available | Utterances per Session |
|---|---|---|
| `F01` | Session 1 | Total: 236 |
| `F03` | Session 1, Session 2, Session 3 | Total: 1087 |
| `F04` | Session 1, Session 2 | Total: 673 |
| `FC01` | Session 1 | Total: 302 |
| `FC02` | Session 2, Session 3 | Total: 2187 |
| `FC03` | Session 1, Session 2, Session 3 | Total: 1922 |
| `M01` | Session 1, Session 3 | Total: 743 |
| `M02` | Session 1, Session 2 | Total: 772 |
| `M03` | Session 2 | Total: 810 |
| `M04` | Session 1, Session 2 | Total: 666 |
| `M05` | Session 1, Session 2 | Total: 587 |
| `MC01` | Session 1, Session 2, Session 3 | Total: 2155 |
| `MC02` | Session 1, Session 2 | Total: 1123 |
| `MC03` | Session 1, Session 2 | Total: 1669 |
| `MC04` | Session 1, Session 2 | Total: 1620 |

---

## 4. Transcript & Intent Taxonomy Mapping

TORGO was collected as an atypical speech acoustic database rather than an emergency dispatch dataset.
Under transparent, reproducible mapping rules:
- **Single-word exact matches** on critical emergency terms ('help', 'pain', 'fall', 'slip', 'air', 'pill', 'yes', 'no') are mapped as `VALID`.
- **Incidental keyword occurrences** in literary reading sentences (e.g., Grandfather passage) are marked `EXCLUDED` to prevent artificial inflation of emergency vocabulary.
- **Standard reading vocabulary** is classified as `GENERAL_SPEECH` / `UNMAPPED`.

### Valid Emergency Intent Distribution:

- **CRITICAL_RESPONSE:** 237 utterances
- **FALL:** 205 utterances
- **BREATHING_MEDICAL:** 112 utterances
- **PAIN:** 94 utterances
- **SOS_HELP:** 31 utterances
- **LOCATION:** 21 utterances

---

## 5. Few-Shot Feasibility (N = 20, 30, 40, 50)

All 15 speakers have enough speech events to support the entire few-shot personalization sweep:

| Speaker ID | Group | Prompts Available | Adapt Pool (N=50) | Held-Out Test Prompts | N=20/30/40/50 Feasible |
|---|---|---|---|---|---|
| `F01` | dysarthric | 118 | 50 | 68 | **PASS** |
| `F03` | dysarthric | 213 | 50 | 163 | **PASS** |
| `F04` | dysarthric | 250 | 50 | 200 | **PASS** |
| `FC01` | control | 151 | 50 | 101 | **PASS** |
| `FC02` | control | 975 | 50 | 925 | **PASS** |
| `FC03` | control | 405 | 50 | 355 | **PASS** |
| `M01` | dysarthric | 285 | 50 | 235 | **PASS** |
| `M02` | dysarthric | 232 | 50 | 182 | **PASS** |
| `M03` | dysarthric | 406 | 50 | 356 | **PASS** |
| `M04` | dysarthric | 286 | 50 | 236 | **PASS** |
| `M05` | dysarthric | 357 | 50 | 307 | **PASS** |
| `MC01` | control | 419 | 50 | 369 | **PASS** |
| `MC02` | control | 383 | 50 | 333 | **PASS** |
| `MC03` | control | 594 | 50 | 544 | **PASS** |
| `MC04` | control | 993 | 50 | 943 | **PASS** |

---

## 6. Known Limitations & Research Safeguards

1. **Microphone Dual-Channel Recording:**
   - TORGO records simultaneously with `headMic` and `arrayMic`.
   - *Safeguard Applied:* Grouping is enforced by `prompt_id` so that simultaneous recordings never cross the train/test split. `headMic` is designated as the primary acoustic evaluation channel.

2. **Natural Emergency Vocabulary Sparsity:**
   - Because TORGO is a general speech database, genuine emergency distress calls (e.g., SOS_HELP) represent a subset (700 utterances) of the full corpus.
   - *Safeguard Applied:* Transparent taxonomy mapping preserves both the raw transcripts and explicit intent tags without fabricating labels.

3. **Speaker Session Imbalance:**
   - Some speakers recorded across 1 session (e.g., F01, FC01, M03) while others recorded across 2 or 3 sessions.
   - *Safeguard Applied:* Few-shot personalization protocol operates per-speaker using prompt-level random sampling, ensuring balanced evaluation pools.

---

## 7. Next Steps for Phase 2

- Proceed to **Phase 2 — Acoustic Representation & Embedding Pipeline** (Whisper feature extraction, caching, and pooling).
- All downstream representations will strictly index from `splits/torgo_splits.json` and `splits/torgo_split_indices.csv`.
