# TORGO Dataset Physical Inventory Report

**Date:** 2026-09-30  
**Dataset Root:** `D:\DL\TORGO DATASET`  
**Source Format:** Apache Parquet (Hugging Face Datasets export)  

---

## 1. High-Level Summary

- **Total Audio Utterances:** 16,552
- **Total Unique Filenames:** 16,552
- **Duplicate Filenames:** 0
- **Malformed Filenames:** 0
- **Total Audio Duration:** 13.68 hours (49,238.30 seconds)
- **Mean Utterance Duration:** 2.97 seconds
- **Sampling Rate:** 16000 Hz
- **Channels:** 1 (Mono)
- **Audio Encoding:** Linear PCM (WAV)
- **Total Speakers:** 15 (8 dysarthric, 7 control)
- **Microphone Channels:** arrayMic: 8,697, headMic: 7,855
- **Transcripts:** 967 unique, 0 null, 0 empty

---

## 2. Raw File Manifest

| File | Size (MB) |
|------|-----------|
| `train-00000-of-00004.parquet` | 353.24 MB |
| `train-00001-of-00004.parquet` | 378.99 MB |
| `train-00002-of-00004.parquet` | 339.83 MB |
| `train-00003-of-00004.parquet` | 420.32 MB |

---

## 3. Speaker Breakdown

| Speaker ID | Group | Gender | Total Recs | HeadMic | ArrayMic | Unique Prompts | Unique Transcripts | Total Dur (s) |
|------------|-------|--------|------------|---------|----------|----------------|--------------------|---------------|
| `F01` | dysarthria | female | 236 | 118 | 118 | 118 | 97 | 573.62 |
| `F03` | dysarthria | female | 1087 | 546 | 541 | 213 | 426 | 2985.65 |
| `F04` | dysarthria | female | 673 | 244 | 429 | 250 | 354 | 2202.22 |
| `FC01` | healthy | female | 302 | 151 | 151 | 151 | 131 | 1082.92 |
| `FC02` | healthy | female | 2187 | 967 | 1220 | 975 | 903 | 6221.79 |
| `FC03` | healthy | female | 1922 | 963 | 959 | 405 | 698 | 5000.87 |
| `M01` | dysarthria | male | 743 | 371 | 372 | 285 | 320 | 2935.93 |
| `M02` | dysarthria | male | 772 | 388 | 384 | 232 | 335 | 2836.81 |
| `M03` | dysarthria | male | 810 | 406 | 404 | 406 | 348 | 2544.56 |
| `M04` | dysarthria | male | 666 | 277 | 389 | 286 | 231 | 2872.64 |
| `M05` | dysarthria | male | 587 | 470 | 117 | 357 | 386 | 2705.76 |
| `MC01` | healthy | male | 2155 | 1077 | 1078 | 419 | 786 | 6512.2 |
| `MC02` | healthy | male | 1123 | 449 | 674 | 383 | 538 | 3974.91 |
| `MC03` | healthy | male | 1669 | 799 | 870 | 594 | 655 | 3536.25 |
| `MC04` | healthy | male | 1620 | 629 | 991 | 993 | 718 | 3252.16 |

---

## 4. Microphone Configuration & Channel Pairings

TORGO records each vocalization simultaneously across two channels:
1. **headMic (Head-Mounted Microphone):** Close proximity to the lips, minimizes ambient room reverberation, higher SNR.
2. **arrayMic (Microphone Array):** Fixed desktop acoustic array, captures environmental reverberation.

> [!IMPORTANT]
> Multi-channel pairings for the same prompt event MUST be grouped together during train/test splitting to prevent acoustic data leakage across splits.
