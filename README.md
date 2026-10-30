# Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qiskit 2.5](https://img.shields.io/badge/Qiskit-2.5-6929C4.svg?style=flat&logo=qiskit&logoColor=white)](https://qiskit.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.133-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-460%2B%20Passing-success.svg?style=flat&logo=pytest&logoColor=white)](tests/)
[![Status: Case B Verified](https://img.shields.io/badge/Verdict-Case%20B%20(Equivalence)-blue.svg?style=flat)](reports/)

---

## 1. Project Overview & Research Question

### The Core Research Question
> **Can a fidelity quantum kernel derived from an $n$-qubit parameterized quantum circuit provide superior few-shot personalization for atypical (dysarthric) speech compared to classical adaptation baselines when only extremely scarce labeled samples ($N \in \{20, 30, 40, 50\}$) are available per individual speaker?**

This repository contains the complete experimental framework, forensic auditing pipelines, statistical evaluation suites, and an interactive **Gate B.2 Engineering Demonstration Application** investigating hybrid quantum-classical speech recognition for motor-speech disorders (TORGO dysarthria corpus).

### Scientific Principle & Empirical Verdict (Case B)
The quantum method is evaluated strictly as an **empirical hypothesis to test**, not an assumed advantage. 
- **Scientific Finding:** Empirical evaluation demonstrates **Case B Equivalence**. Across 15 speakers (dysarthric cohorts F01, F03, F04, M01, M02, M03, M04, M05 and matched control cohorts) and matched budgets ($N=20..50$), the Quantum Support Vector Classifier (`QSVC`, $C=1.0$, ZZ Pauli feature map) achieves performance statistically indistinguishable from optimal Classical Support Vector Classifiers (RBF-SVM) operating on identical frozen 384-dimensional Whisper speech representations ($p > 0.05$ across Wilcoxon signed-rank and McNemar tests).
- **Practical Implication:** Classical linear adapters and RBF kernels maintain orders-of-magnitude faster inference latency with equivalent accuracy. However, the quantum kernel confirms theoretical feature-space separability without degradation.

---

## 2. Key Features

- **Frozen Zero-Leakage Representation Layer:** Offline feature extraction via frozen `openai/whisper-tiny` encoder (384-d L2-normalized representations) and baseline MFCCs (78-d mean+std pooled vectors).
- **Rigorous Few-Shot Personalization Framework:** Exact matched evaluation across $N \in \{20, 30, 40, 50\}$ sample budgets per speaker across 15 distinct speakers (60 deterministic split conditions).
- **Quantum Machine Learning Pipeline:** Dimensionality compression (PCA) $\to$ 2nd-order ZZ Pauli Feature Map (`ZZFeatureMap`) $\to$ Fidelity Quantum Kernel $\to$ Quantum Support Vector Classifier (`QSVC`).
- **Acoustic Noise Robustness Sweeps (Protocol B):** Matched testing across Clean ($\infty$ dB), 15 dB, 10 dB, and 5 dB SNR additive noise conditions (720 evaluated conditions).
- **Gate B.2 Interactive Demonstration Prototype:** High-performance local FastAPI backend with WebSocket streaming, live microphone audio capture, decoupled Whisper STT, and transparent quantum-only operational routing.
- **Fail-Closed Safety Governance:** Explicit model readiness guards returning HTTP 503 (`MODEL_NOT_READY`) when reference sidecars are unmaterialized and HTTP 400 (`MODEL_NOT_ALLOWED`) for classical operational requests.
- **Exhaustive Automated Verification:** Over 460 automated unit, regression, cryptographic protocol consistency, and gate qualification tests with 100% pass rates.

---

## 3. Technology Stack

| Domain | Components & Frameworks | Description |
|---|---|---|
| **Programming Language** | Python 3.11+ | Standard modern runtime environment |
| **Acoustic & DL Models** | PyTorch, HuggingFace Transformers | Frozen Whisper encoder (`openai/whisper-tiny`) |
| **Audio Processing** | Librosa, SoundFile, SciPy | 16 kHz mono resampling, SNR noise sweeps, MFCCs |
| **Quantum Computing** | Qiskit 2.5, Qiskit Machine Learning, PennyLane | Circuit synthesis, StatevectorSampler, ZZFeatureMap |
| **Classical Machine Learning** | Scikit-Learn, NumPy, Pandas | MLP Adapter (AdamW), RBF-SVM, PCA, statistical metrics |
| **Application & API** | FastAPI, Uvicorn, Pydantic v2, WebSockets | Asynchronous RESTful & streaming server |
| **Demonstration Frontend** | HTML5, Vanilla JavaScript, Modern CSS | Real-time waveform visualizer, model inspection UI |
| **Testing & Quality Assurance** | Pytest, Pytest-Asyncio, AnyIO | Deterministic testing, cryptographic hash validation |

---

## 4. System Architecture Overview

```
                                      +---------------------------------------------+
                                      |         Atypical Audio Input (16kHz)        |
                                      +---------------------------------------------+
                                                             |
                                                             v
                                      +---------------------------------------------+
                                      |   Offline Frozen Speech Encoder (Whisper)   |
                                      +---------------------------------------------+
                                                             |
                                          384-dimensional Acoustic Embeddings
                                                             |
                                      +----------------------+----------------------+
                                      |                                             |
                                      v                                             v
                       +-----------------------------+               +-----------------------------+
                       |     Classical Baselines     |               |    Quantum Kernel Pipeline  |
                       +-----------------------------+               +-----------------------------+
                       |  - Trainable MLP Adapter    |               |  - PCA Projection (q=2..8)  |
                       |  - RBF-Kernel SVM (C=1.0)   |               |  - ZZ Pauli Feature Map     |
                       +-----------------------------+               |  - Fidelity Quantum Kernel  |
                                      |                              |  - QSVC Classification      |
                                      |                              +-----------------------------+
                                      |                                             |
                                      +----------------------+----------------------+
                                                             |
                                                             v
                                      +---------------------------------------------+
                                      |  Model Governance & Decision Policy         |
                                      |  - Strict Single-Class & Confidence Filter  |
                                      |  - Intent Preservation Rate (IPER)          |
                                      |  - False Emergency Rate (FER <= 15% Bound)  |
                                      +---------------------------------------------+
                                                             |
                                                             v
                                      +---------------------------------------------+
                                      |  FastAPI Operational Demonstration (Gate B) |
                                      |  - Audio Transcription (/audio/transcribe)  |
                                      |  - Quantum Inference (/api/v1/predict)      |
                                      |  - Interactive Web Demo UI (/demo)          |
                                      +---------------------------------------------+
```

---

## 5. Repository Directory Structure

```
d:/DL/
├── assets/                       # UI demonstration screenshots, logs, and artifacts
│   └── second_review/            # Verified application execution screenshots & manifest
├── config/                       # Authoritative Phase 7/8 research protocols & model registries
│   ├── continual/                # Continual learning governance and drift policies
│   ├── phase8_1/                 # Safety recovery and cross-validation protocol
│   ├── phase8_2_1/               # Corrected threshold protocol v2 (FER decoupled)
│   ├── model_registry.yaml       # Single source of truth model registry
│   ├── research_protocol.yaml    # Core research protocol definitions
│   └── taxonomy.yaml             # Canonical 7-class emergency intent taxonomy
├── configs/                      # Historical configurations for Phases 1 through 5
├── data/                         # Data manifests, synthetic samples, and splits
│   ├── manifests/                # Canonical manifests (torgo_manifest.csv, etc.)
│   └── synthetic_emergency_audio/# 10 standardized atypical emergency validation WAVs
├── docs/                         # Architecture blueprints, context, and phase tracking
│   ├── application/              # Gate B.2 app architecture & API reference
│   ├── continual/                # Phase 7 continual learning documentation
│   └── PHASE_STATUS.md           # Detailed milestone verification log
├── embeddings/                   # Pre-computed Whisper (384-d) and MFCC (78-d) representations
├── final_package/                # Audited research closure, limitations, and evidence maps
├── models/                       # Checkpoints, reference sidecars, and serialized weights
│   ├── continual/                # Adaptive continually trained checkpoints
│   └── quantum/qsvc_q8_r1/       # Frozen reference PCA weights & support vectors
├── notebooks/                    # Jupyter validation notebooks
├── phase5/                       # Phase 5 acoustic noise robustness artifacts
├── phase6/                       # Phase 6 statistical evaluation and hypothesis tests
├── phase6_5/                     # Phase 6.5 diagnostic, rescue, and recovery artifacts
├── reports/                      # Full suite of markdown and CSV research reports
├── research/                     # Phase 7 and 8 protocol audit documents
├── results/                      # Raw experimental prediction CSVs and kernel cache matrices
├── scripts/                      # Automated pipeline runners, extractors, and report builders
├── splits/                       # Deterministic few-shot split definitions
├── src/                          # Production source code
│   ├── app/                      # FastAPI application backend and web frontend
│   │   ├── routes/               # Modular endpoints (health, audio, inference, research)
│   │   ├── static/               # Demonstration UI (index.html, app.js, style.css)
│   │   ├── audio_service.py      # Resampling, quality inspection, Whisper STT
│   │   ├── model_manager.py      # Classical model manager
│   │   └── quantum_model_manager.py # Quantum-only operational model manager
│   ├── continual/                # Continual adaptation, drift detection, replay memory
│   ├── data/                     # Data ingest, validation, and manifest generation
│   ├── embeddings/               # Whisper and MFCC feature extractors
│   ├── models/                   # Classical adapter neural networks and RBF-SVM
│   ├── quantum/                  # Quantum kernel encoding and QSVC classification
│   └── utils/                    # Noise injection and reproducibility utilities
├── tests/                        # Pytest automated test suites (Gates B, B2, C, D, Phases 1-8)
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git exclusion rules (excluding NO NEED and bulky datasets)
├── pyrefly.toml                  # Language server and linter configuration
├── requirements.txt              # Pinned Python package dependencies
├── run_prototype.py              # Application startup runner
└── README.md                     # Master project documentation
```

*(Note: Locally archived obsolete prompts, scratch scripts, and duplicates are preserved in `NO NEED/`, strictly excluded from Git tracking).*

---

## 6. Installation & Environment Setup

### Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python:** Version 3.10 or 3.11 (Python 3.11.15 recommended)
- **Audio Codecs:** `libsndfile` / ffmpeg (for audio processing)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd <repository-root>
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Package Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Environment Configuration
Copy `.env.example` to create `.env`:
```bash
# Windows
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```
All default configurations are bound to local loopback (`127.0.0.1:8000`) for controlled research demonstration. No external API keys or cloud credentials are required.

---

## 7. Running the Application Prototype

The Gate B.2 research demonstration prototype provides both an interactive web dashboard and an OpenAPI/RESTful backend.

### Launching the Application
Execute the dedicated prototype launcher:
```bash
python run_prototype.py
```
*Alternatively, start directly via Uvicorn:*
```bash
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Accessing Interfaces
- **Interactive Speech Intent Lab:** [http://127.0.0.1:8000/demo](http://127.0.0.1:8000/demo)
- **Interactive Swagger API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Technical Specification:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Readiness Audit Endpoint:** [http://127.0.0.1:8000/ready](http://127.0.0.1:8000/ready)

---

## 8. API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Service liveness and timestamp status |
| `/ready` | `GET` | Subsystem readiness and model inventory audit |
| `/audio/transcribe` | `POST` | Local offline Whisper speech-to-text transcription |
| `/audio/extract-embedding` | `POST` | Deterministic 384-d Whisper feature extraction |
| `/audio/validate` | `POST` | Waveform quality, duration, and clipping inspection |
| `/api/v1/predict` | `POST` | Operational prediction (locked to `qsvc_q8_r1`) |
| `/api/v1/predict/file` | `POST` | Direct WAV file operational inference |
| `/api/v1/research/phases` | `GET` | Historical research phases and status report |
| `/api/v1/research/metrics` | `GET` | Macro F1, IPER, and FER baseline benchmarks |
| `/api/v1/research/operating-points` | `GET` | Calibrated decision thresholds and safety points |
| `/api/v1/research/figures` | `GET` | Registry of generated publication-grade figures |
| `/ws/stream` | `WebSocket` | Real-time audio streaming and progressive prediction |

---

## 9. Running Experiments & Reproducing Results

### 1. Verify Environment & Smoke Tests
```bash
python scripts/check_environment.py
```

### 2. Verify Synthetic Emergency Audio Test Suite
```bash
python scripts/verify_synthetic_samples.py
```

### 3. Run Classical Baselines (Phase 3)
```bash
# Pilot sweep (2 speakers)
python scripts/run_classical_baselines.py --pilot

# Full 15-speaker sweep (120 conditions)
python scripts/run_classical_baselines.py
```

### 4. Run Quantum Kernel Personalization (Phase 4)
```bash
# Quantum pilot run
python scripts/run_quantum_kernel.py --pilot

# Full 15-speaker quantum evaluation (240 conditions across q=2..8)
python scripts/run_quantum_kernel.py
```

### 5. Run Acoustic Noise Robustness Sweeps (Phase 5)
```bash
python scripts/run_phase5_noise_sweep.py
```

### 6. Generate Statistical Analysis & Reports (Phase 6)
```bash
python scripts/run_phase6_statistical_analysis.py
```

---

## 10. Automated Testing & Verification

The repository contains an exhaustive test suite organized into modular gates:

```bash
# 1. Run all Final Acceptance and Configuration Integrity Tests
python -m pytest tests/test_final_*.py -v

# 2. Run Gate B & B.2 Application Readiness Tests
python -m pytest tests/gate_b/ tests/gate_b2/ -v

# 3. Run Gate C & D Safety Governance Qualification Tests
python -m pytest tests/gate_c/ tests/gate_d/ -v

# 4. Run Phase 8 Forensic Repair & Threshold Protocol Tests
python -m pytest tests/phase8_1/ tests/phase8_2/ tests/phase8_2_1/ tests/phase8_80_20/ -v

# 5. Run Entire Test Suite (Passes 100%)
python -m pytest tests/ -v
```

---

## 11. Visual Demonstration & Screenshots

Interactive screens captured during verified test execution (located in `assets/second_review/`):

| Screen | Description |
|---|---|
| **Gate B.2 Test Run Verification** | Automated execution showing 100% test pass rate across application routes |
| **Inference Interface** | Real-time waveform visualizer, persona selection, and live classification |
| **Emergency Intent Response** | High-contrast confirmation modal with transparent confidence calibration |
| **Model Comparison Matrix** | Macro F1 comparison between Classical Adapter, RBF-SVM, and QSVC |
| **Acoustic Noise Heatmap** | SNR robustness evaluation across Clean, 15 dB, 10 dB, and 5 dB |

---

## 12. Deployment Governance & Safety Disclaimer

| Gate | Scope | Status | Qualification Criteria |
|---|---|---|---|
| **Gate A** | Technical Architecture | **PASSED** | Design principles and offline components verified |
| **Gate B.2** | Controlled Demo Application | **ACTIVE** | Local prototype running strictly on loopback (127.0.0.1) |
| **Gate C** | Real-World User Pilot | **STRICTLY BLOCKED** | Requires external non-TORGO confirmatory data |
| **Gate D** | Clinical / 911 Dispatch Deployment | **STRICTLY BLOCKED** | Prohibited (Emergency recall limited under 15% FER bound) |

> [!CAUTION]
> **Regulatory and Ethical Notice:**  
> This software is an academic research demonstration prototype designed strictly for non-clinical evaluation. It has not been approved by the FDA, CE, or any regulatory medical authority. It must NOT be used for clinical triage, life-critical emergency dispatch, or unsupervised medical care.

---

## 13. License & Academic Attribution

This project is released for academic, research, and non-commercial educational purposes. 

### Citation
If you utilize this codebase, methodology, or benchmark results in your research, please cite:
```bibtex
@article{atypical_speech_quantum_2026,
  title={Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning},
  author={Kernal Research Team},
  year={2026},
  journal={Academic Research Demonstration Archive}
}
```