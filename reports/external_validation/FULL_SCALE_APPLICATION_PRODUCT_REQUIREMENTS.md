# Product Requirements Document (PRD): Full-Scale Research Application
## Controlled Scientific & Engineering Platform for Low-Resource Atypical Speech Intelligence

**Document Identifier:** PRD-FULL-SCALE-APP-2026-V1  
**Status:** **PROPOSED ARCHITECTURAL BLUEPRINT — PLANNING ONLY (NO IMPLEMENTATION AUTHORIZED)**  
**Date:** 2026-10-11  
**Authoring Roles:** Principal ML Systems Architect, Clinical Software Engineering Lead, Product Manager  
**Baseline Foundation:** Gate B.2 Application Architecture ([`src/app/`](file:///d:/DL/src/app/))  

---

> [!CAUTION]
> **GOVERNANCE & BOUNDARY DIRECTIVE:**  
> This Product Requirements Document defines the target scope, technical specifications, and safety constraints for a future full-scale application build. **NO APPLICATION CODE, DATABASE MIGRATION, OR BACKEND RESTRUCTURING IS AUTHORIZED DURING THIS TASK.** The application remains strictly a local research and engineering platform. Marketing or deploying this system as an emergency response, clinical diagnostic, or 911 dispatch tool is categorically forbidden.

---

## 1. Intended Product Vision & Scope Boundaries

### 1.1 Product Vision
To establish an enterprise-grade, locally deployable, fully reproducible **Speech Machine Learning Research & Clinical Engineering Platform**. The platform enables speech-language pathologists, biostatisticians, and ML engineers to explore atypical speech representations, benchmark personalized acoustic adapters, manage dataset governance ledgers, and audit model performance under strict scientific discipline.

### 1.2 Boundary Enforcement (What the Product IS and IS NOT)
- **What it IS:**
  - An offline, private, localhost research laboratory.
  - An interactive exploration interface for 60 materialized Phase 7.5 adapters and Whisper embeddings.
  - A cryptographic ledger and provenance explorer for frozen historical findings (Case B affirmed).
  - A guarded dataset intake staging tool that validates schemas and flags ethics/DUA deficiencies before processing.
  - An environment for pre-flight preregistration auditing.
- **What it IS NOT (Categorically Out of Scope):**
  - An emergency call dispatch system (zero 911, 112, CAD, or telephony connections).
  - A real-time life-safety alert engine (zero SMS, push alerts, or paging).
  - A certified medical device (SaMD) for patient monitoring.
  - A public cloud SaaS processing unconsented voice biometrics.

---

## 2. Current-to-Target Architecture

```mermaid
flowchart TD
    subgraph Current [Current Hardened Gate B.2 Prototype]
        UI_B2[FastAPI Static SPA: HTML/CSS/JS]
        ROUTER_B2[FastAPI Router: /audio, /inference, /research]
        MODELS_B2[In-Memory ModelManager: 60 Adapters + 503 Guard]
        AUDIO_B2[AudioService: Local Whisper-Base 16kHz Decoding]
        STORE_B2[Ephemeral Local Filesystem: No Persistence]
    end

    subgraph Target [Target Full-Scale Architecture (ALL COMPONENTS PROPOSED)]
        UI_PROP["PROPOSED: Modular Accessible Research UI (Vue/React or Hardened Vanilla JS)<br>WCAG 2.1 AA Compliant, Screen Reader & Switch-Access Ready"]
        GATEWAY_PROP["PROPOSED: Secure Local ASGI Gateway<br>TLS 1.3 Loopback, Strict RBAC, Session Encryption"]
        
        subgraph Core_Services [Core Application Micro-Modules (PROPOSED)]
            INTAKE_PROP["PROPOSED: Guarded Dataset Intake & Validator Service<br>Disabled until lawful DUA/Ethics verified; SHA-256 Manifest Hashing"]
            REGISTRY_PROP["PROPOSED: Cryptographic Model & Checkpoint Ledger<br>Fingerprint Verification, Weight Lineage, Fail-Closed 503 Handler"]
            AUDIT_PROP["PROPOSED: Tamper-Evident Run Registry & Audit Daemon<br>Append-Only SQLite Log, SHA-256 Chained Event Blocks"]
            PREREG_PROP["PROPOSED: Protocol & Preregistration Management Console<br>OSF / Zenodo Sync, Frozen Protocol Diff & Hashing"]
            ENGINE_PROP["PROPOSED: Offline Inference & Calibration Engine<br>Personalized Routing, Ephemeral Buffer Purging, Abstention Guard"]
        end

        subgraph Secure_Storage [Isolated Local Persistence (PROPOSED)]
            DB_PROP["PROPOSED: Local Encrypted SQLite / SQLCipher Database<br>Zero Raw Audio; Manifests, Checksums & Audit Logs Only"]
            VAULT_PROP["PROPOSED: Sealed Confirmatory Vault (Encrypted AES-256 Volume)<br>Access Locked by Single-Pass Cryptographic Token Barrier"]
        end
    end

    UI_B2 -.-> UI_PROP
    ROUTER_B2 -.-> GATEWAY_PROP
    MODELS_B2 -.-> REGISTRY_PROP
    AUDIO_B2 -.-> ENGINE_PROP
    STORE_B2 -.-> DB_PROP
    GATEWAY_PROP --> INTAKE_PROP & REGISTRY_PROP & AUDIT_PROP & PREREG_PROP & ENGINE_PROP
    INTAKE_PROP --> DB_PROP
    AUDIT_PROP --> DB_PROP
    ENGINE_PROP --> VAULT_PROP
```

---

## 3. Comprehensive Feature Matrix

| Feature Module | Specific Capability | Implementation Status | Gate Dependency | Governance Rule |
| :--- | :--- | :---: | :---: | :--- |
| **Overview & Vitals** | Research overview dashboard & vitals | `EXISTING` | Gate B.2 | Truthful display of frozen findings |
| **Audio Lab** | Microphone recording & WAV upload | `EXISTING` | Gate B.2 | 16 kHz mono, max 10 MB, 30s limit |
| **Audio Lab** | Local Whisper ASR transcription | `EXISTING` | Gate B.2 | Local offline decoding only |
| **Audio Lab** | 384-D Whisper embedding extraction | `EXISTING` | Gate B.2 | On-demand vector inspector modal |
| **Audio Lab** | Continual adapter inference (F01–MC04) | `EXISTING` | Gate B.2 | Genuine forward pass on 60 adapters |
| **Model Registry** | Searchable catalog of 67 models | `EXISTING` | Gate B.2 | Distinct weights per speaker |
| **Model Registry** | HTTP 503 guard on absent candidates | `EXISTING` | Gate B.2 | Zero weight fabrication |
| **Results Explorer** | Historical phase metrics & figures | `EXISTING` | Gate B.2 | Mounted directly from frozen reports |
| **Diagnostics** | Real-time latency & WebSocket tester | `EXISTING` | Gate B.2 | Ephemeral logging only |
| **Dataset Intake** | External manifest schema validator | `PROPOSED` | Gate C | **Disabled until DUA/Ethics confirmed** |
| **Dataset Intake** | Case-insensitive TORGO overlap check | `PROPOSED` | Gate C | Immediate fail-closed abort on collision |
| **Dataset Intake** | Multi-vector synthetic data rejection | `PROPOSED` | Gate C | Bar mock fixtures from confirmatory runs |
| **Dataset Intake** | Acoustic perceptual duplicate screening | `PROPOSED` | Gate C | Chromaprint / Dejavu acoustic hash check |
| **Preregistration** | Protocol repository & diff inspector | `PROPOSED` | Gate C | Distinguishes draft from frozen |
| **Preregistration** | OSF / Zenodo API synchronizer | `PROPOSED` | Gate C | Pre-registers SHA-256 before data unlock |
| **Audit Ledger** | Tamper-evident append-only event log | `PROPOSED` | Gate B.2 | SQLite ledger with hashed events |
| **Security** | Encrypted SQLite storage (SQLCipher) | `PROPOSED` | Gate B.2 | Encryption at rest for app config/logs |
| **Evaluation Gate**| Guarded confirmatory evaluation runner | `BLOCKED` | Gate C | Requires explicit owner token & prereg |
| **Live Dispatch** | Telephony / SMS / CAD 911 alert engine| `NOT AUTHORIZED`| Gate D | **Permanently excluded from scope** |

---

## 4. User Roles, Personas, and Access Control (RBAC)

The application enforces strict local Role-Based Access Control:

```mermaid
flowchart TD
    subgraph Roles [Defined System Personas]
        R1[Lead Investigator / Project Owner]
        R2[Speech-Language Pathologist / Clinician]
        R3[Lead Biostatistician]
        R4[Machine Learning Engineer]
        R5[Independent Auditor / Regulatory Reviewer]
    end

    subgraph Permissions [Permitted Capabilities]
        P1[View Research Findings & Model Registry]
        P2[Execute Local Audio Demo & Inspector]
        P3[Author & Review Protocol / Preregistration Drafts]
        P4[Perform Intake Schema Validation on Dev Manifests]
        P5[Export Verification Audit Dossier]
        P6[Issue Live Confirmatory Run Token (Restricted)]
    end

    R1 --> P1 & P2 & P3 & P5 & P6
    R2 --> P1 & P2 & P3 & P5
    R3 --> P1 & P3 & P4 & P5
    R4 --> P1 & P2 & P4
    R5 --> P1 & P3 & P5
```

1. **Project Owner:** Full visibility, system administration, and exclusive authority to configure `GATE_C_OWNER_AUTH_TOKEN`.
2. **Clinical Investigator (CCC-SLP):** Protocol authoring, clinical intent adjudication, review of failure modes and abstentions.
3. **Lead Biostatistician:** SAP formulation, sample size sensitivity audits, threshold derivation on dev data, and power modeling.
4. **Machine Learning Engineer:** Local adapter maintenance, acoustic preprocessing pipelines, performance profiling, and unit tests.
5. **Independent Auditor:** Read-only cryptographic verification of checksums, manifests, and tamper-evident event logs.

---

## 5. Data Flow, Trust Boundaries, and Threat Modeling

### 5.1 Data Flow & Trust Boundaries
- **Boundary 0 (Host Operating System):** Bound strictly to loopback interface (`127.0.0.1:8000`). No inbound traffic permitted from LAN or Internet.
- **Boundary 1 (Browser to Backend API):** Single-user local browser session. Ephemeral session tokens; CORS restricted to `http://127.0.0.1:8000`.
- **Boundary 2 (Audio Processing Pipeline):** Audio ingested via multipart POST or binary WebSocket. Stored in memory or ephemeral temporary file with restrictive permissions (`0600`).
- **Boundary 3 (Model Execution Sandbox):** PyTorch execution in-process with gradient tracking permanently disabled (`requires_grad = False`).

### 5.2 Threat Model & Failure Mitigation (STRIDE)

| Threat Category | Specific Attack / Failure Scenario | Architectural Control / Mitigation |
| :--- | :--- | :--- |
| **Spoofing** | Attacker impersonates Project Owner to unlock confirmatory runner. | High-entropy owner authorization token (min 16 chars) required; dummy tokens rejected; rate-limited token check. |
| **Tampering** | User alters frozen historical report or metric to show positive quantum result. | 28 historical files protected by SHA-256 checksum audit; test suite fails if any file is modified. |
| **Repudiation** | User executes unapproved confirmatory run and denies action. | Append-only tamper-evident audit log with SHA-256 event chaining; single-pass run lock file permanently created. |
| **Information Disclosure** | Private audio or patient transcripts leaked via web or telemetry. | Ephemeral processing with automatic buffer purging; zero outbound telemetry; loopback binding only. |
| **Denial of Service** | Oversized audio upload exhausts server RAM. | Strict payload limit (10 MB, 30s max); non-conforming audio rejected with HTTP 400/413. |
| **Elevation of Privilege** | User uses dataset intake UI to trigger unapproved confirmatory evaluation. | Formal feature gate decoupling intake validation from confirmatory execution; runner requires separate CLI authorization. |

---

## 6. API Contract Plan

The proposed application extends the Gate B.2 API with versioned, change-controlled endpoints:

```
FastAPI Routing Structure:
/api/v1/
├── /health                       (GET: System health, Whisper status, adapter counts)
├── /readiness                    (GET: Detailed Gate B/C/D verification status)
├── /models                       (GET: Catalog of 67 models with availability flags)
├── /models/{model_id}/info       (GET: Specific model provenance, parameters, tau status)
├── /audio/transcribe             (POST: Local offline Whisper transcription)
├── /audio/extract-embedding      (POST: 384-D vector extraction with opt-in full vector)
├── /inference/predict-file       (POST: Audio file classification on served adapter)
├── /research/phases              (GET: Historical phase milestones and descriptions)
├── /research/metrics             (GET: Matched comparison tables and Case B verdicts)
├── /research/operating-points    (GET: Phase 8.1 safety operating points ledger)
├── /research/figures             (GET: Catalog of verified publication figures)
├── /provenance/summary           (GET: 28 historical checksums and TORGO split manifest)
│
├── [PROPOSED] /dataset/validate  (POST: Upload manifest & metadata for pre-intake audit)
├── [PROPOSED] /dataset/intake    (POST: Staged intake — DISABLED without DUA verification)
├── [PROPOSED] /prereg/status     (GET: Preregistration protocol hash & registry sync)
└── [PROPOSED] /audit/events      (GET: Paginated read-only view of tamper-evident log)
```

---

## 7. Storage, Persistence, and Packaging Strategy

1. **Storage Choice (Encrypted Local SQLite / SQLCipher):**
   - **Rationale:** Serverless, zero external network daemon, zero cloud footprint. Enables transparent column-level or database-level AES-256 encryption.
   - **Stored Data:** Intake manifest records, metadata audit results, system configuration, model inventory, and tamper-evident audit trails.
   - **Excluded Data:** Raw speech audio and speech transcripts are **NEVER** stored in the database.
2. **Packaging & Delivery:**
   - Standalone Python package with strict `pyproject.toml` dependency pinning.
   - Optional local containerization via single-container Docker Compose bound strictly to host loopback (`127.0.0.1:8000:8000`).
   - Automated startup verification script performing checksum and test suite sanity checks before launching uvicorn daemon.

---

## 8. Accessibility & Usability Criteria (WCAG 2.1 AA)

Individuals with dysarthria frequently present with concomitant motor disabilities (e.g. cerebral palsy, tremors). The UI must meet the highest accessibility standards:
- **WCAG 2.1 AA Compliance:** Minimum color contrast ratio of $4.5:1$ for normal text and $3:1$ for large text and UI components.
- **Full Keyboard Navigation:** Complete interface operability via Tab, Enter, Space, and Esc without mouse interaction. Visible focus indicators on all interactive elements.
- **Switch-Access & Assistive Tech Support:** Explicit ARIA attributes (`aria-expanded`, `aria-controls`, `role="tab"`, `aria-live="polite"`).
- **Reduced Motion Support:** Respects `prefers-reduced-motion` media queries by disabling non-essential transitions and micro-animations.
- **Screen Reader Announcements:** Dynamic result updates, error alerts, and modal dialogs announced via dedicated `aria-live` containers.
