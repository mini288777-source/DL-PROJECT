# Preregistration Integrity and Confirmatory Runner Readiness Audit
## Precondition Audit, Bypass Vulnerability Analysis, and Governance Enforcement

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Document Identifier:** AUDIT-PREREG-CONFIRM-READINESS-2026-V1  
**Audit Date:** 2026-10-11  
**Authors:** Senior Safety-Critical Software Assurance Lead, Independent Clinical-ML Auditor, Data Governance Officer  

---

> [!CAUTION]
> **GOVERNANCE DIRECTIVE & VERDICT:**  
> Confirmatory evaluation execution is **STRICTLY BLOCKED**. Although repository-side validation tooling and fail-closed security controls are operational, essential external preconditions (independent dysarthric cohort, executed bilateral DUA, institutional ethics approval, clinical SLP oversight, public preregistration deposit, and explicit owner authorization token) remain completely open. The confirmatory runner has NOT been executed.

---

## 1. Executive Summary & Readiness Verdict

$$\mathbf{CONFIRMATORY\; RUNNER\; STATUS:\; BLOCKED\_EXTERNAL\_PREREQUISITES}$$

The confirmatory evaluation subsystem comprises:
1. **Dataset Integrity Gate:** [`src/data/external_dataset_validator.py`](file:///d:/DL/src/data/external_dataset_validator.py) (manifest schema validation, provenance metadata checking, case-insensitive TORGO overlap detection, and multi-vector synthetic data rejection).
2. **Guarded Evaluation Harness:** [`src/evaluation/confirmatory_runner.py`](file:///d:/DL/src/evaluation/confirmatory_runner.py) (protocol hash matching, secure owner token resolution with secret redaction, candidate weight verification, and single-pass cryptographic run lock).

An exhaustive audit of repository source code, configuration files, and test suites confirms that **the repository tooling enforces fail-closed behavior across all simulated defect states**. However, because no independent human speech dataset exists in the repository, live confirmatory evaluation cannot proceed.

---

## 2. Requirement-to-Evidence Precondition Matrix Summary

As documented in [`reports/external_validation/CONFIRMATORY_RUNNER_PRECONDITION_MATRIX.csv`](file:///d:/DL/reports/external_validation/CONFIRMATORY_RUNNER_PRECONDITION_MATRIX.csv):
- **Satisfied on Tooling / Repository Level (12 items):** Manifest schema validation, cryptographic hashing, synthetic rejection across paths and metadata, TORGO speaker collision screening, partition leakage checking, token format verification, token redaction in logs, single-pass run lock, checkpoint presence check, threshold provenance decoupling, signal quality abstention, and regression test suite pass.
- **Blocked Pending External Institutional Action (8 items):** Lawful independent dataset acquisition, executed bilateral DUA, accredited IRB/IEC protocol approval, licensed CCC-SLP clinical appointment, clinical double-annotation ledger, cohort size sufficiency ($K \ge 109$), public preregistration deposit (OSF DOI), and explicit Project Owner authorization memo.

---

## 3. Granular Source Code Audit: Bypass Vulnerability Analysis & Hardening

All code paths in `src/data/external_dataset_validator.py` and `src/evaluation/confirmatory_runner.py` were audited for potential bypasses:

### 3.1 Audit of Synthetic Record Detection (`contains_synthetic_records`)
- **Initial Defect Discovered:** The initial validator implementation checked only 10 column names and a narrow set of values (`"true"`, `"1"`, `"yes"`, `"synthetic"`, `"mock"`, `"simulated"`, `"tone"`, `"sine_wave"`). Datasets declaring synthetic nature via metadata (`metadata.json`), alternative columns (`data_nature`, `generation_type`, `synthesis_method`), or advanced TTS terminology (`tts`, `vits`, `fastspeech`, `elevenlabs`, `vocoder`) could evade detection.
- **Fail-Closed Hardening Executed:**
  1. Expanded `SYNTHETIC_PATH_KEYWORDS` to 19 keywords covering `"sine"`, `"synthesized"`, `"generated"`, `"artificial"`, `"tts"`, `"text_to_speech"`, `"fastspeech"`, `"vits"`, `"elevenlabs"`, and `"vocoder"`.
  2. Expanded `synth_col_keys` to 22 column names and `synth_val_indicators` to 21 indicators.
  3. Added explicit synthetic detection in `_check_provenance()`: If `metadata.json` declares `"is_synthetic": true`, `"data_type": "synthetic"`, or `"data_nature": "synthetic"`, the cohort is flagged and rejected fail-closed.
  4. Added empty/whitespace `audio_path` validation.
- **Verification Evidence:** 17 unit tests in [`tests/gate_c/test_gate_c_infrastructure.py`](file:///d:/DL/tests/gate_c/test_gate_c_infrastructure.py) pass with 100% success (0 regressions).

### 3.2 Audit of Owner Authorization Token Resolution
- **Implementation:** `ConfirmatoryRunner._resolve_and_verify_token()` inspects `config.owner_authorization_token` or environment variable `GATE_C_OWNER_AUTH_TOKEN`.
- **Security Protections:**
  - Enforces minimum length of 16 characters.
  - Normalizes delimiters and rejects dummy tokens containing `"mock"`, `"test"`, `"placeholder"`, `"dummy"`, `"null"`, `"admin"`, `"password"`, `"secret"`, or `"12345"`.
  - In audit logs and JSON reports, the raw token is **NEVER logged or output**. It is strictly redacted to:
    `[VERIFIED_REDACTED_LEN_<length>]` or `"NOT_PROVIDED"`.
- **Finding:** Token verification is secure and prevents accidental secret exposure.

### 3.3 Audit of Single-Pass Manifest-Hash Run Lock
- **Lock Semantics:** The runner computes `manifest_hash_prefix = manifest_sha256[:8]` and creates `confirmatory_lock_<manifest_hash_prefix>.json`.
- **Behavior Across Operational Scenarios:**
  - *Dry-Run Mode (`dry_run_only = True`):* Inspects lock existence but does not abort, allowing pre-flight audits.
  - *Live Evaluation Mode (`dry_run_only = False`):* Aborts immediately with `SINGLE_PASS_VIOLATION` if the lock exists.
  - *Race Conditions / Concurrent Invocations:* The runner writes a temporary lock file (`.tmp`) and uses atomic filesystem replacement (`temp_lock.replace(lock_file)`), preventing partial-write corruption.
  - *Crash & Retry Handling:* If a crash occurs before the lock is finalized, the run can be restarted; once finalized, re-evaluation on the same manifest is permanently locked out.
  - *Changed Manifest or Protocol:* If the manifest is altered (even by 1 byte), `manifest_sha256` changes. Therefore, protocol preregistration MUST bind the evaluation to the public registry registration DOI, not merely the local manifest hash.

---

## 4. Public Preregistration Registry Selection Analysis

The study requires public cryptographic deposit prior to data access:

| Registry Platform | Jurisdiction & Target Scope | Suitability for this Study | Recommendation |
| :--- | :--- | :--- | :--- |
| **Open Science Framework (OSF)** | International; open-access scientific and observational protocols. | **OPTIMAL.** Provides timestamped, immutable DOIs; supports uploading frozen YAML configs and SHA-256 ledgers. | **RECOMMENDED PRIMARY REGISTRY** |
| **ClinicalTrials.gov** | US FDA / National Library of Medicine. Mandated for interventional clinical trials (IND/IDE drugs and devices). | **NOT APPLICABLE AT THIS STAGE.** The current study is an offline, non-interventional diagnostic machine learning validation, not an interventional clinical trial. | **DEFER UNTIL PROSPECTIVE TRIAL** |
| **ClinicalML Registry / Zenodo** | Open machine learning research repositories. | **SUITABLE SECONDARY ARCHIVE.** Zenodo provides immutable archival DOIs for code and frozen protocol hashes. | **RECOMMENDED SECONDARY ARCHIVE** |

---

## 5. Confirmatory Execution Safeguards: Non-Invocation Affirmation

In strict accordance with Task Objective Section 1:
- The confirmatory evaluation runner was **NOT** executed during this task.
- Zero mock tokens (`GATE_C_OWNER_AUTH_TOKEN`) were generated or injected into the live runtime environment.
- Zero held-out data were unblinded or evaluated.
- All testing was restricted to unit tests utilizing isolated temporary directory fixtures (`tmp_path`).
