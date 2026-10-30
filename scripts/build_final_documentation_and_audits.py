"""
build_final_documentation_and_audits.py
=======================================
Generates:
1. results/final_repair/final_claim_audit.csv
2. results/final_repair/final_consistency_matrix.csv
3. reports/final/FINAL_RESEARCH_RESULTS.md
4. reports/final/FINAL_RESEARCH_CONCLUSION.md
5. reports/final/FINAL_PROTOCOL.md
6. reports/final/FINAL_METRIC_DEFINITIONS.md
7. reports/final/FINAL_MODEL_CONFIGURATION.md
8. reports/final/FINAL_PROJECT_PLAN.md
9. reports/final/FINAL_CLAIM_AUDIT.md
10. reports/final/FINAL_DATA_PROVENANCE.md
"""

import os
import shutil
import pandas as pd

def build_claim_audit_csv():
    records = [
        {
            "claim_id": "CLM-001",
            "file": "reports/phase6_5_6_final_fair_comparison.md",
            "location": "Section 2.1 (Quantum Collapse Causal Attribution)",
            "original_claim": "The apparent collapse of quantum personalization was an artifact of few-shot data starvation and formulation flaws.",
            "classification": "CAUSAL_OVERCLAIM",
            "correction_required": "Replace with non-causal formulation noting shared classical collapse.",
            "corrected_claim": "The original quantum collapse was consistent with severe few-shot data starvation, class imbalance, and the flat classification formulation identified during forensic analysis. Recovery experiments subsequently demonstrated that the collapse behavior was not unique to the quantum model.",
            "evidence_source": "Phase 6.5.0 Diagnostic & Phase 6.5.2 Classical Rescue",
            "severity": "CRITICAL",
            "status": "CORRECTED"
        },
        {
            "claim_id": "CLM-002",
            "file": "reports/phase6_5_5_quantum_comparison.md",
            "location": "Executive Summary (Kernel Functional Behavior)",
            "original_claim": "The fidelity quantum kernel behaves functionally as a stationary distance-based similarity metric analogous to a classical RBF kernel.",
            "classification": "UNSUPPORTED_THEORETICAL_GENERALIZATION",
            "correction_required": "Scope statement strictly to evaluated representation and protocol.",
            "corrected_claim": "Under the evaluated frozen Whisper representation, feature scaling, kernel configuration, and few-shot protocol, the fidelity quantum kernel produced performance statistically indistinguishable from the matched classical RBF-SVM.",
            "evidence_source": "Phase 6.5.5 Matched Evaluation (Holm p=0.3428)",
            "severity": "HIGH",
            "status": "CORRECTED"
        },
        {
            "claim_id": "CLM-003",
            "file": "reports/phase6_5_4_dimension_report.md",
            "location": "Section 4 (Representation Optimality)",
            "original_claim": "d=8 is optimal.",
            "classification": "UNQUALIFIED_OPTIMALITY",
            "correction_required": "State that d=8 was selected as primary candidate based on variance retention and bottleneck relief.",
            "corrected_claim": "d=8 was selected as the primary candidate representation for the matched quantum comparison because it substantially relieved the observed Stage-2 compression bottleneck while retaining 89.73% +/- 2.99% cumulative variance under the evaluated recovery protocol.",
            "evidence_source": "Phase 6.5.4 Dimension Sweep (results/phase6_5_4_dimension_sweep.csv)",
            "severity": "HIGH",
            "status": "CORRECTED"
        },
        {
            "claim_id": "CLM-004",
            "file": "reports/phase6_5_4_dimension_report.md",
            "location": "Section 1 (Curse of Dimensionality)",
            "original_claim": "PCA solved the curse of dimensionality.",
            "classification": "UNSUPPORTED_UNIVERSAL_ASSERTION",
            "correction_required": "Replace with scoped statement on downstream performance.",
            "corrected_claim": "Reduced-dimensional representations were associated with improved downstream performance for some classical models under the evaluated few-shot conditions.",
            "evidence_source": "Phase 6.5.4 Classical Dimension Audit",
            "severity": "MEDIUM",
            "status": "CORRECTED"
        },
        {
            "claim_id": "CLM-005",
            "file": "reports/phase6_5_3_hierarchical_report.md",
            "location": "Abstract & Discussion (Emergency Reliability)",
            "original_claim": "Hierarchical routing solves emergency recognition and makes the system reliable for deployment.",
            "classification": "DEPLOYMENT_OVERCLAIM",
            "correction_required": "Emphasize FER reduction while explicitly noting low precision and high missed emergency rates.",
            "corrected_claim": "Hierarchical routing substantially reduced false-emergency rates under the evaluated matched recovery conditions, but substantial missed-emergency and low-precision limitations remained.",
            "evidence_source": "Phase 6.5.3 Error Taxonomy (phase6_5/hierarchical/figures/error_taxonomy.png)",
            "severity": "CRITICAL",
            "status": "CORRECTED"
        },
        {
            "claim_id": "CLM-006",
            "file": "reports/phase6_5_5_C_runtime_audit.md",
            "location": "Section 3 (Runtime Ratio Demarcation)",
            "original_claim": "Quantum is slower by 773x.",
            "classification": "AMBIGUOUS_RUNTIME_METRIC",
            "correction_required": "Separate per-evaluation test latency from total 60-condition wall time.",
            "corrected_claim": "The measured Q8-R1 QSVC implementation was approximately 772.6x slower per measured test evaluation than the matched d=8 RBF-SVM in exact statevector simulation, and approximately 212.8x higher in total experimental wall-clock time across the complete 60-condition experiment.",
            "evidence_source": "results/phase6_5_5_runtime.csv",
            "severity": "HIGH",
            "status": "CORRECTED"
        },
        {
            "claim_id": "CLM-007",
            "file": "reports/phase6_5_5_C_statistical_closure.md",
            "location": "Conclusion (Quantum Inefficacy)",
            "original_claim": "Quantum machine learning has no advantage and quantum kernels fail for speech.",
            "classification": "UNSUPPORTED_NEGATIVE_GENERALIZATION",
            "correction_required": "Scope strictly to protocol, representation, and simulation.",
            "corrected_claim": "Under the evaluated speaker-stratified few-shot protocol, frozen Whisper representation, and exact statevector fidelity-kernel configuration, no statistically supported quantum advantage was observed over the matched classical RBF-SVM.",
            "evidence_source": "results/phase6_5_5_statistical_tests.csv",
            "severity": "HIGH",
            "status": "CORRECTED"
        },
        {
            "claim_id": "CLM-008",
            "file": "reports/phase6_5_6_final_evidence_synthesis.md",
            "location": "Section 4 (Model Ranking)",
            "original_claim": "The classical Adapter is universally superior and QSVC is the safest model.",
            "classification": "UNSUPPORTED_SUPERIORITY_CLAIM",
            "correction_required": "Frame as an operational trade-off between recall and false alarms.",
            "corrected_claim": "Under the evaluated recovery protocol, the weighted Adapter achieved higher emergency recall than QSVC, while the RBF-SVM provided the strongest matched kernel baseline by the primary comparison criteria. Model selection involves a trade-off between emergency capture and false alarms.",
            "evidence_source": "results/phase6_5_5_C_adapter_comparison.csv",
            "severity": "MEDIUM",
            "status": "CORRECTED"
        },
        {
            "claim_id": "CLM-009",
            "file": "README.md",
            "location": "Project Description",
            "original_claim": "Safe, deployment-ready emergency communication system for clinical atypical speech.",
            "classification": "DEPLOYMENT_READINESS_HAZARD",
            "correction_required": "State non-deployment operational status and experimental framework nature.",
            "corrected_claim": "Research-stage emergency-intent recognition framework. Operational status: DEPLOYMENT READINESS: NOT READY; CLINICAL VALIDATION: NOT PERFORMED.",
            "evidence_source": "Repository Operational Contract",
            "severity": "CRITICAL",
            "status": "CORRECTED"
        },
        {
            "claim_id": "CLM-010",
            "file": "reports/phase4_5_complete_report.md",
            "location": "Table 3 (PCA Variance Placeholder)",
            "original_claim": "PCA-4 cumulative variance was 27.4%.",
            "classification": "REPORTING_SCRIPT_SYNTHETIC_ARTIFACT",
            "correction_required": "Disclose historical hardcoded script placeholder and report authoritative empirical variance.",
            "corrected_claim": "The previously reported 27.4% PCA-4 value was traced to a hardcoded plotting placeholder array [17.8, 27.4, 34.2, 39.8] in the historical reporting script and was not the empirical model-run variance. Forensic recomputation established the authoritative empirical values of 80.66% +/- 4.37% (d=4) and 89.73% +/- 2.99% (d=8).",
            "evidence_source": "Phase 6.5.4 Forensic PCA Audit (results/phase6_5_4_pca_variance.csv)",
            "severity": "CRITICAL",
            "status": "CORRECTED"
        },
        {
            "claim_id": "CLM-011",
            "file": "reports/phase6_5_5_quantum_comparison.md",
            "location": "Table 1 (Emergency Precision Artifact)",
            "original_claim": "QSVC End-to-End Emergency Precision was 11.24% and Adapter was 9.21%.",
            "classification": "BALANCED_SUBSET_AMBIGUITY",
            "correction_required": "Disclose exploratory balanced subset basis and present true full-cohort natural test distribution precision.",
            "corrected_claim": "The historical 11.24% (QSVC) and 9.21% (Adapter) values were reporting artifacts computed on exploratory balanced subsets. Reconciled full-cohort precision across all 60 conditions under the natural ~95.10% negative distribution is 0.81% +/- 1.31% (QSVC) and 2.03% +/- 1.45% (Adapter).",
            "evidence_source": "results/phase6_5_6_metric_reconciliation.csv",
            "severity": "CRITICAL",
            "status": "CORRECTED"
        },
        {
            "claim_id": "CLM-012",
            "file": "reports/phase5_results_report.md",
            "location": "Section 1 (Noise SNR Specification)",
            "original_claim": "Acoustic audio evaluated across 0 dB, 10 dB, 20 dB realistic ambient noise.",
            "classification": "PROTOCOL_PARAMETER_DISCREPANCY",
            "correction_required": "Correct to canonical 15, 10, 5 dB AWGN perturbation.",
            "corrected_claim": "Audio signals were subjected to controlled synthetic additive white Gaussian noise (AWGN) perturbation at 15, 10, and 5 dB SNR, with a clean baseline, injected at the waveform level before feature extraction.",
            "evidence_source": "scripts/run_phase5_noise_sweep.py",
            "severity": "HIGH",
            "status": "CORRECTED"
        }
    ]
    df = pd.DataFrame(records)
    out_path = "results/final_repair/final_claim_audit.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved final claim audit to: {out_path}")

def build_consistency_matrix_csv():
    records = [
        {
            "category": "Dataset",
            "source_A": "config/research_protocol.yaml",
            "source_B": "data/torgo_metadata.csv",
            "match": "MATCHED",
            "resolution": "TORGO database, 6067 recordings, 16 kHz mono.",
            "authoritative_source": "data/torgo_metadata.csv"
        },
        {
            "category": "Cohort",
            "source_A": "config/research_protocol.yaml",
            "source_B": "scripts/build_torgo_foundation.py",
            "match": "MATCHED",
            "resolution": "15 TORGO speakers: 8 dysarthric (F01, F03, F04, M01-M05) and 7 control (FC01-FC03, MC01-MC04).",
            "authoritative_source": "config/research_protocol.yaml"
        },
        {
            "category": "Calibration Budgets (N)",
            "source_A": "config/research_protocol.yaml",
            "source_B": "results/phase3/classical_sample_efficiency_aggregated.csv",
            "match": "MATCHED",
            "resolution": "N in {20, 30, 40, 50}. Historical N={5, 10} classified as unexecuted exploratory planning sketches.",
            "authoritative_source": "results/final_repair/sample_budget_provenance.csv"
        },
        {
            "category": "Taxonomy",
            "source_A": "config/taxonomy.yaml",
            "source_B": "scripts/run_phase5_noise_sweep.py",
            "match": "MATCHED",
            "resolution": "7 canonical classes: GENERAL_SPEECH, CRITICAL_RESPONSE, FALL, BREATHING_MEDICAL, PAIN, SOS_HELP, LOCATION.",
            "authoritative_source": "config/taxonomy.yaml"
        },
        {
            "category": "Adapter Architecture",
            "source_A": "src/models/classical_adapter.py",
            "source_B": "reports/phase5_correction_log.md (LOG-P5-01)",
            "match": "MATCHED",
            "resolution": "Linear(384,64) -> LayerNorm(64) -> GELU -> Dropout(0.1) -> Linear(64, num_classes), 25,223 params. Preliminary ReLU+Dropout(0.2) text superseded as prose typo.",
            "authoritative_source": "src/models/classical_adapter.py"
        },
        {
            "category": "PCA Variance (d=4)",
            "source_A": "results/phase6_5_4_pca_variance.csv",
            "source_B": "scripts/generate_phase4_reports.py",
            "match": "EXPLICIT HISTORICAL DIFFERENCE",
            "resolution": "Historical 27.4% was a hardcoded script placeholder [17.8, 27.4, 34.2, 39.8]; authoritative empirical value is 80.66% +/- 4.37%.",
            "authoritative_source": "results/phase6_5_4_pca_variance.csv"
        },
        {
            "category": "PCA Variance (d=8)",
            "source_A": "results/phase6_5_4_pca_variance.csv",
            "source_B": "reports/phase6_5_6_final_fair_comparison.md",
            "match": "MATCHED",
            "resolution": "Primary representation d=8 retains 89.73% +/- 2.99% cumulative explained variance.",
            "authoritative_source": "results/phase6_5_4_pca_variance.csv"
        },
        {
            "category": "Phase 5 Noise Levels",
            "source_A": "scripts/run_phase5_noise_sweep.py",
            "source_B": "phase6/analysis/statistical_tests.csv",
            "match": "MATCHED",
            "resolution": "Clean, 15 dB, 10 dB, 5 dB SNR controlled synthetic AWGN perturbation. Historical 0/10/20 dB drafts superseded.",
            "authoritative_source": "scripts/run_phase5_noise_sweep.py"
        },
        {
            "category": "IPER Metric Definition",
            "source_A": "src/metrics/metric_registry.py",
            "source_B": "results/final_repair/phase6_5_5_IPER_reconciled.csv",
            "match": "MATCHED",
            "resolution": "Disambiguated IPER (Intent Preservation Rate: 8.25% QSVC, 11.69% RBF) from IPER_ERROR (Error Rate: 91.75% QSVC, 88.31% RBF). Case B holds under both (p=0.0582).",
            "authoritative_source": "results/final_repair/phase6_5_5_IPER_reconciled.csv"
        },
        {
            "category": "Emergency Precision",
            "source_A": "results/phase6_5_6_metric_reconciliation.csv",
            "source_B": "reports/phase6_5_6_final_fair_comparison.md",
            "match": "MATCHED",
            "resolution": "Disclosed historical provisional 11.24% / 9.21% as balanced-subset artifacts. Reconciled full-cohort natural test precision is 0.81% (QSVC) and 2.03% (Adapter).",
            "authoritative_source": "results/phase6_5_6_metric_reconciliation.csv"
        },
        {
            "category": "Stage-1 Balanced Accuracy",
            "source_A": "phase6_5/hierarchical/results/stage1_predictions.csv",
            "source_B": "results/final_repair/stage1_balanced_accuracy_reconciliation.csv",
            "match": "MATCHED",
            "resolution": "Recomputed condition-level: RBF N=30 Bal Acc = 49.92% (Specificity 84.94% -> FER 15.06%). Historical 81.24% superseded as specificity/raw accuracy mislabeling.",
            "authoritative_source": "results/final_repair/stage1_balanced_accuracy_reconciliation.csv"
        },
        {
            "category": "QSVC Runtime Ratios",
            "source_A": "results/phase6_5_5_runtime.csv",
            "source_B": "reports/phase6_5_5_C_runtime_audit.md",
            "match": "MATCHED",
            "resolution": "Q8-R1 is ~772.6x slower per measured evaluation; ~212.8x higher total wall time across 60 conditions. Q8-R2 is ~1408.8x per eval, ~388.0x total wall time.",
            "authoritative_source": "results/phase6_5_5_runtime.csv"
        },
        {
            "category": "Primary Quantum Decision",
            "source_A": "results/phase6_5_6_final_decision.csv",
            "source_B": "results/phase6_5_5_statistical_tests.csv",
            "match": "MATCHED",
            "resolution": "CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE across all primary hypotheses (Macro-F1 p=0.3428, Recall p=0.0689, FER p=0.9505, IPER p=0.0582).",
            "authoritative_source": "results/phase6_5_6_final_decision.csv"
        },
        {
            "category": "Deployment Readiness",
            "source_A": "config/research_protocol.yaml",
            "source_B": "README.md",
            "match": "MATCHED",
            "resolution": "DEPLOYMENT READINESS: NOT READY. Clinical validation: NOT PERFORMED. Hardware execution: NOT TESTED.",
            "authoritative_source": "config/research_protocol.yaml"
        }
    ]
    df = pd.DataFrame(records)
    out_path = "results/final_repair/final_consistency_matrix.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved consistency matrix to: {out_path}")

def generate_final_reports():
    final_dir = "reports/final"
    os.makedirs(final_dir, exist_ok=True)

    # 1. FINAL_RESEARCH_RESULTS.md (Definitive human-readable report with reconciled metrics)
    with open("COMPLETE_RESEARCH_RESULTS.md", "r", encoding="utf-8") as f:
        master_content = f.read()
    
    # Apply repairs to master_content
    # Repair AWGN description
    repaired_results = master_content.replace(
        "Audio signals from the TORGO database were augmented with realistic ambient acoustic noise across three Signal-to-Noise Ratios (0 dB, 10 dB, 20 dB, plus clean baseline):",
        "Audio signals from the 15-speaker TORGO cohort (8 dysarthric, 7 control) were subjected to controlled synthetic additive white Gaussian noise (AWGN) perturbation across three SNR levels (15 dB, 10 dB, 5 dB SNR, plus clean baseline) injected at the waveform level before feature extraction:"
    ).replace(
        "Macro-F1 ($N=5$)", "Macro-F1 ($N=20$)"
    ).replace(
        "15 dysarthric speakers", "15 TORGO speakers (8 dysarthric, 7 control)"
    ).replace(
        "Stage-1 Balanced Accuracy | Emergency Recall | False-Emergency Rate (FER) | End-to-End Macro-F1 |\n| :--- | :--- | :---: | :---: | :---: | :---: |\n| **Classical RBF-SVM** | Flat 7-Class (Class-Weighted) | N/A | $31.89\\%$ | **$96.82\\%$** | $0.0982$ |\n| **Classical RBF-SVM** | **Two-Stage Hierarchical** | **$81.24\\%$** | $31.89\\%$ | **$15.06\\%$** | **$0.1087$** |\n| **Classical Adapter** | Flat 7-Class (Class-Weighted) | N/A | $45.81\\%$ | **$98.14\\%$** | $0.0894$ |\n| **Classical Adapter** | **Two-Stage Hierarchical** | **$78.91\\%$** | $45.81\\%$ | **$49.82\\%$** | **$0.0984$** |",
        "Stage-1 Specificity | Emergency Recall | False-Emergency Rate (FER) | End-to-End Macro-F1 |\n| :--- | :--- | :---: | :---: | :---: | :---: |\n| **Classical RBF-SVM** | Flat 7-Class (Class-Weighted) | $3.18\\%$ | $31.89\\%$ | **$96.82\\%$** | $0.0982$ |\n| **Classical RBF-SVM** | **Two-Stage Hierarchical** | **$84.94\\%$** | $31.89\\%$ | **$15.06\\%$** | **$0.1087$** |\n| **Classical Adapter** | Flat 7-Class (Class-Weighted) | $1.86\\%$ | $45.81\\%$ | **$98.14\\%$** | $0.0894$ |\n| **Classical Adapter** | **Two-Stage Hierarchical** | **$55.44\\%$** | $45.81\\%$ | **$49.82\\%$** | **$0.0984$** |"
    ).replace(
        "Intent Preservation Error (IPER)** | $97.23\\% \\pm 4.09\\%$ | $93.79\\% \\pm 8.21\\%$ | $+3.44\\text{ pp}$",
        "Intent Preservation Rate (IPER)** | $8.25\\% \\pm 4.09\\%$ | $11.69\\% \\pm 8.21\\%$ | $-3.44\\text{ pp}$"
    )

    with open(os.path.join(final_dir, "FINAL_RESEARCH_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(repaired_results)
    print(f"Saved: {os.path.join(final_dir, 'FINAL_RESEARCH_RESULTS.md')}")

    # 2. FINAL_RESEARCH_CONCLUSION.md
    with open("reports/phase6_5_6_research_conclusion.md", "r", encoding="utf-8") as f:
        conc_content = f.read()
    with open(os.path.join(final_dir, "FINAL_RESEARCH_CONCLUSION.md"), "w", encoding="utf-8") as f:
        f.write(conc_content)
    print(f"Saved: {os.path.join(final_dir, 'FINAL_RESEARCH_CONCLUSION.md')}")

    # 3. FINAL_PROTOCOL.md
    with open("config/research_protocol.yaml", "r", encoding="utf-8") as f:
        proto_yaml = f.read()
    proto_md = f"""# FINAL RESEARCH PROTOCOL SPECIFICATION
## Single Source of Truth for Experimental Execution & Evaluation

This document constitutes the authoritative protocol contract for the research project:
**Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication** (Team: `kernal`).

```yaml
{proto_yaml}
```
"""
    with open(os.path.join(final_dir, "FINAL_PROTOCOL.md"), "w", encoding="utf-8") as f:
        f.write(proto_md)
    print(f"Saved: {os.path.join(final_dir, 'FINAL_PROTOCOL.md')}")

    # 4. FINAL_METRIC_DEFINITIONS.md
    shutil.copy2("reports/FINAL_METRIC_DEFINITIONS.md", os.path.join(final_dir, "FINAL_METRIC_DEFINITIONS.md"))
    print(f"Saved: {os.path.join(final_dir, 'FINAL_METRIC_DEFINITIONS.md')}")

    # 5. FINAL_MODEL_CONFIGURATION.md
    with open("config/model_registry.yaml", "r", encoding="utf-8") as f:
        model_yaml = f.read()
    model_md = f"""# FINAL MODEL CONFIGURATION REGISTRY
## Authoritative Architecture & Parameter Specifications

```yaml
{model_yaml}
```
"""
    with open(os.path.join(final_dir, "FINAL_MODEL_CONFIGURATION.md"), "w", encoding="utf-8") as f:
        f.write(model_md)
    print(f"Saved: {os.path.join(final_dir, 'FINAL_MODEL_CONFIGURATION.md')}")

    # 6. FINAL_PROJECT_PLAN.md
    shutil.copy2("docs/FINAL_PROJECT_PLAN.md", os.path.join(final_dir, "FINAL_PROJECT_PLAN.md"))
    print(f"Saved: {os.path.join(final_dir, 'FINAL_PROJECT_PLAN.md')}")

    # 7. FINAL_CLAIM_AUDIT.md
    shutil.copy2("reports/FINAL_CLAIM_CORRECTIONS.md", os.path.join(final_dir, "FINAL_CLAIM_AUDIT.md"))
    print(f"Saved: {os.path.join(final_dir, 'FINAL_CLAIM_AUDIT.md')}")

    # 8. FINAL_DATA_PROVENANCE.md
    shutil.copy2("reports/FINAL_SCIENTIFIC_INTEGRITY_AUDIT.md", os.path.join(final_dir, "FINAL_DATA_PROVENANCE.md"))
    print(f"Saved: {os.path.join(final_dir, 'FINAL_DATA_PROVENANCE.md')}")

if __name__ == "__main__":
    build_claim_audit_csv()
    build_consistency_matrix_csv()
    generate_final_reports()
