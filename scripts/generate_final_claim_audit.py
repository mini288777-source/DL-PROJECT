"""
Generate results/final_claim_audit.csv
Machine-readable audit of scientific claims, classifications, corrections, and severities.
"""

import os
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

AUDIT_RECORDS = [
    {
        "claim_id": "CLM-001",
        "file": "reports/phase6_5_6_EXECUTIVE_SUMMARY.md",
        "location": "Section: Most Important Contribution (Line 57)",
        "original_claim": "this study demonstrates that an apparent quantum collapse in personalized speech recognition was actually an artifact of few-shot data starvation and formulation flaws",
        "classification": "CAUSAL_OVERSTATEMENT",
        "correction_required": "YES",
        "corrected_claim": "this study demonstrates that the original quantum collapse was consistent with severe few-shot data starvation, class imbalance, and the flat classification formulation identified during forensic analysis. Recovery experiments subsequently demonstrated that the collapse behavior was not unique to the quantum model.",
        "evidence_source": "Phase 6.5.0 diagnostic and Phase 6.5.2 classical rescue ablation",
        "severity": "CRITICAL"
    },
    {
        "claim_id": "CLM-002",
        "file": "reports/phase6_5_6_research_conclusion.md",
        "location": "Question 5 Answer (Line 35)",
        "original_claim": "The fidelity quantum kernel with ZZFeatureMap acted functionally as a stationary distance-based similarity metric analogous to an RBF kernel.",
        "classification": "OVERLY_BROAD_GENERALIZATION",
        "correction_required": "YES",
        "corrected_claim": "Under the evaluated frozen Whisper representation, feature scaling, kernel configuration, and few-shot protocol, the fidelity quantum kernel produced performance statistically indistinguishable from the matched classical RBF-SVM.",
        "evidence_source": "Phase 6.5.5-C Wilcoxon signed-rank test (W=475.0, raw p=0.1714, Holm p=0.3428)",
        "severity": "HIGH"
    },
    {
        "claim_id": "CLM-003",
        "file": "reports/phase6_5_4_representation_pca_audit.md",
        "location": "Section 4.2 Candidate Selection",
        "original_claim": "d=8 is the optimal dimension for quantum feature mapping",
        "classification": "UNSCOPED_OPTIMALITY",
        "correction_required": "YES",
        "corrected_claim": "d=8 was selected as the primary candidate representation for the matched quantum comparison because it substantially relieved the observed Stage-2 compression bottleneck while retaining 89.73% +/- 2.99% cumulative variance under the evaluated recovery protocol.",
        "evidence_source": "results/phase6_5_4_dimension_ablation.csv and PCA cumulative variance analysis",
        "severity": "MEDIUM"
    },
    {
        "claim_id": "CLM-004",
        "file": "reports/phase6_5_4_representation_pca_audit.md",
        "location": "Section 5 Discussion",
        "original_claim": "PCA dimensionality reduction solved the curse of dimensionality for low-resource adaptation",
        "classification": "BROAD_METHODOLOGICAL_CLAIM",
        "correction_required": "YES",
        "corrected_claim": "Reduced-dimensional representations were associated with improved downstream performance for some classical models under the evaluated few-shot conditions.",
        "evidence_source": "Table 4 dimension sweep comparing d=8, d=32, and unreduced d=384",
        "severity": "MEDIUM"
    },
    {
        "claim_id": "CLM-005",
        "file": "reports/phase6_5_3_hierarchical_report.md",
        "location": "Executive Summary & Conclusion",
        "original_claim": "Hierarchical routing solves the emergency false-alarm problem and makes the system reliable",
        "classification": "DEPLOYMENT_OVERSTATEMENT",
        "correction_required": "YES",
        "corrected_claim": "Hierarchical routing substantially reduced false-emergency rates under the evaluated matched recovery conditions, but substantial missed-emergency and low-precision limitations remained.",
        "evidence_source": "results/phase6_5_3/hierarchical_results.csv showing FER drop from 96.82% to 15.06% but precision <= 2%",
        "severity": "HIGH"
    },
    {
        "claim_id": "CLM-006",
        "file": "reports/phase6_5_5_quantum_recovery_report.md",
        "location": "Section 6 Computational Cost",
        "original_claim": "Quantum kernel evaluation is slower by 770x overall compared to classical SVM",
        "classification": "CONFLATED_MEASUREMENT_BASIS",
        "correction_required": "YES",
        "corrected_claim": "The measured Q8-R1 QSVC implementation was approximately 772.6x slower per measured test evaluation than the matched d=8 RBF-SVM in exact statevector simulation. Across the complete 60-condition experiment, total wall-clock time was approximately 212.8x higher for Q8-R1 than for the matched RBF-SVM.",
        "evidence_source": "results/phase6_5_5_C_runtime_audit.csv",
        "severity": "HIGH"
    },
    {
        "claim_id": "CLM-007",
        "file": "reports/phase4_5_null_result_statement.md",
        "location": "Section 3 Interpretation",
        "original_claim": "Quantum machine learning provides no advantage and quantum kernels fail for atypical speech",
        "classification": "UNIVERSAL_NEGATIVE_OVERCLAIM",
        "correction_required": "YES",
        "corrected_claim": "No statistically supported quantum advantage was observed under the evaluated configuration.",
        "evidence_source": "Phase 6.5.5-C Wilcoxon signed-rank tests across 60 matched conditions (Case B)",
        "severity": "CRITICAL"
    },
    {
        "claim_id": "CLM-008",
        "file": "reports/phase6_5_5_quantum_comparison.md",
        "location": "Section 4 Model Selection",
        "original_claim": "The Classical Adapter is universally superior to quantum SVC and RBF-SVM",
        "classification": "UNSCOPED_SUPERIORITY",
        "correction_required": "YES",
        "corrected_claim": "Under the evaluated recovery protocol, the weighted Adapter achieved higher emergency recall than QSVC, while the RBF-SVM provided the strongest matched kernel baseline by the primary comparison criteria.",
        "evidence_source": "results/phase6_5_5_C_adapter_comparison.csv showing higher recall (45.81% vs 27.52%) but higher FER (49.82% vs 28.12%)",
        "severity": "MEDIUM"
    },
    {
        "claim_id": "CLM-009",
        "file": "reports/phase6_5_6_final_fair_comparison.md",
        "location": "Throughout document terminology",
        "original_claim": "Reliable Emergency Communication System (referring to operational capability of pipeline)",
        "classification": "PREMATURE_DEPLOYMENT_CLAIM",
        "correction_required": "YES",
        "corrected_claim": "Experimental emergency-intent recognition framework / research-stage emergency speech recognition system (Status: NOT READY)",
        "evidence_source": "reports/phase6_5_6_limitations.md (precision <= 2.03%, missed emergency rate >= 54.19%)",
        "severity": "CRITICAL"
    },
    {
        "claim_id": "CLM-010",
        "file": "reports/phase6_5_5_quantum_recovery_report.md",
        "location": "Table 4.1 (Line 85)",
        "original_claim": "Emergency Precision: 11.24% +/- 9.45% (QSVC Q8-R1) and 9.21% +/- 7.15% (Adapter d=8)",
        "classification": "HISTORICAL_REPORTING_ARTIFACT",
        "correction_required": "YES",
        "corrected_claim": "Historical provisional values (11.24% and 9.21%) were reporting transcription artifacts; authoritative final end-to-end emergency precision across all 60 conditions is 0.81% +/- 1.31% (QSVC) and 2.03% +/- 1.45% (Adapter), reflecting extreme 95.10% negative test prevalence.",
        "evidence_source": "results/phase6_5_6_metric_reconciliation.csv and results/phase6_5_5_speaker_results.csv",
        "severity": "CRITICAL"
    },
    {
        "claim_id": "CLM-011",
        "file": "reports/phase4_complete_results.md",
        "location": "Section 2.2 PCA Variance",
        "original_claim": "PCA-4 captures 27.4% cumulative variance of speech embeddings",
        "classification": "HISTORICAL_SCRIPT_ARTIFACT",
        "correction_required": "YES",
        "corrected_claim": "The previously reported 27.4% PCA-4 value was traced to a hardcoded plotting placeholder array [17.8, 27.4, 34.2, 39.8] in the historical reporting script and was not the empirical model-run variance. Forensic recomputation established the authoritative empirical value of 80.66% +/- 4.37% (and 89.73% +/- 2.99% for d=8).",
        "evidence_source": "reports/phase6_5_4_pca_forensic_reconciliation.md",
        "severity": "HIGH"
    },
    {
        "claim_id": "CLM-012",
        "file": "reports/phase6_5_6_final_evidence_synthesis.md",
        "location": "Section 3 Protocol Boundary",
        "original_claim": "Phase 6.5 recovery proved that Phase 3 failure was caused by class imbalance and lack of stratification",
        "classification": "CROSS_PROTOCOL_CAUSAL_CONFLATION",
        "correction_required": "YES",
        "corrected_claim": "The recovery experiments demonstrated that the original collapse was not uniquely attributable to the quantum model and identified data imbalance, few-shot class support, hierarchical formulation, and representation dimensionality as important methodological factors under the recovery protocol.",
        "evidence_source": "Two distinct experimental tracks: Track 1 (Phases 3-6) vs Track 2 (Phases 6.5.0-6.5.5-C)",
        "severity": "HIGH"
    }
]


def main():
    df = pd.DataFrame(AUDIT_RECORDS)
    out_path = os.path.join(RESULTS_DIR, "final_claim_audit.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
