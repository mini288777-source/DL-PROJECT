"""
build_sample_budget_provenance.py
=================================
Generates results/final_repair/sample_budget_provenance.csv
Documents the exact provenance of calibration sample budgets N in {20, 30, 40, 50}
and classifies historical occurrences of N={5, 10}.
"""

import os
import pandas as pd

def build_provenance():
    records = [
        {
            "phase": "Phase 3 (Classical Baselines)",
            "artifact": "results/phase3/classical_sample_efficiency_aggregated.csv",
            "evaluated_budgets": "N in {20, 30, 40, 50}",
            "status": "AUTHORITATIVE_FROZEN",
            "historical_draft_notes": "Canonical benchmark executed strictly across 20, 30, 40, 50 samples."
        },
        {
            "phase": "Phase 4 (Quantum Baseline)",
            "artifact": "results/phase4/quantum_sample_efficiency_aggregated.csv",
            "evaluated_budgets": "N in {20, 30, 40, 50}",
            "status": "AUTHORITATIVE_FROZEN",
            "historical_draft_notes": "Exact 15-speaker sweep executed at N=20, 30, 40, 50."
        },
        {
            "phase": "Phase 5 (Noise Robustness)",
            "artifact": "scripts/run_phase5_noise_sweep.py",
            "evaluated_budgets": "N in {20, 30, 40, 50}",
            "status": "AUTHORITATIVE_FROZEN",
            "historical_draft_notes": "All 720 cells (15 speakers x 4 budgets x 3 models x 4 SNR levels) evaluated at N in {20, 30, 40, 50}."
        },
        {
            "phase": "Phase 6 (Statistical Analysis)",
            "artifact": "phase6/analysis/statistical_tests.csv",
            "evaluated_budgets": "N in {20, 30, 40, 50}",
            "status": "AUTHORITATIVE_FROZEN",
            "historical_draft_notes": "Statistical tests paired strictly across N in {20, 30, 40, 50}."
        },
        {
            "phase": "Phase 6.5.1 - 6.5.3 (Recovery Track)",
            "artifact": "phase6_5/splits/track_b_splits.json",
            "evaluated_budgets": "N in {20, 30, 40, 50}",
            "status": "AUTHORITATIVE_FROZEN",
            "historical_draft_notes": "Track B stratified splits generated strictly for calibration budgets 20, 30, 40, 50."
        },
        {
            "phase": "Phase 6.5.4 (PCA Representation Audit)",
            "artifact": "results/phase6_5_4_dimension_sweep.csv",
            "evaluated_budgets": "N in {20, 30, 40, 50}",
            "status": "AUTHORITATIVE_FROZEN",
            "historical_draft_notes": "Dimensionality sweep evaluated across dimensions 2 to 64 at N in {20, 30, 40, 50}."
        },
        {
            "phase": "Phase 6.5.5 (Quantum Recovery & Closure)",
            "artifact": "results/phase6_5_5_matched_comparison.csv",
            "evaluated_budgets": "N in {20, 30, 40, 50}",
            "status": "AUTHORITATIVE_FROZEN",
            "historical_draft_notes": "60 matched conditions (15 speakers x 4 budgets N in {20, 30, 40, 50})."
        },
        {
            "phase": "Preliminary Project Proposal / Draft Notes",
            "artifact": "historical_sketches/early_proposal_drafts",
            "evaluated_budgets": "N in {5, 10, 20, 50}",
            "status": "HISTORICAL_EXPLORATORY_PROSE_ONLY",
            "historical_draft_notes": "N=5 and N=10 were cited in early conceptual planning sketches prior to data audit. Formal feasibility audits confirmed N < 20 could not provide minimal minority emergency support; canonical implementation started at N=20."
        }
    ]

    df = pd.DataFrame(records)
    out_path = "results/final_repair/sample_budget_provenance.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved sample budget provenance to: {out_path}")

if __name__ == "__main__":
    build_provenance()
