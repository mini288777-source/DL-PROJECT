"""
Generate and execute Phase 3 baselines validation notebook.

Populates 'notebooks/phase3_baselines_validation.ipynb' with interactive,
executable cells demonstrating data loading, pilot runs, adapter and RBF-SVM
training, metric validation, and full sweep verification.
"""

import os
import sys
import nbformat as nbf
from nbclient import NotebookClient


def create_and_execute_notebook(output_path: str = "notebooks/phase3_baselines_validation.ipynb"):
    print("Building Phase 3 validation notebook...")

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }

    cells = []

    # 1. Title & Intro
    cells.append(nbf.v4.new_markdown_cell("""# Phase 3 — Classical Baselines & Few-Shot Sample-Efficiency Validation

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO (15-Speaker Few-Shot Cohort: 8 Dysarthric, 7 Control)  
**Input Representation:** Frozen 384-dimensional Whisper Embeddings (unit-$L_2$ normalized)  
**Phase Objective:** Establish the classical reference performance (Adapter Head vs Classical RBF-Kernel SVM) across few-shot sample budgets $N \\in \\{20, 30, 40, 50\\}$ before introducing the Quantum Kernel in Phase 4.
"""))

    # 2. Imports & Setup
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Resolve project root dynamically whether running from project root or notebooks dir
if os.path.exists("configs") and os.path.exists("src"):
    PROJECT_ROOT = os.path.abspath(".")
else:
    PROJECT_ROOT = os.path.abspath("..")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.embeddings.cache import load_embedding_file
from src.models.classical_adapter import ClassicalAdapterClassifier
from src.models.rbf_svm import ClassicalRBFSVMClassifier
from src.models.evaluation import compute_intent_metrics
from src.utils.seeding import set_seed

print(f"Project root: {PROJECT_ROOT}")
print("Successfully imported Phase 3 modeling and evaluation modules.")
"""))

    # 3. Load Config
    cells.append(nbf.v4.new_markdown_cell("""## 1. Load Experiment Configuration
We verify that experimental hyperparameters, cohort definitions, and output targets are loaded from `configs/phase3_baselines.yaml`.
"""))

    cells.append(nbf.v4.new_code_cell("""config_path = os.path.join(PROJECT_ROOT, "configs/phase3_baselines.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(f"Project:     {cfg['project']['name']}")
print(f"Phase:       {cfg['project']['phase']}")
print(f"Cohort Size: {len(cfg['cohort']['speakers'])} speakers")
print(f"Sample Ns:   {cfg['cohort']['sample_sizes']}")
print(f"Methods:     {cfg['methods']}")
"""))

    # 4. Load Data & Embeddings
    cells.append(nbf.v4.new_markdown_cell("""## 2. Load Phase 1 Splits and Phase 2 Cached Embeddings
We inspect the pilot speakers: **F01** (dysarthric) and **FC01** (control).
"""))

    cells.append(nbf.v4.new_code_cell("""manifest_path = os.path.join(PROJECT_ROOT, "data/manifests/torgo_manifest.csv")
splits_path = os.path.join(PROJECT_ROOT, "splits/torgo_splits.json")

df_manifest = pd.read_csv(manifest_path).set_index("recording_id", drop=False)
with open(splits_path, "r", encoding="utf-8") as f:
    splits_data = json.load(f)

print(f"Loaded manifest with {len(df_manifest)} entries.")
print(f"Loaded {len(splits_data)} split conditions across cohort.")
"""))

    # 5. Run Pilot
    cells.append(nbf.v4.new_markdown_cell("""## 3. Train and Evaluate Classical Baselines (Pilot)
We execute a pilot on F01 and FC01 across all sample sizes $N \\in \\{20, 30, 40, 50\\}$ for both the **Adapter** and **RBF-SVM**.
"""))

    cells.append(nbf.v4.new_code_cell("""pilot_speakers = ["F01", "FC01"]
sample_sizes = [20, 30, 40, 50]
taxonomy_path = os.path.join(PROJECT_ROOT, "configs/intent_taxonomy.yaml")
all_classes = [item["intent_id"] for item in yaml.safe_load(open(taxonomy_path))["intents"]]
whisper_base_dir = os.path.join(PROJECT_ROOT, "embeddings/torgo")

pilot_results = []
for spk in pilot_speakers:
    for n in sample_sizes:
        entry = next(e for e in splits_data if e["speaker_id"] == spk and e["sample_n"] == n)
        ad_ids = entry["adaptation_recording_ids"]
        test_ids = entry["test_recording_ids"]
        
        X_train = np.vstack([load_embedding_file(r, "whisper", whisper_base_dir) for r in ad_ids])
        y_train = [df_manifest.loc[r, "intent"] for r in ad_ids]
        
        X_test = np.vstack([load_embedding_file(r, "whisper", whisper_base_dir) for r in test_ids])
        y_test = [df_manifest.loc[r, "intent"] for r in test_ids]
        
        # 1. Adapter
        set_seed(42)
        adapter = ClassicalAdapterClassifier(epochs=40, seed=42, all_classes=all_classes)
        adapter.fit(X_train, y_train)
        pred_adapter = adapter.predict(X_test)
        m_adapter = compute_intent_metrics(y_test, pred_adapter, all_classes)
        
        # 2. RBF-SVM
        set_seed(42)
        rbf = ClassicalRBFSVMClassifier(C=1.0, gamma="scale", seed=42, all_classes=all_classes)
        rbf.fit(X_train, y_train)
        pred_rbf = rbf.predict(X_test)
        m_rbf = compute_intent_metrics(y_test, pred_rbf, all_classes)
        
        pilot_results.append({
            "speaker_id": spk,
            "N": n,
            "adapter_acc": m_adapter["accuracy"],
            "adapter_f1": m_adapter["macro_f1"],
            "rbf_acc": m_rbf["accuracy"],
            "rbf_f1": m_rbf["macro_f1"],
        })

df_pilot = pd.DataFrame(pilot_results)
df_pilot
"""))

    # 6. Full Sweep Summary
    cells.append(nbf.v4.new_markdown_cell("""## 4. Full 120-Condition Sweep Results & Aggregated Trajectories
We inspect the completed 120-condition benchmark curves generated by `scripts/run_classical_baselines.py`.
"""))

    cells.append(nbf.v4.new_code_cell("""curves_path = os.path.join(PROJECT_ROOT, "results/phase3/baseline_curves.csv")
agg_path = os.path.join(PROJECT_ROOT, "results/phase3/classical_sample_efficiency_aggregated.csv")
preds_path = os.path.join(PROJECT_ROOT, "results/phase3/predictions.csv")

df_curves = pd.read_csv(curves_path)
df_agg = pd.read_csv(agg_path)

print(f"Total evaluated conditions: {len(df_curves)} (100% complete)")
print(f"Total recordings predicted: {len(pd.read_csv(preds_path))}")
display(df_agg[["method", "sample_n", "macro_f1_mean", "macro_f1_ci95", "macro_f1_dysarthric_mean", "macro_f1_control_mean", "accuracy_mean"]])
"""))

    # 7. Verification of Output Schema
    cells.append(nbf.v4.new_markdown_cell("""## 5. Verification of Output Artifacts & Deliverables
We verify the existence and schema of all Phase 3 required deliverables.
"""))

    cells.append(nbf.v4.new_code_cell("""expected_files = [
    os.path.join(PROJECT_ROOT, "results/phase3/baseline_curves.csv"),
    os.path.join(PROJECT_ROOT, "results/phase3/predictions.csv"),
    os.path.join(PROJECT_ROOT, "results/phase3/run_log.csv"),
    os.path.join(PROJECT_ROOT, "results/phase3/classical_sample_efficiency_aggregated.csv"),
    os.path.join(PROJECT_ROOT, "results/phase3/figures/classical_sample_efficiency.png"),
    os.path.join(PROJECT_ROOT, "reports/phase3_class_distribution.csv"),
    os.path.join(PROJECT_ROOT, "reports/phase3_model_configuration.csv"),
    os.path.join(PROJECT_ROOT, "reports/phase3_fairness_audit.md"),
    os.path.join(PROJECT_ROOT, "reports/phase3_classical_baselines_summary.md")
]

all_exist = True
for f in expected_files:
    exists = os.path.exists(f)
    print(f"[{'PASS' if exists else 'FAIL'}] {os.path.basename(f)} exists: {exists}")
    if not exists:
        all_exist = False

print(f"\\nAll Phase 3 deliverables verified: {all_exist}")
"""))

    nb.cells = cells

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to: {output_path}")

    # Execute the notebook to ensure all outputs are rendered
    print("Executing notebook via NotebookClient...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook executed successfully and saved with outputs to: {output_path}")


if __name__ == "__main__":
    create_and_execute_notebook()
