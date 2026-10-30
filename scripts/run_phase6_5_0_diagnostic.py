"""
Subphase 6.5.0 — Forensic Diagnostic Script

Performs full data feasibility, class distribution, kernel collapse,
representation bottleneck, and quantum configuration audit.

Outputs:
  phase6_5/analysis/adaptation_class_support.csv
  phase6_5/analysis/emergency_support.csv
  phase6_5/analysis/speaker_feasibility_matrix.csv
  phase6_5/analysis/representation_diagnostic.csv
  phase6_5/analysis/quantum_config_audit.csv
  phase6_5/analysis/prediction_distribution_by_model.csv
  phase6_5/analysis/pca_explained_variance.csv
"""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
SPLITS_JSON   = "splits/torgo_splits.json"
MANIFEST_CSV  = "data/manifests/torgo_manifest.csv"
INTENT_MAP    = "data/manifests/torgo_intent_mapping.csv"
P5_RESULTS    = "results/phase5/phase5_results.csv"
P5_PREDS      = "results/phase5/phase5_predictions.csv"
P5_PERCLASS   = "results/phase5/phase5_per_class.csv"
EMBED_DIR     = "embeddings/torgo/whisper"
OUT_ANA       = "phase6_5/analysis"
OUT_REP       = "phase6_5/reports"

SPEAKERS = ["F01","F03","F04","FC01","FC02","FC03",
            "M01","M02","M03","M04","M05","MC01","MC02","MC03","MC04"]
DYSARTHRIC = {"F01","F03","F04","M01","M02","M03","M04","M05"}
CONTROL    = {"FC01","FC02","FC03","MC01","MC02","MC03","MC04"}
NS = [20, 30, 40, 50]
CLASSES = ["SOS_HELP","FALL","BREATHING_MEDICAL","PAIN",
           "LOCATION","CRITICAL_RESPONSE","GENERAL_SPEECH"]
EMERGENCY_CLASSES = {"SOS_HELP","FALL","BREATHING_MEDICAL","PAIN",
                     "LOCATION","CRITICAL_RESPONSE"}

os.makedirs(OUT_ANA, exist_ok=True)
os.makedirs(OUT_REP, exist_ok=True)

print("="*70)
print("SUBPHASE 6.5.0 — FORENSIC DIAGNOSTIC")
print("="*70)

# ============================================================
# STEP 1: Load splits
# ============================================================
print("\n[1/8] Loading adaptation splits...")
with open(SPLITS_JSON, "r") as f:
    splits_raw = json.load(f)

# Build lookup: splits_raw is a list of split dicts
# Each entry: {speaker_id, sample_n, adaptation_ids: [...], test_ids: [...]}
split_lookup = {}  # (speaker_id, N) -> {adaptation_ids, test_ids, labels_adapt}
for s in splits_raw:
    key = (s["speaker_id"], s["sample_n"])
    split_lookup[key] = s

print(f"  Loaded {len(splits_raw)} split definitions.")
print(f"  Example keys: {list(split_lookup.keys())[:3]}")

# ============================================================
# STEP 2: Load manifest and intent mapping
# ============================================================
print("\n[2/8] Loading manifest and intent mapping...")
manifest = pd.read_csv(MANIFEST_CSV)
print(f"  Manifest rows: {len(manifest)}")
print(f"  Manifest columns: {list(manifest.columns)[:10]}")

intent_map = pd.read_csv(INTENT_MAP)
print(f"  Intent mapping rows: {len(intent_map)}")
print(f"  Intent mapping columns: {list(intent_map.columns)[:10]}")

# Merge manifest with intent mapping on recording_id
INTENT_COL = "intent_candidate" if "intent_candidate" in intent_map.columns else "intent_label"
if "recording_id" in manifest.columns and "recording_id" in intent_map.columns:
    merged = manifest.merge(intent_map[["recording_id", INTENT_COL]],
                            on="recording_id", how="left")
    merged = merged.rename(columns={INTENT_COL: "intent_label"})
    print(f"  After merge: {len(merged)} rows, intent coverage: {merged['intent_label'].notna().sum()}")
elif "intent" in manifest.columns:
    merged = manifest.copy()
    merged["intent_label"] = merged["intent"]
    print(f"  Intent column found directly in manifest.")
else:
    print(f"  WARNING: Could not merge intent labels. Columns: {list(manifest.columns)}")
    merged = manifest.copy()
    merged["intent_label"] = "UNKNOWN"

# ============================================================
# STEP 3: Per speaker × N adaptation class support analysis
# ============================================================
print("\n[3/8] Computing per-speaker × N adaptation class support...")

rows = []
for spk in SPEAKERS:
    grp = "dysarthric" if spk in DYSARTHRIC else "control"
    for n in NS:
        key = (spk, n)
        if key not in split_lookup:
            print(f"  WARNING: No split found for {spk}, N={n}")
            continue
        split = split_lookup[key]
        adapt_ids = set(split.get("adaptation_recording_ids", split.get("adaptation_ids", [])))
        test_ids  = set(split.get("test_recording_ids", split.get("test_ids", [])))

        # Get labels for adaptation IDs
        adapt_rows = merged[merged["recording_id"].isin(adapt_ids)]
        test_rows  = merged[merged["recording_id"].isin(test_ids)]

        adapt_labels = adapt_rows["intent_label"].dropna().tolist()
        test_labels  = test_rows["intent_label"].dropna().tolist()

        # Count per class
        from collections import Counter
        adapt_counts = Counter(adapt_labels)
        test_counts  = Counter(test_labels)

        gen_count = adapt_counts.get("GENERAL_SPEECH", 0)
        emerg_count = sum(adapt_counts.get(c, 0) for c in EMERGENCY_CLASSES)
        total = len(adapt_labels)
        minority_majority_ratio = emerg_count / gen_count if gen_count > 0 else (1.0 if emerg_count > 0 else 0.0)
        emerg_nonemergen_ratio  = emerg_count / total if total > 0 else 0.0

        n_distinct = len([c for c in CLASSES if adapt_counts.get(c, 0) > 0])
        n_emerg_present = len([c for c in EMERGENCY_CLASSES if adapt_counts.get(c, 0) > 0])
        n_zero_support = len([c for c in CLASSES if adapt_counts.get(c, 0) == 0])

        row = {
            "speaker_id": spk,
            "cohort_group": grp,
            "sample_n": n,
            "adapt_total": total,
            "test_total": len(test_labels),
            "n_distinct_classes_adapt": n_distinct,
            "n_emergency_classes_adapt": n_emerg_present,
            "n_zero_support_classes": n_zero_support,
            "minority_majority_ratio": round(minority_majority_ratio, 4),
            "emergency_nonemergency_ratio": round(emerg_nonemergen_ratio, 4),
        }
        for c in CLASSES:
            row[f"adapt_{c}"] = adapt_counts.get(c, 0)
        for c in CLASSES:
            row[f"test_{c}"] = test_counts.get(c, 0)

        rows.append(row)

adapt_df = pd.DataFrame(rows)
out_path = os.path.join(OUT_ANA, "adaptation_class_support.csv")
adapt_df.to_csv(out_path, index=False)
print(f"  Saved {out_path} ({len(adapt_df)} rows)")

# ============================================================
# STEP 4: Emergency support summary
# ============================================================
print("\n[4/8] Computing emergency support summary...")
emerg_rows = []
for spk in SPEAKERS:
    grp = "dysarthric" if spk in DYSARTHRIC else "control"
    for n in NS:
        sub = adapt_df[(adapt_df["speaker_id"]==spk) & (adapt_df["sample_n"]==n)]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        for ec in sorted(EMERGENCY_CLASSES):
            col = f"adapt_{ec}"
            emerg_rows.append({
                "speaker_id": spk,
                "cohort_group": grp,
                "sample_n": n,
                "emergency_class": ec,
                "adapt_support": int(r.get(col, 0)),
                "zero_support": (r.get(col, 0) == 0),
            })

emerg_df = pd.DataFrame(emerg_rows)
out_path = os.path.join(OUT_ANA, "emergency_support.csv")
emerg_df.to_csv(out_path, index=False)
print(f"  Saved {out_path} ({len(emerg_df)} rows)")

# Zero-support statistics
zero_pivot = emerg_df.groupby(["sample_n", "emergency_class"])["zero_support"].mean().unstack()
print("\n  Zero-support rate per class per N (fraction of 15 speakers):")
print(zero_pivot.round(3).to_string())

# ============================================================
# STEP 5: Speaker feasibility matrix
# ============================================================
print("\n[5/8] Building speaker feasibility matrix...")
feasibility_rows = []
for spk in SPEAKERS:
    grp = "dysarthric" if spk in DYSARTHRIC else "control"
    for n in NS:
        sub = adapt_df[(adapt_df["speaker_id"]==spk) & (adapt_df["sample_n"]==n)]
        if len(sub) == 0:
            feasibility_rows.append({"speaker_id": spk, "cohort_group": grp,
                                      "sample_n": n, "feasibility": "MISSING_DATA"})
            continue
        r = sub.iloc[0]
        n_emerg = int(r["n_emergency_classes_adapt"])
        n_zero  = int(r["n_zero_support_classes"])
        emerg_ratio = float(r["emergency_nonemergency_ratio"])

        if n_emerg >= 3 and n_zero <= 3:
            feasibility = "FEASIBLE"
        elif n_emerg >= 1 and n_zero <= 5:
            feasibility = "PARTIALLY_FEASIBLE"
        else:
            feasibility = "NOT_FEASIBLE"

        feasibility_rows.append({
            "speaker_id": spk,
            "cohort_group": grp,
            "sample_n": n,
            "n_distinct_classes": int(r["n_distinct_classes_adapt"]),
            "n_emergency_classes": n_emerg,
            "n_zero_support_classes": n_zero,
            "emergency_ratio": round(emerg_ratio, 4),
            "feasibility": feasibility,
        })

feas_df = pd.DataFrame(feasibility_rows)
out_path = os.path.join(OUT_ANA, "speaker_feasibility_matrix.csv")
feas_df.to_csv(out_path, index=False)
print(f"  Saved {out_path} ({len(feas_df)} rows)")

# Feasibility summary
summary = feas_df.groupby(["sample_n","feasibility"]).size().unstack(fill_value=0)
print("\n  Feasibility summary (counts per N):")
print(summary.to_string())

# ============================================================
# STEP 6: Prediction distribution analysis (model collapse audit)
# ============================================================
print("\n[6/8] Analyzing prediction distributions (kernel collapse audit)...")
p5_preds = pd.read_csv(P5_PREDS)
print(f"  Loaded {len(p5_preds)} prediction rows.")

# Predicted class distribution per model (CLEAN, all N pooled)
pred_dist_rows = []
for model in ["adapter","rbf_svm","qsvc"]:
    for snr in ["clean","snr15","snr10","snr5"]:
        sub = p5_preds[(p5_preds["model"]==model) & (p5_preds["snr_tag"]==snr)]
        total = len(sub)
        if total == 0:
            continue
        for c in CLASSES:
            cnt = (sub["predicted_intent"] == c).sum()
            row = {
                "model": model,
                "snr_tag": snr,
                "predicted_class": c,
                "count": int(cnt),
                "fraction": round(cnt / total, 5),
                "total_predictions": total,
            }
            pred_dist_rows.append(row)

pred_dist_df = pd.DataFrame(pred_dist_rows)
out_path = os.path.join(OUT_ANA, "prediction_distribution_by_model.csv")
pred_dist_df.to_csv(out_path, index=False)
print(f"  Saved {out_path} ({len(pred_dist_df)} rows)")

# Print collapse diagnosis
print("\n  Predicted class fractions (Clean, all speakers/N pooled):")
clean_dist = pred_dist_df[pred_dist_df["snr_tag"]=="clean"][
    ["model","predicted_class","fraction"]].pivot(index="predicted_class", columns="model", values="fraction")
print(clean_dist.round(5).to_string())

# Emergency recall
print("\n  Emergency class true positive rate (recall) per model [CLEAN]:")
for model in ["adapter","rbf_svm","qsvc"]:
    sub = p5_preds[(p5_preds["model"]==model) & (p5_preds["snr_tag"]=="clean")]
    for ec in sorted(EMERGENCY_CLASSES):
        ec_sub = sub[sub["true_intent"]==ec]
        if len(ec_sub) == 0:
            continue
        recall = (ec_sub["predicted_intent"]==ec).sum() / len(ec_sub)
        print(f"    {model:12s}  {ec:25s}  recall={recall:.3f}  support={len(ec_sub)}")

# ============================================================
# STEP 7: PCA explained variance analysis
# ============================================================
print("\n[7/8] Analyzing PCA explained variance from Phase 5 results...")
p5_res = pd.read_csv(P5_RESULTS)

# Extract PCA info from QSVC rows
qsvc_rows = p5_res[p5_res["model"]=="qsvc"]
print(f"  QSVC rows: {len(qsvc_rows)}")
print(f"  Qubit counts: {qsvc_rows['qubits'].unique()}")
print(f"  Reps: {qsvc_rows['reps'].unique()}")
print(f"  Entanglement: {qsvc_rows['entanglement'].unique()}")
print(f"  Feature map: {qsvc_rows['feature_map'].unique()}")
print(f"  Kernel: {qsvc_rows['kernel'].unique()}")

# Load cached PCA explained variance ratios from kernel cache
pca_cache_dir = "results/phase4/kernel_cache"
pca_rows = []
if os.path.isdir(pca_cache_dir):
    for fname in os.listdir(pca_cache_dir):
        if fname.endswith(".npz"):
            fpath = os.path.join(pca_cache_dir, fname)
            try:
                npz = np.load(fpath, allow_pickle=True)
                meta = npz.get("meta", None)
                if meta is not None:
                    meta_dict = meta.item() if hasattr(meta, "item") else {}
                    pca_ev = meta_dict.get("pca_explained_variance_ratio", [])
                    cumev  = meta_dict.get("pca_cumulative_explained_variance", None)
                    pca_rows.append({
                        "file": fname,
                        "n_components": len(pca_ev),
                        "pca_ev_ratio": str([round(x,4) for x in pca_ev]),
                        "cumulative_ev": round(cumev, 4) if cumev is not None else None,
                    })
            except Exception as e:
                pass

if pca_rows:
    pca_ev_df = pd.DataFrame(pca_rows)
    out_path = os.path.join(OUT_ANA, "pca_explained_variance.csv")
    pca_ev_df.to_csv(out_path, index=False)
    print(f"  Saved PCA EV data: {out_path} ({len(pca_ev_df)} rows)")
    if len(pca_ev_df) > 0:
        print(f"  Example PCA EV ratio: {pca_ev_df.iloc[0]['pca_ev_ratio']}")
        print(f"  Example cumulative EV: {pca_ev_df.iloc[0]['cumulative_ev']}")
else:
    print("  WARNING: No PCA cache found at results/phase4/kernel_cache")
    # Estimate from embeddings if available
    embed_files = [f for f in os.listdir(EMBED_DIR) if f.endswith(".npy")] if os.path.isdir(EMBED_DIR) else []
    print(f"  Embedding files available: {len(embed_files)}")
    if len(embed_files) > 50:
        from sklearn.decomposition import PCA as SkPCA
        # Load a sample of embeddings for representative PCA analysis
        sample_embeds = []
        for fname in embed_files[:500]:  # Use 500 embeddings for diagnosis
            try:
                e = np.load(os.path.join(EMBED_DIR, fname))
                sample_embeds.append(e.flatten())
            except:
                pass
        if len(sample_embeds) > 10:
            X_sample = np.stack(sample_embeds)
            print(f"  Running PCA diagnostic on {len(X_sample)} sample embeddings...")
            pca_rows_out = []
            for n_comp in [2, 4, 6, 8, 16, 32, 64, 128, 256]:
                if n_comp >= min(X_sample.shape):
                    continue
                pca_ = SkPCA(n_components=n_comp, random_state=42)
                pca_.fit(X_sample)
                cum_ev = float(np.sum(pca_.explained_variance_ratio_))
                pca_rows_out.append({
                    "n_components": n_comp,
                    "cumulative_explained_variance": round(cum_ev, 4),
                    "mean_ev_per_component": round(cum_ev / n_comp, 4),
                })
                print(f"    PCA n={n_comp:3d}: cumulative EV = {cum_ev:.4f}")
            pca_ev_df = pd.DataFrame(pca_rows_out)
            out_path = os.path.join(OUT_ANA, "pca_explained_variance.csv")
            pca_ev_df.to_csv(out_path, index=False)
            print(f"  Saved {out_path}")

# ============================================================
# STEP 8: Quantum configuration audit
# ============================================================
print("\n[8/8] Quantum configuration audit from Phase 5 results...")
qsvc_sample = p5_res[p5_res["model"]=="qsvc"].iloc[0] if len(qsvc_rows) > 0 else None

quantum_config = {
    "qubits": int(qsvc_sample["qubits"]) if qsvc_sample is not None else "N/A",
    "reps": int(qsvc_sample["reps"]) if qsvc_sample is not None else "N/A",
    "entanglement": str(qsvc_sample["entanglement"]) if qsvc_sample is not None else "N/A",
    "feature_map": str(qsvc_sample["feature_map"]) if qsvc_sample is not None else "N/A",
    "kernel": str(qsvc_sample["kernel"]) if qsvc_sample is not None else "N/A",
    "pca_dim": int(qsvc_sample["qubits"]) if qsvc_sample is not None else "N/A",  # pca_n_components = qubit_count
    "C": 1.0,
    "simulator": "Qiskit Aer Statevector",
    "shots": "Exact (no sampling noise)",
    "source": "src/models/quantum_kernel.py + src/quantum/fidelity_kernel.py",
}
print("  Quantum config extracted:")
for k, v in quantum_config.items():
    print(f"    {k}: {v}")

qcfg_df = pd.DataFrame([quantum_config])
out_path = os.path.join(OUT_ANA, "quantum_config_audit.csv")
qcfg_df.to_csv(out_path, index=False)
print(f"  Saved {out_path}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

# 1. Class balance in adaptation sets
gen_mean_20 = adapt_df[adapt_df["sample_n"]==20]["adapt_GENERAL_SPEECH"].mean()
gen_mean_50 = adapt_df[adapt_df["sample_n"]==50]["adapt_GENERAL_SPEECH"].mean()
tot_mean_20 = adapt_df[adapt_df["sample_n"]==20]["adapt_total"].mean()
tot_mean_50 = adapt_df[adapt_df["sample_n"]==50]["adapt_total"].mean()
print(f"\n1. Class Balance (adaptation sets, averaged across 15 speakers):")
print(f"   N=20: GENERAL_SPEECH mean = {gen_mean_20:.1f} / {tot_mean_20:.1f} = "
      f"{gen_mean_20/tot_mean_20*100:.1f}%")
print(f"   N=50: GENERAL_SPEECH mean = {gen_mean_50:.1f} / {tot_mean_50:.1f} = "
      f"{gen_mean_50/tot_mean_50*100:.1f}%")

# 2. Zero-support emergency classes at N=20
zero_20 = emerg_df[emerg_df["sample_n"]==20]["zero_support"].mean()
zero_50 = emerg_df[emerg_df["sample_n"]==50]["zero_support"].mean()
print(f"\n2. Zero-support emergency class rate:")
print(f"   N=20: {zero_20*100:.1f}% of (speaker, class) pairs have 0 adaptation samples")
print(f"   N=50: {zero_50*100:.1f}% of (speaker, class) pairs have 0 adaptation samples")

# 3. Feasibility breakdown
for n in NS:
    feas_n = feas_df[feas_df["sample_n"]==n]["feasibility"].value_counts()
    print(f"\n3. Feasibility at N={n}: {dict(feas_n)}")

# 4. Collapse confirmation
for model in ["adapter","rbf_svm","qsvc"]:
    sub = pred_dist_df[(pred_dist_df["model"]==model) & (pred_dist_df["snr_tag"]=="clean")]
    gs_frac = sub[sub["predicted_class"]=="GENERAL_SPEECH"]["fraction"].values
    gs_frac = float(gs_frac[0]) if len(gs_frac) > 0 else 0.0
    print(f"\n4. {model:12s}: GENERAL_SPEECH predicted fraction (Clean) = {gs_frac*100:.2f}%")

print("\n" + "="*70)
print("SUBPHASE 6.5.0 DIAGNOSTIC COMPLETE")
print("="*70)
