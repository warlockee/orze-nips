"""Analyze test predictions from top nexar_collision experiments."""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, confusion_matrix

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
SOLUTION_CSV = Path(os.environ.get("NEXAR_TEST_CSV", "data/solution.csv"))
TEST_FEATURE_DIR = Path(os.environ.get("TEST_FEATURE_DIR", "features/nexar_test/dinov3_vitb16"))

# ── 0. Build video_id order from sorted .pt files (same as train.py load_split) ──
print("=" * 80)
print("BUILDING VIDEO ID MAPPING")
print("=" * 80)
import torch
pt_files = sorted(TEST_FEATURE_DIR.glob("*.pt"))
video_ids_ordered = []
for f in pt_files:
    data = torch.load(f, map_location="cpu", weights_only=False)
    video_ids_ordered.append(int(data.get("video_id", f.stem)))
print(f"Test set: {len(video_ids_ordered)} videos")

# Load solution
sol = pd.read_csv(SOLUTION_CSV)
# Create a mapping from position in npz (sorted .pt order) to solution row
sol_indexed = sol.set_index("id")
# Build aligned arrays: for each npz position, get group/Usage/target from solution
aligned_groups = np.array([sol_indexed.loc[vid, "group"] for vid in video_ids_ordered])
aligned_usage = np.array([sol_indexed.loc[vid, "Usage"] for vid in video_ids_ordered])
aligned_targets = np.array([sol_indexed.loc[vid, "target"] for vid in video_ids_ordered], dtype=float)

# ── 1. Read leaderboard ──
print("\n" + "=" * 80)
print("TOP 10 EXPERIMENTS BY test_map")
print("=" * 80)
with open(RESULTS_DIR / "_leaderboard.json") as f:
    lb = json.load(f)

top_entries = lb["top"][:10]
print(f"{'Rank':<5} {'Idea ID':<16} {'test_mAP':>10} {'val_mAP':>10} {'params':>10} {'time(m)':>8}  Title")
print("-" * 120)
for i, e in enumerate(top_entries):
    em = e["eval_metrics"]
    print(f"{i+1:<5} {e['idea_id']:<16} {em['test_map']:>10.4f} {em['val_map']:>10.4f} {em.get('param_count','?'):>10} {em.get('training_time',0):>8.1f}  {e['title'][:60]}")

# ── 2. Top 5 experiments: load npz, print stats ──
top5_ids = [e["idea_id"] for e in top_entries[:5]]
print("\n" + "=" * 80)
print("TOP 5 EXPERIMENTS: NPZ FILE ANALYSIS")
print("=" * 80)

top5_probs = {}
for idea_id in top5_ids:
    npz_path = RESULTS_DIR / idea_id / "test_predictions.npz"
    if not npz_path.exists():
        print(f"\n--- {idea_id}: NO test_predictions.npz ---")
        continue
    d = np.load(npz_path)
    probs = d["probs"].astype(np.float64)
    labels = d["labels"].astype(np.float64)
    top5_probs[idea_id] = probs

    print(f"\n--- {idea_id} ---")
    print(f"  Keys: {list(d.keys())}")
    print(f"  probs shape={probs.shape} dtype={d['probs'].dtype}  |  labels shape={labels.shape} dtype={d['labels'].dtype}")
    print(f"  probs: min={probs.min():.4f} max={probs.max():.4f} mean={probs.mean():.4f} std={probs.std():.4f}")
    print(f"  probs median={np.median(probs):.4f}  p5={np.percentile(probs,5):.4f}  p95={np.percentile(probs,95):.4f}")
    # Verify labels match
    assert np.allclose(labels, aligned_targets), f"Labels mismatch for {idea_id}!"
    ap = average_precision_score(labels, probs)
    print(f"  Verified mAP = {ap:.6f}")

# ── 3. Best model deep analysis ──
best_id = top5_ids[0]
print("\n" + "=" * 80)
print(f"DETAILED ANALYSIS: {best_id}")
print("=" * 80)

probs = top5_probs[best_id]
labels = aligned_targets

# 3a. Per-group mAP
print("\n--- Per-Group mAP ---")
for g in sorted(np.unique(aligned_groups)):
    mask = aligned_groups == g
    ap = average_precision_score(labels[mask], probs[mask])
    n_pos = int(labels[mask].sum())
    n_neg = int((~labels[mask].astype(bool)).sum())
    print(f"  Group {g}: mAP={ap:.4f}  (n={mask.sum()}, pos={n_pos}, neg={n_neg})")

# 3b. Per-Usage split mAP
print("\n--- Per-Usage (Public/Private) mAP ---")
for usage in ["Public", "Private"]:
    mask = aligned_usage == usage
    ap = average_precision_score(labels[mask], probs[mask])
    n_pos = int(labels[mask].sum())
    n_neg = int((~labels[mask].astype(bool)).sum())
    print(f"  {usage}: mAP={ap:.4f}  (n={mask.sum()}, pos={n_pos}, neg={n_neg})")

# 3c. Most confident wrong predictions
print("\n--- 20 Most Confident Wrong Predictions ---")
preds_binary = (probs >= 0.5).astype(float)
wrong_mask = preds_binary != labels

# Confidence = distance from correct answer
# For FP (pred=1, label=0): confidence = probs[i]
# For FN (pred=0, label=1): confidence = 1 - probs[i]
confidence_of_error = np.where(labels == 0, probs, 1.0 - probs)
confidence_of_error[~wrong_mask] = -1  # ignore correct predictions

top_wrong_idx = np.argsort(-confidence_of_error)[:20]
print(f"  {'Rank':<5} {'VideoID':>8} {'Label':>6} {'Pred':>6} {'Prob':>8} {'Group':>6} {'Usage':>8} {'Type':<5}")
print("  " + "-" * 70)
for rank, idx in enumerate(top_wrong_idx):
    vid = video_ids_ordered[idx]
    lbl = int(labels[idx])
    pred = int(preds_binary[idx])
    prob = probs[idx]
    grp = aligned_groups[idx]
    usg = aligned_usage[idx]
    err_type = "FP" if pred == 1 and lbl == 0 else "FN"
    print(f"  {rank+1:<5} {vid:>8} {lbl:>6} {pred:>6} {prob:>8.4f} {grp:>6} {usg:>8} {err_type:<5}")

# 3d. Calibration
print("\n--- Calibration ---")
pos_mask = labels == 1
neg_mask = labels == 0
print(f"  Mean predicted prob for POSITIVE samples: {probs[pos_mask].mean():.4f} (should be ~1.0)")
print(f"  Mean predicted prob for NEGATIVE samples: {probs[neg_mask].mean():.4f} (should be ~0.0)")
print(f"  Median predicted prob for POSITIVE: {np.median(probs[pos_mask]):.4f}")
print(f"  Median predicted prob for NEGATIVE: {np.median(probs[neg_mask]):.4f}")

# Calibration by decile
print("\n  Calibration by predicted probability bin:")
bins = np.linspace(0, 1, 11)
print(f"  {'Bin':<15} {'Count':>6} {'Actual pos rate':>16} {'Mean pred prob':>16}")
for i in range(len(bins) - 1):
    bin_mask = (probs >= bins[i]) & (probs < bins[i + 1])
    if i == len(bins) - 2:  # last bin includes upper bound
        bin_mask = (probs >= bins[i]) & (probs <= bins[i + 1])
    if bin_mask.sum() > 0:
        actual_rate = labels[bin_mask].mean()
        mean_pred = probs[bin_mask].mean()
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}){'':<3} {bin_mask.sum():>6} {actual_rate:>16.4f} {mean_pred:>16.4f}")

# 3e. Confusion matrix at threshold=0.5
print("\n--- Confusion Matrix (threshold=0.5) ---")
cm = confusion_matrix(labels, preds_binary)
tn, fp, fn, tp = cm.ravel()
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"  Accuracy: {(tp+tn)/(tp+tn+fp+fn):.4f}")
print(f"  Precision: {tp/(tp+fp):.4f}" if (tp+fp) > 0 else "  Precision: N/A")
print(f"  Recall: {tp/(tp+fn):.4f}" if (tp+fn) > 0 else "  Recall: N/A")
print(f"  F1: {2*tp/(2*tp+fp+fn):.4f}" if (2*tp+fp+fn) > 0 else "  F1: N/A")

# ── 4. Ensemble analysis ──
ensemble_path = RESULTS_DIR / "ensemble_top5.npz"
if ensemble_path.exists():
    print("\n" + "=" * 80)
    print("ENSEMBLE (ensemble_top5.npz) ANALYSIS")
    print("=" * 80)
    ed = np.load(ensemble_path)
    eprobs = ed["probs"].astype(np.float64)
    elabels = ed["labels"].astype(np.float64)
    assert np.allclose(elabels, aligned_targets), "Ensemble labels mismatch!"

    eap = average_precision_score(elabels, eprobs)
    print(f"  Overall mAP: {eap:.6f}")
    print(f"  probs: min={eprobs.min():.4f} max={eprobs.max():.4f} mean={eprobs.mean():.4f} std={eprobs.std():.4f}")

    print("\n  Per-Group mAP:")
    for g in sorted(np.unique(aligned_groups)):
        mask = aligned_groups == g
        ap = average_precision_score(elabels[mask], eprobs[mask])
        n_pos = int(elabels[mask].sum())
        print(f"    Group {g}: mAP={ap:.4f}  (n={mask.sum()}, pos={n_pos})")

    print("\n  Per-Usage mAP:")
    for usage in ["Public", "Private"]:
        mask = aligned_usage == usage
        ap = average_precision_score(elabels[mask], eprobs[mask])
        print(f"    {usage}: mAP={ap:.4f}  (n={mask.sum()})")

    print("\n  Confusion Matrix (threshold=0.5):")
    epreds = (eprobs >= 0.5).astype(float)
    cm = confusion_matrix(elabels, epreds)
    tn, fp, fn, tp = cm.ravel()
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(f"    Accuracy: {(tp+tn)/(tp+tn+fp+fn):.4f}")
    print(f"    Precision: {tp/(tp+fp):.4f}  Recall: {tp/(tp+fn):.4f}")

    # Calibration
    pos_mask = elabels == 1
    neg_mask = elabels == 0
    print(f"\n  Calibration:")
    print(f"    Mean pred for POSITIVE: {eprobs[pos_mask].mean():.4f}")
    print(f"    Mean pred for NEGATIVE: {eprobs[neg_mask].mean():.4f}")

    # Compare ensemble vs best single
    print(f"\n  Ensemble vs Best Single ({best_id}):")
    print(f"    Ensemble mAP: {eap:.6f}")
    best_ap = average_precision_score(labels, top5_probs[best_id])
    print(f"    Best single mAP: {best_ap:.6f}")
    print(f"    Delta: {eap - best_ap:+.6f}")

# ── 5. Hard samples analysis across ALL experiments with test_predictions.npz ──
print("\n" + "=" * 80)
print("HARD SAMPLES ANALYSIS (across all experiments with test_predictions.npz)")
print("=" * 80)

# First, load leaderboard to get ordered list of all ideas
all_ideas_by_map = [(e["idea_id"], e["eval_metrics"]["test_map"]) for e in lb["top"]]
# Also find all npz files
all_npz = sorted(RESULTS_DIR.glob("*/test_predictions.npz"))
npz_ideas = {p.parent.name for p in all_npz}
print(f"Total experiments with test_predictions.npz: {len(npz_ideas)}")

# Use top 20 by test_map that have npz
top20_ideas = []
for idea_id, test_map in all_ideas_by_map:
    if idea_id in npz_ideas:
        top20_ideas.append(idea_id)
    if len(top20_ideas) == 20:
        break
print(f"Using top {len(top20_ideas)} experiments for hard sample analysis")

# Load all their probs
n_samples = len(aligned_targets)
wrong_count = np.zeros(n_samples, dtype=int)
model_count = 0

for idea_id in top20_ideas:
    npz_path = RESULTS_DIR / idea_id / "test_predictions.npz"
    d = np.load(npz_path)
    p = d["probs"].astype(np.float64)
    l = d["labels"].astype(np.float64)
    if len(p) != n_samples:
        print(f"  SKIP {idea_id}: shape mismatch {len(p)} vs {n_samples}")
        continue
    if not np.allclose(l, aligned_targets):
        print(f"  SKIP {idea_id}: label mismatch")
        continue
    pred_binary = (p >= 0.5).astype(float)
    wrong = pred_binary != aligned_targets
    wrong_count += wrong.astype(int)
    model_count += 1

print(f"Successfully loaded {model_count} models")

# Find hardest samples
hardest_idx = np.argsort(-wrong_count)[:30]
print(f"\n--- 30 Hardest Samples (most frequently misclassified by top {model_count} models) ---")
print(f"{'Rank':<5} {'VideoID':>8} {'Label':>6} {'Wrong/Total':>12} {'WrongRate':>10} {'Group':>6} {'Usage':>8}")
print("-" * 70)
for rank, idx in enumerate(hardest_idx):
    vid = video_ids_ordered[idx]
    lbl = int(aligned_targets[idx])
    print(f"{rank+1:<5} {vid:>8} {lbl:>6} {wrong_count[idx]:>5}/{model_count:<5} {wrong_count[idx]/model_count:>10.1%} {aligned_groups[idx]:>6} {aligned_usage[idx]:>8}")

# Summary stats on hard samples
print(f"\n--- Hard Sample Summary ---")
for threshold in [model_count, int(model_count * 0.75), int(model_count * 0.5), int(model_count * 0.25)]:
    n_hard = (wrong_count >= threshold).sum()
    print(f"  Misclassified by >= {threshold}/{model_count} models: {n_hard} samples")

# Break down by label
print(f"\n  Among 30 hardest samples:")
hard30_labels = aligned_targets[hardest_idx]
print(f"    Positives (FN-prone): {int(hard30_labels.sum())}")
print(f"    Negatives (FP-prone): {int((1 - hard30_labels).sum())}")

# Hard samples per group
print(f"\n  Hard samples (wrong by >50% of models) by group:")
hard_mask = wrong_count > model_count / 2
for g in sorted(np.unique(aligned_groups)):
    gmask = aligned_groups == g
    n_hard_g = (hard_mask & gmask).sum()
    n_total_g = gmask.sum()
    print(f"    Group {g}: {n_hard_g}/{n_total_g} ({n_hard_g/n_total_g:.1%})")

# Hard samples per Usage
print(f"\n  Hard samples (wrong by >50% of models) by Usage:")
for usage in ["Public", "Private"]:
    umask = aligned_usage == usage
    n_hard_u = (hard_mask & umask).sum()
    n_total_u = umask.sum()
    print(f"    {usage}: {n_hard_u}/{n_total_u} ({n_hard_u/n_total_u:.1%})")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
