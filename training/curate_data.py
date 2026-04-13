#!/usr/bin/env python3
"""Embedding-space data curation for Nexar Collision Detection.

Uses frozen DINOv3 ViT-B/16 features (mean-pooled per video) to identify
noisy/ambiguous training samples via 5-fold cross-validation with logistic
regression. Outputs cleaned_train.csv with per-video quality scores and
flags for removal/downweighting.

This implements the 1st-place insight: "only 2-5% of training data contained
the impactful signals." By finding the model-confusing samples, we can
remove or downweight them before e2e training.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler


FEATURE_DIR = "features/nexar/dinov3_vitb16"
TRAIN_CSV = os.environ.get("NEXAR_TRAIN_CSV", "data/train.csv")
OUTPUT_DIR = "results/_curation"


def load_train_labels(csv_path):
    """Load competition train.csv -> {video_id: target}."""
    labels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["id"]] = int(row["target"])
    return labels


def load_video_embedding(feature_dir, video_id):
    """Load frozen features and mean-pool to single vector."""
    pt_path = os.path.join(feature_dir, f"{video_id}.pt")
    if not os.path.exists(pt_path):
        return None
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    feats = data["features"]  # (T, 768)
    return feats.mean(dim=0).numpy()  # (768,)


def load_video_embedding_temporal(feature_dir, video_id, n_segments=4):
    """Load frozen features with temporal segmentation.

    Splits video into n_segments equal parts, mean-pools each,
    concatenates → (n_segments * 768,) vector.
    Captures temporal structure (e.g., last segment has crash signal).
    """
    pt_path = os.path.join(feature_dir, f"{video_id}.pt")
    if not os.path.exists(pt_path):
        return None
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    feats = data["features"]  # (T, 768)
    T = feats.shape[0]

    if T < n_segments:
        # Pad with repetitions
        feats = feats.repeat((n_segments // T) + 1, 1)[:n_segments]
        T = n_segments

    seg_size = T // n_segments
    segments = []
    for i in range(n_segments):
        start = i * seg_size
        end = start + seg_size if i < n_segments - 1 else T
        segments.append(feats[start:end].mean(dim=0))

    return torch.cat(segments).numpy()  # (n_segments * 768,)


def run_curation(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load labels
    labels = load_train_labels(args.train_csv)
    print(f"Train videos in CSV: {len(labels)}")
    print(f"  Positive: {sum(v for v in labels.values())}")
    print(f"  Negative: {sum(1 - v for v in labels.values())}")

    # Load embeddings
    print(f"\nLoading {args.pool} embeddings from {args.feature_dir}...")
    video_ids = []
    embeddings = []
    targets = []

    for vid, target in sorted(labels.items()):
        if args.pool == "temporal":
            emb = load_video_embedding_temporal(args.feature_dir, vid, args.n_segments)
        else:
            emb = load_video_embedding(args.feature_dir, vid)

        if emb is not None:
            video_ids.append(vid)
            embeddings.append(emb)
            targets.append(target)

    X = np.array(embeddings)
    y = np.array(targets)
    print(f"Loaded embeddings: {X.shape[0]} videos, {X.shape[1]}-d features")
    print(f"  Matched: {X.shape[0]}/{len(labels)} ({100*X.shape[0]/len(labels):.1f}%)")

    # 5-fold CV
    print(f"\nRunning {args.n_folds}-fold CV with LogisticRegression (C={args.C})...")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(y))
    fold_aps = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Train
        clf = LogisticRegression(
            C=args.C,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)

        # Predict
        probs = clf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs

        ap = average_precision_score(y_val, probs)
        fold_aps.append(ap)
        print(f"  Fold {fold+1}: AP={ap:.4f} (val_pos={y_val.sum()}, val_neg={len(y_val)-y_val.sum()})")

    mean_ap = np.mean(fold_aps)
    print(f"\nOOF AP: {mean_ap:.4f} (std={np.std(fold_aps):.4f})")

    # Identify outliers
    fp_mask = (y == 0) & (oof_probs > args.fp_threshold)
    fn_mask = (y == 1) & (oof_probs < args.fn_threshold)
    clean_mask = ~fp_mask & ~fn_mask

    fp_ids = [video_ids[i] for i in np.where(fp_mask)[0]]
    fn_ids = [video_ids[i] for i in np.where(fn_mask)[0]]

    print(f"\n{'='*60}")
    print(f"CURATION RESULTS (fp_thresh={args.fp_threshold}, fn_thresh={args.fn_threshold})")
    print(f"{'='*60}")
    print(f"False positives (neg videos, prob>{args.fp_threshold}): {len(fp_ids)}")
    print(f"False negatives (pos videos, prob<{args.fn_threshold}): {len(fn_ids)}")
    print(f"Clean videos: {clean_mask.sum()}")
    print(f"Removed: {len(fp_ids) + len(fn_ids)} ({100*(len(fp_ids)+len(fn_ids))/len(y):.1f}%)")

    # High-confidence samples (the "impactful signals")
    hc_pos = (y == 1) & (oof_probs > 0.9)
    hc_neg = (y == 0) & (oof_probs < 0.1)
    print(f"\nHigh-confidence positive (prob>0.9): {hc_pos.sum()}")
    print(f"High-confidence negative (prob<0.1): {hc_neg.sum()}")
    print(f"Total high-confidence: {hc_pos.sum() + hc_neg.sum()} ({100*(hc_pos.sum()+hc_neg.sum())/len(y):.1f}%)")

    # Detailed FP analysis
    if len(fp_ids) > 0:
        print(f"\nFalse Positive videos (top 20 by prob):")
        fp_indices = np.where(fp_mask)[0]
        fp_sorted = fp_indices[np.argsort(-oof_probs[fp_indices])]
        for idx in fp_sorted[:20]:
            print(f"  {video_ids[idx]}: prob={oof_probs[idx]:.4f} (label=0)")

    if len(fn_ids) > 0:
        print(f"\nFalse Negative videos (top 20 by lowest prob):")
        fn_indices = np.where(fn_mask)[0]
        fn_sorted = fn_indices[np.argsort(oof_probs[fn_indices])]
        for idx in fn_sorted[:20]:
            print(f"  {video_ids[idx]}: prob={oof_probs[idx]:.4f} (label=1)")

    # Write cleaned_train.csv
    output_csv = os.path.join(args.output_dir, "cleaned_train.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "target", "oof_prob", "status", "weight"])
        for i in range(len(video_ids)):
            if fp_mask[i]:
                status = "FP_REMOVE"
                weight = 0.0
            elif fn_mask[i]:
                status = "FN_REMOVE"
                weight = 0.0
            elif hc_pos[i] or hc_neg[i]:
                status = "HIGH_CONF"
                weight = 2.0  # upweight high-confidence samples
            else:
                status = "KEEP"
                weight = 1.0
            writer.writerow([video_ids[i], targets[i], f"{oof_probs[i]:.6f}", status, weight])

    print(f"\nSaved: {output_csv}")

    # Write detailed report
    report = {
        "oof_ap": float(mean_ap),
        "fold_aps": [float(x) for x in fold_aps],
        "n_videos": len(video_ids),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "fp_threshold": args.fp_threshold,
        "fn_threshold": args.fn_threshold,
        "n_fp_removed": len(fp_ids),
        "n_fn_removed": len(fn_ids),
        "n_clean": int(clean_mask.sum()),
        "n_high_confidence": int(hc_pos.sum() + hc_neg.sum()),
        "pct_removed": float(100 * (len(fp_ids) + len(fn_ids)) / len(y)),
        "fp_video_ids": fp_ids,
        "fn_video_ids": fn_ids,
        "pool": args.pool,
        "C": args.C,
    }
    report_path = os.path.join(args.output_dir, "curation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")

    # Write video exclusion list for train_e2e.py
    exclude_path = os.path.join(args.output_dir, "exclude_videos.txt")
    with open(exclude_path, "w") as f:
        for vid in sorted(fp_ids + fn_ids):
            f.write(f"{vid}\n")
    print(f"Saved: {exclude_path} ({len(fp_ids)+len(fn_ids)} videos)")

    return report


def main():
    parser = argparse.ArgumentParser(description="Embedding-space data curation")
    parser.add_argument("--feature_dir", default=FEATURE_DIR)
    parser.add_argument("--train_csv", default=TRAIN_CSV)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--pool", choices=["mean", "temporal"], default="mean",
                        help="Pooling strategy: mean (768-d) or temporal (n_segments*768-d)")
    parser.add_argument("--n_segments", type=int, default=4,
                        help="Number of temporal segments for temporal pooling")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--C", type=float, default=1.0,
                        help="Regularization for LogisticRegression")
    parser.add_argument("--fp_threshold", type=float, default=0.7,
                        help="Remove negative videos with prob above this")
    parser.add_argument("--fn_threshold", type=float, default=0.3,
                        help="Remove positive videos with prob below this")
    args = parser.parse_args()

    run_curation(args)


if __name__ == "__main__":
    main()
