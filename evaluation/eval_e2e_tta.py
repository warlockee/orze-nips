#!/usr/bin/env python3
"""Enhanced evaluation with TTA (horizontal flip) and softer aggregation.

Usage:
    python3 eval_e2e_tta.py results/e2e_mvit_v2_s_run28_resumed/best_model.pt --stride 30
    python3 eval_e2e_tta.py results/e2e_mvit_v2_s_run*/best_model.pt --stride 30 --tta
    python3 eval_e2e_tta.py results/e2e_mvit_v2_s_run*/best_model.pt --agg top3
"""
import argparse
import sys
import os
import glob
import time
import copy
import collections

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from train_e2e import (
    build_model, load_test_annotations, VideoEvalDataset,
    TEST_CSV, TEST_VIDEO_DIR, set_seed, SEED,
)


def aggregate_preds(preds_list, method="max"):
    if method == "max":
        return max(preds_list)
    elif method == "top3":
        s = sorted(preds_list, reverse=True)
        return np.mean(s[:3])
    elif method == "top5":
        s = sorted(preds_list, reverse=True)
        return np.mean(s[:5])
    elif method == "p95":
        return float(np.percentile(preds_list, 95))
    elif method == "p90":
        return float(np.percentile(preds_list, 90))
    elif method == "mean":
        return np.mean(preds_list)
    else:
        raise ValueError(f"Unknown aggregation: {method}")


@torch.no_grad()
def evaluate_with_options(model, video_ids, labels_dict, video_dir, device,
                          batch_size=8, sample_stride=30, tta_flip=False,
                          agg_method="max"):
    model.eval()

    eval_dataset = VideoEvalDataset(
        video_ids=video_ids,
        video_dir=video_dir,
        sample_stride=sample_stride,
        clip_frames=16,
        clip_stride=4,
    )

    if len(eval_dataset) == 0:
        return 0.0

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    video_preds = collections.defaultdict(list)
    n_total = len(loader)
    t_eval = time.time()

    for batch_idx, (clips, vids) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(clips).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()

        if tta_flip:
            clips_flip = clips.flip(-1)  # Horizontal flip (W dimension)
            with torch.amp.autocast("cuda"):
                logits_flip = model(clips_flip).squeeze(-1)
            probs_flip = torch.sigmoid(logits_flip).cpu().numpy()
            probs = (probs + probs_flip) / 2.0

        for prob, vid in zip(probs, vids):
            video_preds[vid].append(float(prob))

        if (batch_idx + 1) % 100 == 0:
            print(f"    eval batch {batch_idx+1}/{n_total} "
                  f"({time.time()-t_eval:.0f}s)", flush=True)

    y_true = []
    y_score = []
    for vid in video_preds:
        if vid in labels_dict:
            y_true.append(labels_dict[vid])
            y_score.append(aggregate_preds(video_preds[vid], agg_method))

    if len(y_true) == 0 or sum(y_true) == 0:
        return 0.0

    return average_precision_score(y_true, y_score)


def model_soup(ckpt_paths, device):
    """Average model weights from multiple checkpoints (uniform soup)."""
    state_dicts = []
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        state_dicts.append(ckpt["model_state_dict"])

    avg_state = {}
    for key in state_dicts[0]:
        tensors = [sd[key].float() for sd in state_dicts]
        avg_state[key] = torch.mean(torch.stack(tensors), dim=0)

    return avg_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint .pt files")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tta", action="store_true", help="Enable horizontal flip TTA")
    parser.add_argument("--agg", default="max", choices=["max", "top3", "top5", "p95", "p90", "mean"])
    parser.add_argument("--soup", action="store_true", help="Average weights of all checkpoints (model soup)")
    args = parser.parse_args()

    ckpt_paths = []
    for pattern in args.checkpoints:
        expanded = sorted(glob.glob(pattern))
        ckpt_paths.extend(expanded if expanded else [pattern])

    if not ckpt_paths:
        print("No checkpoints found")
        sys.exit(1)

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    test_annots = load_test_annotations(TEST_CSV)
    test_ids = [r["id"] for r in test_annots]
    labels_dict = {r["id"]: r["target"] for r in test_annots}

    model = build_model(pretrained=False)
    model = model.to(device)

    agg_methods = [args.agg] if args.agg != "all" else ["max", "top3", "top5", "p95", "p90", "mean"]

    print(f"Eval stride: {args.stride} | TTA: {args.tta} | Agg: {args.agg} | GPU: {args.gpu}")
    print(f"{'Checkpoint':<60} {'Agg':<6} {'mAP':>8} {'Time':>8}")
    print("-" * 90)

    if args.soup and len(ckpt_paths) > 1:
        # Model soup mode: average weights, eval once
        print(f"Model soup: averaging {len(ckpt_paths)} checkpoints...")
        avg_state = model_soup(ckpt_paths, device)
        model.load_state_dict(avg_state)

        for agg in agg_methods:
            t0 = time.time()
            test_map = evaluate_with_options(
                model, test_ids, labels_dict, TEST_VIDEO_DIR, device,
                batch_size=args.batch_size, sample_stride=args.stride,
                tta_flip=args.tta, agg_method=agg,
            )
            elapsed = time.time() - t0
            label = f"SOUP({len(ckpt_paths)} ckpts)"
            print(f"{label:<60} {agg:<6} {test_map:.4f}  {elapsed:6.0f}s")
    else:
        # Standard mode: eval each checkpoint
        for ckpt_path in ckpt_paths:
            if not os.path.exists(ckpt_path):
                print(f"{ckpt_path:<60} NOT FOUND")
                continue

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

            for agg in agg_methods:
                t0 = time.time()
                test_map = evaluate_with_options(
                    model, test_ids, labels_dict, TEST_VIDEO_DIR, device,
                    batch_size=args.batch_size, sample_stride=args.stride,
                    tta_flip=args.tta, agg_method=agg,
                )
                elapsed = time.time() - t0
                ep = ckpt.get("epoch", "?")
                orig_map = ckpt.get("test_map", ckpt.get("best_test_map", "?"))
                label = f"{os.path.basename(os.path.dirname(ckpt_path))} ep{ep} (orig={orig_map})"
                print(f"{label:<60} {agg:<6} {test_map:.4f}  {elapsed:6.0f}s")


if __name__ == "__main__":
    main()
