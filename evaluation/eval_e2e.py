#!/usr/bin/env python3
"""Re-evaluate saved checkpoints with configurable eval stride.

Usage:
    python3 eval_e2e.py results/e2e_mvit_v2_s_run19/best_model.pt --stride 30
    python3 eval_e2e.py results/e2e_mvit_v2_s_run19/best_model.pt --stride 15 --gpu 1
    python3 eval_e2e.py results/e2e_mvit_v2_s_run*/best_model.pt --stride 30  # glob
"""
import argparse
import sys
import os
import glob
import time

import torch
from train_e2e import (
    build_model, load_test_annotations, evaluate_videos,
    TEST_CSV, TEST_VIDEO_DIR, set_seed, SEED,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint .pt files (supports glob)")
    parser.add_argument("--stride", type=int, default=30, help="Eval sample stride (default: 30)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Expand globs
    ckpt_paths = []
    for pattern in args.checkpoints:
        expanded = sorted(glob.glob(pattern))
        ckpt_paths.extend(expanded if expanded else [pattern])

    if not ckpt_paths:
        print("No checkpoints found")
        sys.exit(1)

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load test annotations
    test_annots = load_test_annotations(TEST_CSV)
    test_ids = [r["id"] for r in test_annots]
    labels_dict = {r["id"]: r["target"] for r in test_annots}

    # Build model once
    model = build_model(pretrained=False)
    model = model.to(device)

    print(f"Eval stride: {args.stride} | GPU: {args.gpu} | Test videos: {len(test_ids)}")
    print(f"{'Checkpoint':<65} {'mAP':>8} {'Time':>8}")
    print("-" * 85)

    for ckpt_path in ckpt_paths:
        if not os.path.exists(ckpt_path):
            print(f"{ckpt_path:<65} NOT FOUND")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        t0 = time.time()
        test_map = evaluate_videos(
            model, test_ids, labels_dict, TEST_VIDEO_DIR, device,
            batch_size=args.batch_size, sample_stride=args.stride,
        )
        elapsed = time.time() - t0

        ep = ckpt.get("epoch", "?")
        orig_map = ckpt.get("test_map", ckpt.get("best_test_map", "?"))
        label = f"{ckpt_path} (ep{ep}, orig={orig_map})"
        print(f"{label:<65} {test_map:.4f}  {elapsed:6.0f}s")


if __name__ == "__main__":
    main()
