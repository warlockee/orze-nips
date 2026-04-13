#!/usr/bin/env python3
"""Model soup: average weights from multiple checkpoints, evaluate.

Usage:
    # Average top-K checkpoints from a single run:
    python3 model_soup.py results/e2e_mvit_v2_s_run35/checkpoint_ep*.pt --top_k 3

    # Average best_model.pt from multiple runs:
    python3 model_soup.py results/e2e_mvit_v2_s_run10/best_model.pt results/e2e_mvit_v2_s_run35/best_model.pt

    # Greedy soup: iteratively add checkpoints that improve mAP
    python3 model_soup.py results/e2e_mvit_v2_s_run35/checkpoint_ep*.pt --greedy
"""
import argparse
import glob
import sys
import os
import copy

import torch
import numpy as np

from train_e2e import (
    build_model, load_test_annotations, evaluate_videos,
    TEST_CSV, TEST_VIDEO_DIR, set_seed, SEED,
)


def load_checkpoints(paths, top_k=None):
    """Load checkpoints, optionally select top-K by test_map."""
    ckpts = []
    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        test_map = ckpt.get("test_map", ckpt.get("best_test_map", 0.0))
        ckpts.append({"path": p, "test_map": test_map, "state_dict": ckpt["model_state_dict"]})
        print(f"  {p}: test_mAP={test_map:.4f}")

    ckpts.sort(key=lambda x: x["test_map"], reverse=True)
    if top_k and top_k < len(ckpts):
        print(f"\nSelecting top {top_k} by test_mAP:")
        ckpts = ckpts[:top_k]
        for c in ckpts:
            print(f"  {c['path']}: {c['test_map']:.4f}")

    return ckpts


def uniform_soup(state_dicts):
    """Average all state dicts uniformly."""
    avg = copy.deepcopy(state_dicts[0])
    for key in avg:
        for i in range(1, len(state_dicts)):
            avg[key] = avg[key] + state_dicts[i][key]
        avg[key] = avg[key] / len(state_dicts)
    return avg


def evaluate_state_dict(state_dict, device, stride, batch_size):
    """Build model, load state dict, evaluate."""
    model = build_model()
    model.load_state_dict(state_dict)
    model = model.to(device)

    labels_dict = load_test_annotations(TEST_CSV)
    video_ids = list(labels_dict.keys())

    mAP = evaluate_videos(
        model, video_ids, labels_dict, TEST_VIDEO_DIR, device,
        batch_size=batch_size, sample_stride=stride,
    )
    return mAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint .pt files")
    parser.add_argument("--top_k", type=int, default=None, help="Use top-K checkpoints by mAP")
    parser.add_argument("--greedy", action="store_true", help="Greedy soup: add if improves")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Expand globs
    paths = []
    for p in args.checkpoints:
        expanded = sorted(glob.glob(p))
        if not expanded:
            print(f"WARNING: no files match {p}")
        paths.extend(expanded)

    if len(paths) < 2:
        print(f"Need at least 2 checkpoints, got {len(paths)}")
        sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    print(f"Loading {len(paths)} checkpoints...")
    ckpts = load_checkpoints(paths, args.top_k)

    if args.greedy:
        print(f"\n{'='*60}")
        print("GREEDY SOUP")
        print(f"{'='*60}")

        # Start with the best single checkpoint
        best_sd = copy.deepcopy(ckpts[0]["state_dict"])
        best_map = evaluate_state_dict(best_sd, device, args.stride, args.batch_size)
        print(f"\nSeed: {ckpts[0]['path']} → mAP={best_map:.4f}")

        included = [ckpts[0]["path"]]
        soup_sds = [ckpts[0]["state_dict"]]

        for i in range(1, len(ckpts)):
            candidate_sds = soup_sds + [ckpts[i]["state_dict"]]
            candidate_avg = uniform_soup(candidate_sds)
            candidate_map = evaluate_state_dict(candidate_avg, device, args.stride, args.batch_size)

            if candidate_map > best_map:
                print(f"  + {ckpts[i]['path']} → mAP={candidate_map:.4f} (+{candidate_map-best_map:.4f}) ✓")
                best_map = candidate_map
                best_sd = candidate_avg
                soup_sds = candidate_sds
                included.append(ckpts[i]["path"])
            else:
                print(f"  - {ckpts[i]['path']} → mAP={candidate_map:.4f} ({candidate_map-best_map:+.4f}) ✗")

        print(f"\nGreedy soup: {len(included)} models, mAP={best_map:.4f}")
    else:
        print(f"\n{'='*60}")
        print(f"UNIFORM SOUP ({len(ckpts)} models)")
        print(f"{'='*60}")

        state_dicts = [c["state_dict"] for c in ckpts]
        avg_sd = uniform_soup(state_dicts)
        soup_map = evaluate_state_dict(avg_sd, device, args.stride, args.batch_size)
        best_single = ckpts[0]["test_map"]
        print(f"\nUniform soup mAP: {soup_map:.4f} (best single: {best_single:.4f}, Δ={soup_map-best_single:+.4f})")


if __name__ == "__main__":
    main()
