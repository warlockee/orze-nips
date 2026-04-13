#!/usr/bin/env python3
"""Analyze SMAC baseline results and compare with existing baselines.

Reads completed SMAC experiments, computes convergence curves,
and generates comparison data for the NeurIPS paper.

Usage:
    python doc/scripts/analyze_smac_baseline.py
    python doc/scripts/analyze_smac_baseline.py --results-dir results/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_NIPS_DIR = _SCRIPT_DIR.parent / "NIPS"
_COMPUTED_DIR = _NIPS_DIR / "computed_values"
_COMPUTED_DIR.mkdir(parents=True, exist_ok=True)


def load_smac_experiments(results_dir):
    """Load all completed SMAC baseline experiments."""
    experiments = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("smac-baseline-"):
            continue
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            if m.get("status") != "COMPLETED":
                continue
            # Extract config from resolved_config.json or metrics.json
            config = m.get("config", {})
            if not config:
                config_path = d / "resolved_config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
            backbone = config.get("backbone", {}).get("name", "unknown")
            encoder = config.get("temporal_encoder", {}).get("type", "unknown")
            experiments.append({
                "idea_id": m.get("idea_id", d.name),
                "backbone": backbone,
                "encoder": encoder,
                "best_val_metric": m.get("best_val_metric", 0.0),
                "config": config,
            })
        except Exception as e:
            print(f"  Skipping {d.name}: {e}", file=sys.stderr)
    return experiments


def load_competition_map(results_dir):
    """Load competition mAP from nexar_comp_report.json if available."""
    comp_results = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("smac-baseline-"):
            continue
        report = d / "nexar_comp_report.json"
        if report.exists():
            try:
                with open(report) as f:
                    r = json.load(f)
                comp_results[d.name] = r.get("competition_mAP", r.get("mAP", None))
            except Exception:
                pass
    return comp_results


def compute_convergence(experiments, metric_key="best_val_metric"):
    """Compute cumulative best curve."""
    vals = [e[metric_key] for e in experiments]
    cummax = np.maximum.accumulate(vals)
    return cummax


def analyze_backbone_distribution(experiments):
    """Count experiments per backbone."""
    from collections import Counter
    return Counter(e["backbone"] for e in experiments)


def analyze_encoder_distribution(experiments):
    """Count experiments per encoder."""
    from collections import Counter
    return Counter(e["encoder"] for e in experiments)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    print("=" * 60)
    print("SMAC Baseline Analysis")
    print("=" * 60)

    # Load experiments
    experiments = load_smac_experiments(results_dir)
    print(f"\nCompleted experiments: {len(experiments)}")

    if not experiments:
        print("No completed experiments found. Waiting for runs to finish.")
        return

    # Validation metric stats
    vals = [e["best_val_metric"] for e in experiments]
    print(f"\nValidation mAP statistics:")
    print(f"  Mean:   {np.mean(vals):.4f}")
    print(f"  Median: {np.median(vals):.4f}")
    print(f"  Max:    {np.max(vals):.4f}")
    print(f"  Min:    {np.min(vals):.4f}")
    print(f"  Std:    {np.std(vals):.4f}")

    # Convergence curve
    cummax = compute_convergence(experiments)
    print(f"\nConvergence (cumulative best val mAP):")
    milestones = [10, 25, 50, 100, 200, 300, 400, 500]
    for m in milestones:
        if m <= len(cummax):
            print(f"  N={m:>4d}: {cummax[m-1]:.4f}")

    # Backbone distribution
    bb_counts = analyze_backbone_distribution(experiments)
    print(f"\nBackbone distribution:")
    for bb, count in bb_counts.most_common():
        bb_vals = [e["best_val_metric"] for e in experiments if e["backbone"] == bb]
        print(f"  {bb:<35s}: n={count:>3d}  mean={np.mean(bb_vals):.4f}  max={np.max(bb_vals):.4f}")

    # Encoder distribution
    enc_counts = analyze_encoder_distribution(experiments)
    print(f"\nEncoder distribution:")
    for enc, count in enc_counts.most_common():
        enc_vals = [e["best_val_metric"] for e in experiments if e["encoder"] == enc]
        print(f"  {enc:<30s}: n={count:>3d}  mean={np.mean(enc_vals):.4f}  max={np.max(enc_vals):.4f}")

    # Top 10 configurations
    top10 = sorted(experiments, key=lambda e: e["best_val_metric"], reverse=True)[:10]
    print(f"\nTop 10 configurations:")
    for i, e in enumerate(top10):
        cfg = e["config"]
        lr = cfg.get("training", {}).get("learning_rate", "?")
        print(f"  {i+1:>2d}. val={e['best_val_metric']:.4f}  {e['backbone']:<25s}  {e['encoder']:<20s}  lr={lr}")

    # Competition mAP (if evaluated)
    comp = load_competition_map(results_dir)
    if comp:
        comp_vals = [v for v in comp.values() if v is not None]
        print(f"\nCompetition mAP (n={len(comp_vals)}):")
        print(f"  Max:  {max(comp_vals):.4f}")
        print(f"  Mean: {np.mean(comp_vals):.4f}")

    # Compare with existing baselines
    print(f"\n{'='*60}")
    print("Comparison with existing baselines")
    print(f"{'='*60}")

    # Load existing baseline data
    expanded_path = _COMPUTED_DIR / "expanded_baselines.json"
    if expanded_path.exists():
        with open(expanded_path) as f:
            eb = json.load(f)
        print(f"  Expanded TPE best:    {eb.get('tpe_best_mAP', '?')}")
        print(f"  Expanded Random best: {eb.get('random_best_mAP', '?')}")
    print(f"  SMAC best val:        {np.max(vals):.4f}")
    print(f"  SMAC total:           {len(experiments)}")
    print(f"\n  Note: SMAC val metric is validation mAP, not competition mAP.")
    print(f"  Run evaluate_nexar_test.py on SMAC experiments to get competition mAP.")

    # Save results
    output = {
        "n_completed": len(experiments),
        "val_mAP_stats": {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "max": float(np.max(vals)),
            "min": float(np.min(vals)),
            "std": float(np.std(vals)),
        },
        "convergence": {str(m): float(cummax[m-1]) for m in milestones if m <= len(cummax)},
        "backbone_counts": dict(bb_counts),
        "encoder_counts": dict(enc_counts),
        "top10": [
            {
                "backbone": e["backbone"],
                "encoder": e["encoder"],
                "val_mAP": e["best_val_metric"],
                "idea_id": e["idea_id"],
            }
            for e in top10
        ],
    }
    if comp:
        comp_vals = [v for v in comp.values() if v is not None]
        output["competition_mAP"] = {
            "n_evaluated": len(comp_vals),
            "max": float(max(comp_vals)) if comp_vals else None,
            "mean": float(np.mean(comp_vals)) if comp_vals else None,
        }

    out_path = _COMPUTED_DIR / "smac_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
