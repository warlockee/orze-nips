#!/usr/bin/env python3
"""Compute ANOVA for E2E fine-tuning ablation (NeurIPS paper).

Reads all e2e-ablation-XXXX results and computes:
  - backbone eta^2 (one-way)
  - encoder eta^2 (one-way)
  - backbone x encoder eta^2 (two-way)
  - lr eta^2 (one-way)
  - weight_decay eta^2 (one-way)
  - Bootstrap 95% CIs on all eta^2 values (10000 resamples)

Saves to: doc/computed_values/e2e_anova.json

Usage:
    python doc/scripts/compute_e2e_anova.py
"""

from __future__ import annotations

import json
import os
import sys
import glob
import yaml
import numpy as np
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "e2e_anova.json")


def one_way_anova(groups, min_group_size=2):
    """Compute one-way ANOVA with eta-squared.

    Parameters
    ----------
    groups : dict of str -> list[float]
        Factor level name -> observed values.
    min_group_size : int
        Minimum observations per group to include.

    Returns
    -------
    dict or None
    """
    valid_groups = {
        k: np.array(v) for k, v in groups.items() if len(v) >= min_group_size
    }
    if len(valid_groups) < 2:
        return None

    all_values = np.concatenate(list(valid_groups.values()))
    grand_mean = np.mean(all_values)
    N = len(all_values)
    k = len(valid_groups)

    ssb = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2 for g in valid_groups.values()
    )
    ssw = sum(np.sum((g - np.mean(g)) ** 2) for g in valid_groups.values())

    df_between = k - 1
    df_within = N - k
    msb = ssb / df_between if df_between > 0 else 0
    msw = ssw / df_within if df_within > 0 else 1e-10
    f_stat = msb / msw

    try:
        from scipy.stats import f as f_dist
        p_value = 1 - f_dist.cdf(f_stat, df_between, df_within)
    except ImportError:
        p_value = float("nan")

    ss_total = ssb + ssw
    eta_squared = ssb / ss_total if ss_total > 0 else 0

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "df_between": int(df_between),
        "df_within": int(df_within),
        "ssb": float(ssb),
        "ssw": float(ssw),
        "msb": float(msb),
        "msw": float(msw),
        "eta_squared": float(eta_squared),
        "n_groups": k,
        "n_total": N,
        "groups_used": {
            k2: len(v2)
            for k2, v2 in sorted(
                valid_groups.items(), key=lambda x: -np.mean(x[1])
            )
        },
        "group_means": {
            k2: float(np.mean(v2))
            for k2, v2 in sorted(
                valid_groups.items(), key=lambda x: -np.mean(x[1])
            )
        },
    }


def two_way_anova(factor_a, factor_b, values):
    """Compute two-way ANOVA (Type I) with eta-squared for interaction.

    Parameters
    ----------
    factor_a, factor_b : list[str]
        Factor levels for each observation.
    values : list[float]
        Observed values.

    Returns
    -------
    dict or None
    """
    assert len(factor_a) == len(factor_b) == len(values)
    N = len(values)
    values = np.array(values)
    grand_mean = np.mean(values)

    # Cell means
    cells = defaultdict(list)
    for a, b, v in zip(factor_a, factor_b, values):
        cells[(a, b)].append(v)

    # Factor A main effect
    a_groups = defaultdict(list)
    for a, _, v in zip(factor_a, factor_b, values):
        a_groups[a].append(v)
    ss_a = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2
        for g in a_groups.values()
    )

    # Factor B main effect
    b_groups = defaultdict(list)
    for _, b, v in zip(factor_a, factor_b, values):
        b_groups[b].append(v)
    ss_b = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2
        for g in b_groups.values()
    )

    # Total SS
    ss_total = np.sum((values - grand_mean) ** 2)

    # Within-cell SS
    ss_within = sum(
        np.sum((np.array(g) - np.mean(g)) ** 2) for g in cells.values()
    )

    # Interaction SS = SS_total - SS_A - SS_B - SS_within
    ss_ab = ss_total - ss_a - ss_b - ss_within
    if ss_ab < 0:
        ss_ab = 0  # numerical precision

    n_a = len(a_groups)
    n_b = len(b_groups)

    eta_sq_a = ss_a / ss_total if ss_total > 0 else 0
    eta_sq_b = ss_b / ss_total if ss_total > 0 else 0
    eta_sq_ab = ss_ab / ss_total if ss_total > 0 else 0

    return {
        "ss_a": float(ss_a),
        "ss_b": float(ss_b),
        "ss_ab": float(ss_ab),
        "ss_within": float(ss_within),
        "ss_total": float(ss_total),
        "eta_sq_a": float(eta_sq_a),
        "eta_sq_b": float(eta_sq_b),
        "eta_sq_ab": float(eta_sq_ab),
        "n_a_levels": n_a,
        "n_b_levels": n_b,
        "n_total": N,
        "a_levels": sorted(a_groups.keys()),
        "b_levels": sorted(b_groups.keys()),
    }


def bootstrap_eta_squared(groups, n_bootstrap=10000, ci=0.95, min_group_size=2):
    """Bootstrap confidence interval for eta-squared.

    Resamples observations within each group, recomputes ANOVA each time.
    """
    valid_groups = {
        k: np.array(v) for k, v in groups.items() if len(v) >= min_group_size
    }
    if len(valid_groups) < 2:
        return None

    rng = np.random.RandomState(42)
    eta_sq_samples = []

    group_names = list(valid_groups.keys())
    group_arrays = [valid_groups[k] for k in group_names]

    for _ in range(n_bootstrap):
        # Resample within each group (with replacement)
        resampled = {}
        for name, arr in zip(group_names, group_arrays):
            idx = rng.randint(0, len(arr), size=len(arr))
            resampled[name] = arr[idx]

        all_vals = np.concatenate(list(resampled.values()))
        gm = np.mean(all_vals)
        ssb = sum(
            len(g) * (np.mean(g) - gm) ** 2 for g in resampled.values()
        )
        ssw = sum(
            np.sum((g - np.mean(g)) ** 2) for g in resampled.values()
        )
        ss_total = ssb + ssw
        eta_sq = ssb / ss_total if ss_total > 0 else 0
        eta_sq_samples.append(eta_sq)

    eta_sq_samples = np.array(eta_sq_samples)
    alpha = 1 - ci
    lo = np.percentile(eta_sq_samples, 100 * alpha / 2)
    hi = np.percentile(eta_sq_samples, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(eta_sq_samples)),
        "std": float(np.std(eta_sq_samples)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def load_e2e_experiments():
    """Load all e2e-ablation-XXXX experiments."""
    experiments = []
    pattern = os.path.join(RESULTS_DIR, "e2e-ablation-*")
    idea_dirs = sorted(glob.glob(pattern))

    print(f"Scanning {len(idea_dirs)} e2e-ablation directories...", file=sys.stderr)

    for idea_dir in idea_dirs:
        idea_id = os.path.basename(idea_dir)

        config_path = os.path.join(idea_dir, "resolved_config.yaml")
        metrics_path = os.path.join(idea_dir, "metrics.json")

        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            continue

        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            with open(metrics_path) as f:
                met = json.load(f)
        except Exception:
            continue

        if cfg is None or met is None:
            continue

        # Only include completed experiments with positive AP
        status = met.get("status", "")
        if status != "COMPLETED":
            continue

        ap = met.get("best_val_metric", 0)
        if ap is None or not isinstance(ap, (int, float)) or ap <= 0:
            continue

        # Extract factors
        backbone = cfg.get("backbone", {}).get("name", "unknown")
        encoder = cfg.get("temporal_encoder", {}).get("type", "unknown")
        lr = cfg.get("optimizer", {}).get("lr", 0)
        wd = cfg.get("optimizer", {}).get("weight_decay", 0)
        seed = cfg.get("training", {}).get("seed", 0)

        experiments.append({
            "idea_id": idea_id,
            "backbone": backbone,
            "encoder": encoder,
            "lr": lr,
            "wd": wd,
            "seed": seed,
            "ap": float(ap),
        })

    return experiments


def main():
    experiments = load_e2e_experiments()
    print(f"Loaded {len(experiments)} completed E2E ablation experiments", file=sys.stderr)

    if len(experiments) < 10:
        print(
            f"WARNING: Only {len(experiments)} experiments found. "
            "Need more data for meaningful ANOVA.",
            file=sys.stderr,
        )
        if len(experiments) == 0:
            # Write empty placeholder
            output = {
                "status": "INSUFFICIENT_DATA",
                "n_experiments": 0,
                "message": "No completed E2E ablation experiments found",
            }
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Saved placeholder to {OUTPUT_PATH}")
            return

    # Build factor groups
    backbone_groups = defaultdict(list)
    encoder_groups = defaultdict(list)
    lr_groups = defaultdict(list)
    wd_groups = defaultdict(list)
    backbone_x_encoder_groups = defaultdict(list)

    backbones_list = []
    encoders_list = []
    aps_list = []

    for exp in experiments:
        ap = exp["ap"]
        bb = exp["backbone"]
        enc = exp["encoder"]
        lr_str = f"{exp['lr']:.0e}"
        wd_str = f"{exp['wd']:.0e}"

        backbone_groups[bb].append(ap)
        encoder_groups[enc].append(ap)
        lr_groups[lr_str].append(ap)
        wd_groups[wd_str].append(ap)
        backbone_x_encoder_groups[f"{bb}+{enc}"].append(ap)

        backbones_list.append(bb)
        encoders_list.append(enc)
        aps_list.append(ap)

    # One-way ANOVAs
    print("\n--- One-way ANOVA: Backbone ---", file=sys.stderr)
    backbone_anova = one_way_anova(backbone_groups)
    if backbone_anova:
        print(
            f"  eta^2 = {backbone_anova['eta_squared']:.4f}, "
            f"F = {backbone_anova['f_statistic']:.2f}, "
            f"p = {backbone_anova['p_value']:.4e}",
            file=sys.stderr,
        )

    print("\n--- One-way ANOVA: Encoder ---", file=sys.stderr)
    encoder_anova = one_way_anova(encoder_groups)
    if encoder_anova:
        print(
            f"  eta^2 = {encoder_anova['eta_squared']:.4f}, "
            f"F = {encoder_anova['f_statistic']:.2f}, "
            f"p = {encoder_anova['p_value']:.4e}",
            file=sys.stderr,
        )

    print("\n--- One-way ANOVA: Learning Rate ---", file=sys.stderr)
    lr_anova = one_way_anova(lr_groups)
    if lr_anova:
        print(
            f"  eta^2 = {lr_anova['eta_squared']:.4f}, "
            f"F = {lr_anova['f_statistic']:.2f}, "
            f"p = {lr_anova['p_value']:.4e}",
            file=sys.stderr,
        )

    print("\n--- One-way ANOVA: Weight Decay ---", file=sys.stderr)
    wd_anova = one_way_anova(wd_groups)
    if wd_anova:
        print(
            f"  eta^2 = {wd_anova['eta_squared']:.4f}, "
            f"F = {wd_anova['f_statistic']:.2f}, "
            f"p = {wd_anova['p_value']:.4e}",
            file=sys.stderr,
        )

    print("\n--- One-way ANOVA: Backbone x Encoder ---", file=sys.stderr)
    bxe_anova = one_way_anova(backbone_x_encoder_groups)
    if bxe_anova:
        print(
            f"  eta^2 = {bxe_anova['eta_squared']:.4f}, "
            f"F = {bxe_anova['f_statistic']:.2f}, "
            f"p = {bxe_anova['p_value']:.4e}",
            file=sys.stderr,
        )

    # Two-way ANOVA: backbone x encoder
    print("\n--- Two-way ANOVA: Backbone x Encoder ---", file=sys.stderr)
    two_way = two_way_anova(backbones_list, encoders_list, aps_list)
    if two_way:
        print(
            f"  eta^2_backbone = {two_way['eta_sq_a']:.4f}, "
            f"eta^2_encoder = {two_way['eta_sq_b']:.4f}, "
            f"eta^2_interaction = {two_way['eta_sq_ab']:.4f}",
            file=sys.stderr,
        )

    # Bootstrap CIs
    print("\n--- Bootstrap 95% CIs (10000 resamples) ---", file=sys.stderr)
    bootstrap_backbone = bootstrap_eta_squared(backbone_groups)
    bootstrap_encoder = bootstrap_eta_squared(encoder_groups)
    bootstrap_lr = bootstrap_eta_squared(lr_groups)
    bootstrap_wd = bootstrap_eta_squared(wd_groups)
    bootstrap_bxe = bootstrap_eta_squared(backbone_x_encoder_groups)

    for name, bs in [
        ("backbone", bootstrap_backbone),
        ("encoder", bootstrap_encoder),
        ("lr", bootstrap_lr),
        ("wd", bootstrap_wd),
        ("backbone_x_encoder", bootstrap_bxe),
    ]:
        if bs:
            print(
                f"  {name}: eta^2 = {bs['mean']:.4f} "
                f"[{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]",
                file=sys.stderr,
            )

    # Assemble output
    output = {
        "status": "OK",
        "n_experiments": len(experiments),
        "n_backbones": len(backbone_groups),
        "n_encoders": len(encoder_groups),
        "one_way_anova": {
            "backbone": backbone_anova,
            "encoder": encoder_anova,
            "learning_rate": lr_anova,
            "weight_decay": wd_anova,
            "backbone_x_encoder": bxe_anova,
        },
        "two_way_anova_backbone_encoder": two_way,
        "bootstrap_95ci": {
            "backbone": bootstrap_backbone,
            "encoder": bootstrap_encoder,
            "learning_rate": bootstrap_lr,
            "weight_decay": bootstrap_wd,
            "backbone_x_encoder": bootstrap_bxe,
        },
        "summary": {
            "overall_mean_ap": float(np.mean(aps_list)),
            "overall_std_ap": float(np.std(aps_list)),
            "per_backbone": {
                bb: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
                for bb, vals in backbone_groups.items()
            },
            "per_encoder": {
                enc: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
                for enc, vals in encoder_groups.items()
            },
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
