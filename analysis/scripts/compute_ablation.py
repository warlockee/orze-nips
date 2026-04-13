#!/usr/bin/env python3
"""
Script 1: Compute the component ablation table (Table 2 in the paper).

Reads resolved_config.yaml and ken_test_report.json from each experiment,
groups by backbone, temporal encoder, loss type, and pooling type, then
computes statistics and significance tests.

Outputs:
  - doc/computed_values/ablation.json
  - Prints formatted LaTeX table to stdout
"""

import json
import os
import sys
import glob
import yaml
import numpy as np
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
RESULTS_DIR = os.path.abspath(RESULTS_DIR)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'computed_values')
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_backbone(name):
    """Map raw backbone names to canonical categories."""
    if name is None:
        return None
    name = name.lower().strip()
    if 'vjepa' in name:
        return 'VJepa2'
    if 'dinov3' in name and ('large' in name or '_l' in name or 'vitl' in name):
        return 'DINOv3-L'
    if 'dinov3' in name:
        return 'DINOv3-B'
    if 'dinov2' in name:
        return 'DINOv2'
    if 'siglip' in name:
        return 'SigLIP2'
    if 'intern' in name:
        return 'InternViT'
    if 'convnext' in name:
        return 'ConvNeXt'
    if 'eva' in name:
        return 'EVA02'
    if 'swin' in name:
        return 'Swin'
    if 'hiera' in name:
        return 'Hiera'
    if 'mobile' in name:
        return 'MobileViT'
    if 'fast' in name:
        return 'FastViT'
    return name


def normalize_encoder(enc_type):
    """Map raw encoder types to canonical categories."""
    if enc_type is None:
        return None
    enc_type = enc_type.lower().strip()
    if 'zipformer' in enc_type:
        return 'Zipformer'
    if 'hybrid' in enc_type or 'retention_mamba' in enc_type or 'retention-mamba' in enc_type:
        return 'Hybrid R-M'
    if 'retention' in enc_type or 'retnet' in enc_type:
        return 'Retention'
    if 'bimamba' in enc_type or 'bi_mamba' in enc_type or 'mamba' in enc_type:
        return 'BiMamba'
    if 'gru' in enc_type:
        return 'GRU'
    return enc_type


def normalize_loss(loss_cfg):
    """Classify loss into Focal (with gamma >= 2) vs BCE."""
    if loss_cfg is None:
        return None
    if isinstance(loss_cfg, dict):
        cls = loss_cfg.get('classification', loss_cfg)
        if isinstance(cls, dict):
            loss_type = cls.get('type', '').lower()
            gamma = cls.get('gamma', 0)
            if 'focal' in loss_type:
                if gamma is not None and float(gamma) >= 2.0:
                    return 'Focal (g>=2)'
                return 'Focal (g<2)'
            return 'BCE'
    return None


def normalize_pooling(heads_cfg):
    """Extract pooling type from heads config."""
    if heads_cfg is None:
        return None
    if isinstance(heads_cfg, dict):
        cls = heads_cfg.get('classification', heads_cfg)
        if isinstance(cls, dict):
            pooling = cls.get('pooling', 'mean').lower()
            if 'attention' in pooling:
                return 'Attention'
            return 'Mean'
    return None


def extract_backbone_from_config(cfg):
    """Handle both single and multi-backbone configs."""
    backbone = cfg.get('backbone', {})
    if isinstance(backbone, dict):
        if 'type' in backbone and 'multi' in str(backbone.get('type', '')).lower():
            # Multi-backbone fusion
            names = []
            for b in backbone.get('backbones', []):
                names.append(normalize_backbone(b.get('name', '')))
            return 'Multi-Backbone'
        return normalize_backbone(backbone.get('name', ''))
    return None


def load_all_experiments():
    """Load experiment configs and AP values from result directories."""
    experiments = []
    idea_dirs = glob.glob(os.path.join(RESULTS_DIR, 'idea-*'))
    print(f"Scanning {len(idea_dirs)} result directories...", file=sys.stderr)

    loaded = 0
    skipped_no_config = 0
    skipped_no_eval = 0
    skipped_no_ap = 0

    for idea_dir in idea_dirs:
        idea_id = os.path.basename(idea_dir)

        # Load resolved config
        config_path = os.path.join(idea_dir, 'resolved_config.yaml')
        if not os.path.exists(config_path):
            skipped_no_config += 1
            continue

        # Load ken_test_report.json for AP
        eval_path = os.path.join(idea_dir, 'ken_test_report.json')
        if not os.path.exists(eval_path):
            skipped_no_eval += 1
            continue

        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            with open(eval_path) as f:
                eval_data = json.load(f)
        except Exception:
            continue

        if cfg is None or eval_data is None:
            continue

        # Extract AP
        metrics = eval_data.get('metrics', {})
        ap = metrics.get('average_precision')
        if ap is None or not isinstance(ap, (int, float)) or np.isnan(ap):
            skipped_no_ap += 1
            continue

        # Extract config components
        backbone = extract_backbone_from_config(cfg)
        encoder = normalize_encoder(cfg.get('temporal_encoder', {}).get('type'))
        loss = normalize_loss(cfg.get('loss'))
        pooling = normalize_pooling(cfg.get('heads'))
        seq_len = cfg.get('training', {}).get('sequence_length')

        # Also extract claim timestamp for ordering
        claim_path = os.path.join(idea_dir, 'claim.json')
        claimed_at = None
        if os.path.exists(claim_path):
            try:
                with open(claim_path) as f:
                    claim = json.load(f)
                claimed_at = claim.get('claimed_at')
            except Exception:
                pass

        experiments.append({
            'idea_id': idea_id,
            'ap': float(ap),
            'backbone': backbone,
            'encoder': encoder,
            'loss': loss,
            'pooling': pooling,
            'seq_len': seq_len,
            'claimed_at': claimed_at,
        })
        loaded += 1

    print(f"Loaded: {loaded}, no config: {skipped_no_config}, "
          f"no eval: {skipped_no_eval}, no AP: {skipped_no_ap}", file=sys.stderr)
    return experiments


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, stat_fn=np.mean):
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    if len(values) < 2:
        m = stat_fn(values)
        return m, m, m
    rng = np.random.default_rng(42)
    boot_stats = np.array([
        stat_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_stats, alpha * 100)
    hi = np.percentile(boot_stats, (1 - alpha) * 100)
    return stat_fn(values), lo, hi


def mann_whitney_u(x, y):
    """Mann-Whitney U test (two-sided)."""
    from scipy.stats import mannwhitneyu
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    stat, p = mannwhitneyu(x, y, alternative='two-sided')
    return float(stat), float(p)


def compute_group_stats(experiments, key, values_of_interest=None, top_n=50):
    """Compute stats for each value of a given key."""
    groups = defaultdict(list)
    for exp in experiments:
        val = exp.get(key)
        if val is not None:
            groups[val].append(exp['ap'])

    results = {}
    for val, aps in sorted(groups.items(), key=lambda x: -max(x[1])):
        if values_of_interest and val not in values_of_interest:
            continue
        aps = np.array(aps)
        sorted_aps = np.sort(aps)[::-1]
        top_aps = sorted_aps[:top_n]

        mean_val, ci_lo, ci_hi = bootstrap_ci(top_aps)
        results[val] = {
            'count': len(aps),
            'best_ap': float(np.max(aps)),
            'mean_ap_top50': float(mean_val),
            'ci_95_lo': float(ci_lo),
            'ci_95_hi': float(ci_hi),
            'median_ap': float(np.median(aps)),
            'std_ap': float(np.std(aps)),
            'mean_ap_all': float(np.mean(aps)),
        }
    return results


def run_pairwise_tests(experiments, key, pairs):
    """Run Mann-Whitney U between specified pairs."""
    groups = defaultdict(list)
    for exp in experiments:
        val = exp.get(key)
        if val is not None:
            groups[val].append(exp['ap'])

    results = {}
    for a, b in pairs:
        if a in groups and b in groups:
            u_stat, p_val = mann_whitney_u(
                np.array(groups[a]), np.array(groups[b])
            )
            results[f"{a} vs {b}"] = {
                'U_statistic': u_stat,
                'p_value': p_val,
                'n_a': len(groups[a]),
                'n_b': len(groups[b]),
            }
    return results


def format_ci(lo, hi):
    """Format a 95% CI as [lo, hi]."""
    return f"[{lo:.4f}, {hi:.4f}]"


def main():
    experiments = load_all_experiments()
    if not experiments:
        print("ERROR: No experiments loaded!", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal experiments with AP and config: {len(experiments)}", file=sys.stderr)

    # ---- Backbone ablation ----
    backbone_stats = compute_group_stats(
        experiments, 'backbone',
        values_of_interest=['VJepa2', 'DINOv3-B', 'DINOv3-L', 'DINOv2', 'SigLIP2', 'Multi-Backbone']
    )

    # ---- Encoder ablation ----
    encoder_stats = compute_group_stats(
        experiments, 'encoder',
        values_of_interest=['Zipformer', 'Retention', 'BiMamba', 'Hybrid R-M']
    )

    # ---- Loss ablation ----
    loss_stats = compute_group_stats(
        experiments, 'loss',
        values_of_interest=['Focal (g>=2)', 'Focal (g<2)', 'BCE']
    )

    # ---- Pooling ablation ----
    pooling_stats = compute_group_stats(
        experiments, 'pooling',
        values_of_interest=['Attention', 'Mean']
    )

    # ---- Pairwise tests ----
    backbone_tests = run_pairwise_tests(experiments, 'backbone', [
        ('VJepa2', 'DINOv3-B'),
        ('VJepa2', 'DINOv3-L'),
        ('DINOv3-B', 'DINOv3-L'),
    ])
    encoder_tests = run_pairwise_tests(experiments, 'encoder', [
        ('Zipformer', 'Retention'),
        ('Zipformer', 'BiMamba'),
        ('Zipformer', 'Hybrid R-M'),
        ('Retention', 'BiMamba'),
    ])
    loss_tests = run_pairwise_tests(experiments, 'loss', [
        ('Focal (g>=2)', 'BCE'),
        ('Focal (g>=2)', 'Focal (g<2)'),
    ])
    pooling_tests = run_pairwise_tests(experiments, 'pooling', [
        ('Attention', 'Mean'),
    ])

    # ---- Compile output ----
    output = {
        'total_experiments': len(experiments),
        'backbone': backbone_stats,
        'encoder': encoder_stats,
        'loss': loss_stats,
        'pooling': pooling_stats,
        'pairwise_tests': {
            'backbone': backbone_tests,
            'encoder': encoder_tests,
            'loss': loss_tests,
            'pooling': pooling_tests,
        },
    }

    # Save JSON
    out_path = os.path.join(OUTPUT_DIR, 'ablation.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}", file=sys.stderr)

    # ---- Print LaTeX table ----
    print("\n% === ABLATION TABLE (Table 2) ===")
    print("% Dimension & Value & Best AP & Mean AP (top-50) & 95% CI & Count")

    def print_row(dimension, value, stats):
        s = stats.get(value, {})
        if not s:
            print(f"% {dimension} & {value} & -- & -- & -- & -- \\\\")
            return
        ci = format_ci(s['ci_95_lo'], s['ci_95_hi'])
        print(f"& {value} & {s['best_ap']:.4f} & {s['mean_ap_top50']:.4f} & {ci} & {s['count']} \\\\")

    print("\\multirow{3}{*}{Backbone}")
    for v in ['VJepa2', 'DINOv3-B', 'DINOv3-L']:
        print_row('Backbone', v, backbone_stats)

    print("\\midrule")
    print("\\multirow{4}{*}{Encoder}")
    for v in ['Zipformer', 'Retention', 'BiMamba', 'Hybrid R-M']:
        print_row('Encoder', v, encoder_stats)

    print("\\midrule")
    print("\\multirow{2}{*}{Loss}")
    for v in ['Focal (g>=2)', 'BCE']:
        print_row('Loss', v, loss_stats)

    print("\\midrule")
    print("\\multirow{2}{*}{Pooling}")
    for v in ['Attention', 'Mean']:
        print_row('Pooling', v, pooling_stats)

    # ---- Print pairwise tests ----
    print("\n% === PAIRWISE MANN-WHITNEY U TESTS ===")
    for category, tests in output['pairwise_tests'].items():
        for pair_name, result in tests.items():
            p = result['p_value']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"% {pair_name}: U={result['U_statistic']:.0f}, p={p:.2e} {sig} "
                  f"(n={result['n_a']},{result['n_b']})")

    # Print summary for paper
    print("\n% === KEY VALUES FOR PAPER ===")
    for dim_name, stats in [('backbone', backbone_stats), ('encoder', encoder_stats),
                             ('loss', loss_stats), ('pooling', pooling_stats)]:
        for val, s in stats.items():
            print(f"% {dim_name}/{val}: best={s['best_ap']:.4f}, "
                  f"mean_top50={s['mean_ap_top50']:.4f}, "
                  f"ci=[{s['ci_95_lo']:.4f},{s['ci_95_hi']:.4f}], "
                  f"n={s['count']}")


if __name__ == '__main__':
    main()
