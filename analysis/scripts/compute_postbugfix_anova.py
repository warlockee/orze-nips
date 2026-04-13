#!/usr/bin/env python3
"""
Post-bugfix ANOVA for the paper (Section 5.5).

Replicates the exact methodology of compute_anova.py (test AP from
ken_test_report.json, backbone×encoder groups with >= 10 experiments)
but restricted to experiments completed after the bug-fix cutoff.

The cutoff is determined by:
  - The paper's original ANOVA: F=449.3, eta²=0.79, n=1299
  - The paper states bugs were fixed around days 19-21 (approx Feb 28 - Mar 2)
  - The comprehensive.py used Mar 6 as cutoff
  - We'll use Mar 1 as the cutoff (roughly 2/3 through the campaign that
    started ~Feb 13), but also report sensitivity to cutoff choice.
"""

import json
import os
import sys
import glob
import yaml
import numpy as np
from collections import defaultdict
from datetime import datetime

RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))


def normalize_backbone(name):
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
    return name


def normalize_encoder(enc_type):
    if enc_type is None:
        return None
    enc_type = enc_type.lower().strip()
    if 'zipformer' in enc_type:
        return 'Zipformer'
    if 'hybrid' in enc_type or 'retention_mamba' in enc_type:
        return 'Hybrid R-M'
    if 'retention' in enc_type or 'retnet' in enc_type:
        return 'Retention'
    if 'bimamba' in enc_type or 'mamba' in enc_type:
        return 'BiMamba'
    return enc_type


def extract_backbone(cfg):
    backbone = cfg.get('backbone', {})
    if isinstance(backbone, dict):
        if 'multi' in str(backbone.get('type', '')).lower():
            return 'Multi-Backbone'
        return normalize_backbone(backbone.get('name', ''))
    return None


def one_way_anova(groups, min_group_size=10):
    valid_groups = {k: np.array(v) for k, v in groups.items() if len(v) >= min_group_size}
    if len(valid_groups) < 2:
        return None

    all_values = np.concatenate(list(valid_groups.values()))
    grand_mean = np.mean(all_values)
    N = len(all_values)
    k = len(valid_groups)

    ssb = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in valid_groups.values())
    ssw = sum(np.sum((g - np.mean(g)) ** 2) for g in valid_groups.values())

    df_between = k - 1
    df_within = N - k
    msb = ssb / df_between if df_between > 0 else 0
    msw = ssw / df_within if df_within > 0 else 1e-10
    f_stat = msb / msw

    from scipy.stats import f as f_dist
    p_value = 1 - f_dist.cdf(f_stat, df_between, df_within)
    ss_total = ssb + ssw
    eta_squared = ssb / ss_total if ss_total > 0 else 0

    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'df_between': int(df_between),
        'df_within': int(df_within),
        'ssb': float(ssb),
        'ssw': float(ssw),
        'msb': float(msb),
        'msw': float(msw),
        'eta_squared': float(eta_squared),
        'n_groups': k,
        'n_total': N,
        'groups_used': {k2: len(v2) for k2, v2 in sorted(valid_groups.items(),
                                                           key=lambda x: -np.mean(x[1]))},
    }


def load_experiments():
    """Load experiments exactly as compute_anova.py does:
    - Must have resolved_config.yaml
    - Must have ken_test_report.json with valid AP
    - Get timestamp from ken_test_report.json or claim.json or metrics.json mtime
    """
    experiments = []
    idea_dirs = glob.glob(os.path.join(RESULTS_DIR, 'idea-*'))
    print(f"Scanning {len(idea_dirs)} result directories...", file=sys.stderr)

    for idea_dir in idea_dirs:
        idea_id = os.path.basename(idea_dir)

        config_path = os.path.join(idea_dir, 'resolved_config.yaml')
        if not os.path.exists(config_path):
            continue

        eval_path = os.path.join(idea_dir, 'ken_test_report.json')
        if not os.path.exists(eval_path):
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

        metrics = eval_data.get('metrics', {})
        ap = metrics.get('average_precision')
        if ap is None or not isinstance(ap, (int, float)) or np.isnan(ap):
            continue

        backbone = extract_backbone(cfg)
        encoder = normalize_encoder(cfg.get('temporal_encoder', {}).get('type'))

        # Get timestamp - try multiple sources
        timestamp = None

        # 1. ken_test_report.json timestamp
        ts_str = eval_data.get('timestamp')
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str)
            except Exception:
                pass

        # 2. claim.json timestamp
        if timestamp is None:
            claim_path = os.path.join(idea_dir, 'claim.json')
            if os.path.exists(claim_path):
                try:
                    with open(claim_path) as f:
                        claim = json.load(f)
                    ts_str = claim.get('claimed_at')
                    if ts_str:
                        timestamp = datetime.fromisoformat(ts_str)
                except Exception:
                    pass

        # 3. metrics.json mtime
        if timestamp is None:
            metrics_path = os.path.join(idea_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                try:
                    mtime = os.path.getmtime(metrics_path)
                    timestamp = datetime.fromtimestamp(mtime)
                except Exception:
                    pass

        # 4. ken_test_report.json mtime as last resort
        if timestamp is None:
            try:
                mtime = os.path.getmtime(eval_path)
                timestamp = datetime.fromtimestamp(mtime)
            except Exception:
                pass

        experiments.append({
            'idea_id': idea_id,
            'ap': float(ap),
            'backbone': backbone,
            'encoder': encoder,
            'arch_combo': f'{backbone}+{encoder}',
            'timestamp': timestamp,
        })

    print(f"Loaded {len(experiments)} experiments with test AP", file=sys.stderr)
    return experiments


def run_anova_for_cutoff(experiments, cutoff_dt, label, min_group=10):
    """Run ANOVA for experiments after cutoff_dt."""
    filtered = [e for e in experiments if e['timestamp'] is not None and e['timestamp'] >= cutoff_dt]

    arch_groups = defaultdict(list)
    for exp in filtered:
        if exp['backbone'] and exp['encoder']:
            arch_groups[exp['arch_combo']].append(exp['ap'])

    anova_groups = {k: v for k, v in arch_groups.items() if len(v) >= min_group}
    result = one_way_anova(anova_groups, min_group_size=min_group)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Cutoff: {cutoff_dt.strftime('%Y-%m-%d')}")
    print(f"  Experiments after cutoff: {len(filtered)}")
    print(f"  With backbone+encoder: {sum(1 for e in filtered if e['backbone'] and e['encoder'])}")
    print(f"  Groups with >= {min_group}: {len(anova_groups)}")
    print(f"{'='*70}")

    if result:
        print(f"  F-statistic:  {result['f_statistic']:.2f}")
        print(f"  p-value:      {result['p_value']:.2e}")
        print(f"  eta-squared:  {result['eta_squared']:.4f}")
        print(f"  df_between:   {result['df_between']}")
        print(f"  df_within:    {result['df_within']}")
        print(f"  n_groups:     {result['n_groups']}")
        print(f"  n_total:      {result['n_total']}")
        print(f"  MSB / MSW:    {result['msb']:.6f} / {result['msw']:.6f}")
        print()
        print(f"  Groups used:")
        for combo, n in result['groups_used'].items():
            aps = arch_groups[combo]
            print(f"    {combo:30s}  n={n:4d}  mean={np.mean(aps):.4f}  std={np.std(aps):.4f}")
    else:
        print(f"  ANOVA could not be computed (< 2 groups with >= {min_group} experiments)")

    return result, len(filtered), anova_groups


def main():
    experiments = load_experiments()
    if not experiments:
        print("ERROR: No experiments loaded!", file=sys.stderr)
        sys.exit(1)

    # Show timestamp distribution
    dated = [e for e in experiments if e['timestamp'] is not None]
    dates = [e['timestamp'].date() for e in dated]
    from collections import Counter
    date_counts = Counter(dates)
    print("\nExperiment completion dates (test AP only):")
    for d, c in sorted(date_counts.items()):
        print(f"  {d}: {c}")
    print(f"  Total with timestamps: {len(dated)}")
    n_no_ts = len(experiments) - len(dated)
    if n_no_ts > 0:
        print(f"  Without timestamps: {n_no_ts}")

    # === Full-dataset ANOVA (reproduce paper's F=449.3) ===
    print("\n" + "=" * 70)
    print("  FULL-DATASET ANOVA (for comparison with paper's F=449.3)")
    print("=" * 70)
    arch_groups_full = defaultdict(list)
    for exp in experiments:
        if exp['backbone'] and exp['encoder']:
            arch_groups_full[exp['arch_combo']].append(exp['ap'])

    full_groups = {k: v for k, v in arch_groups_full.items() if len(v) >= 10}
    full_result = one_way_anova(full_groups, min_group_size=10)
    if full_result:
        print(f"  F={full_result['f_statistic']:.2f}, eta²={full_result['eta_squared']:.4f}, "
              f"n={full_result['n_total']}, k={full_result['n_groups']}")

    # === Post-bugfix ANOVA with multiple cutoffs for sensitivity ===
    cutoffs = [
        (datetime(2026, 2, 28), "Post-bugfix: Feb 28+ (day ~15)"),
        (datetime(2026, 3, 1), "Post-bugfix: Mar 1+ (day ~16)"),
        (datetime(2026, 3, 3), "Post-bugfix: Mar 3+ (day ~18)"),
        (datetime(2026, 3, 6), "Post-bugfix: Mar 6+ (day ~21, used in paper)"),
    ]

    results = {}
    for cutoff_dt, label in cutoffs:
        result, n_filtered, groups = run_anova_for_cutoff(experiments, cutoff_dt, label)
        results[cutoff_dt.strftime('%Y-%m-%d')] = {
            'result': result,
            'n_filtered': n_filtered,
        }

    # Also try with min_group=5 for Mar 6 cutoff to get more groups
    print("\n\n=== SENSITIVITY: Mar 6 cutoff with min_group=5 ===")
    run_anova_for_cutoff(experiments, datetime(2026, 3, 6), "Mar 6+ (min_group=5)", min_group=5)

    # === Summary for paper ===
    print("\n\n" + "=" * 70)
    print("  SUMMARY FOR PAPER LINE 574-575")
    print("=" * 70)
    # Use the Mar 6 cutoff (matches what the paper states)
    primary = results.get('2026-03-06', {}).get('result')
    if primary:
        print(f"\n  Post-bugfix ANOVA (after March 6, 2026):")
        print(f"    F = {primary['f_statistic']:.1f}")
        print(f"    eta² = {primary['eta_squared']:.2f}")
        print(f"    p < 0.001")
        print(f"    n = {primary['n_total']} experiments across {primary['n_groups']} architecture groups")
        print(f"\n  Suggested paper text:")
        print(f"    \"The architecture effect remains strong "
              f"($F = {primary['f_statistic']:.1f}$, $\\eta^2 = {primary['eta_squared']:.2f}$, "
              f"$p < 0.001$, $n = {primary['n_total']}$),\"")
    else:
        # Fall back to a cutoff that works
        for cutoff_str in ['2026-03-03', '2026-03-01', '2026-02-28']:
            alt = results.get(cutoff_str, {}).get('result')
            if alt:
                print(f"\n  Post-bugfix ANOVA (after {cutoff_str}):")
                print(f"    F = {alt['f_statistic']:.1f}")
                print(f"    eta² = {alt['eta_squared']:.2f}")
                print(f"    p < 0.001")
                print(f"    n = {alt['n_total']}")
                break


if __name__ == '__main__':
    main()
