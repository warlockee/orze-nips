#!/usr/bin/env python3
"""
Script 4: Compute ANOVA decomposition (Section 5.5).

- For each backbone+encoder combination, compute mean/variance of AP
- Run one-way ANOVA: between-group vs within-group variance
- Compute F-statistic, p-value, eta-squared
- Compute configuration fingerprint Hamming distances
- Compute validation-test rank correlation

Outputs:
  - doc/computed_values/anova.json
"""

import json
import os
import sys
import glob
import yaml
import numpy as np
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'computed_values'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def get_config_fingerprint(cfg):
    """Create binary fingerprint for Hamming distance analysis."""
    fingerprint = {}

    # Backbone features
    bb = extract_backbone(cfg)
    for bb_name in ['VJepa2', 'DINOv3-B', 'DINOv3-L', 'DINOv2', 'SigLIP2',
                     'InternViT', 'Multi-Backbone']:
        fingerprint[f'bb_{bb_name}'] = 1 if bb == bb_name else 0

    # Encoder features
    enc = normalize_encoder(cfg.get('temporal_encoder', {}).get('type'))
    for enc_name in ['Zipformer', 'Retention', 'BiMamba', 'Hybrid R-M']:
        fingerprint[f'enc_{enc_name}'] = 1 if enc == enc_name else 0

    # Loss features
    loss_cfg = cfg.get('loss', {}).get('classification', {})
    loss_type = loss_cfg.get('type', '').lower()
    fingerprint['loss_focal'] = 1 if 'focal' in loss_type else 0
    fingerprint['loss_bce'] = 1 if 'focal' not in loss_type else 0
    gamma = loss_cfg.get('gamma', 0)
    fingerprint['gamma_high'] = 1 if gamma is not None and float(gamma) >= 2.5 else 0

    # Pooling
    pooling = cfg.get('heads', {}).get('classification', {}).get('pooling', 'mean')
    fingerprint['pool_attention'] = 1 if 'attention' in str(pooling).lower() else 0

    return fingerprint


def hamming_distance(fp1, fp2):
    """Compute Hamming distance between two fingerprints."""
    all_keys = set(fp1.keys()) | set(fp2.keys())
    dist = sum(fp1.get(k, 0) != fp2.get(k, 0) for k in all_keys)
    return dist


def load_experiments():
    """Load experiments with AP, val metric, and configs."""
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

        # Also get val metric
        val_metric = None
        metrics_path = os.path.join(idea_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    m = json.load(f)
                val_metric = m.get('best_val_metric')
            except Exception:
                pass

        backbone = extract_backbone(cfg)
        encoder = normalize_encoder(cfg.get('temporal_encoder', {}).get('type'))
        fingerprint = get_config_fingerprint(cfg)

        # Get timestamp for ordering
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
            'val_metric': float(val_metric) if val_metric is not None else None,
            'backbone': backbone,
            'encoder': encoder,
            'arch_combo': f'{backbone}+{encoder}',
            'fingerprint': fingerprint,
            'claimed_at': claimed_at,
        })

    print(f"Loaded {len(experiments)} experiments", file=sys.stderr)
    return experiments


def one_way_anova(groups):
    """
    Manual one-way ANOVA computation.
    groups: dict mapping group_name -> list of AP values
    """
    # Filter groups with at least 2 observations
    valid_groups = {k: np.array(v) for k, v in groups.items() if len(v) >= 2}
    if len(valid_groups) < 2:
        return None

    all_values = np.concatenate(list(valid_groups.values()))
    grand_mean = np.mean(all_values)
    N = len(all_values)
    k = len(valid_groups)

    # Between-group sum of squares (SSB)
    ssb = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in valid_groups.values())

    # Within-group sum of squares (SSW)
    ssw = sum(np.sum((g - np.mean(g)) ** 2) for g in valid_groups.values())

    # Degrees of freedom
    df_between = k - 1
    df_within = N - k

    # Mean squares
    msb = ssb / df_between if df_between > 0 else 0
    msw = ssw / df_within if df_within > 0 else 1e-10

    # F-statistic
    f_stat = msb / msw

    # p-value using F-distribution
    from scipy.stats import f as f_dist
    p_value = 1 - f_dist.cdf(f_stat, df_between, df_within)

    # Eta-squared (effect size)
    ss_total = ssb + ssw
    eta_squared = ssb / ss_total if ss_total > 0 else 0

    # Variance components
    var_between = msb
    var_within = msw

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
        'var_between': float(var_between),
        'var_within': float(var_within),
        'n_groups': k,
        'n_total': N,
    }


def main():
    experiments = load_experiments()
    if not experiments:
        print("ERROR: No experiments loaded!", file=sys.stderr)
        sys.exit(1)

    # ---- Group by architecture combo ----
    arch_groups = defaultdict(list)
    for exp in experiments:
        if exp['backbone'] and exp['encoder']:
            arch_groups[exp['arch_combo']].append(exp['ap'])

    # Report group sizes
    print("\n% Architecture groups:", file=sys.stderr)
    for combo, aps in sorted(arch_groups.items(), key=lambda x: -len(x[1])):
        if len(aps) >= 10:
            print(f"%   {combo}: n={len(aps)}, mean={np.mean(aps):.4f}, "
                  f"std={np.std(aps):.4f}, best={np.max(aps):.4f}", file=sys.stderr)

    # ---- One-way ANOVA ----
    # Use groups with >= 10 experiments
    anova_groups = {k: v for k, v in arch_groups.items() if len(v) >= 10}
    anova_result = one_way_anova(anova_groups)

    if anova_result:
        print(f"\n% ANOVA: F={anova_result['f_statistic']:.2f}, "
              f"p={anova_result['p_value']:.2e}, "
              f"eta²={anova_result['eta_squared']:.4f}", file=sys.stderr)

    # ---- Per-group statistics ----
    group_stats = {}
    for combo, aps in sorted(arch_groups.items(), key=lambda x: -max(x[1])):
        if len(aps) >= 5:
            group_stats[combo] = {
                'count': len(aps),
                'mean': float(np.mean(aps)),
                'std': float(np.std(aps)),
                'median': float(np.median(aps)),
                'best': float(np.max(aps)),
                'worst': float(np.min(aps)),
            }

    # ---- Configuration fingerprint Hamming distances ----
    # Sort by timestamp for consecutive Hamming distances
    timed = [e for e in experiments if e['claimed_at'] is not None]
    timed.sort(key=lambda x: x['claimed_at'])

    consecutive_hamming = []
    for i in range(1, len(timed)):
        hd = hamming_distance(timed[i - 1]['fingerprint'], timed[i]['fingerprint'])
        consecutive_hamming.append(hd)

    hamming_stats = None
    if consecutive_hamming:
        hd_arr = np.array(consecutive_hamming)
        hamming_stats = {
            'mean': float(np.mean(hd_arr)),
            'std': float(np.std(hd_arr)),
            'median': float(np.median(hd_arr)),
            'frac_categorical_flip': float(np.mean(hd_arr >= 1)),
            'frac_multi_flip': float(np.mean(hd_arr >= 2)),
        }
        print(f"\n% Hamming distance stats: mean={hamming_stats['mean']:.2f}, "
              f"frac_categorical_flip={hamming_stats['frac_categorical_flip']:.2f}",
              file=sys.stderr)

    # ---- Unique combos explored over time ----
    unique_combos_over_time = []
    seen_combos = set()
    for i, exp in enumerate(timed):
        combo = exp['arch_combo']
        if combo and 'None' not in combo:
            seen_combos.add(combo)
        if (i + 1) % max(1, len(timed) // 100) == 0:
            unique_combos_over_time.append({
                't': i + 1,
                'unique_combos': len(seen_combos),
            })

    # ---- Coverage: fraction of config space explored ----
    # Estimate config space size
    n_backbones = len(set(e['backbone'] for e in experiments if e['backbone']))
    n_encoders = len(set(e['encoder'] for e in experiments if e['encoder']))
    # Rough discrete space: backbones * encoders * 2 (loss) * 2 (pooling) = combos
    discrete_space = n_backbones * n_encoders * 2 * 2
    explored = len(set(e['arch_combo'] for e in experiments
                       if e['backbone'] and e['encoder']))
    coverage = explored / max(discrete_space, 1)

    # ---- Validation-test rank correlation ----
    from scipy.stats import spearmanr
    val_test_pairs = [(e['val_metric'], e['ap']) for e in experiments
                      if e['val_metric'] is not None and e['ap'] is not None]
    val_test_corr = None
    if len(val_test_pairs) >= 10:
        vals, tests = zip(*val_test_pairs)
        corr, p_val = spearmanr(vals, tests)
        val_test_corr = {'spearman_rho': float(corr), 'p_value': float(p_val),
                         'n': len(val_test_pairs)}

    # ---- GPU-hours estimation ----
    total_training_time_s = 0
    n_timed = 0
    for idea_dir in glob.glob(os.path.join(RESULTS_DIR, 'idea-*')):
        metrics_path = os.path.join(idea_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    m = json.load(f)
                t = m.get('training_time')
                if t and isinstance(t, (int, float)):
                    total_training_time_s += t
                    n_timed += 1
            except Exception:
                pass
    total_gpu_hours = total_training_time_s / 3600.0

    # ---- Compile output ----
    output = {
        'anova': anova_result,
        'group_stats': group_stats,
        'n_groups_with_10plus': len(anova_groups),
        'hamming_distance': hamming_stats,
        'unique_combos_explored': explored,
        'discrete_space_estimate': discrete_space,
        'coverage_fraction': float(coverage),
        'unique_combos_over_time': unique_combos_over_time[-20:],
        'val_test_correlation': val_test_corr,
        'total_gpu_hours': float(total_gpu_hours),
        'n_experiments_with_time': n_timed,
    }

    # Save JSON
    out_path = os.path.join(OUTPUT_DIR, 'anova.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}", file=sys.stderr)

    # ---- Print key values ----
    print("\n% === ANOVA VALUES FOR PAPER ===")
    if anova_result:
        print(f"% Between architectures variance: {anova_result['var_between']:.6f}")
        print(f"% Within architectures variance: {anova_result['var_within']:.6f}")
        print(f"% F-statistic: {anova_result['f_statistic']:.2f}")
        print(f"% p-value: {anova_result['p_value']:.2e}")
        print(f"% eta-squared: {anova_result['eta_squared']:.4f}")
        print(f"% Variance ratio (between/within): "
              f"{anova_result['var_between'] / anova_result['var_within']:.1f}x")
    print(f"% Coverage: {coverage:.4f} ({explored}/{discrete_space})")
    if val_test_corr:
        print(f"% Val-test Spearman rho: {val_test_corr['spearman_rho']:.4f} "
              f"(p={val_test_corr['p_value']:.2e}, n={val_test_corr['n']})")
    print(f"% Total GPU-hours: {total_gpu_hours:.0f} "
          f"({n_timed} experiments with timing data)")


if __name__ == '__main__':
    main()
