#!/usr/bin/env python3
"""
Compute cost-efficiency deep dive for the Nexar competition campaign.

Analyses:
1. Per-experiment GPU cost (training_time -> GPU-hours)
2. Baseline GPU-hour justification (TPE, BOHB, Random, LLM)
3. Marginal cost per SSC step (VJepa2, Multi-backbone fusion, GRU/LSTM)
4. Cumulative cost curve (GPU-hours vs running-max mAP)

Output: doc/computed_values/cost_efficiency_deep.json
"""

import json
import os
import sys
import glob
import re
import yaml
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NEXAR_RESULTS = os.path.join(BASE_DIR, 'nexar_comp', 'results')
BASELINE_RESULTS = os.path.join(BASE_DIR, 'results')
CACHE_PATH = os.path.join(NEXAR_RESULTS, '_results_cache.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'doc', 'computed_values')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_baseline_experiments(prefix):
    """Load baseline experiments from results/ matching a prefix pattern."""
    pattern = os.path.join(BASELINE_RESULTS, f'{prefix}*')
    dirs = sorted(glob.glob(pattern))

    experiments = []
    for d in dirs:
        metrics_path = os.path.join(d, 'metrics.json')
        if not os.path.exists(metrics_path):
            continue
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except Exception:
            continue

        training_time = metrics.get('training_time', 0)
        if training_time is None:
            training_time = 0

        # Try to get competition mAP from nexar_comp_report.json
        comp_map = None
        report_path = os.path.join(d, 'nexar_comp_report.json')
        if os.path.exists(report_path):
            try:
                with open(report_path) as f:
                    report = json.load(f)
                comp_map = report.get('metrics', {}).get('mAP')
            except Exception:
                pass

        experiments.append({
            'id': os.path.basename(d),
            'training_time': float(training_time),
            'gpu_hours': float(training_time) / 3600.0,
            'competition_mAP': comp_map,
            'best_val_metric': metrics.get('best_val_metric'),
            'timestamp': metrics.get('timestamp'),
        })

    return experiments


def normalize_backbone(name):
    """Normalize backbone name."""
    if name is None:
        return 'unknown'
    name = name.lower().strip()
    if '+' in name or ',' in name or name == 'multi' or 'fusion' in name:
        return 'multi-backbone'
    if 'vjepa' in name:
        return 'vjepa2'
    if 'dinov3' in name:
        return 'dinov3'
    if 'dinov2' in name:
        return 'dinov2'
    if 'siglip' in name:
        return 'siglip2'
    if 'mvit' in name:
        return 'mvitv2'
    if 'videomae' in name:
        return 'videomae_v2'
    return 'other'


def normalize_encoder(enc):
    """Normalize encoder type."""
    if enc is None:
        return 'unknown'
    enc = enc.lower().strip()
    if 'gru' in enc:
        return 'gru'
    if 'lstm' in enc:
        return 'lstm'
    if 'zipformer' in enc:
        return 'zipformer'
    if 'mamba' in enc:
        return 'mamba'
    if 'transformer' in enc:
        return 'transformer'
    return 'other'


def load_backbone_encoder(idea_id):
    """Load backbone and encoder from resolved_config.yaml."""
    config_path = os.path.join(NEXAR_RESULTS, idea_id, 'resolved_config.yaml')
    if not os.path.exists(config_path):
        return None, None
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        try:
            with open(config_path) as f:
                cfg = yaml.unsafe_load(f)
        except Exception:
            return None, None

    backbone = cfg.get('backbone', {}).get('name', 'unknown')
    encoder = cfg.get('temporal_encoder', {}).get('type', 'unknown')
    return backbone, encoder


def main():
    # ---- Load LLM campaign experiments from cache ----
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    print(f"Loaded {len(cache)} LLM experiments from cache", file=sys.stderr)

    llm_experiments = []
    for idea_id, entry in cache.items():
        row = entry.get('row', {})
        values = row.get('values', {})
        metrics = row.get('metrics', {})

        comp_map = values.get('mAP')
        if comp_map is None or comp_map <= 0:
            continue

        timestamp = metrics.get('timestamp')
        training_time = metrics.get('training_time', 0)
        if training_time is None:
            training_time = 0

        backbone, encoder = load_backbone_encoder(idea_id)

        llm_experiments.append({
            'id': idea_id,
            'timestamp': timestamp,
            'mAP': float(comp_map),
            'training_time': float(training_time),
            'gpu_hours': float(training_time) / 3600.0,
            'backbone': backbone,
            'encoder': encoder,
            'backbone_norm': normalize_backbone(backbone),
            'encoder_norm': normalize_encoder(encoder),
        })

    # Sort by timestamp
    llm_experiments.sort(key=lambda x: x.get('timestamp') or '')
    print(f"LLM experiments with mAP: {len(llm_experiments)}", file=sys.stderr)

    # ---- 1. Per-policy cost stats ----
    # TPE baselines
    tpe_exps = load_baseline_experiments('tpe-baseline-')
    # Also check tpe- without baseline
    tpe_exps2 = load_baseline_experiments('tpe-')
    # Deduplicate: tpe-baseline-XXXX overlaps with tpe- prefix
    seen_ids = set(e['id'] for e in tpe_exps)
    for e in tpe_exps2:
        if e['id'] not in seen_ids:
            tpe_exps.append(e)
            seen_ids.add(e['id'])

    bohb_exps = load_baseline_experiments('bohb-')
    random_exps = load_baseline_experiments('random-baseline-')

    def policy_stats(exp_list):
        if not exp_list:
            return {'n': 0, 'total_gpu_hrs': 0, 'mean_per_experiment': 0}
        total = sum(e['gpu_hours'] for e in exp_list)
        return {
            'n': len(exp_list),
            'total_gpu_hrs': round(total, 4),
            'mean_per_experiment': round(total / len(exp_list), 4),
        }

    per_policy = {
        'tpe': policy_stats(tpe_exps),
        'bohb': policy_stats(bohb_exps),
        'random': policy_stats(random_exps),
        'llm': policy_stats(llm_experiments),
    }

    print(f"\nPer-policy stats:", file=sys.stderr)
    for name, s in per_policy.items():
        print(f"  {name}: n={s['n']}, total={s['total_gpu_hrs']:.1f} GPU-hrs, "
              f"mean={s['mean_per_experiment']:.3f} GPU-hrs/exp", file=sys.stderr)

    # ---- 2. Marginal cost per SSC step ----
    # Define SSC events by their characteristics
    # Event 1: VJepa2 backbone discovery
    # Event 2: Multi-backbone fusion
    # Event 3: GRU/LSTM encoders

    # Split experiments into phases based on when features were introduced
    # Find first VJepa2 experiment
    first_vjepa2_idx = None
    first_fusion_idx = None
    first_gru_lstm_idx = None

    for i, e in enumerate(llm_experiments):
        if e['backbone_norm'] == 'vjepa2' and first_vjepa2_idx is None:
            first_vjepa2_idx = i
        if e['backbone_norm'] == 'multi-backbone' and first_fusion_idx is None:
            first_fusion_idx = i
        enc = e['encoder_norm']
        if enc in ('gru', 'lstm') and first_gru_lstm_idx is None:
            first_gru_lstm_idx = i

    print(f"\nSSC event indices (chronological):", file=sys.stderr)
    print(f"  First VJepa2: idx={first_vjepa2_idx}", file=sys.stderr)
    print(f"  First multi-backbone: idx={first_fusion_idx}", file=sys.stderr)
    print(f"  First GRU/LSTM: idx={first_gru_lstm_idx}", file=sys.stderr)

    # For each event, compute:
    # - GPU hours for the event's experiments (all experiments using that feature)
    # - mAP before (running max just before the event)
    # - mAP after (running max including event experiments)
    ssc_events = []

    # Running max up to each point
    running_max = []
    rm = 0.0
    for e in llm_experiments:
        rm = max(rm, e['mAP'])
        running_max.append(rm)

    # Event 1: VJepa2
    # "before" = running max just before the first VJepa2 experiment.
    # If VJepa2 is the very first experiment, use the best non-VJepa2 mAP from
    # experiments in the first 10% of the campaign as the "before" baseline.
    if first_vjepa2_idx is not None:
        vjepa2_exps = [e for e in llm_experiments if e['backbone_norm'] == 'vjepa2']
        vjepa2_gpu_hrs = sum(e['gpu_hours'] for e in vjepa2_exps)
        if first_vjepa2_idx > 0:
            mAP_before = running_max[first_vjepa2_idx - 1]
        else:
            # VJepa2 was one of the first features tried. Use the best non-VJepa2
            # mAP from the early campaign as baseline.
            early_cutoff = max(50, len(llm_experiments) // 10)
            early_non_vjepa2 = [e['mAP'] for e in llm_experiments[:early_cutoff]
                                if e['backbone_norm'] != 'vjepa2']
            mAP_before = max(early_non_vjepa2) if early_non_vjepa2 else 0

        best_vjepa2_mAP = max(e['mAP'] for e in vjepa2_exps)
        gain = best_vjepa2_mAP - mAP_before

        ssc_events.append({
            'event': 'VJepa2 backbone',
            'n_experiments': len(vjepa2_exps),
            'gpu_hrs': round(vjepa2_gpu_hrs, 4),
            'mAP_before': round(mAP_before, 6),
            'mAP_after': round(best_vjepa2_mAP, 6),
            'mAP_gain': round(gain, 6),
            'cost_per_mAP_point': round(vjepa2_gpu_hrs / (gain * 100), 4) if gain > 0 else None,
        })

    # Event 2: Multi-backbone fusion
    # "before" = running max just before the first fusion experiment
    if first_fusion_idx is not None:
        fusion_exps = [e for e in llm_experiments if e['backbone_norm'] == 'multi-backbone']
        fusion_gpu_hrs = sum(e['gpu_hours'] for e in fusion_exps)
        mAP_before = running_max[first_fusion_idx - 1] if first_fusion_idx > 0 else 0
        # "after" = running max at the end of the campaign (fusion contributed to further gains)
        best_fusion_mAP = max(e['mAP'] for e in fusion_exps)
        # Use overall running max after fusion was introduced
        mAP_after = running_max[-1]
        gain = mAP_after - mAP_before

        ssc_events.append({
            'event': 'Multi-backbone fusion',
            'n_experiments': len(fusion_exps),
            'gpu_hrs': round(fusion_gpu_hrs, 4),
            'mAP_before': round(mAP_before, 6),
            'mAP_after': round(mAP_after, 6),
            'mAP_gain': round(gain, 6),
            'cost_per_mAP_point': round(fusion_gpu_hrs / (gain * 100), 4) if gain > 0 else None,
        })

    # Event 3: GRU/LSTM encoders
    # "before" = running max just before the first GRU/LSTM experiment
    if first_gru_lstm_idx is not None:
        gru_lstm_exps = [e for e in llm_experiments if e['encoder_norm'] in ('gru', 'lstm')]
        gru_lstm_gpu_hrs = sum(e['gpu_hours'] for e in gru_lstm_exps)
        if first_gru_lstm_idx > 0:
            mAP_before = running_max[first_gru_lstm_idx - 1]
        else:
            # GRU/LSTM was among the first experiments; use early non-GRU/LSTM as baseline
            early_cutoff = max(50, len(llm_experiments) // 10)
            early_non_gru = [e['mAP'] for e in llm_experiments[:early_cutoff]
                             if e['encoder_norm'] not in ('gru', 'lstm')]
            mAP_before = max(early_non_gru) if early_non_gru else 0

        # "after" = running max at end of campaign
        mAP_after = running_max[-1]
        best_gru_lstm_mAP = max(e['mAP'] for e in gru_lstm_exps)
        gain = mAP_after - mAP_before

        ssc_events.append({
            'event': 'GRU/LSTM encoders',
            'n_experiments': len(gru_lstm_exps),
            'gpu_hrs': round(gru_lstm_gpu_hrs, 4),
            'mAP_before': round(mAP_before, 6),
            'mAP_after': round(mAP_after, 6),
            'mAP_gain': round(gain, 6),
            'cost_per_mAP_point': round(gru_lstm_gpu_hrs / (gain * 100), 4) if gain > 0 else None,
        })

    print(f"\nSSC Events:", file=sys.stderr)
    for ev in ssc_events:
        print(f"  {ev['event']}: {ev['n_experiments']} exps, "
              f"{ev['gpu_hrs']:.1f} GPU-hrs, "
              f"mAP {ev['mAP_before']:.4f} -> {ev['mAP_after']:.4f} "
              f"(+{ev['mAP_gain']:.4f})", file=sys.stderr)

    # ---- 3. Cumulative cost curve ----
    cost_curve = []
    cumulative_gpu_hrs = 0.0
    rm = 0.0
    for e in llm_experiments:
        cumulative_gpu_hrs += e['gpu_hours']
        rm = max(rm, e['mAP'])
        cost_curve.append([round(cumulative_gpu_hrs, 4), round(rm, 6)])

    # Subsample for output (every 10th point + last)
    if len(cost_curve) > 500:
        step = len(cost_curve) // 500
        cost_curve_sparse = cost_curve[::step]
        if cost_curve_sparse[-1] != cost_curve[-1]:
            cost_curve_sparse.append(cost_curve[-1])
    else:
        cost_curve_sparse = cost_curve

    # ---- Compile output ----
    output = {
        'per_policy_cost': per_policy,
        'marginal_ssc_cost': ssc_events,
        'cost_curve': cost_curve_sparse,
        'total_llm_gpu_hrs': round(sum(e['gpu_hours'] for e in llm_experiments), 4),
        'total_baseline_gpu_hrs': round(
            sum(e['gpu_hours'] for e in tpe_exps) +
            sum(e['gpu_hours'] for e in bohb_exps) +
            sum(e['gpu_hours'] for e in random_exps), 4),
    }

    out_path = os.path.join(OUTPUT_DIR, 'cost_efficiency_deep.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}", file=sys.stderr)

    # Print summary
    print("\n=== Cost Efficiency Summary ===", file=sys.stderr)
    print(f"Total LLM campaign: {output['total_llm_gpu_hrs']:.1f} GPU-hrs "
          f"({len(llm_experiments)} experiments)", file=sys.stderr)
    print(f"Total baselines: {output['total_baseline_gpu_hrs']:.1f} GPU-hrs", file=sys.stderr)
    print(f"Final running-max mAP: {cost_curve[-1][1] if cost_curve else 0:.4f}", file=sys.stderr)


if __name__ == '__main__':
    main()
