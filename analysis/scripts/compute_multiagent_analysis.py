#!/usr/bin/env python3
"""
Compute multi-agent quantitative analysis for the Nexar competition campaign.

Analyses:
1. Agent ID mapping (Claude vs Gemini)
2. Per-agent running-max competition mAP curves
3. Per-agent statistics table
4. Single-agent counterfactual comparison
5. JSD-innovation correlation

Output: doc/computed_values/multiagent_quantitative.json
"""

import json
import os
import sys
import glob
import re
import yaml
import numpy as np
from collections import defaultdict
from datetime import datetime
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NEXAR_RESULTS = os.path.join(BASE_DIR, 'nexar_comp', 'results')
CACHE_PATH = os.path.join(NEXAR_RESULTS, '_results_cache.json')
CLAUDE_LOGS = os.path.join(NEXAR_RESULTS, '_research_claude_logs')
ANTHROPIC_LOGS = os.path.join(NEXAR_RESULTS, '_research_anthropic_logs')
GEMINI_LOGS = os.path.join(NEXAR_RESULTS, '_research_gemini_logs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'doc', 'computed_values')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_agent_mapping():
    """Parse research logs to map idea IDs -> agent (claude/gemini)."""
    idea_to_agent = {}

    for agent_name, log_dir in [('claude', CLAUDE_LOGS), ('claude', ANTHROPIC_LOGS), ('gemini', GEMINI_LOGS)]:
        if not os.path.isdir(log_dir):
            print(f"Warning: {log_dir} not found", file=sys.stderr)
            continue

        log_files = sorted(glob.glob(os.path.join(log_dir, 'cycle_*.log')))
        for log_file in log_files:
            try:
                with open(log_file) as f:
                    content = f.read()
            except Exception:
                continue

            # Extract idea IDs: lines like "  idea-XXXXXX: Title..."
            for m in re.finditer(r'(idea-[a-f0-9]+):', content):
                idea_id = m.group(1)
                if idea_id not in idea_to_agent:
                    idea_to_agent[idea_id] = agent_name

    print(f"Agent mapping: {len(idea_to_agent)} base ideas "
          f"(claude={sum(1 for v in idea_to_agent.values() if v == 'claude')}, "
          f"gemini={sum(1 for v in idea_to_agent.values() if v == 'gemini')})",
          file=sys.stderr)
    return idea_to_agent


def resolve_agent(idea_id, idea_to_agent):
    """Resolve agent for an experiment ID, including -ht- variants."""
    if idea_id in idea_to_agent:
        return idea_to_agent[idea_id]
    # Strip -ht-N suffix
    base = re.sub(r'-ht-\d+$', '', idea_id)
    return idea_to_agent.get(base, None)


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


def normalize_backbone(name):
    """Normalize backbone name for grouping."""
    if name is None:
        return 'unknown'
    name = name.lower().strip()
    # Multi-backbone detection
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


def compute_jsd(p_dict, q_dict):
    """Jensen-Shannon divergence between two distributions."""
    all_keys = set(p_dict.keys()) | set(q_dict.keys())
    if not all_keys:
        return 0.0
    p = np.array([p_dict.get(k, 0) for k in all_keys], dtype=float)
    q = np.array([q_dict.get(k, 0) for k in all_keys], dtype=float)

    if p.sum() == 0 or q.sum() == 0:
        return 0.0

    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def main():
    # Load results cache
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    print(f"Loaded {len(cache)} experiments from cache", file=sys.stderr)

    # Build agent mapping
    idea_to_agent = parse_agent_mapping()

    # Build experiment list: (idea_id, timestamp, mAP, agent, backbone, encoder)
    experiments = []
    for idea_id, entry in cache.items():
        row = entry.get('row', {})
        values = row.get('values', {})
        metrics = row.get('metrics', {})

        comp_map = values.get('mAP')
        if comp_map is None or comp_map <= 0:
            continue

        timestamp = metrics.get('timestamp')
        if not timestamp:
            continue

        training_time = metrics.get('training_time', 0)
        agent = resolve_agent(idea_id, idea_to_agent)
        if agent is None:
            continue  # Skip unattributed experiments

        backbone, encoder = load_backbone_encoder(idea_id)

        experiments.append({
            'idea_id': idea_id,
            'timestamp': timestamp,
            'mAP': float(comp_map),
            'agent': agent,
            'backbone': backbone,
            'encoder': encoder,
            'backbone_norm': normalize_backbone(backbone),
            'training_time': float(training_time) if training_time else 0,
        })

    # Sort chronologically
    experiments.sort(key=lambda x: x['timestamp'])
    print(f"Total agent-attributed experiments with mAP: {len(experiments)}", file=sys.stderr)
    print(f"  claude: {sum(1 for e in experiments if e['agent'] == 'claude')}", file=sys.stderr)
    print(f"  gemini: {sum(1 for e in experiments if e['agent'] == 'gemini')}", file=sys.stderr)

    # ---- 1. Per-agent running-max competition mAP ----
    claude_exps = [e for e in experiments if e['agent'] == 'claude']
    gemini_exps = [e for e in experiments if e['agent'] == 'gemini']

    def running_max_curve(exp_list):
        curve = []
        best = 0.0
        for i, e in enumerate(exp_list):
            best = max(best, e['mAP'])
            curve.append([i + 1, round(best, 6)])
        return curve

    claude_curve = running_max_curve(claude_exps)
    gemini_curve = running_max_curve(gemini_exps)
    combined_curve = running_max_curve(experiments)

    # ---- 2. Per-agent statistics ----
    def agent_stats(exp_list, all_exps_sorted):
        if not exp_list:
            return {}

        n = len(exp_list)
        combos = set()
        first_vjepa2 = None
        running_best = 0.0
        innovations = 0

        for i, e in enumerate(exp_list):
            combo = (e['backbone_norm'], e.get('encoder', 'unknown'))
            combos.add(combo)

            if e['backbone_norm'] == 'vjepa2' and first_vjepa2 is None:
                first_vjepa2 = i + 1  # 1-indexed

            if e['mAP'] > running_best:
                running_best = e['mAP']
                innovations += 1

        return {
            'n_experiments': n,
            'n_unique_combos': len(combos),
            'best_mAP': round(max(e['mAP'] for e in exp_list), 6),
            'first_vjepa2_trial': first_vjepa2,
            'innovation_rate': round(innovations / n, 6) if n > 0 else 0,
        }

    claude_stats = agent_stats(claude_exps, experiments)
    gemini_stats = agent_stats(gemini_exps, experiments)

    # ---- 3. Single-agent counterfactual ----
    def n_to_peak(curve):
        if not curve:
            return None
        peak = max(c[1] for c in curve)
        for n, v in curve:
            if v >= peak:
                return n
        return len(curve)

    synergy = {
        'combined_peak': round(max(c[1] for c in combined_curve), 6) if combined_curve else 0,
        'claude_only_peak': round(max(c[1] for c in claude_curve), 6) if claude_curve else 0,
        'gemini_only_peak': round(max(c[1] for c in gemini_curve), 6) if gemini_curve else 0,
        'combined_n_to_peak': n_to_peak(combined_curve),
        'claude_n_to_peak': n_to_peak(claude_curve),
        'gemini_n_to_peak': n_to_peak(gemini_curve),
    }

    # ---- 4. JSD-innovation correlation ----
    # Sliding windows of 100 experiments
    window_size = 100
    step = 25
    windows = []
    combined_running_max = 0.0

    # Precompute per-experiment: is it an innovation?
    is_innovation = []
    rm = 0.0
    for e in experiments:
        if e['mAP'] > rm:
            rm = e['mAP']
            is_innovation.append(True)
        else:
            is_innovation.append(False)

    for start in range(0, len(experiments) - window_size + 1, step):
        end = start + window_size
        window = experiments[start:end]
        window_innovations = is_innovation[start:end]

        # Backbone distributions per agent in this window
        claude_bb = defaultdict(int)
        gemini_bb = defaultdict(int)
        for e in window:
            bb = e['backbone_norm']
            if e['agent'] == 'claude':
                claude_bb[bb] += 1
            else:
                gemini_bb[bb] += 1

        jsd = compute_jsd(dict(claude_bb), dict(gemini_bb))
        innov_rate = sum(window_innovations) / window_size

        center = start + window_size // 2
        windows.append({
            'center': center,
            'jsd': round(jsd, 6),
            'innovation_rate': round(innov_rate, 6),
        })

    # Spearman correlation
    spearman_rho = None
    p_value = None
    if len(windows) > 3:
        jsd_vals = [w['jsd'] for w in windows]
        innov_vals = [w['innovation_rate'] for w in windows]
        rho, pval = stats.spearmanr(jsd_vals, innov_vals)
        spearman_rho = round(float(rho), 6) if not np.isnan(rho) else None
        p_value = round(float(pval), 6) if not np.isnan(pval) else None

    # ---- Compile output ----
    output = {
        'per_agent': {
            'claude': claude_stats,
            'gemini': gemini_stats,
        },
        'running_max_curves': {
            'claude': claude_curve,
            'gemini': gemini_curve,
            'combined': combined_curve,
        },
        'synergy': synergy,
        'jsd_innovation_correlation': {
            'spearman_rho': spearman_rho,
            'p_value': p_value,
            'n_windows': len(windows),
            'window_size': window_size,
            'windows': windows,
        },
    }

    out_path = os.path.join(OUTPUT_DIR, 'multiagent_quantitative.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}", file=sys.stderr)

    # Print summary
    print("\n=== Multi-Agent Quantitative Summary ===", file=sys.stderr)
    print(f"Claude: {claude_stats.get('n_experiments', 0)} exps, "
          f"best mAP={claude_stats.get('best_mAP', 0):.4f}, "
          f"innovation rate={claude_stats.get('innovation_rate', 0):.4f}", file=sys.stderr)
    print(f"Gemini: {gemini_stats.get('n_experiments', 0)} exps, "
          f"best mAP={gemini_stats.get('best_mAP', 0):.4f}, "
          f"innovation rate={gemini_stats.get('innovation_rate', 0):.4f}", file=sys.stderr)
    print(f"Combined peak: {synergy['combined_peak']:.4f} (n={synergy['combined_n_to_peak']})", file=sys.stderr)
    print(f"Claude-only peak: {synergy['claude_only_peak']:.4f} (n={synergy['claude_n_to_peak']})", file=sys.stderr)
    print(f"Gemini-only peak: {synergy['gemini_only_peak']:.4f} (n={synergy['gemini_n_to_peak']})", file=sys.stderr)
    if spearman_rho is not None:
        print(f"JSD-innovation Spearman rho={spearman_rho:.4f}, p={p_value:.4f}", file=sys.stderr)


if __name__ == '__main__':
    main()
