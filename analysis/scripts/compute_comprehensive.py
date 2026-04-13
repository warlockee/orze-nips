#!/usr/bin/env python3
"""
Comprehensive analysis script addressing gaps in existing analysis.

Analyses:
  1. Post-bugfix ANOVA (experiments after 2026-03-06)
  2. Test AP for top configs (from ken_test_report.json)
  3. Updated convergence with file-mtime timestamps
  4. Full-data ANOVA (backbone, encoder, backbone×encoder)
  5. Agent attribution (Claude vs Gemini vs Unknown)
  6. Nexar competition context (public/private mAP)

Outputs:
  - doc/computed_values/comprehensive.json
  - Human-readable summary to stdout
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
from pathlib import Path

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'computed_values'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUGFIX_CUTOFF = datetime(2026, 3, 6)


# ── Normalization helpers ──────────────────────────────────────────────

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


# ── ANOVA helper ───────────────────────────────────────────────────────

def one_way_anova(groups, min_group_size=2):
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
        'eta_squared': float(eta_squared),
        'n_groups': k,
        'n_total': N,
        'groups_used': {k2: len(v2) for k2, v2 in valid_groups.items()},
    }


# ── Agent mapping from research logs ──────────────────────────────────

def build_agent_map():
    """Parse research logs to map idea IDs → agent (claude/gemini)."""
    agent_map = {}

    # Claude (anthropic) logs
    claude_dir = os.path.join(RESULTS_DIR, '_research_logs')
    if os.path.isdir(claude_dir):
        for logfile in glob.glob(os.path.join(claude_dir, '*.log')):
            try:
                with open(logfile) as f:
                    text = f.read()
                # Check it's actually anthropic
                if '(anthropic)' in text or 'claude' in text.lower() or 'Anthropic' in text:
                    for m in re.finditer(r'idea-([a-f0-9]+)', text):
                        agent_map[f'idea-{m.group(1)}'] = 'Claude'
            except Exception:
                pass

    # Gemini logs
    gemini_dir = os.path.join(RESULTS_DIR, '_research_gemini_logs')
    if os.path.isdir(gemini_dir):
        for logfile in glob.glob(os.path.join(gemini_dir, '*.log')):
            try:
                with open(logfile) as f:
                    text = f.read()
                if '(gemini)' in text or 'Gemini' in text:
                    for m in re.finditer(r'idea-([a-f0-9]+)', text):
                        agent_map[f'idea-{m.group(1)}'] = 'Gemini'
            except Exception:
                pass

    return agent_map


# ── Main data loading ─────────────────────────────────────────────────

def load_all_experiments():
    """Load ALL experiments with maximum data coverage."""
    experiments = []
    idea_dirs = glob.glob(os.path.join(RESULTS_DIR, 'idea-*'))
    stats = {
        'total_dirs': len(idea_dirs),
        'has_metrics_json': 0,
        'has_ken_test': 0,
        'has_config': 0,
        'has_claim': 0,
        'has_any_ap': 0,
        'skipped_no_ap': 0,
        'skipped_parse_error': 0,
    }

    print(f"Scanning {len(idea_dirs)} result directories...", file=sys.stderr)

    for idea_dir in idea_dirs:
        idea_id = os.path.basename(idea_dir)

        # --- metrics.json ---
        val_metric = None
        metrics_timestamp = None
        training_time = None
        metrics_path = os.path.join(idea_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            stats['has_metrics_json'] += 1
            try:
                with open(metrics_path) as f:
                    mdata = json.load(f)
                val_metric = mdata.get('best_val_metric')
                training_time = mdata.get('training_time')
                ts = mdata.get('timestamp')
                if ts:
                    metrics_timestamp = ts
            except Exception:
                stats['skipped_parse_error'] += 1
                continue
            # File mtime as fallback timestamp
            try:
                mtime = os.path.getmtime(metrics_path)
                mtime_dt = datetime.fromtimestamp(mtime)
            except Exception:
                mtime_dt = None
        else:
            mtime_dt = None

        # --- ken_test_report.json ---
        test_ap = None
        test_metrics_full = None
        ken_path = os.path.join(idea_dir, 'ken_test_report.json')
        if os.path.exists(ken_path):
            stats['has_ken_test'] += 1
            try:
                with open(ken_path) as f:
                    ken_data = json.load(f)
                km = ken_data.get('metrics', {})
                test_ap = km.get('average_precision') or km.get('AP') or km.get('mAP')
                test_metrics_full = km
            except Exception:
                pass

        # --- resolved_config.yaml ---
        config = None
        backbone = None
        encoder = None
        loss_type = None
        pooling = None
        focal_gamma = None
        lr = None
        config_path = os.path.join(idea_dir, 'resolved_config.yaml')
        if os.path.exists(config_path):
            stats['has_config'] += 1
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                if config:
                    backbone = extract_backbone(config)
                    encoder = normalize_encoder(config.get('temporal_encoder', {}).get('type'))
                    loss_cfg = config.get('loss', {}).get('classification', {})
                    loss_type = loss_cfg.get('type', 'unknown')
                    focal_gamma = loss_cfg.get('gamma')
                    pooling = config.get('heads', {}).get('classification', {}).get('pooling', 'unknown')
                    lr = config.get('optimizer', {}).get('lr')
            except Exception:
                pass

        # --- claim.json ---
        claimed_at = None
        claimed_by = None
        claim_path = os.path.join(idea_dir, 'claim.json')
        if os.path.exists(claim_path):
            stats['has_claim'] += 1
            try:
                with open(claim_path) as f:
                    claim = json.load(f)
                claimed_at = claim.get('claimed_at')
                claimed_by = claim.get('claimed_by')
            except Exception:
                pass

        # --- Determine best available AP ---
        # Primary: test AP from ken_test_report. Fallback: val metric from metrics.json.
        primary_ap = test_ap if test_ap is not None else val_metric
        if primary_ap is None or not isinstance(primary_ap, (int, float)):
            stats['skipped_no_ap'] += 1
            continue
        try:
            if np.isnan(primary_ap) or np.isinf(primary_ap):
                stats['skipped_no_ap'] += 1
                continue
        except Exception:
            stats['skipped_no_ap'] += 1
            continue

        stats['has_any_ap'] += 1

        experiments.append({
            'idea_id': idea_id,
            'val_metric': float(val_metric) if val_metric is not None else None,
            'test_ap': float(test_ap) if test_ap is not None else None,
            'primary_ap': float(primary_ap),
            'backbone': backbone,
            'encoder': encoder,
            'loss_type': loss_type,
            'pooling': pooling,
            'focal_gamma': focal_gamma,
            'lr': lr,
            'training_time': training_time,
            'claimed_at': claimed_at,
            'claimed_by': claimed_by,
            'mtime': mtime_dt.isoformat() if mtime_dt else None,
            'mtime_epoch': mtime_dt.timestamp() if mtime_dt else None,
            'test_metrics_full': test_metrics_full,
        })

    return experiments, stats


# ── Analysis 1: Post-Bugfix ANOVA ─────────────────────────────────────

def analysis_post_bugfix_anova(experiments):
    """ANOVA restricted to experiments after 2026-03-06."""
    post_bugfix = []
    for exp in experiments:
        ts = exp.get('mtime') or exp.get('claimed_at')
        if ts is None:
            continue
        try:
            dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
            if dt >= BUGFIX_CUTOFF:
                post_bugfix.append(exp)
        except Exception:
            continue

    # Group by backbone+encoder
    arch_groups = defaultdict(list)
    for exp in post_bugfix:
        if exp['backbone'] and exp['encoder']:
            combo = f"{exp['backbone']}+{exp['encoder']}"
            arch_groups[combo].append(exp['primary_ap'])

    anova_result = one_way_anova({k: v for k, v in arch_groups.items() if len(v) >= 5}, min_group_size=5)

    # Also do full-dataset ANOVA for comparison
    full_arch_groups = defaultdict(list)
    for exp in experiments:
        if exp['backbone'] and exp['encoder']:
            combo = f"{exp['backbone']}+{exp['encoder']}"
            full_arch_groups[combo].append(exp['primary_ap'])

    full_anova = one_way_anova({k: v for k, v in full_arch_groups.items() if len(v) >= 5}, min_group_size=5)

    group_stats = {}
    for combo, aps in sorted(arch_groups.items(), key=lambda x: -np.mean(x[1])):
        if len(aps) >= 5:
            group_stats[combo] = {
                'count': len(aps),
                'mean': float(np.mean(aps)),
                'std': float(np.std(aps)),
                'best': float(np.max(aps)),
            }

    return {
        'n_post_bugfix': len(post_bugfix),
        'n_with_arch': sum(1 for e in post_bugfix if e['backbone'] and e['encoder']),
        'post_bugfix_anova': anova_result,
        'full_dataset_anova': full_anova,
        'post_bugfix_group_stats': group_stats,
    }


# ── Analysis 2: Test AP for Top Configs ────────────────────────────────

def analysis_test_ap_top_configs(experiments):
    """Report test AP for top-10 by val AP, compute val-test correlation."""
    from scipy.stats import spearmanr

    # Experiments with both val and test AP
    both = [e for e in experiments if e['val_metric'] is not None and e['test_ap'] is not None]

    # Top 10 by val metric (exclude obviously-leaked: val=1.0 with no config)
    valid_val = [e for e in experiments
                 if e['val_metric'] is not None
                 and not (e['val_metric'] >= 0.999 and e['backbone'] is None)]
    by_val = sorted(valid_val, key=lambda x: -x['val_metric'])
    top10 = []
    for exp in by_val[:10]:
        top10.append({
            'idea_id': exp['idea_id'],
            'val_ap': exp['val_metric'],
            'test_ap': exp['test_ap'],
            'backbone': exp['backbone'],
            'encoder': exp['encoder'],
            'loss_type': exp['loss_type'],
        })

    # Val-test Spearman correlation
    val_test_corr = None
    if len(both) >= 10:
        vals = [e['val_metric'] for e in both]
        tests = [e['test_ap'] for e in both]
        rho, pval = spearmanr(vals, tests)
        val_test_corr = {
            'spearman_rho': float(rho),
            'p_value': float(pval),
            'n': len(both),
        }

    # Test AP: VJepa2 vs others
    vjepa_test = [e['test_ap'] for e in experiments if e['test_ap'] is not None and e['backbone'] == 'VJepa2']
    other_test = [e['test_ap'] for e in experiments if e['test_ap'] is not None and e['backbone'] is not None and e['backbone'] != 'VJepa2']

    def ci95(arr):
        arr = np.array(arr)
        mean = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(len(arr))
        return float(mean - 1.96 * se), float(mean + 1.96 * se)

    vjepa_stats = None
    if vjepa_test:
        lo, hi = ci95(vjepa_test) if len(vjepa_test) > 1 else (np.mean(vjepa_test), np.mean(vjepa_test))
        vjepa_stats = {
            'n': len(vjepa_test),
            'mean': float(np.mean(vjepa_test)),
            'std': float(np.std(vjepa_test)),
            'ci95': [lo, hi],
            'best': float(np.max(vjepa_test)),
        }

    other_stats = None
    if other_test:
        lo, hi = ci95(other_test) if len(other_test) > 1 else (np.mean(other_test), np.mean(other_test))
        other_stats = {
            'n': len(other_test),
            'mean': float(np.mean(other_test)),
            'std': float(np.std(other_test)),
            'ci95': [lo, hi],
            'best': float(np.max(other_test)),
        }

    # Top 10 by test AP
    by_test = sorted([e for e in experiments if e['test_ap'] is not None],
                     key=lambda x: -x['test_ap'])
    top10_test = []
    for exp in by_test[:10]:
        top10_test.append({
            'idea_id': exp['idea_id'],
            'test_ap': exp['test_ap'],
            'val_ap': exp['val_metric'],
            'backbone': exp['backbone'],
            'encoder': exp['encoder'],
        })

    return {
        'n_with_test_ap': len([e for e in experiments if e['test_ap'] is not None]),
        'n_with_both': len(both),
        'top10_by_val': top10,
        'top10_by_test': top10_test,
        'val_test_correlation': val_test_corr,
        'vjepa2_test_stats': vjepa_stats,
        'other_backbones_test_stats': other_stats,
    }


# ── Analysis 3: Convergence with Better Timestamps ────────────────────

def _run_convergence_on_subset(subset, label):
    """Run convergence metrics on a time-sorted subset of experiments."""
    if not subset:
        return {'error': f'No experiments in {label}'}

    # Running best AP
    running_best = []
    best_so_far = -1.0
    for exp in subset:
        if exp['primary_ap'] > best_so_far:
            best_so_far = exp['primary_ap']
        running_best.append(best_so_far)

    # AP@N
    checkpoints = [100, 500, 1000, 5000, 10000, 20000]
    ap_at_n = {}
    for n in checkpoints:
        if n <= len(running_best):
            ap_at_n[f'AP@{n}'] = float(running_best[n - 1])
        else:
            ap_at_n[f'AP@{n}'] = float(running_best[-1]) if running_best else None

    best_final = running_best[-1]

    # Power law fit: gap(n) = best_final - best(n) ~ b * n^(-c)
    fit_result = None
    try:
        xs, ys = [], []
        for i, rb in enumerate(running_best):
            gap = best_final - rb
            if gap > 1e-8 and (i + 1) >= 10:
                xs.append(np.log(i + 1))
                ys.append(np.log(gap))
        if len(xs) > 20:
            xs, ys = np.array(xs), np.array(ys)
            coeffs = np.polyfit(xs, ys, 1)
            fit_result = {
                'power_law_exponent': float(-coeffs[0]),
                'b_coefficient': float(np.exp(coeffs[1])),
                'best_final': float(best_final),
                'n_points_for_fit': len(xs),
            }
    except Exception as e:
        fit_result = {'error': str(e)}

    # LLM ordering vs random permutation
    n_random_trials = 200
    n_check = min(1000, len(running_best))
    llm_ap = running_best[n_check - 1]

    random_aps = []
    rng = np.random.RandomState(42)
    primary_aps = [e['primary_ap'] for e in subset]
    for _ in range(n_random_trials):
        perm = rng.permutation(len(primary_aps))
        best_rand = -1.0
        for idx in range(n_check):
            if primary_aps[perm[idx]] > best_rand:
                best_rand = primary_aps[perm[idx]]
        random_aps.append(best_rand)

    llm_vs_random = {
        'n_check': n_check,
        'llm_ap': float(llm_ap),
        'random_mean': float(np.mean(random_aps)),
        'random_std': float(np.std(random_aps)),
        'llm_advantage': float(llm_ap - np.mean(random_aps)),
        'llm_percentile': float(np.mean(np.array(random_aps) < llm_ap) * 100),
    }

    return {
        'n_experiments': len(subset),
        'ap_at_n': ap_at_n,
        'power_law_fit': fit_result,
        'llm_vs_random': llm_vs_random,
        'best_ap': float(best_final),
        'first_time': subset[0]['mtime'],
        'last_time': subset[-1]['mtime'],
    }


def analysis_convergence(experiments):
    """Power law fit and AP@N — on full data AND clean (post-bugfix) subset."""
    # Full dataset by mtime
    timed_all = sorted(
        [e for e in experiments if e['mtime_epoch'] is not None],
        key=lambda x: x['mtime_epoch'],
    )

    # Clean subset: post-bugfix only (excludes leaked val=1.0 experiments)
    timed_clean = sorted(
        [e for e in experiments
         if e['mtime_epoch'] is not None
         and ((e.get('mtime') and e['mtime'] >= BUGFIX_CUTOFF.isoformat())
              or (e.get('claimed_at') and e['claimed_at'] >= BUGFIX_CUTOFF.isoformat()))],
        key=lambda x: x['mtime_epoch'],
    )

    # Test-AP-only subset (most trustworthy)
    timed_test = sorted(
        [e for e in experiments
         if e['mtime_epoch'] is not None and e['test_ap'] is not None],
        key=lambda x: x['mtime_epoch'],
    )

    return {
        'full_dataset': _run_convergence_on_subset(timed_all, 'full'),
        'post_bugfix': _run_convergence_on_subset(timed_clean, 'post_bugfix'),
        'test_ap_only': _run_convergence_on_subset(timed_test, 'test_ap_only'),
    }


# ── Analysis 4: Full ANOVA ────────────────────────────────────────────

def analysis_full_anova(experiments):
    """ANOVA on full data: backbone only, encoder only, backbone×encoder."""
    with_config = [e for e in experiments if e['backbone'] is not None and e['encoder'] is not None]

    # Backbone only
    bb_groups = defaultdict(list)
    for exp in with_config:
        bb_groups[exp['backbone']].append(exp['primary_ap'])
    bb_anova = one_way_anova({k: v for k, v in bb_groups.items() if len(v) >= 10})

    # Encoder only
    enc_groups = defaultdict(list)
    for exp in with_config:
        enc_groups[exp['encoder']].append(exp['primary_ap'])
    enc_anova = one_way_anova({k: v for k, v in enc_groups.items() if len(v) >= 10})

    # Backbone × Encoder
    combo_groups = defaultdict(list)
    for exp in with_config:
        combo = f"{exp['backbone']}+{exp['encoder']}"
        combo_groups[combo].append(exp['primary_ap'])
    combo_anova = one_way_anova({k: v for k, v in combo_groups.items() if len(v) >= 10})

    # Summary stats per group
    bb_stats = {}
    for name, aps in sorted(bb_groups.items(), key=lambda x: -np.mean(x[1])):
        bb_stats[name] = {
            'count': len(aps),
            'mean': float(np.mean(aps)),
            'std': float(np.std(aps)),
            'best': float(np.max(aps)),
        }

    enc_stats = {}
    for name, aps in sorted(enc_groups.items(), key=lambda x: -np.mean(x[1])):
        enc_stats[name] = {
            'count': len(aps),
            'mean': float(np.mean(aps)),
            'std': float(np.std(aps)),
            'best': float(np.max(aps)),
        }

    return {
        'n_with_config_and_ap': len(with_config),
        'backbone_anova': bb_anova,
        'encoder_anova': enc_anova,
        'backbone_x_encoder_anova': combo_anova,
        'backbone_stats': bb_stats,
        'encoder_stats': enc_stats,
    }


# ── Analysis 5: Agent Attribution ─────────────────────────────────────

def analysis_agent_attribution(experiments, agent_map):
    """Count experiments by agent with performance stats."""
    # Assign agents
    for exp in experiments:
        exp['agent'] = agent_map.get(exp['idea_id'], 'Unknown')

    agent_stats = defaultdict(lambda: {'count': 0, 'aps': [], 'backbones': defaultdict(int)})
    for exp in experiments:
        a = exp['agent']
        agent_stats[a]['count'] += 1
        agent_stats[a]['aps'].append(exp['primary_ap'])
        if exp['backbone']:
            agent_stats[a]['backbones'][exp['backbone']] += 1

    result = {}
    for agent, data in sorted(agent_stats.items()):
        aps = np.array(data['aps'])
        result[agent] = {
            'count': data['count'],
            'mean_ap': float(np.mean(aps)),
            'std_ap': float(np.std(aps)),
            'best_ap': float(np.max(aps)),
            'median_ap': float(np.median(aps)),
            'backbone_distribution': dict(data['backbones']),
        }

    return {
        'agent_map_size': len(agent_map),
        'agent_stats': result,
    }


# ── Analysis 6: Nexar Competition Context ─────────────────────────────

def analysis_nexar_competition(experiments):
    """Check for public/private mAP competition scores."""
    competition_scores = []

    # Check all ken_test_report.json and metrics.json for competition fields
    for idea_dir in glob.glob(os.path.join(RESULTS_DIR, 'idea-*')):
        idea_id = os.path.basename(idea_dir)
        for fname in ['ken_test_report.json', 'metrics.json', 'fedex_test_report.json']:
            fpath = os.path.join(idea_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
                # Flatten and search for competition keys
                text = json.dumps(data)
                for key in ['public_mAP', 'private_mAP', 'leaderboard_score',
                             'competition_score', 'nexar_score']:
                    if key in text:
                        competition_scores.append({
                            'idea_id': idea_id,
                            'file': fname,
                            'key': key,
                        })
            except Exception:
                pass

    # Also check leaderboard file
    lb_path = os.path.join(RESULTS_DIR, '_leaderboard.json')
    leaderboard_info = None
    if os.path.exists(lb_path):
        try:
            with open(lb_path) as f:
                lb = json.load(f)
            leaderboard_info = {
                'metric': lb.get('metric'),
                'n_entries': len(lb.get('top', [])),
            }
            if lb.get('top'):
                leaderboard_info['top_entries'] = lb['top'][:5]
        except Exception:
            pass

    return {
        'competition_scores_found': len(competition_scores),
        'competition_details': competition_scores[:20],
        'leaderboard_info': leaderboard_info,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70, file=sys.stderr)
    print("COMPREHENSIVE ANALYSIS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Load data
    experiments, data_stats = load_all_experiments()
    print(f"\n--- Data Coverage ---", file=sys.stderr)
    for k, v in data_stats.items():
        print(f"  {k}: {v}", file=sys.stderr)

    if not experiments:
        print("ERROR: No experiments loaded!", file=sys.stderr)
        sys.exit(1)

    # Build agent map
    print("\nBuilding agent map from research logs...", file=sys.stderr)
    agent_map = build_agent_map()
    print(f"  Mapped {len(agent_map)} idea IDs to agents", file=sys.stderr)

    # Run analyses
    print("\n--- Running Analysis 1: Post-Bugfix ANOVA ---", file=sys.stderr)
    a1 = analysis_post_bugfix_anova(experiments)
    print(f"  Post-bugfix experiments: {a1['n_post_bugfix']}", file=sys.stderr)

    print("\n--- Running Analysis 2: Test AP for Top Configs ---", file=sys.stderr)
    a2 = analysis_test_ap_top_configs(experiments)
    print(f"  Experiments with test AP: {a2['n_with_test_ap']}", file=sys.stderr)

    print("\n--- Running Analysis 3: Convergence ---", file=sys.stderr)
    a3 = analysis_convergence(experiments)
    for k in ['full_dataset', 'post_bugfix', 'test_ap_only']:
        n = a3.get(k, {}).get('n_experiments', 0)
        print(f"  {k}: {n} experiments", file=sys.stderr)

    print("\n--- Running Analysis 4: Full ANOVA ---", file=sys.stderr)
    a4 = analysis_full_anova(experiments)
    print(f"  Experiments with config+AP: {a4['n_with_config_and_ap']}", file=sys.stderr)

    print("\n--- Running Analysis 5: Agent Attribution ---", file=sys.stderr)
    a5 = analysis_agent_attribution(experiments, agent_map)
    print(f"  Agent map size: {a5['agent_map_size']}", file=sys.stderr)

    print("\n--- Running Analysis 6: Nexar Competition ---", file=sys.stderr)
    a6 = analysis_nexar_competition(experiments)
    print(f"  Competition scores found: {a6['competition_scores_found']}", file=sys.stderr)

    # Compile output
    output = {
        'data_coverage': data_stats,
        'analysis_1_post_bugfix_anova': a1,
        'analysis_2_test_ap_top_configs': a2,
        'analysis_3_convergence': a3,
        'analysis_4_full_anova': a4,
        'analysis_5_agent_attribution': a5,
        'analysis_6_nexar_competition': a6,
        'generated_at': datetime.now().isoformat(),
    }

    # Save JSON
    out_path = os.path.join(OUTPUT_DIR, 'comprehensive.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}", file=sys.stderr)

    # ── Human-readable summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("=" * 70)

    print(f"\n--- Data Coverage ---")
    print(f"  Total idea directories scanned: {data_stats['total_dirs']}")
    print(f"  With metrics.json: {data_stats['has_metrics_json']}")
    print(f"  With ken_test_report.json: {data_stats['has_ken_test']}")
    print(f"  With resolved_config.yaml: {data_stats['has_config']}")
    print(f"  With claim.json: {data_stats['has_claim']}")
    print(f"  Experiments with any AP metric: {data_stats['has_any_ap']}")
    print(f"  Skipped (no AP): {data_stats['skipped_no_ap']}")
    print(f"  Skipped (parse error): {data_stats['skipped_parse_error']}")

    print(f"\n--- Analysis 1: Post-Bugfix ANOVA (after 2026-03-06) ---")
    print(f"  Post-bugfix experiments: {a1['n_post_bugfix']}")
    print(f"  Post-bugfix with backbone+encoder: {a1['n_with_arch']}")
    if a1['post_bugfix_anova']:
        pba = a1['post_bugfix_anova']
        print(f"  Post-bugfix ANOVA: F={pba['f_statistic']:.2f}, p={pba['p_value']:.2e}, "
              f"eta^2={pba['eta_squared']:.4f}, groups={pba['n_groups']}, N={pba['n_total']}")
    else:
        print(f"  Post-bugfix ANOVA: insufficient data")
    if a1['full_dataset_anova']:
        fda = a1['full_dataset_anova']
        print(f"  Full-dataset ANOVA: F={fda['f_statistic']:.2f}, p={fda['p_value']:.2e}, "
              f"eta^2={fda['eta_squared']:.4f}, groups={fda['n_groups']}, N={fda['n_total']}")
    if a1['post_bugfix_group_stats']:
        print(f"  Post-bugfix group means (top 5):")
        for i, (combo, st) in enumerate(sorted(a1['post_bugfix_group_stats'].items(),
                                                key=lambda x: -x[1]['mean'])):
            if i >= 5:
                break
            print(f"    {combo}: mean={st['mean']:.4f}, std={st['std']:.4f}, "
                  f"best={st['best']:.4f}, n={st['count']}")

    print(f"\n--- Analysis 2: Test AP for Top Configs ---")
    print(f"  Experiments with test AP: {a2['n_with_test_ap']}")
    print(f"  Experiments with both val+test: {a2['n_with_both']}")
    if a2['val_test_correlation']:
        vtc = a2['val_test_correlation']
        print(f"  Val-test Spearman rho: {vtc['spearman_rho']:.4f} (p={vtc['p_value']:.2e}, n={vtc['n']})")
    print(f"  Top 10 by validation AP:")
    for entry in a2['top10_by_val']:
        test_str = f"{entry['test_ap']:.4f}" if entry['test_ap'] is not None else 'N/A'
        print(f"    {entry['idea_id']}: val={entry['val_ap']:.4f}, test={test_str}, "
              f"bb={entry['backbone']}, enc={entry['encoder']}")
    print(f"  Top 10 by test AP:")
    for entry in a2['top10_by_test']:
        val_str = f"{entry['val_ap']:.4f}" if entry['val_ap'] is not None else 'N/A'
        print(f"    {entry['idea_id']}: test={entry['test_ap']:.4f}, val={val_str}, "
              f"bb={entry['backbone']}, enc={entry['encoder']}")
    if a2['vjepa2_test_stats']:
        vs = a2['vjepa2_test_stats']
        print(f"  VJepa2 test AP: mean={vs['mean']:.4f}, CI95={vs['ci95']}, n={vs['n']}, best={vs['best']:.4f}")
    if a2['other_backbones_test_stats']:
        os_ = a2['other_backbones_test_stats']
        print(f"  Other backbones test AP: mean={os_['mean']:.4f}, CI95={os_['ci95']}, n={os_['n']}, best={os_['best']:.4f}")

    print(f"\n--- Analysis 3: Convergence ---")
    for subset_name in ['full_dataset', 'post_bugfix', 'test_ap_only']:
        sub = a3.get(subset_name, {})
        if 'error' in sub:
            print(f"  [{subset_name}] {sub['error']}")
            continue
        print(f"  [{subset_name}] N={sub['n_experiments']}, "
              f"best={sub['best_ap']:.4f}, "
              f"range: {sub['first_time']} to {sub['last_time']}")
        if 'ap_at_n' in sub:
            vals = ', '.join(f"{k}={v}" for k, v in sub['ap_at_n'].items() if v is not None)
            print(f"    AP@N: {vals}")
        if sub.get('power_law_fit') and 'power_law_exponent' in sub['power_law_fit']:
            plf = sub['power_law_fit']
            print(f"    Power law: exponent={plf['power_law_exponent']:.4f}, "
                  f"b={plf['b_coefficient']:.6f}")
        if 'llm_vs_random' in sub:
            lr = sub['llm_vs_random']
            print(f"    LLM AP@{lr['n_check']}: {lr['llm_ap']:.4f}, "
                  f"Random: {lr['random_mean']:.4f} +/- {lr['random_std']:.4f}, "
                  f"advantage: {lr['llm_advantage']:.4f}, "
                  f"percentile: {lr['llm_percentile']:.1f}%")

    print(f"\n--- Analysis 4: Full ANOVA ---")
    print(f"  Experiments with config+AP: {a4['n_with_config_and_ap']}")
    for name, key in [('Backbone only', 'backbone_anova'),
                      ('Encoder only', 'encoder_anova'),
                      ('Backbone x Encoder', 'backbone_x_encoder_anova')]:
        anova = a4[key]
        if anova:
            print(f"  {name}: F={anova['f_statistic']:.2f}, p={anova['p_value']:.2e}, "
                  f"eta^2={anova['eta_squared']:.4f}, groups={anova['n_groups']}, N={anova['n_total']}")
        else:
            print(f"  {name}: insufficient data")
    print(f"  Backbone stats:")
    for name, st in a4['backbone_stats'].items():
        print(f"    {name}: mean={st['mean']:.4f}, std={st['std']:.4f}, "
              f"best={st['best']:.4f}, n={st['count']}")
    print(f"  Encoder stats:")
    for name, st in a4['encoder_stats'].items():
        print(f"    {name}: mean={st['mean']:.4f}, std={st['std']:.4f}, "
              f"best={st['best']:.4f}, n={st['count']}")

    print(f"\n--- Analysis 5: Agent Attribution ---")
    print(f"  Ideas mapped from research logs: {a5['agent_map_size']}")
    for agent, st in a5['agent_stats'].items():
        print(f"  {agent}: count={st['count']}, mean_ap={st['mean_ap']:.4f}, "
              f"best_ap={st['best_ap']:.4f}, median={st['median_ap']:.4f}")
        bb_dist = st['backbone_distribution']
        top_bbs = sorted(bb_dist.items(), key=lambda x: -x[1])[:3]
        bb_str = ', '.join(f"{k}:{v}" for k, v in top_bbs)
        print(f"    Top backbones: {bb_str}")

    print(f"\n--- Analysis 6: Nexar Competition Context ---")
    print(f"  Competition scores found: {a6['competition_scores_found']}")
    if a6['competition_scores_found'] > 0:
        for entry in a6['competition_details'][:5]:
            print(f"    {entry['idea_id']}: {entry['key']} in {entry['file']}")
    if a6['leaderboard_info']:
        li = a6['leaderboard_info']
        print(f"  Leaderboard: metric={li['metric']}, entries={li['n_entries']}")
        if li.get('top_entries'):
            for entry in li['top_entries']:
                print(f"    {entry}")

    print(f"\n{'=' * 70}")
    print(f"Output saved to: {out_path}")


if __name__ == '__main__':
    main()
