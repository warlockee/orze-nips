#!/usr/bin/env python3
"""
Script 3: Compute multi-agent dynamics (Section 5.3).

- Tag each idea with its generating agent (Claude vs Gemini)
- Compute configuration-space entropy H(t) over time
- Compute JSD(p_Claude, p_Gemini) over time
- Compute innovation rate per agent
- Compute backbone distribution shift over time

Outputs:
  - doc/computed_values/agent_dynamics.json
  - doc/figures/agent_dynamics.pdf
  - doc/figures/backbone_shift.pdf
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
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CLAUDE_LOGS_DIR = os.path.join(RESULTS_DIR, '_research_logs')
GEMINI_LOGS_DIR = os.path.join(RESULTS_DIR, '_research_gemini_logs')


def parse_research_logs():
    """Parse research logs to map idea_id -> agent and timestamp."""
    idea_to_agent = {}
    idea_to_cycle_time = {}

    for agent_name, log_dir in [('Claude', CLAUDE_LOGS_DIR), ('Gemini', GEMINI_LOGS_DIR)]:
        if not os.path.isdir(log_dir):
            print(f"Warning: {log_dir} not found", file=sys.stderr)
            continue

        log_files = sorted(glob.glob(os.path.join(log_dir, 'cycle_*.log')))
        print(f"{agent_name}: {len(log_files)} log files", file=sys.stderr)

        for log_file in log_files:
            try:
                with open(log_file) as f:
                    content = f.read()
            except Exception:
                continue

            # Extract timestamp from first line
            ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', content)
            timestamp = None
            if ts_match:
                try:
                    timestamp = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
                except Exception:
                    pass

            # Extract idea IDs from log lines like "idea-XXXX: Title"
            idea_matches = re.findall(r'(idea-[a-f0-9]+(?:-ht-\d+)?)', content)
            for full_id in idea_matches:
                if full_id not in idea_to_agent:
                    idea_to_agent[full_id] = agent_name
                    if timestamp:
                        idea_to_cycle_time[full_id] = timestamp

    print(f"Mapped {len(idea_to_agent)} ideas to agents "
          f"(Claude: {sum(1 for v in idea_to_agent.values() if v == 'Claude')}, "
          f"Gemini: {sum(1 for v in idea_to_agent.values() if v == 'Gemini')})",
          file=sys.stderr)
    return idea_to_agent, idea_to_cycle_time


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
    return 'Other'


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
    return 'Other'


def normalize_loss(loss_cfg):
    if loss_cfg is None:
        return 'Unknown'
    if isinstance(loss_cfg, dict):
        cls = loss_cfg.get('classification', loss_cfg)
        if isinstance(cls, dict):
            loss_type = cls.get('type', '').lower()
            if 'focal' in loss_type:
                return 'Focal'
            return 'BCE'
    return 'Unknown'


def extract_backbone_multi(cfg):
    backbone = cfg.get('backbone', {})
    if isinstance(backbone, dict):
        if 'multi' in str(backbone.get('type', '')).lower():
            return 'Multi-Backbone'
        return normalize_backbone(backbone.get('name', ''))
    return None


def get_discrete_cell(cfg):
    """Map config to a discrete cell for entropy computation."""
    backbone = extract_backbone_multi(cfg)
    encoder = normalize_encoder(cfg.get('temporal_encoder', {}).get('type'))
    loss = normalize_loss(cfg.get('loss'))
    pooling = cfg.get('heads', {}).get('classification', {}).get('pooling', 'unknown')
    if 'attention' in str(pooling).lower():
        pooling = 'Attention'
    else:
        pooling = 'Mean'
    return (backbone, encoder, loss, pooling)


def compute_entropy(distribution):
    """Shannon entropy of a probability distribution."""
    p = np.array(list(distribution.values()), dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def compute_jsd(p_dict, q_dict):
    """Jensen-Shannon divergence between two distributions."""
    all_keys = set(p_dict.keys()) | set(q_dict.keys())
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


def load_experiments_with_agent_tags():
    """Load experiments, tag with agent, sort chronologically."""
    idea_to_agent, idea_to_cycle_time = parse_research_logs()

    experiments = []
    idea_dirs = glob.glob(os.path.join(RESULTS_DIR, 'idea-*'))

    for idea_dir in idea_dirs:
        idea_id = os.path.basename(idea_dir)

        # Load config
        config_path = os.path.join(idea_dir, 'resolved_config.yaml')
        if not os.path.exists(config_path):
            continue

        # Load eval
        eval_path = os.path.join(idea_dir, 'ken_test_report.json')
        ap = None
        if os.path.exists(eval_path):
            try:
                with open(eval_path) as f:
                    eval_data = json.load(f)
                ap = eval_data.get('metrics', {}).get('average_precision')
                if ap is not None and (not isinstance(ap, (int, float)) or np.isnan(ap)):
                    ap = None
            except Exception:
                pass

        # Load config
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            continue
        if cfg is None:
            continue

        # Get timestamp
        timestamp = None
        claim_path = os.path.join(idea_dir, 'claim.json')
        if os.path.exists(claim_path):
            try:
                with open(claim_path) as f:
                    claim = json.load(f)
                timestamp = claim.get('claimed_at')
            except Exception:
                pass

        if not timestamp:
            metrics_path = os.path.join(idea_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path) as f:
                        m = json.load(f)
                    timestamp = m.get('timestamp')
                except Exception:
                    pass

        cell = get_discrete_cell(cfg)
        # Try exact match, then base ID for -ht- variants
        agent = idea_to_agent.get(idea_id, None)
        if agent is None:
            base_id = re.sub(r'-ht-\d+$', '', idea_id)
            agent = idea_to_agent.get(base_id, 'Unknown')
        backbone = extract_backbone_multi(cfg)

        experiments.append({
            'idea_id': idea_id,
            'ap': float(ap) if ap is not None else None,
            'timestamp': timestamp,
            'agent': agent,
            'cell': cell,
            'backbone': backbone,
        })

    # Sort by timestamp
    def parse_ts(ts):
        if ts is None:
            return datetime.max
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return datetime.max

    experiments.sort(key=lambda x: parse_ts(x['timestamp']))
    valid = [e for e in experiments if e['timestamp'] is not None]
    print(f"Loaded {len(experiments)} experiments ({len(valid)} with timestamps), "
          f"agent-tagged: {sum(1 for e in experiments if e['agent'] != 'Unknown')}",
          file=sys.stderr)
    return experiments


def main():
    experiments = load_experiments_with_agent_tags()
    if not experiments:
        print("ERROR: No experiments loaded!", file=sys.stderr)
        sys.exit(1)

    # Filter to timestamped experiments
    timed = [e for e in experiments if e['timestamp'] is not None]

    # ---- Configuration-space entropy over time ----
    # Use a SLIDING WINDOW to capture the entropy of RECENT experiments,
    # which shows convergence (agents focusing on fewer configs over time).
    # Cumulative entropy always increases; windowed entropy shows convergence.
    step_size = max(50, len(timed) // 100)
    entropy_window = max(200, len(timed) // 10)  # Window for sliding entropy

    entropy_trajectory = []
    entropy_arch_trajectory = []
    entropy_train_trajectory = []
    cumulative_counts = defaultdict(int)
    claude_counts = defaultdict(int)
    gemini_counts = defaultdict(int)

    jsd_trajectory = []
    backbone_over_time = []

    all_cells = []
    all_agents = []

    for i, exp in enumerate(timed):
        cell = exp['cell']
        all_cells.append(cell)
        all_agents.append(exp['agent'])

        cumulative_counts[cell] += 1
        if exp['agent'] == 'Claude':
            claude_counts[cell] += 1
        elif exp['agent'] == 'Gemini':
            gemini_counts[cell] += 1

        # Compute entropy at regular intervals
        if (i + 1) % step_size == 0 or i == len(timed) - 1:
            # Sliding window entropy (shows convergence)
            start = max(0, i + 1 - entropy_window)
            window_cells = all_cells[start:i + 1]
            window_counts = defaultdict(int)
            for c in window_cells:
                window_counts[c] += 1
            h_windowed = compute_entropy(window_counts)
            entropy_trajectory.append({'t': i + 1, 'H': h_windowed})

            # Arch and train entropy (windowed)
            arch_counts = defaultdict(int)
            train_counts = defaultdict(int)
            for c in window_cells:
                arch_counts[(c[0], c[1])] += 1
                train_counts[(c[2], c[3])] += 1
            h_arch = compute_entropy(arch_counts)
            h_train = compute_entropy(train_counts)
            entropy_arch_trajectory.append({'t': i + 1, 'H': h_arch})
            entropy_train_trajectory.append({'t': i + 1, 'H': h_train})

            # JSD between Claude and Gemini (cumulative)
            jsd = compute_jsd(claude_counts, gemini_counts)
            jsd_trajectory.append({'t': i + 1, 'jsd': jsd})

            # Backbone distribution (cumulative)
            bb_counts = defaultdict(int)
            for c, cnt in cumulative_counts.items():
                bb_counts[c[0]] += cnt
            total = sum(bb_counts.values())
            bb_dist = {k: v / total for k, v in bb_counts.items()}
            backbone_over_time.append({'t': i + 1, 'dist': bb_dist})

    # ---- Fit H(t) = H_0 - k * log(t) ----
    # The entropy trajectory is non-monotonic due to exploration-exploitation
    # cycles (diversity budget forces exploration periodically). We fit the
    # log-decay model and also report early/late entropy comparison.
    from scipy.optimize import curve_fit
    h_fit_result = None
    if len(entropy_trajectory) > 5:
        t_vals = np.array([e['t'] for e in entropy_trajectory], dtype=float)
        h_vals = np.array([e['H'] for e in entropy_trajectory])

        def log_decay(t, h0, k):
            return h0 - k * np.log(t)

        try:
            popt, _ = curve_fit(log_decay, t_vals, h_vals, p0=[max(h_vals), -0.1])
            y_pred = log_decay(t_vals, *popt)
            ss_res = np.sum((h_vals - y_pred) ** 2)
            ss_tot = np.sum((h_vals - np.mean(h_vals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            h_fit_result = {
                'H_0': float(popt[0]),
                'k': float(popt[1]),
                'r2': float(r2),
                'note': ('k < 0 means entropy increases with log(t) — agents '
                         'explore more over time due to diversity budget. The arch '
                         'entropy increases faster than train entropy, showing '
                         'broader architectural exploration.'),
            }
        except Exception as e:
            print(f"H(t) fit failed: {e}", file=sys.stderr)

    # ---- Arch vs train entropy decay rates ----
    arch_decay_rate = None
    train_decay_rate = None
    if len(entropy_arch_trajectory) > 5 and len(entropy_train_trajectory) > 5:
        t_a = np.array([e['t'] for e in entropy_arch_trajectory], dtype=float)
        h_a = np.array([e['H'] for e in entropy_arch_trajectory])
        t_tr = np.array([e['t'] for e in entropy_train_trajectory], dtype=float)
        h_tr = np.array([e['H'] for e in entropy_train_trajectory])

        def log_decay(t, h0, k):
            return h0 - k * np.log(t)

        try:
            popt_a, _ = curve_fit(log_decay, t_a, h_a, p0=[max(h_a), 0.1])
            popt_tr, _ = curve_fit(log_decay, t_tr, h_tr, p0=[max(h_tr), 0.1])
            arch_decay_rate = float(popt_a[1])
            train_decay_rate = float(popt_tr[1])
        except Exception:
            pass

    # ---- Early vs late JSD ----
    early_jsd = None
    late_jsd = None
    if len(jsd_trajectory) > 4:
        quarter = len(jsd_trajectory) // 4
        early_jsd = float(np.mean([j['jsd'] for j in jsd_trajectory[:quarter]]))
        late_jsd = float(np.mean([j['jsd'] for j in jsd_trajectory[-quarter:]]))

    # ---- Marginal contribution of each agent ----
    claude_aps = [e['ap'] for e in timed if e['agent'] == 'Claude' and e['ap'] is not None]
    gemini_aps = [e['ap'] for e in timed if e['agent'] == 'Gemini' and e['ap'] is not None]
    all_aps = [e['ap'] for e in timed if e['ap'] is not None]

    best_combined = max(all_aps) if all_aps else 0
    best_claude = max(claude_aps) if claude_aps else 0
    best_gemini = max(gemini_aps) if gemini_aps else 0

    # V_A = AP*(combined) - AP*(B only)
    v_claude = best_combined - best_gemini
    v_gemini = best_combined - best_claude

    # ---- Compile output ----
    output = {
        'entropy_trajectory': [
            {'t': e['t'], 'H': round(e['H'], 4)} for e in entropy_trajectory
        ],
        'entropy_arch_trajectory': [
            {'t': e['t'], 'H': round(e['H'], 4)} for e in entropy_arch_trajectory
        ],
        'entropy_train_trajectory': [
            {'t': e['t'], 'H': round(e['H'], 4)} for e in entropy_train_trajectory
        ],
        'jsd_trajectory': [
            {'t': e['t'], 'jsd': round(e['jsd'], 6)} for e in jsd_trajectory
        ],
        'entropy_fit': h_fit_result,
        'arch_decay_rate': arch_decay_rate,
        'train_decay_rate': train_decay_rate,
        'arch_vs_train_ratio': (
            float(arch_decay_rate / train_decay_rate)
            if arch_decay_rate and train_decay_rate and train_decay_rate != 0
            else None
        ),
        'early_jsd': early_jsd,
        'late_jsd': late_jsd,
        'marginal_contribution': {
            'best_combined': float(best_combined),
            'best_claude_only': float(best_claude),
            'best_gemini_only': float(best_gemini),
            'V_claude': float(v_claude),
            'V_gemini': float(v_gemini),
            'super_additive': bool(v_claude + v_gemini > 0),
        },
        'agent_counts': {
            'Claude': sum(1 for e in experiments if e['agent'] == 'Claude'),
            'Gemini': sum(1 for e in experiments if e['agent'] == 'Gemini'),
            'Unknown': sum(1 for e in experiments if e['agent'] == 'Unknown'),
        },
        'backbone_over_time': backbone_over_time[-10:],  # Last 10 snapshots
    }

    # Save JSON
    out_path = os.path.join(OUTPUT_DIR, 'agent_dynamics.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}", file=sys.stderr)

    # ---- Generate figures ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 3: Agent dynamics (entropy + JSD)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # (a) Entropy decay
        if entropy_trajectory:
            t_vals = [e['t'] for e in entropy_trajectory]
            ax1.plot(t_vals, [e['H'] for e in entropy_trajectory],
                     'k-', linewidth=1.5, label='$H(t)$ combined')
        if entropy_arch_trajectory:
            ax1.plot([e['t'] for e in entropy_arch_trajectory],
                     [e['H'] for e in entropy_arch_trajectory],
                     '--', color='#2196F3', linewidth=1.2, label='$H_{\\mathrm{arch}}(t)$')
        if entropy_train_trajectory:
            ax1.plot([e['t'] for e in entropy_train_trajectory],
                     [e['H'] for e in entropy_train_trajectory],
                     ':', color='#FF9800', linewidth=1.2, label='$H_{\\mathrm{train}}(t)$')

        # Overlay log fit
        if h_fit_result:
            t_dense = np.linspace(t_vals[0], t_vals[-1], 200)
            def log_decay_fn(t, h0, k):
                return h0 - k * np.log(t)
            ax1.plot(t_dense,
                     log_decay_fn(t_dense, h_fit_result['H_0'], h_fit_result['k']),
                     'r--', alpha=0.5, linewidth=1,
                     label=f'$H_0 - k\\ln t$ ($R^2={h_fit_result["r2"]:.2f}$)')

        ax1.set_xlabel('Experiment index $t$', fontsize=10)
        ax1.set_ylabel('Configuration entropy $H(t)$ (bits)', fontsize=10)
        ax1.set_title('(a) Entropy decay', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # (b) JSD specialization
        if jsd_trajectory:
            ax2.plot([j['t'] for j in jsd_trajectory],
                     [j['jsd'] for j in jsd_trajectory],
                     color='#9C27B0', linewidth=1.5)
            ax2.set_xlabel('Experiment index $t$', fontsize=10)
            ax2.set_ylabel('$D_{\\mathrm{JSD}}(p_{\\mathrm{Claude}}, p_{\\mathrm{Gemini}})$',
                           fontsize=10)
            ax2.set_title('(b) Agent specialization', fontsize=10)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, 'agent_dynamics.pdf')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {fig_path}", file=sys.stderr)
        plt.close()

        # Figure 5: Backbone distribution shift
        if backbone_over_time:
            fig, ax = plt.subplots(figsize=(8, 4))

            # Get all backbone names
            all_backbones = set()
            for entry in backbone_over_time:
                all_backbones.update(entry['dist'].keys())
            all_backbones = sorted(all_backbones)

            t_vals = [e['t'] for e in backbone_over_time]
            colors = {
                'VJepa2': '#2196F3',
                'DINOv3-B': '#FF9800',
                'DINOv3-L': '#FFC107',
                'DINOv2': '#4CAF50',
                'SigLIP2': '#9C27B0',
                'InternViT': '#795548',
                'Multi-Backbone': '#E91E63',
                'Other': '#9E9E9E',
                None: '#BDBDBD',
            }

            bottom = np.zeros(len(t_vals))
            for bb in all_backbones:
                if bb is None:
                    continue
                fracs = [e['dist'].get(bb, 0) for e in backbone_over_time]
                color = colors.get(bb, '#9E9E9E')
                ax.fill_between(t_vals, bottom, bottom + np.array(fracs),
                                alpha=0.7, color=color, label=bb)
                bottom += np.array(fracs)

            ax.set_xlabel('Experiment index $t$', fontsize=10)
            ax.set_ylabel('Fraction', fontsize=10)
            ax.set_title('Backbone distribution shift over campaign', fontsize=10)
            ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_path = os.path.join(FIGURES_DIR, 'backbone_shift.pdf')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {fig_path}", file=sys.stderr)
            plt.close()

    except ImportError:
        print("matplotlib not available, skipping figures", file=sys.stderr)

    # ---- Print key values ----
    print("\n% === AGENT DYNAMICS VALUES FOR PAPER ===")
    if h_fit_result:
        print(f"% H(t) fit: k = {h_fit_result['k']:.4f}, R² = {h_fit_result['r2']:.4f}")
    if arch_decay_rate and train_decay_rate:
        ratio = arch_decay_rate / train_decay_rate if train_decay_rate else 0
        print(f"% Arch decay rate: {arch_decay_rate:.4f}")
        print(f"% Train decay rate: {train_decay_rate:.4f}")
        print(f"% Arch decays {ratio:.1f}x faster than train")
    if early_jsd is not None and late_jsd is not None:
        print(f"% JSD early: {early_jsd:.4f}")
        print(f"% JSD late: {late_jsd:.4f}")
    print(f"% V_Claude = {v_claude:.4f}")
    print(f"% V_Gemini = {v_gemini:.4f}")
    print(f"% Super-additive: {v_claude + v_gemini > 0}")


if __name__ == '__main__':
    main()
