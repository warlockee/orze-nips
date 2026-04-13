#!/usr/bin/env python3
"""
Representativeness test: attributed vs. unattributed experiment subsets.

Addresses reviewer concern about selection bias in the multi-agent dynamics
analysis (Section 5.3). Tests whether the 688 experiments with agent
attribution are statistically distinguishable from the 2,511 unattributed
experiments on observable dimensions.

Tests:
  1. Backbone distribution — chi-squared
  2. Encoder distribution — chi-squared
  3. AP distribution — KS test
  4. Temporal position — KS test

Outputs:
  - doc/computed_values/representativeness.json
"""

import json
import os
import sys
import numpy as np
from collections import Counter
from scipy import stats

# Reuse loading logic from agent dynamics script
sys.path.insert(0, os.path.dirname(__file__))
from compute_agent_dynamics import (
    load_experiments_with_agent_tags,
    normalize_encoder,
)

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'computed_values'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def chi2_test(counter_a, counter_b, label):
    """Chi-squared test on two category distributions.

    Uses all categories present in either group.  Merges categories with
    expected count < 5 into 'Other' per standard practice.
    """
    all_keys = sorted(set(counter_a.keys()) | set(counter_b.keys()))
    obs_a = np.array([counter_a.get(k, 0) for k in all_keys], dtype=float)
    obs_b = np.array([counter_b.get(k, 0) for k in all_keys], dtype=float)

    # Merge small categories (expected < 5 in either row)
    total_a, total_b = obs_a.sum(), obs_b.sum()
    total = total_a + total_b
    expected_a = (obs_a + obs_b) * total_a / total
    expected_b = (obs_a + obs_b) * total_b / total

    small_mask = (expected_a < 5) | (expected_b < 5)
    if small_mask.any() and not small_mask.all():
        merged_keys = [k for k, s in zip(all_keys, small_mask) if not s]
        merged_keys.append('_Other_merged')
        new_a = [counter_a.get(k, 0) for k, s in zip(all_keys, small_mask) if not s]
        new_a.append(sum(obs_a[small_mask]))
        new_b = [counter_b.get(k, 0) for k, s in zip(all_keys, small_mask) if not s]
        new_b.append(sum(obs_b[small_mask]))
        obs_a = np.array(new_a, dtype=float)
        obs_b = np.array(new_b, dtype=float)
        all_keys = merged_keys

    contingency = np.array([obs_a, obs_b])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    # Compute proportions for characterization
    prop_a = {k: float(v / total_a) if total_a > 0 else 0
              for k, v in zip(all_keys, obs_a)}
    prop_b = {k: float(v / total_b) if total_b > 0 else 0
              for k, v in zip(all_keys, obs_b)}

    # Cramér's V for effect size
    n = contingency.sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 and n > 0 else 0.0

    return {
        'test': 'chi-squared',
        'dimension': label,
        'chi2': float(chi2),
        'p_value': float(p),
        'dof': int(dof),
        'cramers_v': round(cramers_v, 4),
        'n_attributed': int(total_a),
        'n_unattributed': int(total_b),
        'proportions_attributed': {k: round(v, 4) for k, v in prop_a.items()},
        'proportions_unattributed': {k: round(v, 4) for k, v in prop_b.items()},
        'significant_at_005': bool(p < 0.05),
    }


def ks_test(values_a, values_b, label):
    """Two-sample KS test on continuous distributions."""
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)

    result = stats.ks_2samp(a, b)

    return {
        'test': 'KS',
        'dimension': label,
        'ks_statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'n_attributed': len(a),
        'n_unattributed': len(b),
        'mean_attributed': float(np.mean(a)),
        'mean_unattributed': float(np.mean(b)),
        'median_attributed': float(np.median(a)),
        'median_unattributed': float(np.median(b)),
        'std_attributed': float(np.std(a)),
        'std_unattributed': float(np.std(b)),
        'significant_at_005': bool(result.pvalue < 0.05),
    }


def main():
    experiments = load_experiments_with_agent_tags()
    if not experiments:
        print("ERROR: No experiments loaded!", file=sys.stderr)
        sys.exit(1)

    # Split into attributed vs unattributed
    attributed = [e for e in experiments if e['agent'] in ('Claude', 'Gemini')]
    unattributed = [e for e in experiments if e['agent'] == 'Unknown']

    print(f"\nAttributed: {len(attributed)}", file=sys.stderr)
    print(f"Unattributed: {len(unattributed)}", file=sys.stderr)

    results = {
        'n_total': len(experiments),
        'n_attributed': len(attributed),
        'n_unattributed': len(unattributed),
        'fraction_attributed': round(len(attributed) / len(experiments), 4),
        'tests': [],
    }

    # --- 1. Backbone distribution (chi-squared) ---
    bb_attr = Counter(e['backbone'] for e in attributed if e['backbone'] is not None)
    bb_unattr = Counter(e['backbone'] for e in unattributed if e['backbone'] is not None)
    test_bb = chi2_test(bb_attr, bb_unattr, 'backbone')
    results['tests'].append(test_bb)
    print(f"\nBackbone chi2={test_bb['chi2']:.2f}, p={test_bb['p_value']:.4f}, "
          f"V={test_bb['cramers_v']:.4f}", file=sys.stderr)

    # --- 2. Encoder distribution (chi-squared) ---
    def get_encoder(exp):
        return exp['cell'][1]  # cell = (backbone, encoder, loss, pooling)

    enc_attr = Counter(get_encoder(e) for e in attributed if get_encoder(e) is not None)
    enc_unattr = Counter(get_encoder(e) for e in unattributed if get_encoder(e) is not None)
    test_enc = chi2_test(enc_attr, enc_unattr, 'encoder')
    results['tests'].append(test_enc)
    print(f"Encoder chi2={test_enc['chi2']:.2f}, p={test_enc['p_value']:.4f}, "
          f"V={test_enc['cramers_v']:.4f}", file=sys.stderr)

    # --- 3. AP distribution (KS test) ---
    ap_attr = [e['ap'] for e in attributed if e['ap'] is not None]
    ap_unattr = [e['ap'] for e in unattributed if e['ap'] is not None]
    if ap_attr and ap_unattr:
        test_ap = ks_test(ap_attr, ap_unattr, 'average_precision')
        results['tests'].append(test_ap)
        print(f"AP KS={test_ap['ks_statistic']:.4f}, p={test_ap['p_value']:.4f}", file=sys.stderr)
    else:
        print("WARNING: not enough AP values for KS test", file=sys.stderr)

    # --- 4. Temporal position (KS test on ordinal index) ---
    # Assign ordinal index to each experiment (already sorted by timestamp)
    timed = [e for e in experiments if e['timestamp'] is not None]
    id_to_ordinal = {e['idea_id']: i for i, e in enumerate(timed)}

    ord_attr = [id_to_ordinal[e['idea_id']] for e in attributed
                if e['idea_id'] in id_to_ordinal]
    ord_unattr = [id_to_ordinal[e['idea_id']] for e in unattributed
                  if e['idea_id'] in id_to_ordinal]

    if ord_attr and ord_unattr:
        test_time = ks_test(ord_attr, ord_unattr, 'temporal_position')
        results['tests'].append(test_time)
        print(f"Temporal KS={test_time['ks_statistic']:.4f}, p={test_time['p_value']:.4f}",
              file=sys.stderr)

        # Characterize temporal difference if significant
        if test_time['p_value'] < 0.05:
            n_total_timed = len(timed)
            q1 = n_total_timed * 0.25
            q2 = n_total_timed * 0.50
            q3 = n_total_timed * 0.75

            attr_quartiles = [
                sum(1 for x in ord_attr if x < q1),
                sum(1 for x in ord_attr if q1 <= x < q2),
                sum(1 for x in ord_attr if q2 <= x < q3),
                sum(1 for x in ord_attr if x >= q3),
            ]
            unattr_quartiles = [
                sum(1 for x in ord_unattr if x < q1),
                sum(1 for x in ord_unattr if q1 <= x < q2),
                sum(1 for x in ord_unattr if q2 <= x < q3),
                sum(1 for x in ord_unattr if x >= q3),
            ]

            test_time['temporal_characterization'] = {
                'attributed_quartile_counts': attr_quartiles,
                'unattributed_quartile_counts': unattr_quartiles,
                'attributed_quartile_fracs': [round(x / len(ord_attr), 4)
                                              for x in attr_quartiles],
                'unattributed_quartile_fracs': [round(x / len(ord_unattr), 4)
                                                for x in unattr_quartiles],
                'note': 'Quartiles Q1-Q4 of campaign timeline (early to late)',
            }

    # --- 5. Temporal-stratified retest (controls for temporal confound) ---
    # The attributed subset is concentrated in the later campaign (Q3+Q4).
    # Re-run backbone, encoder, and AP tests restricted to the same temporal
    # window to separate genuine composition bias from temporal evolution.
    if ord_attr and ord_unattr:
        n_total_timed = len(timed)
        q2_cutoff = n_total_timed * 0.50  # second half of campaign

        late_attr = [e for e in attributed
                     if e['idea_id'] in id_to_ordinal
                     and id_to_ordinal[e['idea_id']] >= q2_cutoff]
        late_unattr = [e for e in unattributed
                       if e['idea_id'] in id_to_ordinal
                       and id_to_ordinal[e['idea_id']] >= q2_cutoff]

        print(f"\nTemporal-stratified retest (second half only): "
              f"attr={len(late_attr)}, unattr={len(late_unattr)}", file=sys.stderr)

        stratified_tests = []

        # Backbone (stratified)
        bb_attr_s = Counter(e['backbone'] for e in late_attr if e['backbone'] is not None)
        bb_unattr_s = Counter(e['backbone'] for e in late_unattr if e['backbone'] is not None)
        if sum(bb_attr_s.values()) > 10 and sum(bb_unattr_s.values()) > 10:
            t_bb_s = chi2_test(bb_attr_s, bb_unattr_s, 'backbone_stratified')
            stratified_tests.append(t_bb_s)
            print(f"  Backbone(strat) chi2={t_bb_s['chi2']:.2f}, p={t_bb_s['p_value']:.4f}, "
                  f"V={t_bb_s['cramers_v']:.4f}", file=sys.stderr)

        # Encoder (stratified)
        enc_attr_s = Counter(get_encoder(e) for e in late_attr if get_encoder(e) is not None)
        enc_unattr_s = Counter(get_encoder(e) for e in late_unattr if get_encoder(e) is not None)
        if sum(enc_attr_s.values()) > 10 and sum(enc_unattr_s.values()) > 10:
            t_enc_s = chi2_test(enc_attr_s, enc_unattr_s, 'encoder_stratified')
            stratified_tests.append(t_enc_s)
            print(f"  Encoder(strat) chi2={t_enc_s['chi2']:.2f}, p={t_enc_s['p_value']:.4f}, "
                  f"V={t_enc_s['cramers_v']:.4f}", file=sys.stderr)

        # AP (stratified)
        ap_attr_s = [e['ap'] for e in late_attr if e['ap'] is not None]
        ap_unattr_s = [e['ap'] for e in late_unattr if e['ap'] is not None]
        if len(ap_attr_s) > 10 and len(ap_unattr_s) > 10:
            t_ap_s = ks_test(ap_attr_s, ap_unattr_s, 'average_precision_stratified')
            stratified_tests.append(t_ap_s)
            print(f"  AP(strat) KS={t_ap_s['ks_statistic']:.4f}, "
                  f"p={t_ap_s['p_value']:.4f}", file=sys.stderr)

        # Apply Bonferroni within stratified block
        n_strat = len(stratified_tests)
        for t in stratified_tests:
            t['bonferroni_significant'] = bool(t['p_value'] < 0.05 / n_strat) if n_strat > 0 else False

        n_sig_strat_bonf = sum(1 for t in stratified_tests if t.get('bonferroni_significant'))
        max_effect_strat = 0.0
        for t in stratified_tests:
            if t['test'] == 'chi-squared':
                max_effect_strat = max(max_effect_strat, t.get('cramers_v', 0))
            elif t['test'] == 'KS':
                max_effect_strat = max(max_effect_strat, t.get('ks_statistic', 0))

        results['stratified_tests'] = stratified_tests
        results['stratified_summary'] = {
            'temporal_window': 'second_half (experiments 1600-3199)',
            'n_attributed_in_window': len(late_attr),
            'n_unattributed_in_window': len(late_unattr),
            'n_tests': n_strat,
            'n_significant_bonferroni': n_sig_strat_bonf,
            'max_effect_size': round(max_effect_strat, 4),
            'verdict': (
                'NOT_DISTINGUISHABLE' if n_sig_strat_bonf == 0
                else 'WEAKLY_DISTINGUISHABLE' if max_effect_strat < 0.1
                else 'DISTINGUISHABLE'
            ),
        }

    # --- Summary verdict ---
    n_sig = sum(1 for t in results['tests'] if t.get('significant_at_005', False))
    n_tests = len(results['tests'])

    # Also compute Bonferroni-corrected significance
    for t in results['tests']:
        t['bonferroni_significant'] = bool(t['p_value'] < 0.05 / n_tests)

    n_sig_bonferroni = sum(1 for t in results['tests'] if t.get('bonferroni_significant', False))

    # Effect sizes for significant tests
    max_effect = 0.0
    for t in results['tests']:
        if t['test'] == 'chi-squared':
            max_effect = max(max_effect, t.get('cramers_v', 0))
        elif t['test'] == 'KS':
            max_effect = max(max_effect, t.get('ks_statistic', 0))

    results['summary'] = {
        'n_tests': n_tests,
        'n_significant_nominal': n_sig,
        'n_significant_bonferroni': n_sig_bonferroni,
        'max_effect_size': round(max_effect, 4),
        'verdict': (
            'NOT_DISTINGUISHABLE' if n_sig_bonferroni == 0
            else 'WEAKLY_DISTINGUISHABLE' if max_effect < 0.1
            else 'DISTINGUISHABLE'
        ),
        'interpretation': '',  # filled below
    }

    if n_sig_bonferroni == 0:
        results['summary']['interpretation'] = (
            f'None of {n_tests} tests significant after Bonferroni correction '
            f'(alpha=0.05/{n_tests}={0.05/n_tests:.4f}). The attributed subset '
            f'is statistically representative of the full population.'
        )
    elif max_effect < 0.1:
        results['summary']['interpretation'] = (
            f'{n_sig_bonferroni}/{n_tests} tests significant after Bonferroni correction, '
            f'but maximum effect size is {max_effect:.4f} (< 0.1 threshold for small effect). '
            f'The differences, while statistically detectable with N={len(experiments)}, '
            f'are negligible in practical terms.'
        )
    else:
        results['summary']['interpretation'] = (
            f'{n_sig_bonferroni}/{n_tests} tests significant after Bonferroni correction '
            f'with max effect size {max_effect:.4f}. The attributed subset shows meaningful '
            f'differences from the unattributed population — see individual test details.'
        )

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'representativeness.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}", file=sys.stderr)

    # Print key values for paper
    print("\n% === REPRESENTATIVENESS TEST VALUES FOR PAPER ===")
    print(f"% N_total = {results['n_total']}")
    print(f"% N_attributed = {results['n_attributed']} ({results['fraction_attributed']*100:.1f}%)")
    print(f"% N_unattributed = {results['n_unattributed']}")
    for t in results['tests']:
        if t['test'] == 'chi-squared':
            print(f"% {t['dimension']}: chi2({t['dof']})={t['chi2']:.2f}, "
                  f"p={t['p_value']:.4f}, V={t['cramers_v']:.4f}"
                  f"{' *' if t['bonferroni_significant'] else ''}")
        elif t['test'] == 'KS':
            print(f"% {t['dimension']}: D={t['ks_statistic']:.4f}, "
                  f"p={t['p_value']:.4f}"
                  f"{' *' if t['bonferroni_significant'] else ''}")
    print(f"% Verdict: {results['summary']['verdict']}")
    print(f"% {results['summary']['interpretation']}")

    # Print stratified results
    if 'stratified_tests' in results:
        print(f"\n% === TEMPORAL-STRATIFIED RETEST (second half only) ===")
        ss = results['stratified_summary']
        print(f"% Window: {ss['temporal_window']}")
        print(f"% N_attributed = {ss['n_attributed_in_window']}, "
              f"N_unattributed = {ss['n_unattributed_in_window']}")
        for t in results['stratified_tests']:
            if t['test'] == 'chi-squared':
                print(f"% {t['dimension']}: chi2({t['dof']})={t['chi2']:.2f}, "
                      f"p={t['p_value']:.4f}, V={t['cramers_v']:.4f}"
                      f"{' *' if t['bonferroni_significant'] else ''}")
            elif t['test'] == 'KS':
                print(f"% {t['dimension']}: D={t['ks_statistic']:.4f}, "
                      f"p={t['p_value']:.4f}"
                      f"{' *' if t['bonferroni_significant'] else ''}")
        print(f"% Stratified verdict: {ss['verdict']}")


if __name__ == '__main__':
    main()
