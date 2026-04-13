#!/usr/bin/env python3
"""
Script 5: Compute idea genealogy analysis (Section 5.4).

- Parse parent-child relationships from idea_lake.db
- Build DAG
- Compute depth of each idea
- Compute Spearman correlation between depth and AP
- Trace lineage of champion idea (idea-2ec818) back to root
- Compute branching factor statistics

Outputs:
  - doc/computed_values/genealogy.json
"""

import json
import os
import sys
import re
import glob
import sqlite3
import numpy as np
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'idea_lake.db'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'computed_values'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_genealogy_from_db():
    """Load parent-child relationships from idea_lake.db."""
    parent_map = {}
    idea_titles = {}
    idea_categories = {}
    idea_created = {}

    try:
        db = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
        cur = db.cursor()

        cur.execute('''
            SELECT idea_id, title, parent, category, created_at
            FROM ideas
            WHERE parent IS NOT NULL AND parent != 'none'
        ''')
        for idea_id, title, parent, category, created_at in cur.fetchall():
            parent_map[idea_id] = parent
            idea_titles[idea_id] = title
            if category:
                idea_categories[idea_id] = category
            if created_at:
                idea_created[idea_id] = created_at

        # Also load titles for parents that may not be in the parent map
        cur.execute('SELECT idea_id, title, category, created_at FROM ideas')
        for idea_id, title, category, created_at in cur.fetchall():
            if idea_id not in idea_titles:
                idea_titles[idea_id] = title
            if category and idea_id not in idea_categories:
                idea_categories[idea_id] = category
            if created_at and idea_id not in idea_created:
                idea_created[idea_id] = created_at

        db.close()
    except Exception as e:
        print(f"Database error: {e}", file=sys.stderr)
        print("Falling back to ideas.md parsing...", file=sys.stderr)
        return load_genealogy_from_md()

    print(f"Loaded {len(parent_map)} parent-child links from database, "
          f"{len(idea_titles)} total ideas", file=sys.stderr)
    return parent_map, idea_titles, idea_categories, idea_created


def load_genealogy_from_md():
    """Fallback: parse ideas.md."""
    IDEAS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ideas.md'))
    parent_map = {}
    idea_titles = {}
    current_idea_id = None

    with open(IDEAS_PATH) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('## '):
                title = stripped[3:].strip()
                id_match = re.match(r'(idea-[a-f0-9]+)', title)
                if id_match:
                    current_idea_id = id_match.group(1)
                    idea_titles[current_idea_id] = title
                else:
                    current_idea_id = None
            elif stripped.startswith('- **Parent**:'):
                parent_val = stripped.split(':', 1)[1].strip()
                if parent_val and parent_val.lower() != 'none':
                    parent_match = re.search(r'(idea-[a-f0-9]+)', parent_val)
                    if parent_match and current_idea_id:
                        parent_map[current_idea_id] = parent_match.group(1)

    return parent_map, idea_titles, {}, {}


def build_dag(parent_map):
    """Build DAG from parent map. Returns children map and depths."""
    children = defaultdict(list)
    all_nodes = set(parent_map.keys()) | set(parent_map.values())

    for child, parent in parent_map.items():
        children[parent].append(child)

    # Compute depth iteratively (handle cycles gracefully)
    depth = {}

    def get_depth(node, visited=None):
        if visited is None:
            visited = set()
        if node in depth:
            return depth[node]
        if node in visited:
            return 0  # Cycle detected
        visited.add(node)
        if node not in parent_map:
            depth[node] = 0
            return 0
        parent = parent_map[node]
        d = get_depth(parent, visited) + 1
        depth[node] = d
        return d

    for node in all_nodes:
        if node not in depth:
            get_depth(node)

    roots = set()
    for node in all_nodes:
        if node not in parent_map:
            roots.add(node)

    return children, depth, roots


def trace_lineage(idea_id, parent_map, idea_titles):
    """Trace the lineage of an idea back to its root."""
    lineage = [idea_id]
    current = idea_id
    visited = set()
    while current in parent_map and current not in visited:
        visited.add(current)
        current = parent_map[current]
        lineage.append(current)
    lineage.reverse()
    return [{
        'idea_id': idea,
        'title': idea_titles.get(idea, 'Unknown'),
    } for idea in lineage]


def load_idea_aps():
    """Load AP values for ideas that have them."""
    idea_aps = {}
    idea_dirs = glob.glob(os.path.join(RESULTS_DIR, 'idea-*'))

    for idea_dir in idea_dirs:
        idea_id = os.path.basename(idea_dir)
        eval_path = os.path.join(idea_dir, 'ken_test_report.json')
        if not os.path.exists(eval_path):
            continue
        try:
            with open(eval_path) as f:
                eval_data = json.load(f)
            ap = eval_data.get('metrics', {}).get('average_precision')
            if ap is not None and isinstance(ap, (int, float)) and not np.isnan(ap):
                idea_aps[idea_id] = float(ap)
        except Exception:
            continue

    return idea_aps


def main():
    parent_map, idea_titles, idea_categories, idea_created = load_genealogy_from_db()
    children, depth, roots = build_dag(parent_map)
    idea_aps = load_idea_aps()

    all_depths = list(depth.values())
    max_depth = max(all_depths) if all_depths else 0
    print(f"Roots: {len(roots)}, Max depth: {max_depth}", file=sys.stderr)

    # ---- Depth statistics ----
    depth_counts = defaultdict(int)
    for d in all_depths:
        depth_counts[d] += 1

    # ---- Branching factor ----
    branching_factors = []
    for node, kids in children.items():
        if len(kids) > 0:
            branching_factors.append(len(kids))

    bf_stats = {}
    if branching_factors:
        bf_arr = np.array(branching_factors)
        bf_stats = {
            'mean': float(np.mean(bf_arr)),
            'median': float(np.median(bf_arr)),
            'std': float(np.std(bf_arr)),
            'max': int(np.max(bf_arr)),
            'n_parents': len(branching_factors),
        }

    # ---- Depth vs AP correlation ----
    from scipy.stats import spearmanr
    depth_ap_pairs = []
    for idea_id, ap in idea_aps.items():
        base_id = re.sub(r'-ht-\d+$', '', idea_id)
        if base_id in depth:
            depth_ap_pairs.append((depth[base_id], ap))
        elif idea_id in depth:
            depth_ap_pairs.append((depth[idea_id], ap))

    depth_ap_corr = None
    if len(depth_ap_pairs) >= 10:
        d_vals, ap_vals = zip(*depth_ap_pairs)
        # Check if either array is constant
        if len(set(d_vals)) > 1 and len(set(ap_vals)) > 1:
            corr, p_val = spearmanr(d_vals, ap_vals)
            depth_ap_corr = {
                'spearman_rho': float(corr),
                'p_value': float(p_val),
                'n': len(depth_ap_pairs),
            }
            print(f"Depth-AP Spearman: rho={corr:.4f}, p={p_val:.2e}", file=sys.stderr)

    # ---- Depth-stratified AP stats ----
    depth_ap_stats = defaultdict(list)
    for d, ap in depth_ap_pairs:
        depth_ap_stats[d].append(ap)
    depth_ap_summary = {}
    for d in sorted(depth_ap_stats.keys()):
        aps = depth_ap_stats[d]
        if len(aps) >= 3:
            depth_ap_summary[str(d)] = {
                'count': len(aps),
                'mean_ap': float(np.mean(aps)),
                'std_ap': float(np.std(aps)),
                'best_ap': float(np.max(aps)),
            }

    # ---- Champion lineage ----
    champion_id = 'idea-2ec818'
    champion_lineage = trace_lineage(champion_id, parent_map, idea_titles)

    # Top lineages
    top_lineages = {}
    sorted_ideas = sorted(idea_aps.items(), key=lambda x: -x[1])[:10]
    for idea_id, ap in sorted_ideas:
        base_id = re.sub(r'-ht-\d+$', '', idea_id)
        lineage = trace_lineage(base_id, parent_map, idea_titles)
        top_lineages[idea_id] = {
            'ap': ap,
            'depth': depth.get(base_id, depth.get(idea_id, 0)),
            'lineage': lineage,
        }

    # ---- Refinement success rate ----
    improved_count = 0
    parent_child_pairs = 0
    for child_id, parent_id in parent_map.items():
        child_ap = idea_aps.get(child_id)
        parent_ap = idea_aps.get(parent_id)
        if child_ap is not None and parent_ap is not None:
            parent_child_pairs += 1
            if child_ap > parent_ap:
                improved_count += 1

    refinement_success_rate = improved_count / max(parent_child_pairs, 1)

    # ---- Category distribution ----
    cat_counts = defaultdict(int)
    for cat in idea_categories.values():
        cat_counts[cat] += 1
    total_cats = sum(cat_counts.values())
    cat_dist = {k: {'count': v, 'fraction': float(v / total_cats)}
                for k, v in sorted(cat_counts.items(), key=lambda x: -x[1])}

    # ---- Compile output ----
    output = {
        'n_ideas_total': len(idea_titles),
        'n_ideas_with_parents': len(parent_map),
        'n_roots': len(roots),
        'max_depth': int(max_depth),
        'mean_depth': float(np.mean(all_depths)) if all_depths else 0,
        'depth_distribution': {str(k): v for k, v in sorted(depth_counts.items())},
        'branching_factor': bf_stats,
        'depth_ap_correlation': depth_ap_corr,
        'depth_ap_summary': depth_ap_summary,
        'champion_lineage': {
            'idea_id': champion_id,
            'ap': idea_aps.get(champion_id),
            'depth': depth.get(champion_id, 0),
            'lineage': champion_lineage,
        },
        'top_10_lineages': top_lineages,
        'refinement_analysis': {
            'parent_child_pairs_with_ap': parent_child_pairs,
            'improved': improved_count,
            'success_rate': float(refinement_success_rate),
        },
        'category_distribution': cat_dist,
    }

    # Save JSON
    out_path = os.path.join(OUTPUT_DIR, 'genealogy.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}", file=sys.stderr)

    # ---- Print key values ----
    print("\n% === GENEALOGY VALUES FOR PAPER ===")
    print(f"% Total ideas: {len(idea_titles)}")
    print(f"% Ideas with parent pointers: {len(parent_map)} "
          f"({100 * len(parent_map) / max(len(idea_titles), 1):.1f}%)")
    print(f"% Max depth: {max_depth}")
    print(f"% Mean depth: {np.mean(all_depths):.2f}" if all_depths else "% Mean depth: 0")
    if bf_stats:
        print(f"% Mean branching factor: {bf_stats['mean']:.2f}")
        print(f"% Max branching factor: {bf_stats['max']}")
    if depth_ap_corr:
        print(f"% Depth-AP Spearman: rho={depth_ap_corr['spearman_rho']:.4f}, "
              f"p={depth_ap_corr['p_value']:.2e}")
    print(f"% Champion (idea-2ec818) depth: {depth.get(champion_id, 0)}")
    if champion_lineage:
        print(f"% Champion lineage: {' -> '.join(l['idea_id'] for l in champion_lineage)}")
    print(f"% Refinement success rate: {refinement_success_rate:.1%} "
          f"({improved_count}/{parent_child_pairs})")
    print(f"% Top category: architecture ({cat_dist.get('architecture', {}).get('fraction', 0):.1%})")


if __name__ == '__main__':
    main()
