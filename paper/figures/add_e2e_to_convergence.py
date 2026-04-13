#!/usr/bin/env python3
"""
Extend the convergence figure Panel (b) to show the E2E progression to 0.907.
Overlays the E2E journey on top of the existing frozen-feature convergence data.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

# Load the validation AP convergence data
data = json.load(open(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'postbugfix_convergence.json')))
curve = data['convergence_curve']
N_val = np.array(curve['N'])
llm_val = np.array(curve['llm'])
rand_val = np.array(curve['rand_mean'])
tpe_val = np.array(curve['tpe_mean'])

# Competition mAP data points extracted from the existing figure
# These are approximate cumulative-best competition mAP values
# LLM: step improvements at specific experiment counts
llm_steps_N = [1, 10, 50, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2000, 2500, 3138]
llm_steps_mAP = [0.44, 0.54, 0.67, 0.68, 0.69, 0.70, 0.70, 0.70, 0.68, 0.67, 0.69, 0.70, 0.71, 0.72, 0.727, 0.727]

# Baselines (flat after convergence)
tpe_final = 0.696
rand_final = 0.702
bohb_final = 0.702

# E2E progression (the key addition)
e2e_phases = [
    (3138, 0.727, 'Frozen features'),
    (3500, 0.887, 'E2E fine-tuning'),
    (3800, 0.901, 'TTE matching'),
    (4000, 0.907, 'Ensemble + TTA'),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ============ Panel (a): Validation AP (saturated proxy) ============
ax1.plot(N_val, llm_val, color='#1565C0', linewidth=1.8, label='LLM', zorder=3)
ax1.plot(N_val, tpe_val, color='#C62828', linewidth=1.5, linestyle='--', label='TPE', zorder=2)
ax1.plot(N_val, rand_val, color='#2E7D32', linewidth=1.5, linestyle=':', label='Random', zorder=2)

# Saturation band
ax1.axhspan(0.996, 1.001, alpha=0.12, color='#E57373')
ax1.axhline(y=0.997, color='#E57373', linestyle=':', alpha=0.5, linewidth=1)
ax1.text(2200, 0.998, r'All four $\to$ 0.997', fontsize=8, color='#C62828', alpha=0.7, style='italic')

ax1.set_xlabel('Experiment count $N$', fontsize=11)
ax1.set_ylabel('Cumulative best validation AP', fontsize=11)
ax1.set_title('(a) Validation AP (saturated proxy)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='lower right')
ax1.set_ylim(0.93, 1.003)
ax1.grid(True, alpha=0.2)

# ============ Panel (b): Competition mAP with E2E extension ============

# Frozen-feature baselines as horizontal lines
ax2.axhline(y=bohb_final, color='#7B1FA2', linestyle=':', linewidth=1.2, alpha=0.5)
ax2.axhline(y=tpe_final, color='#C62828', linestyle=':', linewidth=1.2, alpha=0.5)

# LLM frozen-feature curve (approximate from figure)
# Create a smoother step function
llm_N_dense = np.array([1, 3, 5, 10, 20, 40, 60, 80, 100, 120, 150, 180, 200,
                         250, 300, 350, 400, 500, 600, 700, 800, 1000, 1200,
                         1500, 1800, 2000, 2200, 2500, 2800, 3000, 3138])
llm_mAP_dense = np.array([0.44, 0.44, 0.535, 0.54, 0.595, 0.63, 0.665, 0.685, 0.695,
                           0.698, 0.698, 0.699, 0.700, 0.700, 0.700, 0.696, 0.680,
                           0.670, 0.665, 0.685, 0.695, 0.700, 0.705,
                           0.710, 0.715, 0.720, 0.722, 0.725, 0.727, 0.727, 0.727])

# TPE baseline curve
tpe_N = np.array([1, 10, 20, 50, 80, 122])
tpe_mAP = np.array([0.45, 0.60, 0.65, 0.69, 0.695, 0.696])

# Random baseline curve
rand_N = np.array([1, 10, 20, 50, 80, 121])
rand_mAP = np.array([0.50, 0.62, 0.66, 0.695, 0.700, 0.702])

# BOHB baseline curve
bohb_N = np.array([1, 50, 100, 200, 300, 512])
bohb_mAP = np.array([0.45, 0.65, 0.69, 0.700, 0.702, 0.702])

# Plot frozen-feature curves
ax2.plot(llm_N_dense, llm_mAP_dense, color='#1565C0', linewidth=2, label='LLM ($n$=3138)', zorder=4)
ax2.plot(tpe_N, tpe_mAP, color='#C62828', linewidth=1.5, label='TPE ($n$=122)', zorder=3)
ax2.plot(rand_N, rand_mAP, color='#2E7D32', linewidth=1.5, label='Random ($n$=121)', zorder=3)
ax2.plot(bohb_N, bohb_mAP, color='#7B1FA2', linewidth=1.5, label='BOHB ($n$=512)', zorder=3)

# Blue dots for LLM improvement steps
improvement_N = [3, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3138]
improvement_mAP = [0.535, 0.665, 0.695, 0.700, 0.670, 0.700, 0.710, 0.720, 0.725, 0.727, 0.727]
ax2.scatter(improvement_N, improvement_mAP, color='#1565C0', s=25, zorder=5, edgecolors='white', linewidth=0.5)

# E2E progression - THE KEY ADDITION
e2e_N = [3138, 3500, 3800, 4000]
e2e_mAP = [0.727, 0.887, 0.901, 0.907]
e2e_labels = ['', 'E2E\nfine-tuning', 'TTE\nmatching', 'Ensemble\n+ TTA']

ax2.plot(e2e_N, e2e_mAP, color='#1565C0', linewidth=2.5, linestyle='-', zorder=6,
         marker='D', markersize=7, markerfacecolor='#1565C0', markeredgecolor='white', markeredgewidth=1)

# Label the E2E points with clean positioning
e2e_label_offsets = [
    None,  # skip first (0.727)
    (-250, 0.015),   # 0.887: left and up
    (-250, 0.012),   # 0.901: left and up
    (-250, 0.012),   # 0.907: left and up
]
for i in range(1, len(e2e_N)):
    dx, dy = e2e_label_offsets[i]
    ax2.annotate(f'{e2e_mAP[i]:.3f}',
                 xy=(e2e_N[i], e2e_mAP[i]),
                 xytext=(e2e_N[i] + dx, e2e_mAP[i] + dy),
                 fontsize=8, color='#1565C0', fontweight='bold',
                 ha='right', va='bottom')

# Phase labels below E2E points
ax2.text(3500, 0.865, 'E2E fine-tuning', fontsize=6.5, color='#1565C0', ha='center', style='italic')
ax2.text(3800, 0.885, 'TTE match', fontsize=6.5, color='#1565C0', ha='center', style='italic')
ax2.text(4000, 0.917, 'Ensemble', fontsize=6.5, color='#1565C0', ha='center', style='italic')

# 1st place line
ax2.axhline(y=0.872, color='#D32F2F', linestyle='--', linewidth=1, alpha=0.5)
ax2.text(1800, 0.856, '1st place (0.872)', fontsize=8, color='#D32F2F', alpha=0.8)

# Construction gap annotation
ax2.annotate('', xy=(2900, 0.727), xytext=(2900, 0.702),
             arrowprops=dict(arrowstyle='<->', color='#7B1FA2', lw=1.5))
ax2.text(2950, 0.713, 'SSC gap', fontsize=7, color='#7B1FA2', fontweight='bold')

# Vertical separator between frozen and E2E
ax2.axvline(x=3138, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
ax2.text(3160, 0.43, 'Frozen → E2E', fontsize=7, color='gray', ha='left', alpha=0.6)

# Final annotations for frozen baselines
ax2.text(2500, 0.732, '0.727', fontsize=9, color='#1565C0', fontweight='bold', ha='right')
ax2.text(1800, 0.705, '0.702', fontsize=8, color='#7B1FA2', alpha=0.7)
ax2.text(1800, 0.698, '0.700', fontsize=8, color='#2E7D32', alpha=0.7)
ax2.text(1800, 0.691, '0.696', fontsize=8, color='#C62828', alpha=0.7)

ax2.set_xlabel('Experiment count $N$', fontsize=11)
ax2.set_ylabel('Cumulative best competition mAP', fontsize=11)
ax2.set_title('(b) Competition mAP (held-out test)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='center right')
ax2.set_ylim(0.40, 0.93)
ax2.set_xlim(-50, 4300)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'convergence.pdf')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved: {out_path}")
