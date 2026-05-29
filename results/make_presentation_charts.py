import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

results = pd.read_csv('results/results.csv')
per_class = pd.read_csv('results/per_class_f1.csv')

COLORS = {'random': '#7B9EC2', 'vision': '#E8A87C', 'multimodal': '#7EC8A0', 'full': '#B0B0B0'}
PHASE_SHORT = {
    'CalotTriangleDissection': 'Calot\nDissection',
    'CleaningCoagulation': 'Cleaning\nCoag.',
    'ClippingCutting': 'Clipping\nCutting',
    'GallbladderDissection': 'Gallbladder\nDissection',
    'GallbladderExtraction': 'Gallbladder\nExtraction',
    'GallbladderPackaging': 'Gallbladder\nPackaging',
    'Preparation': 'Preparation',
}
PHASES = list(PHASE_SHORT.keys())

# ── 1. Macro F1 vs Budget ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for method in ['random', 'vision', 'multimodal']:
    sub = results[results['method'] == method].sort_values('budget')
    ax.plot(sub['budget'] * 100, sub['macro_f1'], marker='o', linewidth=2.5,
            markersize=8, color=COLORS[method], label=method.capitalize())
full_f1 = float(results[results['method'] == 'full']['macro_f1'])
ax.axhline(full_f1, linestyle='--', color=COLORS['full'], linewidth=1.8, label=f'Full data ({full_f1:.3f})')
ax.set_xlabel('Training Budget (%)', fontsize=13)
ax.set_ylabel('Macro F1', fontsize=13)
ax.set_title('Macro F1 vs. Training Budget\n(CholecTrack20, CLIP + Logistic Regression)', fontsize=13)
ax.legend(fontsize=11)
ax.set_xticks([10, 25, 50])
ax.set_ylim(0.42, 0.65)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('results/chart_macro_f1_vs_budget.png', dpi=150)
plt.close()
print("Saved chart_macro_f1_vs_budget.png")

# ── 2. Accuracy vs Budget ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for method in ['random', 'vision', 'multimodal']:
    sub = results[results['method'] == method].sort_values('budget')
    ax.plot(sub['budget'] * 100, sub['accuracy'] * 100, marker='o', linewidth=2.5,
            markersize=8, color=COLORS[method], label=method.capitalize())
full_acc = float(results[results['method'] == 'full']['accuracy']) * 100
ax.axhline(full_acc, linestyle='--', color=COLORS['full'], linewidth=1.8, label=f'Full data ({full_acc:.1f}%)')
ax.set_xlabel('Training Budget (%)', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Accuracy vs. Training Budget\n(CholecTrack20, CLIP + Logistic Regression)', fontsize=13)
ax.legend(fontsize=11)
ax.set_xticks([10, 25, 50])
ax.set_ylim(50, 70)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('results/chart_accuracy_vs_budget.png', dpi=150)
plt.close()
print("Saved chart_accuracy_vs_budget.png")

# ── 3. Per-class F1 at 10% budget (grouped bar) ──────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
budget = 0.1
methods = ['random', 'vision', 'multimodal']
x = np.arange(len(PHASES))
width = 0.25
for i, method in enumerate(methods):
    row = per_class[(per_class['method'] == method) & (np.isclose(per_class['budget'], budget))]
    vals = [float(row[p]) for p in PHASES]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=method.capitalize(),
                  color=COLORS[method], edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([PHASE_SHORT[p] for p in PHASES], fontsize=9.5)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Per-Phase F1 at 10% Training Budget', fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim(0, 0.85)
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('results/chart_per_class_f1_10pct.png', dpi=150)
plt.close()
print("Saved chart_per_class_f1_10pct.png")

# ── 4. Per-class F1 at 50% budget (grouped bar) ──────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
budget = 0.5
for i, method in enumerate(methods):
    row = per_class[(per_class['method'] == method) & (np.isclose(per_class['budget'], budget))]
    vals = [float(row[p]) for p in PHASES]
    ax.bar(x + (i - 1) * width, vals, width, label=method.capitalize(),
           color=COLORS[method], edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([PHASE_SHORT[p] for p in PHASES], fontsize=9.5)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Per-Phase F1 at 50% Training Budget', fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim(0, 0.85)
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('results/chart_per_class_f1_50pct.png', dpi=150)
plt.close()
print("Saved chart_per_class_f1_50pct.png")

# ── 5. Summary bar: mean macro F1 per method ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
non_full = results[results['method'] != 'full']
mean_f1 = non_full.groupby('method')['macro_f1'].mean().reindex(['random', 'vision', 'multimodal'])
bars = ax.bar(mean_f1.index, mean_f1.values,
              color=[COLORS[m] for m in mean_f1.index],
              edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars, mean_f1.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Macro F1 (across budgets)', fontsize=12)
ax.set_title('Average Macro F1 by Sampling Method\n(averaged over 10%, 25%, 50% budgets)', fontsize=12)
ax.set_ylim(0.50, 0.535)
ax.set_xticklabels(['Random', 'Vision-only\nK-means', 'Multimodal\nK-means'], fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('results/chart_mean_f1_summary.png', dpi=150)
plt.close()
print("Saved chart_mean_f1_summary.png")

print("\nAll charts generated!")
