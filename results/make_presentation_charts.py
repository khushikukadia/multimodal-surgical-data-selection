import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results = pd.read_csv('results/results.csv')
finetune = pd.read_csv('results/results_finetune.csv')

COLORS = {'random': '#7B9EC2', 'vision': '#E8A87C', 'multimodal': '#7EC8A0', 'stratified': '#B088C9', 'full': '#B0B0B0'}
METHODS = ['random', 'stratified', 'vision', 'multimodal']
METHOD_LABELS = {'random': 'Random', 'stratified': 'Stratified', 'vision': 'Vision K-means', 'multimodal': 'Multimodal K-means'}

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

# ── 1. Macro F1 vs Budget (frozen, all budgets) ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for method in METHODS:
    sub = results[results['method'] == method].sort_values('budget')
    ax.errorbar(sub['budget'] * 100, sub['macro_f1'], yerr=sub['macro_f1_std'],
                marker='o', linewidth=2.5, markersize=7, capsize=4,
                color=COLORS[method], label=METHOD_LABELS[method])
full_f1 = float(results[results['method'] == 'full']['macro_f1'].iloc[0])
ax.axhline(full_f1, linestyle='--', color=COLORS['full'], linewidth=1.8, label=f'Full data ({full_f1:.3f})')
ax.set_xlabel('Training Budget (%)', fontsize=13)
ax.set_ylabel('Macro F1', fontsize=13)
ax.set_title('Macro F1 vs. Training Budget\n(Frozen CLIP + Logistic Regression, mean ± std over 5 seeds)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xticks([1, 5, 10, 25, 50])
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('results/chart_macro_f1_vs_budget.png', dpi=150)
plt.close()
print("Saved chart_macro_f1_vs_budget.png")

# ── 2. Frozen vs Fine-tuned comparison (grouped bar) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
for ax, budget, budget_label in zip(axes, [0.05, 0.1], ['5% Budget', '10% Budget']):
    x = np.arange(len(METHODS))
    width = 0.35
    frozen_vals = [float(results[(results['method']==m) & (np.isclose(results['budget'], budget))]['macro_f1'].iloc[0]) for m in METHODS]
    frozen_std  = [float(results[(results['method']==m) & (np.isclose(results['budget'], budget))]['macro_f1_std'].iloc[0]) for m in METHODS]
    ft_vals = [float(finetune[(finetune['method']==m) & (np.isclose(finetune['budget'], budget))]['macro_f1'].iloc[0]) for m in METHODS]
    ft_std  = [float(finetune[(finetune['method']==m) & (np.isclose(finetune['budget'], budget))]['macro_f1_std'].iloc[0]) for m in METHODS]

    bars1 = ax.bar(x - width/2, frozen_vals, width, yerr=frozen_std, capsize=4,
                   color=[COLORS[m] for m in METHODS], alpha=0.5, edgecolor='grey', label='Frozen')
    bars2 = ax.bar(x + width/2, ft_vals, width, yerr=ft_std, capsize=4,
                   color=[COLORS[m] for m in METHODS], alpha=1.0, edgecolor='grey', label='Fine-tuned')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=9)
    ax.set_ylabel('Macro F1', fontsize=12)
    ax.set_title(f'Frozen vs Fine-tuned — {budget_label}', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0.25, 0.55)

fig.suptitle('Fine-tuning boosts all methods — does selection method matter more?', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig('results/chart_frozen_vs_finetuned.png', dpi=150)
plt.close()
print("Saved chart_frozen_vs_finetuned.png")

# ── 3. Fine-tuning gain per method ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(METHODS))
width = 0.35
for i, (budget, label) in enumerate([(0.05, '5% Budget'), (0.1, '10% Budget')]):
    gains = []
    for m in METHODS:
        frozen = float(results[(results['method']==m) & (np.isclose(results['budget'], budget))]['macro_f1'].iloc[0])
        ft = float(finetune[(finetune['method']==m) & (np.isclose(finetune['budget'], budget))]['macro_f1'].iloc[0])
        gains.append(ft - frozen)
    ax.bar(x + (i - 0.5) * width, gains, width, label=label,
           color=['#5B9BD5', '#ED7D31'][i], alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=11)
ax.set_ylabel('Macro F1 Gain (Fine-tuned − Frozen)', fontsize=12)
ax.set_title('Fine-tuning Gain by Sampling Method', fontsize=13)
ax.legend(fontsize=11)
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('results/chart_finetune_gain.png', dpi=150)
plt.close()
print("Saved chart_finetune_gain.png")

# ── 4. Summary bar: mean macro F1 per method (frozen) ───────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
non_full = results[results['method'].isin(METHODS)]
mean_f1 = non_full.groupby('method')['macro_f1'].mean().reindex(METHODS)
bars = ax.bar([METHOD_LABELS[m] for m in METHODS], mean_f1.values,
              color=[COLORS[m] for m in METHODS], edgecolor='white', width=0.5)
for bar, val in zip(bars, mean_f1.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Macro F1 (across budgets)', fontsize=12)
ax.set_title('Average Macro F1 by Sampling Method\n(Frozen CLIP, averaged over all budgets & 5 seeds)', fontsize=12)
ax.set_ylim(0.30, 0.46)
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('results/chart_mean_f1_summary.png', dpi=150)
plt.close()
print("Saved chart_mean_f1_summary.png")

# ── 5. Per-class F1 at 10% budget (frozen) ───────────────────────────────────
per_class = pd.read_csv('results/per_class_f1.csv')
fig, ax = plt.subplots(figsize=(12, 5.5))
budget = 0.1
x = np.arange(len(PHASES))
width = 0.2
for i, method in enumerate(METHODS):
    row = per_class[(per_class['method'] == method) & (np.isclose(per_class['budget'], budget))]
    if row.empty: continue
    vals = [float(row[p].iloc[0]) for p in PHASES]
    ax.bar(x + (i - 1.5) * width, vals, width, label=METHOD_LABELS[method],
           color=COLORS[method], edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([PHASE_SHORT[p] for p in PHASES], fontsize=9.5)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Per-Phase F1 at 10% Training Budget (Frozen CLIP)', fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.85)
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('results/chart_per_class_f1_10pct.png', dpi=150)
plt.close()
print("Saved chart_per_class_f1_10pct.png")

print("\nAll charts generated!")
