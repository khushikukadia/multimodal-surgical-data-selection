import pandas as pd
import numpy as np
import os

results  = pd.read_csv('results/results.csv')
finetune = pd.read_csv('results/results_finetune.csv')
meta     = pd.read_csv('data/cholectrack20_metadata.csv')

os.makedirs('results/paper_figures', exist_ok=True)

methods = ['random', 'stratified', 'vision', 'multimodal']
mlabels = {
    'random':     'Random',
    'stratified': 'Stratified',
    'vision':     'Vision K-means',
    'multimodal': 'Multimodal K-means',
}

def bld(m, s):
    return (r'\textbf{' + s + r'}') if m == 'multimodal' else s

lines = []
lines += [
    '% ============================================================',
    '% ALL FIGURES & TABLES — ordered by appearance in the paper',
    '% Upload all files in this folder to your Overleaf project',
    '% ============================================================',
    '',
]

# ── Section 4: Phase examples ────────────────────────────────────────────────
lines += [
    '% --- Fig 1  (Section 4: Dataset) ---',
    r'\begin{figure}[t]',
    r'  \centering',
    r'  \includegraphics[width=\linewidth]{fig1_phase_examples.png}',
    r'  \caption{Representative 1-fps frames for each of the seven CholecTrack20 surgical phases. '
    r'Calot Triangle Dissection and Gallbladder Dissection are visually similar, '
    r'motivating macro-F1 as the primary metric.}',
    r'  \label{fig:phases}',
    r'\end{figure}',
    '',
]

# ── Section 4: Per-phase frame count table ───────────────────────────────────
counts = meta.groupby('phase').size().reset_index(name='count').sort_values('count', ascending=False)
total  = int(counts['count'].sum())
lines += [
    '% --- Table (Section 4: Dataset) ---',
    r'\begin{table}[t]',
    r'  \centering',
    r'  \caption{Per-phase frame counts in the 12-video CholecTrack20 subset.}',
    r'  \label{tab:phases}',
    r'  \begin{tabular}{lcc}',
    r'  \toprule',
    r'  Phase & Frames & \% of total \\',
    r'  \midrule',
]
for _, row in counts.iterrows():
    pct = row['count'] / total * 100
    lines.append(f"  {row['phase']} & {row['count']:,} & {pct:.1f}\\% \\\\")
lines += [
    r'  \midrule',
    f'  Total & {total:,} & 100\\% \\\\',
    r'  \bottomrule',
    r'  \end{tabular}',
    r'\end{table}',
    '',
]

# ── Section 5: Performance vs budget ─────────────────────────────────────────
lines += [
    '% --- Fig 2  (Section 5: Frozen results) ---',
    r'\begin{figure}[t]',
    r'  \centering',
    r'  \includegraphics[width=\linewidth]{fig2_perf_vs_budget.png}',
    r'  \caption{Macro-F1 vs.\ training-data budget for all four samplers '
    r'(mean $\pm$ std over 5 seeds). The dashed line is the full-data upper bound. '
    r'Multimodal K-means leads at the 5\% budget; methods converge at 10\% and above.}',
    r'  \label{fig:perf_vs_budget}',
    r'\end{figure}',
    '',
]

# ── Section 5: Table X frozen ────────────────────────────────────────────────
full = results[results['method'] == 'full'].iloc[0]
lines += [
    '% --- Table X  (Section 5: Frozen results) ---',
    r'\begin{table}[t]',
    r'  \centering',
    r'  \caption{Frozen-feature macro-F1 (mean $\pm$ std, 5 seeds) at the 5\% and 10\% budgets.}',
    r'  \label{tab:frozen}',
    r'  \begin{tabular}{lcc}',
    r'  \toprule',
    r'  Method & 5\% Budget & 10\% Budget \\',
    r'  \midrule',
]
for m in methods:
    r5  = results[(results['method'] == m) & np.isclose(results['budget'], 0.05)].iloc[0]
    r10 = results[(results['method'] == m) & np.isclose(results['budget'], 0.10)].iloc[0]
    c5  = f"{r5['macro_f1']:.3f} $\\pm$ {r5['macro_f1_std']:.3f}"
    c10 = f"{r10['macro_f1']:.3f} $\\pm$ {r10['macro_f1_std']:.3f}"
    lines.append(f"  {bld(m, mlabels[m])} & {bld(m, c5)} & {c10} \\\\")
lines += [
    r'  \midrule',
    f"  Full data (100\\%) & --- & {full['macro_f1']:.3f} \\\\",
    r'  \bottomrule',
    r'  \end{tabular}',
    r'\end{table}',
    '',
]

# ── Section 5: Frozen vs fine-tuned figure ───────────────────────────────────
lines += [
    '% --- Fig 3  (Section 5: Fine-tuning results) ---',
    r'\begin{figure}[t]',
    r'  \centering',
    r'  \includegraphics[width=\linewidth]{fig3_frozen_vs_finetuned.png}',
    r'  \caption{Macro-F1 of frozen (light) vs.\ fine-tuned (solid) classifiers at the 5\% '
    r'and 10\% budgets. Fine-tuning amplifies all methods; multimodal selection gains most '
    r'at the 5\% budget.}',
    r'  \label{fig:frozen_vs_ft}',
    r'\end{figure}',
    '',
]

# ── Section 5: Table Y fine-tuned ────────────────────────────────────────────
lines += [
    '% --- Table Y  (Section 5: Fine-tuning results) ---',
    r'\begin{table}[t]',
    r'  \centering',
    r'  \caption{Fine-tuned macro-F1 (mean $\pm$ std, 2 seeds) at the 5\% and 10\% budgets.}',
    r'  \label{tab:finetuned}',
    r'  \begin{tabular}{lcc}',
    r'  \toprule',
    r'  Method & 5\% Budget & 10\% Budget \\',
    r'  \midrule',
]
for m in methods:
    r5  = finetune[(finetune['method'] == m) & np.isclose(finetune['budget'], 0.05)].iloc[0]
    r10 = finetune[(finetune['method'] == m) & np.isclose(finetune['budget'], 0.10)].iloc[0]
    c5  = f"{r5['macro_f1']:.3f} $\\pm$ {r5['macro_f1_std']:.3f}"
    c10 = f"{r10['macro_f1']:.3f} $\\pm$ {r10['macro_f1_std']:.3f}"
    lines.append(f"  {bld(m, mlabels[m])} & {bld(m, c5)} & {c10} \\\\")
lines += [
    r'  \bottomrule',
    r'  \end{tabular}',
    r'\end{table}',
    '',
]

# ── Section 5: Per-class F1 ───────────────────────────────────────────────────
lines += [
    '% --- Fig 4  (Section 5: Qualitative analysis) ---',
    r'\begin{figure}[t]',
    r'  \centering',
    r'  \includegraphics[width=\linewidth]{fig4_per_class_f1.png}',
    r'  \caption{Per-phase F1 at the 10\% training budget (frozen CLIP). '
    r'CleaningCoagulation and GallbladderDissection are consistently the hardest phases '
    r'across all sampling methods.}',
    r'  \label{fig:perclass}',
    r'\end{figure}',
    '',
]

# ── Section 5: Confusion matrices ────────────────────────────────────────────
lines += [
    '% --- Fig 5  (Section 5: Qualitative analysis) ---',
    r'\begin{figure}[t]',
    r'  \centering',
    r'  \includegraphics[width=\linewidth]{fig5_confusion_matrices.png}',
    r'  \caption{Confusion matrices for the stratified sampler at 10\% budget (left) '
    r'and multimodal K-means at 50\% budget (right). GallbladderDissection is '
    r'systematically confused with CalotTriangleDissection due to visual similarity.}',
    r'  \label{fig:cm}',
    r'\end{figure}',
    '',
]

out = '\n'.join(lines)
with open('results/paper_figures/latex_figures_and_tables.tex', 'w') as f:
    f.write(out)
print("Saved latex_figures_and_tables.tex")
