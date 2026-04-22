# 1. Load responses (after annotators complete survey)
responses = pd.read_csv("survey_responses.csv")
# Expected columns: annotator_id, pair_id, fluency, relevance, realism

# 2. Merge with master to get labels
master = pd.read_csv(f"{base}\\master_with_conditions.csv")

results = responses.merge(
    master[['annotator_id', 'pair_id', 'condition', 'is_shared']],
    on=['annotator_id', 'pair_id'],
    how='left'
)

# 3. Krippendorff's alpha — gold items only
gold = results[results['is_shared'] == True]

# Pivot to matrix: rows = pair_id, columns = annotator_id
for metric in ['fluency', 'relevance', 'realism']:
    matrix = gold.pivot(index='pair_id', columns='annotator_id', values=metric).values
    # pass matrix to krippendorff alpha function
    print(f"Krippendorff's α ({metric}): ...")

# 4. Real vs synthetic comparison — all items
real = results[results['condition'] == 'real']
syn  = results[results['condition'] == 'synthetic']

for metric in ['fluency', 'relevance', 'realism']:
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(real[metric], syn[metric], alternative='two-sided')
    print(f"{metric}: real={real[metric].mean():.2f}, "
          f"synthetic={syn[metric].mean():.2f}, p={p:.3f}")
    
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── After merging results with master ────────────────────────────────
# results has columns: condition, fluency, relevance, realism

metrics = ['fluency', 'relevance', 'realism']

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

colors     = {'real': '#E8A87C', 'synthetic': '#7B9EC9'}  # orange, blue
hatches    = {'real': '***',     'synthetic': 'xxx'}
conditions = ['real', 'synthetic']
labels     = ['Real', 'Synthetic']

for ax, metric in zip(axes, metrics):

    data_to_plot = [
        results[results['condition'] == cond][metric].dropna().values
        for cond in conditions
    ]

    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        widths=0.4,
        showfliers=True,
        medianprops=dict(color='black', linewidth=2)
    )

    # Apply colors and hatches
    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(colors[cond])
        patch.set_hatch(hatches[cond])
        patch.set_alpha(0.85)

    ax.set_title(metric.capitalize(), fontsize=13, fontweight='bold')
    ax.set_xticks([])                          # remove x ticks (legend handles it)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(1, 5.2)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

# ── Legend ────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=colors[cond], hatch=hatches[cond],
                   alpha=0.85, label=label)
    for cond, label in zip(conditions, labels)
]

fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=2,
    fontsize=11,
    frameon=True,
    bbox_to_anchor=(0.5, -0.05)
)

plt.suptitle('Human Evaluation: Real vs Synthetic Comments', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{base}\\human_eval_boxplots.png", dpi=300, bbox_inches='tight')
plt.show()