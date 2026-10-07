import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mannwhitneyu

results_base = r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\human_evaluation\results"
subsets_base = r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\human_evaluation\subsets"
metrics = ['fluency', 'relevance', 'humanness']

# Circular sliding window overlap pairs (Ann 7 → Ann 1 closes the loop)
overlap_pairs = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,1)]

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: PARSE GOOGLE FORM RESPONSES
# ═══════════════════════════════════════════════════════════════════════

def parse_annotator_response(csv_path, annotator_id, subset_path=None):
    """
    Parse Google Form CSV export.
    Columns are repeated question text with pandas suffixes (.1, .2, ... .19).
    Groups of 3 columns per item: fluency, relevance, humanness.
    item_order is looked up from the annotator subset file to recover pair_id.
    """
    df  = pd.read_csv(csv_path)
    row = df.iloc[0].values  # one submission per annotator

    if subset_path is None:
        subset_path = f"{subsets_base}\\annotator_{annotator_id:02d}.csv"

    if subset_path.endswith('.json'):
        with open(subset_path, encoding='utf-8') as f:
            subset = pd.DataFrame(json.load(f))
    else:
        subset = pd.read_csv(subset_path)

    order_to_pair = dict(zip(subset['item_order'], subset['pair_id']))
    n_items = len(subset)

    records = []
    for item_order in range(1, n_items + 1):
        pair_id  = order_to_pair[item_order]
        base_idx = 1 + (item_order - 1) * 3   # skip Timestamp at index 0
        records.append({
            'annotator_id': annotator_id,
            'pair_id':      pair_id,
            'fluency':      int(row[base_idx]),
            'relevance':    int(row[base_idx + 1]),
            'humanness':    int(row[base_idx + 2]),
        })

    return pd.DataFrame(records)

# Load all 7 annotators
all_responses = []
for ann_id in range(1, 8):
    if ann_id == 1:
        # Ann 1's subset was redesigned mid-study to close the circular loop.
        # The current unified 20-item submission maps via annotator_01_new.json.
        ann_df = parse_annotator_response(
            csv_path=f"{results_base}\\annotator_01_responses.csv",
            annotator_id=1,
            subset_path=f"{subsets_base}\\annotator_01_new.json",
        )
    else:
        path   = f"{results_base}\\annotator_{ann_id:02d}_responses.csv"
        ann_df = parse_annotator_response(path, ann_id)
    all_responses.append(ann_df)
    print(f"Annotator {ann_id}: {len(ann_df)} items parsed ✓")

responses = pd.concat(all_responses, ignore_index=True)
print(f"\nTotal responses: {len(responses)} (expect 140)")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: MERGE WITH MASTER TO GET CONDITION LABELS
# ═══════════════════════════════════════════════════════════════════════

master = pd.read_csv(f"{subsets_base}\\master_with_conditions.csv")

# Join on pair_id only — condition/item_id are properties of the pair,
# not the annotator. Ann 1's new items (pair_ids from Ann 7's block) exist
# in master under annotator_id=7, so joining on annotator_id+pair_id would miss them.
pair_info = (master[['pair_id', 'condition', 'item_id', 'post', 'comment']]
             .drop_duplicates('pair_id'))

results = responses.merge(pair_info, on='pair_id', how='left')

print(f"Null conditions: {results['condition'].isna().sum()} (expect 0)")
print(f"Condition counts:\n{results['condition'].value_counts()}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: COHEN'S KAPPA — SLIDING WINDOW OVERLAP PAIRS
# ═══════════════════════════════════════════════════════════════════════
# Only consecutive annotator pairs share items (overlap of 10)
# κ computed ONLY on shared items between each pair

print("\n=== INTER-ANNOTATOR AGREEMENT (Quadratic Weighted Cohen's κ) ===")

kappa_records = []

for ann_a, ann_b in overlap_pairs:

    # Get shared item_ids between this pair
    items_a = set(results[results['annotator_id'] == ann_a]['item_id'])
    items_b = set(results[results['annotator_id'] == ann_b]['item_id'])
    shared  = items_a & items_b

    print(f"\n  Pair ({ann_a},{ann_b}) — {len(shared)} shared items:")

    if len(shared) == 0:
        print(f"    WARNING: no shared items found!")
        continue

    scores_a = (results[(results['annotator_id'] == ann_a) &
                        (results['item_id'].isin(shared))]
                .sort_values('item_id'))

    scores_b = (results[(results['annotator_id'] == ann_b) &
                        (results['item_id'].isin(shared))]
                .sort_values('item_id'))

    for metric in metrics:
        kappa = cohen_kappa_score(
            scores_a[metric].values,
            scores_b[metric].values,
            weights='quadratic'
        )
        kappa_records.append({
            'pair':   f"({ann_a},{ann_b})",
            'metric': metric,
            'kappa':  round(kappa, 3)
        })
        print(f"    {metric:>10}: κ = {kappa:.3f}")

kappa_df = pd.DataFrame(kappa_records)

# Summary
print(f"\nMean κ per metric (across all overlap pairs):")
for metric, k in kappa_df.groupby('metric')['kappa'].mean().items():
    interpretation = (
        'slight'          if k < 0.20 else
        'fair'            if k < 0.40 else
        'moderate'        if k < 0.60 else
        'substantial'     if k < 0.80 else
        'almost perfect'
    )
    print(f"  {metric:>10}: κ = {k:.3f} ({interpretation})")

print(f"\nOverall mean κ: {kappa_df['kappa'].mean():.3f}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: AVERAGE SCORES PER ITEM
# ═══════════════════════════════════════════════════════════════════════
# Overlap items (seen by 2 annotators) → average of 2 ratings
# Edge items (seen by 1 annotator)     → single rating

item_scores = (results
               .groupby(['item_id', 'pair_id', 'condition'])[metrics]
               .mean()
               .reset_index())

print(f"\n=== MEAN SCORES PER CONDITION ===")
print(item_scores.groupby('condition')[metrics].mean().round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: MANN-WHITNEY U — REAL VS SYNTHETIC
# ═══════════════════════════════════════════════════════════════════════

print("\n=== REAL vs SYNTHETIC (Mann-Whitney U) ===")

real = item_scores[item_scores['condition'] == 'real']
syn  = item_scores[item_scores['condition'] == 'synthetic']

stat_records = []

for metric in metrics:
    stat, p = mannwhitneyu(
        real[metric].values,
        syn[metric].values,
        alternative='two-sided'
    )
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    stat_records.append({
        'metric':    metric,
        'real_mean': round(real[metric].mean(), 3),
        'syn_mean':  round(syn[metric].mean(), 3),
        'U':         stat,
        'p':         round(p, 4),
        'sig':       sig
    })
    print(f"  {metric:>10}: real={real[metric].mean():.2f}, "
          f"synthetic={syn[metric].mean():.2f}, "
          f"U={stat:.0f}, p={p:.3f} {sig}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 6: BOXPLOTS
# ═══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      10,
    'axes.linewidth': 0.8
})

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

colors     = {'real': '#F4A460', 'synthetic': '#9EB9D4'}
hatches    = {'real': '///',     'synthetic': '...'}
conditions = ['real', 'synthetic']
labels     = ['Real', 'Synthetic']

for ax, metric in zip(axes, metrics):

    data_to_plot = [
        item_scores[item_scores['condition'] == cond][metric].dropna().values
        for cond in conditions
    ]

    bp = ax.boxplot(
        data_to_plot,
        positions=[1, 2],
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker='o', markersize=4,
                        markerfacecolor='none',
                        markeredgewidth=0.8),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        boxprops=dict(linewidth=0.8)
    )

    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(colors[cond])
        patch.set_hatch(hatches[cond])
        patch.set_alpha(0.9)

    # Significance bar + stars
    stat_row = next(s for s in stat_records if s['metric'] == metric)
    y_bar = max(
        item_scores[item_scores['condition'] == 'real'][metric].max(),
        item_scores[item_scores['condition'] == 'synthetic'][metric].max()
    ) + 0.2
    ax.plot([1, 2], [y_bar, y_bar], color='black', linewidth=0.8)
    ax.text(1.5, y_bar + 0.05, stat_row['sig'],
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title(metric.capitalize(), fontsize=11, fontweight='bold', pad=6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel('Condition', fontsize=9, labelpad=4)
    ax.set_ylabel('Score', fontsize=9)
    ax.set_ylim(1, 5.8)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)

    # Legend inside first plot only
    if ax == axes[0]:
        legend_handles = [
            mpatches.Patch(
                facecolor=colors[cond],
                hatch=hatches[cond],
                alpha=0.9,
                label=label,
                linewidth=0.8
            )
            for cond, label in zip(conditions, labels)
        ]
        ax.legend(
            handles=legend_handles,
            loc='upper right',
            fontsize=8,
            frameon=True,
            framealpha=0.9,
            edgecolor='gray'
        )

plt.suptitle(
    'Figure X: Distribution of human evaluation scores on three qualitative metrics.',
    fontsize=9,
    y=-0.02,
    style='italic'
)

plt.tight_layout()
plt.savefig(f"{results_base}\\human_eval_boxplots.png", dpi=300, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════
# STEP 7: SAVE ALL RESULTS
# ═══════════════════════════════════════════════════════════════════════

item_scores.to_csv(f"{results_base}\\item_scores_averaged.csv", index=False)
kappa_df.to_csv(f"{results_base}\\kappa_results.csv", index=False)
pd.DataFrame(stat_records).to_csv(f"{results_base}\\mannwhitney_results.csv", index=False)

print(f"\n✓ Saved:")
print(f"  item_scores_averaged.csv  — averaged scores per item")
print(f"  kappa_results.csv         — Cohen's κ per overlap pair per metric")
print(f"  mannwhitney_results.csv   — Mann-Whitney U results")
print(f"  human_eval_boxplots.png   — figure for paper")