import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu
from sklearn.metrics import cohen_kappa_score

base = r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\Fine Tune gemma\objective"

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
# Google Form exports one CSV per annotator
# Each CSV has one row (the annotator's submission)
# Columns: Timestamp, Item 1 of 20 [ID:X] Fluency, Item 1 of 20 [ID:X] Relevance, etc.

master = pd.read_csv(f"{base}\\master_with_conditions.csv")

annotator_pairs = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8)
]

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: PARSE GOOGLE FORM RESPONSES
# ═══════════════════════════════════════════════════════════════════════
# Google Form responses come in wide format (one row per annotator,
# one column per question). We parse and reshape to long format.

import re

def parse_annotator_response(csv_path, annotator_id):
    """
    Parse a single annotator's Google Form response CSV.
    Extracts pair_id from column names and reshapes to long format.
    
    Returns DataFrame with columns:
        annotator_id, pair_id, fluency, relevance, realism
    """
    df = pd.read_csv(csv_path)
    
    # Take first data row (one submission per annotator)
    row = df.iloc[0]
    
    records = []
    
    # Column names look like: "Item 3 of 20 [ID:47] Fluency"
    fluency_cols   = [c for c in df.columns if re.search(r'\[ID:\d+\]', c) and 'Fluency'   in c]
    relevance_cols = [c for c in df.columns if re.search(r'\[ID:\d+\]', c) and 'Relevance' in c]
    realism_cols   = [c for c in df.columns if re.search(r'\[ID:\d+\]', c) and 'Realism'   in c]
    
    for f_col, rel_col, rea_col in zip(fluency_cols, relevance_cols, realism_cols):
        # Extract pair_id from column name
        pair_id = int(re.search(r'\[ID:(\d+)\]', f_col).group(1))
        
        records.append({
            'annotator_id': annotator_id,
            'pair_id':      pair_id,
            'fluency':      int(row[f_col]),
            'relevance':    int(row[rel_col]),
            'realism':      int(row[rea_col])
        })
    
    return pd.DataFrame(records)

# Load all 8 annotator responses
all_responses = []
for ann_id in range(1, 9):
    csv_path = f"{base}\\annotator_{ann_id:02d}_responses.csv"
    ann_df   = parse_annotator_response(csv_path, ann_id)
    all_responses.append(ann_df)
    print(f"Annotator {ann_id}: {len(ann_df)} items parsed ✓")

responses = pd.concat(all_responses, ignore_index=True)
print(f"\nTotal responses: {len(responses)} (expect 160)")

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: MERGE WITH MASTER TO GET CONDITION LABELS
# ═══════════════════════════════════════════════════════════════════════

results = responses.merge(
    master[['annotator_id', 'pair_id', 'condition', 'item_id']],
    on=['annotator_id', 'pair_id'],
    how='left'
)

# Sanity check
print(f"\nMerge check — null conditions: {results['condition'].isna().sum()} (expect 0)")
print(f"Condition counts:\n{results['condition'].value_counts().to_string()}")

metrics = ['fluency', 'relevance', 'realism']

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: COHEN'S KAPPA — INTER-ANNOTATOR AGREEMENT
# ═══════════════════════════════════════════════════════════════════════
# Each item was seen by exactly 2 annotators (one pair)
# Compute Cohen's κ per pair per metric
# Then report mean κ across all pairs

print("\n=== INTER-ANNOTATOR AGREEMENT (Cohen's κ) ===")

kappa_records = []

for ann_a, ann_b in annotator_pairs:
    scores_a = results[results['annotator_id'] == ann_a].sort_values('item_id')
    scores_b = results[results['annotator_id'] == ann_b].sort_values('item_id')
    
    for metric in metrics:
        kappa = cohen_kappa_score(
            scores_a[metric].values,
            scores_b[metric].values,
            weights='quadratic'   # quadratic weighted κ for ordinal scales
        )
        kappa_records.append({
            'pair':   f"({ann_a},{ann_b})",
            'metric': metric,
            'kappa':  kappa
        })
        print(f"Pair ({ann_a},{ann_b}) — {metric:>9}: κ = {kappa:.3f}")

kappa_df = pd.DataFrame(kappa_records)

print(f"\nMean κ per metric:")
mean_kappa = kappa_df.groupby('metric')['kappa'].mean()
for metric, k in mean_kappa.items():
    print(f"  {metric:>9}: κ = {k:.3f}")

print(f"\nOverall mean κ: {kappa_df['kappa'].mean():.3f}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: AVERAGE SCORES PER ITEM (MEAN OF 2 ANNOTATORS)
# ═══════════════════════════════════════════════════════════════════════
# For analysis and plotting, average the two annotators' scores per item

item_scores = results.groupby(['item_id', 'pair_id', 'condition'])[metrics].mean().reset_index()

print(f"\n=== MEAN SCORES (averaged across 2 annotators per item) ===")
print(item_scores.groupby('condition')[metrics].mean().round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════
# STEP 6: REAL VS SYNTHETIC COMPARISON — MANN-WHITNEY U TEST
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
    stat_records.append({
        'metric':    metric,
        'real_mean': real[metric].mean(),
        'syn_mean':  syn[metric].mean(),
        'U':         stat,
        'p':         p
    })
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"{metric:>9}: real={real[metric].mean():.2f}, "
          f"synthetic={syn[metric].mean():.2f}, "
          f"U={stat:.0f}, p={p:.3f} {sig}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 7: BOXPLOTS
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

colors     = {'real': '#E8A87C', 'synthetic': '#7B9EC9'}
hatches    = {'real': '***',     'synthetic': 'xxx'}
conditions = ['real', 'synthetic']
labels     = ['Real', 'Synthetic']

for ax, metric in zip(axes, metrics):

    data_to_plot = [
        item_scores[item_scores['condition'] == cond][metric].dropna().values
        for cond in conditions
    ]

    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        widths=0.4,
        showfliers=True,
        medianprops=dict(color='black', linewidth=2)
    )

    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(colors[cond])
        patch.set_hatch(hatches[cond])
        patch.set_alpha(0.85)

    # Add significance stars above boxes
    stat_row = next(s for s in stat_records if s['metric'] == metric)
    p = stat_row['p']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.text(1.5, 5.05, sig, ha='center', fontsize=13, fontweight='bold')
    ax.plot([1, 2], [4.9, 4.9], color='black', linewidth=1)

    ax.set_title(metric.capitalize(), fontsize=13, fontweight='bold')
    ax.set_xticks([])
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(1, 5.4)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

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

plt.suptitle('Human Evaluation: Real vs Synthetic Comments',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{base}\\human_eval_boxplots.png", dpi=300, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════
# STEP 8: SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════

item_scores.to_csv(f"{base}\\item_scores_averaged.csv", index=False)
kappa_df.to_csv(f"{base}\\kappa_results.csv", index=False)
pd.DataFrame(stat_records).to_csv(f"{base}\\mannwhitney_results.csv", index=False)

print(f"\n✓ Saved:")
print(f"  item_scores_averaged.csv   — averaged scores per item")
print(f"  kappa_results.csv          — Cohen's κ per pair per metric")
print(f"  mannwhitney_results.csv    — real vs synthetic test results")
print(f"  human_eval_boxplots.png    — figure for paper")