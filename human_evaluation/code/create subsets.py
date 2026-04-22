import pandas as pd

data = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\Fine Tune gemma\objective\objective_generated_commentsv2.csv")

data = data[~((data["true_comment"].str.startswith("<PERSON>")) | (data["generated_comment"].str.startswith("<PERSON>")))]

data = data.drop_duplicates(subset = ["post"], keep = "first")

import pandas as pd

# ── 1. SAMPLE 110 POSTS ───────────────────────────────────────────────
sample = data.sample(n=110, random_state=42).reset_index(drop=True)
sample['pair_id'] = range(1, 111)

# ── 2. SPLIT INTO SHARED (10) AND UNIQUE (100) ────────────────────────
shared_posts = sample.iloc[:10].copy()
unique_posts  = sample.iloc[10:].copy().reset_index(drop=True)

# ── 3. CREATE SHARED ITEMS (10 posts, 1 comment each, no post repeated)
shared_real = shared_posts.iloc[:5][['pair_id', 'post', 'true_comment']].copy()
shared_real.columns = ['pair_id', 'post', 'comment']
shared_real['condition'] = 'real'

shared_syn = shared_posts.iloc[5:10][['pair_id', 'post', 'generated_comment']].copy()
shared_syn.columns = ['pair_id', 'post', 'comment']
shared_syn['condition'] = 'synthetic'

shared_items = pd.concat([shared_real, shared_syn], ignore_index=True)
shared_items['is_shared'] = True

# ── 4. CREATE UNIQUE ITEMS (100 posts, 1 comment each, alternating real/synthetic)
unique_items_list = []
for i, row in unique_posts.iterrows():
    comment_col = 'true_comment' if i % 2 == 0 else 'generated_comment'
    condition   = 'real'         if i % 2 == 0 else 'synthetic'
    unique_items_list.append({
        'pair_id':   row['pair_id'],
        'post':      row['post'],
        'comment':   row[comment_col],
        'condition': condition,
        'is_shared': False
    })

unique_items = pd.DataFrame(unique_items_list)

# ── 5. ASSIGN 10 UNIQUE ITEMS PER ANNOTATOR ───────────────────────────
annotator_assignments = []

for annotator_id in range(1, 11):
    start = (annotator_id - 1) * 10
    end   = start + 10

    unique_subset = unique_items.iloc[start:end].copy()
    annotator_items = pd.concat([shared_items, unique_subset], ignore_index=True)
    annotator_items = annotator_items.sample(frac=1, random_state=annotator_id).reset_index(drop=True)
    annotator_items.insert(0, 'annotator_id', annotator_id)
    annotator_items['item_order'] = range(1, 21)
    annotator_assignments.append(annotator_items)

# ── 6. SANITY CHECKS ──────────────────────────────────────────────────
master = pd.concat(annotator_assignments, ignore_index=True)

print("=== SANITY CHECKS ===")
print(f"\nItems per annotator (expect 20):")
print(master.groupby('annotator_id').size().to_string())

print(f"\nUnique posts per annotator (expect 20):")
print(master.groupby('annotator_id')['pair_id'].nunique().to_string())

print(f"\nShared items per annotator (expect 10):")
print(master.groupby('annotator_id')['is_shared'].sum().to_string())

print(f"\nReal/Synthetic per annotator (expect 10 each):")
print(master.groupby(['annotator_id', 'condition']).size().unstack())

# ── 7. SAVE ───────────────────────────────────────────────────────────
base = r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\Fine Tune gemma\objective"

for ann_df in annotator_assignments:
    ann_id = ann_df['annotator_id'].iloc[0]
    ann_df.drop(columns=['condition', 'is_shared']).to_csv(
        f"{base}\\annotator_{ann_id:02d}.csv", index=False
    )

master.to_csv(f"{base}\\master_with_conditions.csv", index=False)
print(f"\n✓ Saved 10 annotator CSVs + master to:\n  {base}")