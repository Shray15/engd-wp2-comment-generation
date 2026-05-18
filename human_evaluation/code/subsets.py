import pandas as pd
import numpy as np
import re

base = r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\human_evaluation\subsets"

# ── LOAD & FILTER ─────────────────────────────────────────────────────
data = pd.read_csv(
    r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\Fine Tune gemma\objective\objective_generated_commentsv2.csv"
)

def is_only_person_tags(text):
    if pd.isna(text):
        return True
    return re.sub(r'<PERSON>', '', str(text)).strip() == ''

def is_meaningful(text, min_words=5):
    if pd.isna(text):
        return False
    cleaned = re.sub(r'[^\w\s]', '', re.sub(r'<PERSON>', '', str(text))).strip()
    return len([w for w in cleaned.split() if w]) >= min_words

def replace_org(text):
    if pd.isna(text):
        return text
    return re.sub(r'Stichting Woonbedrijf|Woonbedrijf|woonbedrijf|#Woonbedrijf|#woonbedrijf', '<ORG>', str(text))

def replace_url(text):
    if pd.isna(text):
        return text
    return re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE).sub('<URL>', str(text))

def replace_phone(text):
    if pd.isna(text):
        return text
    # Matches Dutch phone formats:
    # 06-12345678, 06 12345678, 0612345678
    # 010-1234567, +31612345678, +31 6 1234 5678
    return re.sub(
        r'(\+31|0031|0)[\s\-]?'      # country/area code
        r'(\d[\s\-]?){8,9}',         # 8-9 digits with optional spaces/dashes
        '<PHONE>',
        str(text)
    )

def is_only_url_or_org(text, min_words=5):
    """Returns True if text has no content beyond <URL>, <ORG>, <PERSON> tags."""
    if pd.isna(text):
        return True
    # Remove all tags
    cleaned = re.sub(r'<URL>|<ORG>|<PERSON>|<PHONE>', '', str(text)).strip()
    # Remove punctuation and whitespace
    cleaned = re.sub(r'[^\w\s]', '', cleaned).strip()
    words = [w for w in cleaned.split() if w]
    return len(words) < min_words

def replace_email(text):
    if pd.isna(text):
        return text
    return re.sub(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
        '<EMAIL>',
        str(text)
    )


                 



# Apply filters
data = data[~data['true_comment'].apply(is_only_person_tags)]
data = data[~data['generated_comment'].apply(is_only_person_tags)]
data = data[data['true_comment'].apply(is_meaningful)]
data = data[data['generated_comment'].apply(is_meaningful)]

for col in ['post', 'true_comment', 'generated_comment']:
    data[col] = data[col].apply(replace_org).apply(replace_url).apply(replace_phone).apply(replace_email)

before = len(data)
data = data[~data['post'].apply(is_only_url_or_org)]
print(f"After removing tag-only posts: {len(data)} rows (removed {before - len(data)})")

data = data.drop_duplicates(subset=['post'], keep='first').reset_index(drop=True)
print(f"Available posts after filtering: {len(data)}")
assert len(data) >= 80, f"Need at least 80 posts, only have {len(data)}"

# ── SAMPLE 80 POSTS ───────────────────────────────────────────────────
sample = data.sample(n=80, random_state=42).reset_index(drop=True)
sample['pair_id'] = range(1, 81)

# ── CREATE 80 ITEMS (40 real + 40 synthetic) ──────────────────────────
# First 40 posts → real comment
# Last  40 posts → synthetic comment
# No post appears twice → no annotator sees same post twice

real_items = sample.iloc[:40][['pair_id', 'post', 'true_comment']].copy()
real_items.columns = ['pair_id', 'post', 'comment']
real_items['condition'] = 'real'

syn_items = sample.iloc[40:][['pair_id', 'post', 'generated_comment']].copy()
syn_items.columns = ['pair_id', 'post', 'comment']
syn_items['condition'] = 'synthetic'

# Interleave real and synthetic so each window of 20 has ~10 real + 10 synthetic
# Alternate: real, synthetic, real, synthetic...
real_items = real_items.reset_index(drop=True)
syn_items  = syn_items.reset_index(drop=True)

interleaved = []
for i in range(40):
    interleaved.append(real_items.iloc[i])
    interleaved.append(syn_items.iloc[i])

all_items = pd.DataFrame(interleaved).reset_index(drop=True)
all_items['item_id'] = range(1, 81)

print(f"Items created: {len(all_items)}")
print(f"Condition distribution:\n{all_items['condition'].value_counts()}")

# ── SLIDING WINDOW ASSIGNMENT ─────────────────────────────────────────
# 7 annotators, window=20, stride=10
# Ann 1: items  1–20  (index 0–19)
# Ann 2: items 11–30  (index 10–29)
# Ann 3: items 21–40  (index 20–39)
# ...
# Ann 7: items 61–80  (index 60–79)

n_annotators = 7
window       = 20
stride       = 10

annotator_assignments = {}

for ann_id in range(1, n_annotators + 1):
    start = (ann_id - 1) * stride        # 0, 10, 20, 30, 40, 50, 60
    end   = start + window               # 20, 30, 40, 50, 60, 70, 80

    block = all_items.iloc[start:end].copy()

    # Verify balance
    real_count = len(block[block['condition'] == 'real'])
    syn_count  = len(block[block['condition'] == 'synthetic'])
    print(f"Annotator {ann_id}: items {start+1}–{end} "
          f"| real={real_count}, synthetic={syn_count}")

    # Shuffle item order so overlap items aren't always first
    block = block.sample(frac=1, random_state=ann_id).reset_index(drop=True)
    block.insert(0, 'annotator_id', ann_id)
    block['item_order'] = range(1, 21)

    annotator_assignments[ann_id] = block

# ── SANITY CHECKS ─────────────────────────────────────────────────────
master = pd.concat(annotator_assignments.values(), ignore_index=True)

print("\n=== SANITY CHECKS ===")

print(f"\n1. Items per annotator (expect 20):")
print(master.groupby('annotator_id').size().to_string())

print(f"\n2. Unique posts per annotator (expect 20):")
print(master.groupby('annotator_id')['pair_id'].nunique().to_string())

print(f"\n3. Real/Synthetic per annotator (expect 10 each):")
print(master.groupby(['annotator_id', 'condition']).size().unstack().to_string())

print(f"\n4. Annotators per item (overlap items expect 2, edge items expect 1):")
print(master.groupby('item_id')['annotator_id'].count().value_counts().to_string())

print(f"\n5. Total annotations: {len(master)} (expect 140)")

# ── OVERLAP VERIFICATION ──────────────────────────────────────────────
print(f"\n6. Overlap between consecutive annotators (expect 10 each):")
for ann_id in range(1, n_annotators):
    items_a = set(annotator_assignments[ann_id]['item_id'])
    items_b = set(annotator_assignments[ann_id + 1]['item_id'])
    overlap = items_a & items_b
    print(f"  Ann {ann_id} & Ann {ann_id+1}: {len(overlap)} overlapping items ✓")

# ── SAVE OUTPUTS ──────────────────────────────────────────────────────
for ann_id, ann_df in annotator_assignments.items():
    ann_df.drop(columns=['condition']).to_csv(
        f"{base}\\annotator_{ann_id:02d}.csv", index=False
    )

master.to_csv(f"{base}\\master_with_conditions.csv", index=False)

print(f"\n✓ Saved 7 annotator CSVs + master to:\n  {base}")