
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\data\final_training_set_cleaned.csv")

# ── helpers ──────────────────────────────────────────────────────────────────
def word_count(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

def char_count(text):
    if pd.isna(text):
        return 0
    return len(str(text))

def has_mention(text):
    if pd.isna(text):
        return 0
    if text.startswith("<PERSON>"):
        return 1

def has_url(text):
    if pd.isna(text):
        return 0
    return int(bool(re.search(r'http\S+|www\.\S+', str(text))))

# ── word / char features on comments ─────────────────────────────────────────
data['_words']    = data['text'].apply(word_count)
data['_chars']    = data['text'].apply(char_count)
data['_mention']  = data['text'].apply(has_mention)

# URL: use existing has_urls column where available, fall back to regex

data['_url'] = data['text'].apply(has_url)

# ── date range ────────────────────────────────────────────────────────────────
date_cols = [c for c in ['time', 'Comment_Time'] if c in data.columns]
all_dates = pd.concat([pd.to_datetime(data[c], errors='coerce') for c in date_cols]).dropna()
date_min, date_max = all_dates.min(), all_dates.max()

# ── unique posts & mean comments/post ────────────────────────────────────────
unique_posts = data['text_ha'].dropna().nunique()
comments_per_post = data.groupby('text_ha').size()
mean_cpp  = comments_per_post.mean()
median_cpp = comments_per_post.median()
max_cpp   = comments_per_post.max()

# ── word-count buckets ────────────────────────────────────────────────────────
total = len(data)
w1_20  = (data['_words'] <= 20).sum()
w21_60 = ((data['_words'] >= 21) & (data['_words'] <= 60)).sum()
w61p   = (data['_words'] >= 61).sum()

# ── print table ───────────────────────────────────────────────────────────────
print("=" * 65)
print(f"{'Attribute':<30} {'Value':>12}  Details")
print("=" * 65)
print("\nCorpus Statistics")
print(f"{'Total comments':<30} {total:>12,}")
print(f"{'Total unique posts':<30} {unique_posts:>12,}")
print(f"{'Date range':<30} {date_min.strftime('%Y-%m-%d')} – {date_max.strftime('%Y-%m-%d')}")
print(f"{'Mean comments/post':<30} {mean_cpp:>12.1f}  Median: {median_cpp:.0f}, Max: {max_cpp:.0f}")

print("\nComment Characteristics")
mean_w = data['_words'].mean()
med_w  = data['_words'].median()
std_w  = data['_words'].std()
q1, q3 = data['_words'].quantile(0.25), data['_words'].quantile(0.75)
print(f"{'Mean words per comment':<30} {mean_w:>12.1f}  SD: {std_w:.1f}")
print(f"{'Median words':<30} {med_w:>12.1f}  IQR: {q1:.0f}–{q3:.0f}")
print(f"{'1–20 words':<30} {w1_20:>9,} ({100*w1_20/total:.0f}%)")
print(f"{'21–60 words':<30} {w21_60:>9,} ({100*w21_60/total:.0f}%)")
print(f"{'61+ words':<30} {w61p:>9,} ({100*w61p/total:.0f}%)")

print("\nLinguistic Features")
n_mention = data['_mention'].sum()
n_url     = data['_url'].sum()
mean_c    = data['_chars'].mean()
std_c     = data['_chars'].std()
print(f"{'Contains @mention':<30} {n_mention:>9,} ({100*n_mention/total:.0f}%)")
print(f"{'Contains URL':<30} {n_url:>9,} ({100*n_url/total:.0f}%)")
print(f"{'Avg. characters':<30} {mean_c:>12.1f}  SD: {std_c:.1f}")
print("=" * 65)

############################################################################################
# ── Train / Val split ────────────────────────────────────────────────────────
INTENT_MAP = {
    "Criticism": 0, "Statement": 1,
    "Inquiry": 2, "Forward": 3, "Appreciation": 4
}

# Keep a string-label copy for the distribution table, then encode numerically
data["Intent_label"] = data["Intent"]
data["Intent"] = data["Intent"].map(INTENT_MAP)

total = len(data)

train_df, eval_df = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["Intent"]
)

train_df['_words']    = train_df['text'].apply(word_count)
eval_df['_words']     = eval_df['text'].apply(word_count)

train_mean_w = train_df['_words'].mean()
train_med_w  = train_df['_words'].median()
train_std_w  = train_df['_words'].std()

eval_mean_w = eval_df['_words'].mean()
eval_med_w  = eval_df['_words'].median()
eval_std_w  = eval_df['_words'].std()

print("\nDataset Sizes")
print(f"{'Train':<10} {len(train_df):>12,} ({100*len(train_df)/total:.0f}%)")
print(f"{'Eval':<10} {len(eval_df):>12,} ({100*len(eval_df)/total:.0f}%)")

print("\nTrain vs Eval Word Count")
print(f"{'Dataset':<10} {'Mean':>8} {'Median':>8} {'Std Dev':>8}")
print(f"{'Train':<10} {train_mean_w:8.1f} {train_med_w:8.1f} {train_std_w:8.1f}")
print(f"{'Eval':<10} {eval_mean_w:8.1f} {eval_med_w:8.1f} {eval_std_w:8.1f}")

############################################################################################
# ── Table: Dimension / Category / Train / Validation ────────────────────────

SENTIMENT_MAP = {-1.0: "Negative", 0.0: "Neutral", 1.0: "Positive"}
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
INTENT_ORDER    = ["Criticism", "Statement", "Inquiry", "Forward", "Appreciation"]

def fmt(count, total):
    """Return 'N (XX.X%)' string."""
    return f"{count} ({100 * count / total:.1f}%)"

n_tr = len(train_df)
n_ev = len(eval_df)

# Sentiment distributions
train_sent = train_df["Sentiment"].map(SENTIMENT_MAP).value_counts()
eval_sent  = eval_df["Sentiment"].map(SENTIMENT_MAP).value_counts()

# Intent distributions (use original string labels)
train_intent = train_df["Intent_label"].value_counts()
eval_intent  = eval_df["Intent_label"].value_counts()

# Relatedness (Similarity column)
train_sim_avail = train_df["Similarity"].notna().sum()
eval_sim_avail  = eval_df["Similarity"].notna().sum()
train_sim_mean  = train_df["Similarity"].mean()
eval_sim_mean   = eval_df["Similarity"].mean()

# ── Print table ──────────────────────────────────────────────────────────────
COL = 26   # category column width
VAL = 22   # train/val column width

sep = "-" * (10 + COL + VAL * 2)
print("\n" + "=" * (10 + COL + VAL * 2))
print(f"{'Dimension':<10} {'Category':<{COL}} {'Train':>{VAL}} {'Validation':>{VAL}}")
print("=" * (10 + COL + VAL * 2))

# Sentiment block
for i, label in enumerate(SENTIMENT_ORDER):
    code = list(SENTIMENT_MAP.keys())[list(SENTIMENT_MAP.values()).index(label)]
    dim_str = "Sentiment" if i == 0 else ""
    cat_str = f"{label} ({int(code) if code != -1.0 else 0 if label=='Negative' else (1 if label=='Neutral' else 2)})"
    # remap label codes to 0/1/2 for display
    display_code = {-1.0: 0, 0.0: 1, 1.0: 2}[code]
    cat_str = f"{label} ({display_code})"
    tr_val = fmt(train_sent.get(label, 0), n_tr)
    ev_val = fmt(eval_sent.get(label, 0), n_ev)
    print(f"{dim_str:<10} {cat_str:<{COL}} {tr_val:>{VAL}} {ev_val:>{VAL}}")

print(sep)

# Intent block
for i, label in enumerate(INTENT_ORDER):
    dim_str = "Intent" if i == 0 else ""
    cat_str = f"{label} ({i})"
    tr_val = fmt(train_intent.get(label, 0), n_tr)
    ev_val = fmt(eval_intent.get(label, 0), n_ev)
    print(f"{dim_str:<10} {cat_str:<{COL}} {tr_val:>{VAL}} {ev_val:>{VAL}}")

print(sep)

# Relatedness block
rel_tr_avail = f"{train_sim_avail}/{n_tr} ({100*train_sim_avail/n_tr:.1f}%)"
rel_ev_avail = f"{eval_sim_avail}/{n_ev} ({100*eval_sim_avail/n_ev:.1f}%)"
print(f"{'Relatedness':<10} {'Available':<{COL}} {rel_tr_avail:>{VAL}} {rel_ev_avail:>{VAL}}")
print(f"{'':10} {'Mean score':<{COL}} {train_sim_mean:>{VAL}.3f} {eval_sim_mean:>{VAL}.3f}")

print("=" * (10 + COL + VAL * 2))

