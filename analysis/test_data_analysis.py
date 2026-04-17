
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\data\final_test_set.csv")

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
# ── Table: Dimension / Category / Test ────────────────────────

SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
INTENT_ORDER    = ["Criticism", "Statement", "Inquiry", "Forward", "Appreciation"]
data["Intent_label"] = data["Intent"]

def fmt(count, total):
    """Return 'N (XX.X%)' string."""
    return f"{count} ({100 * count / total:.1f}%)"

n_test = len(data)

# Sentiment distributions — test set stores labels as strings already
test_sent = data["Sentiment"].value_counts()

# Intent distributions (use original string labels)
test_intent = data["Intent_label"].value_counts()

# Relatedness (Similarity column)
test_sim_avail = data["Similarity"].notna().sum()
test_sim_mean  = data["Similarity"].mean()

# ── Print table ──────────────────────────────────────────────────────────────
COL = 26   # category column width
VAL = 22   # train/val column width

sep = "-" * (10 + COL + VAL * 2)
print("\n" + "=" * (10 + COL + VAL * 2))
print(f"{'Dimension':<10} {'Category':<{COL}} {'Test':>{VAL}}")
print("=" * (10 + COL + VAL * 2))

# Sentiment block
DISPLAY_CODE = {"Negative": 0, "Neutral": 1, "Positive": 2}
for i, label in enumerate(SENTIMENT_ORDER):
    dim_str = "Sentiment" if i == 0 else ""
    cat_str = f"{label} ({DISPLAY_CODE[label]})"
    tr_val = fmt(test_sent.get(label, 0), n_test)
    print(f"{dim_str:<10} {cat_str:<{COL}} {tr_val:>{VAL}}")

print(sep)

# Intent block
for i, label in enumerate(INTENT_ORDER):
    dim_str = "Intent" if i == 0 else ""
    cat_str = f"{label} ({i})"
    val = fmt(test_intent.get(label, 0), n_test)
    print(f"{dim_str:<10} {cat_str:<{COL}} {val:>{VAL}}")

print(sep)

# Relatedness block
rel_tr_avail = f"{test_sim_avail}/{n_test} ({100*test_sim_avail/n_test:.1f}%)"
print(f"{'Relatedness':<10} {'Available':<{COL}} {rel_tr_avail:>{VAL}}")
print(f"{'':10} {'Mean score':<{COL}} {test_sim_mean:>{VAL}.3f}")

print("=" * (10 + COL + VAL * 2))

