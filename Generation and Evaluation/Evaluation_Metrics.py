import pandas as pd
import numpy as np
import evaluate
from tqdm.auto import tqdm

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
data = pd.read_csv(
    r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\data\final_data_cleaned_with_author_names.csv"
)
data_gen = pd.read_csv(
    r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\Few Shot Prompt\Mistral_3_3B_instruct\Generated_comments_Mistral_3_3B_Instruct.csv"
)

data_gen = data_gen.dropna(subset=["Post", "Comments"])

# Group indices by post text
posts1 = data.groupby("text_ha")["text"].groups       # human comments
posts2 = data_gen.groupby("Post")["Comments"].groups  # generated comments

# -------------------------------------------------------------------
# 2. Load metrics
# -------------------------------------------------------------------
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")

# Per-post scores (all pairwise comparisons)
per_post_scores_rouge1 = []
per_post_scores_rouge2 = []
per_post_scores_rougeL = []
per_post_scores_rougesum = []

per_post_scores_meteor = []
per_post_scores_bertscore = []

# Maximum scores per post (original approach)
maximum_scores_rouge1 = []
maximum_scores_rouge2 = []
maximum_scores_rougeL = []
maximum_scores_rougesum = []


maximum_scores_meteor = []
maximum_scores_bertscore = []

# Only evaluate posts that exist in both datasets
common_posts = [p for p in posts2.keys() if p in posts1.keys()]

# -------------------------------------------------------------------
# 3. Main loop with progress bar
# -------------------------------------------------------------------
for post_text in tqdm(common_posts, desc="Evaluating posts"):

    values1 = posts1[post_text]
    values2 = posts2[post_text]

    actual_values1 = data.loc[values1, "text"].tolist()        # human comments
    actual_values2 = data_gen.loc[values2, "Comments"].tolist()  # generated comments

    # Per-post score collections for all pairwise comparisons
    post_rouge1_scores = []
    post_rouge2_scores = []
    post_rougeL_scores = []
    post_rougesum_scores = []

    post_meteor_scores = []

    max_rouge1 = 0.0
    max_rouge2 = 0.0
    max_rougeL = 0.0
    max_rougesum = 0.0

    max_meteor = 0.0
    max_bs = 0.0

    # ---------------------------------------------------------------
    # (A) Pairwise loop for ROUGE, BLEU, METEOR (fast enough)
    # ---------------------------------------------------------------
    for ref in actual_values1:          # human
        for pred in actual_values2:     # generated

            # ROUGE
            res_rouge = rouge.compute(predictions=[pred], references=[ref])
            rouge1_score = res_rouge["rouge1"]
            rouge2_score = res_rouge["rouge2"]
            rougeL_score = res_rouge["rougeL"]
            rougesum_score = res_rouge["rougeLsum"]
            
            # Collect all scores for per-post analysis
            post_rouge1_scores.append(rouge1_score)
            post_rouge2_scores.append(rouge2_score)
            post_rougeL_scores.append(rougeL_score)
            post_rougesum_scores.append(rougesum_score)
            
            # Update maximums
            max_rouge1 = max(max_rouge1, rouge1_score)
            max_rouge2 = max(max_rouge2, rouge2_score)
            max_rougeL = max(max_rougeL, rougeL_score)
            max_rougesum = max(max_rougesum, rougesum_score)

            # METEOR
            res_meteor = meteor.compute(predictions=[pred], references=[ref])
            meteor_score = res_meteor["meteor"]
            post_meteor_scores.append(meteor_score)
            max_meteor = max(max_meteor, meteor_score)

    # ---------------------------------------------------------------
    # (B) BERTScore in batch for all combinations (big speed-up)
    # ---------------------------------------------------------------
    pair_preds = []
    pair_refs = []
    for ref in actual_values1:
        for pred in actual_values2:
            pair_preds.append(pred)
            pair_refs.append(ref)

    if pair_preds:  # safety check
        res_bs = bertscore.compute(
            predictions=pair_preds,
            references=pair_refs,
            lang="nl"
        )
        bertscore_f1_scores = res_bs["f1"]  # list of F1 scores
        max_bs = max(bertscore_f1_scores)
    else:
        bertscore_f1_scores = []

    # Store per-post averages (all pairwise comparisons for this post)
    per_post_scores_rouge1.append(np.mean(post_rouge1_scores) if post_rouge1_scores else 0.0)
    per_post_scores_rouge2.append(np.mean(post_rouge2_scores) if post_rouge2_scores else 0.0)
    per_post_scores_rougeL.append(np.mean(post_rougeL_scores) if post_rougeL_scores else 0.0)
    per_post_scores_rougesum.append(np.mean(post_rougesum_scores) if post_rougesum_scores else 0.0)
    per_post_scores_meteor.append(np.mean(post_meteor_scores) if post_meteor_scores else 0.0)
    per_post_scores_bertscore.append(np.mean(bertscore_f1_scores) if bertscore_f1_scores else 0.0)

    # Store maximum scores for this post (original approach)
    maximum_scores_rouge1.append(max_rouge1)
    maximum_scores_rouge2.append(max_rouge2)
    maximum_scores_rougeL.append(max_rougeL)
    maximum_scores_rougesum.append(max_rougesum)

    maximum_scores_meteor.append(max_meteor)
    maximum_scores_bertscore.append(max_bs)

# -------------------------------------------------------------------
# 4. Print aggregate statistics
# -------------------------------------------------------------------
print("=" * 80)
print("PER-POST AVERAGE SCORES (All pairwise comparisons)")
print("=" * 80)
print("MEAN of per-post averages:")
print(f"ROUGE-1: {np.mean(per_post_scores_rouge1):.4f}")
print(f"ROUGE-2: {np.mean(per_post_scores_rouge2):.4f}")
print(f"ROUGE-L: {np.mean(per_post_scores_rougeL):.4f}")
print(f"ROUGE-Lsum: {np.mean(per_post_scores_rougesum):.4f}")
print(f"METEOR: {np.mean(per_post_scores_meteor):.4f}")
print(f"BERTScore F1: {np.mean(per_post_scores_bertscore):.4f}")

print("\nMAX of per-post averages:")
print(f"ROUGE-1: {np.max(per_post_scores_rouge1):.4f}")
print(f"ROUGE-2: {np.max(per_post_scores_rouge2):.4f}")
print(f"ROUGE-L: {np.max(per_post_scores_rougeL):.4f}")
print(f"ROUGE-Lsum: {np.max(per_post_scores_rougesum):.4f}")
print(f"METEOR: {np.max(per_post_scores_meteor):.4f}")
print(f"BERTScore F1: {np.max(per_post_scores_bertscore):.4f}")

print("\n" + "=" * 80)
print("MAXIMUM SCORES PER POST (Best single pair)")
print("=" * 80)
print("MEAN of maximum scores:")
print(
    np.mean(maximum_scores_rouge1),
    np.mean(maximum_scores_rouge2),
    np.mean(maximum_scores_rougeL),
    np.mean(maximum_scores_rougesum),
    np.mean(maximum_scores_meteor),
    np.mean(maximum_scores_bertscore),
)

print("\nMAX of maximum scores:")
print(
    np.max(maximum_scores_rouge1),
    np.max(maximum_scores_rouge2),
    np.max(maximum_scores_rougeL),
    np.max(maximum_scores_rougesum),
    np.max(maximum_scores_meteor),
    np.max(maximum_scores_bertscore),
)

print("\n" + "=" * 80)
print(f"Evaluated {len(common_posts)} posts")
print("=" * 80)

# -------------------------------------------------------------------
# 5. Create DataFrame with per-post mean and max scores
# -------------------------------------------------------------------
print("\nCreating DataFrame with per-post statistics...")

# Create DataFrame with all per-post scores
results_df = pd.DataFrame({
    'Post': common_posts,
    'Mean_ROUGE1': per_post_scores_rouge1,
    'Mean_ROUGE2': per_post_scores_rouge2,
    'Mean_ROUGEL': per_post_scores_rougeL,
    'Mean_ROUGELsum': per_post_scores_rougesum,
    'Mean_METEOR': per_post_scores_meteor,
    'Mean_BERTScore_F1': per_post_scores_bertscore,
    'Max_ROUGE1': maximum_scores_rouge1,
    'Max_ROUGE2': maximum_scores_rouge2,
    'Max_ROUGEL': maximum_scores_rougeL,
    'Max_ROUGELsum': maximum_scores_rougesum,
    'Max_METEOR': maximum_scores_meteor,
    'Max_BERTScore_F1': maximum_scores_bertscore
})

# Save per-post metrics to CSV
output_path = r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\Few Shot Prompt\Mistral_3_3B_instruct\per_post_similarity_metrics.csv"
results_df.to_csv(output_path, index=False)
print(f"✅ Saved per-post metrics to: {output_path}")
print(f"   DataFrame shape: {results_df.shape}")
print(f"\nFirst few rows:")
print(results_df.head())

# -------------------------------------------------------------------
# 6. Save aggregate statistics (final output summary)
# -------------------------------------------------------------------
print("\nCreating aggregate statistics summary...")

# Create summary DataFrame with overall statistics
aggregate_stats = pd.DataFrame({
    'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum', 'METEOR', 'BERTScore F1'],
    'Mean_of_PerPost_Averages': [
        np.mean(per_post_scores_rouge1),
        np.mean(per_post_scores_rouge2),
        np.mean(per_post_scores_rougeL),
        np.mean(per_post_scores_rougesum),
        np.mean(per_post_scores_meteor),
        np.mean(per_post_scores_bertscore)
    ],
    'Max_of_PerPost_Averages': [
        np.max(per_post_scores_rouge1),
        np.max(per_post_scores_rouge2),
        np.max(per_post_scores_rougeL),
        np.max(per_post_scores_rougesum),
        np.max(per_post_scores_meteor),
        np.max(per_post_scores_bertscore)
    ],
    'Mean_of_Maximum_Scores': [
        np.mean(maximum_scores_rouge1),
        np.mean(maximum_scores_rouge2),
        np.mean(maximum_scores_rougeL),
        np.mean(maximum_scores_rougesum),
        np.mean(maximum_scores_meteor),
        np.mean(maximum_scores_bertscore)
    ],
    'Max_of_Maximum_Scores': [
        np.max(maximum_scores_rouge1),
        np.max(maximum_scores_rouge2),
        np.max(maximum_scores_rougeL),
        np.max(maximum_scores_rougesum),
        np.max(maximum_scores_meteor),
        np.max(maximum_scores_bertscore)
    ]
})

# Save aggregate statistics
aggregate_output_path = r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\Few Shot Prompt\Mistral_3_3B_instruct\aggregate_similarity_statistics_summary.csv"
aggregate_stats.to_csv(aggregate_output_path, index=False)
print(f"✅ Saved aggregate statistics to: {aggregate_output_path}")
print(f"\nAggregate Statistics:")
print(aggregate_stats.to_string(index=False))
print(f"\n📊 Total posts evaluated: {len(common_posts)}")
