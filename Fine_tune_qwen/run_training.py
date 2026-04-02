"""
run_training.py — Single-file fine-tuning script for Qwen3.5-4B-Base.

Covers: data loading → stratified split → training → test set evaluation.

Usage on server:
    python run_training.py
    python run_training.py --train_csv /path/to/final_training_set.csv \
                           --test_csv  /path/to/final_test_set.csv \
                           --output_dir outputs/qwen3_5_4b_base \
                           --hf_token hf_xxx
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

# =============================================================================
# 0.  CONFIGURATION  —  edit these defaults before running on server
# =============================================================================

DEFAULTS = dict(
    # Data paths (relative to this script or absolute)
    train_csv   = "../data/final_training_set.csv",
    test_csv    = "../data/final_test_set.csv",

    # Model
    model_name  = "Qwen/Qwen3.5-4B-Base",

    # Output
    output_dir  = "outputs/qwen3_5_4b_base",

    # LoRA
    lora_r      = 16,
    lora_alpha  = 32,
    lora_dropout= 0.05,

    # Sequence
    max_seq_length = 2048,
    max_comments   = 5,       # cap per post; training range communicated to model

    # Training
    num_train_epochs            = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    learning_rate               = 2e-4,
    warmup_ratio                = 0.05,
    lr_scheduler_type           = "cosine",
    weight_decay                = 0.01,
    logging_steps               = 10,
    eval_steps                  = 50,
    save_steps                  = 100,
    save_total_limit            = 2,

    # Data split
    val_split = 0.2,
    seed      = 42,

    # Evaluation
    n_generate      = 5,     # comments to generate per test post
    eval_max_posts  = 9999,  # generate on full test set (capped by actual test set size)
    # max_new_tokens is now dynamic per comment (sampled from training distribution)
    # This fallback is only used if length distribution cannot be computed
    max_new_tokens  = 150,
    temperature     = 0.8,
    top_p           = 0.9,

    # HuggingFace token (or set HF_TOKEN env var)
    hf_token = None,
)

SYSTEM_PROMPT = (
    "You are a Facebook user commenting on posts from a Dutch housing association. "
    "Write realistic, varied comments that reflect how real users respond."
)
SEPARATOR = "---"

INTENT_MAP = {
    "Criticism":    0,
    "Statement":    1,
    "Inquiry":      2,
    "Forward":      3,
    "Appreciation": 4,
}
INTENT_NAMES = {v: k for k, v in INTENT_MAP.items()}

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# =============================================================================
# 1.  LENGTH DISTRIBUTION  (mirrors Gemma approach)
# =============================================================================

def compute_length_distribution(posts: List[Dict]) -> np.ndarray:
    """
    Compute word-count distribution from real training comments.
    Returns a numpy array of per-comment word counts drawn from all posts.
    This is used at inference to sample a realistic max_tokens per comment,
    so generated comment lengths match the real training distribution.
    """
    word_counts = []
    for post in posts:
        for comment in post["comments"]:
            word_counts.append(len(str(comment).split()))
    wc = np.array(word_counts)

    print(f"\n  Length distribution (from {len(wc)} training comments):")
    print(f"    mean={wc.mean():.1f}  median={np.median(wc):.0f}  "
          f"std={wc.std():.1f}  p25={np.percentile(wc,25):.0f}  "
          f"p75={np.percentile(wc,75):.0f}  p90={np.percentile(wc,90):.0f}")
    return wc


def sample_max_tokens(word_counts: np.ndarray) -> int:
    """
    Sample one target word count from the real distribution,
    then convert to a max_tokens budget (tokens ≈ words * 1.5 for Dutch).

    Buckets match the training data range:
      ≤ 5 words  → max_tokens = 10
      ≤ 15 words → max_tokens = 25
      ≤ 30 words → max_tokens = 50
      > 30 words → max_tokens = min(word_count * 1.6, 150)
    """
    target_words = int(np.random.choice(word_counts))
    if target_words <= 5:
        return 10
    elif target_words <= 15:
        return 25
    elif target_words <= 30:
        return 50
    else:
        return min(int(target_words * 1.6), 150)


# =============================================================================
# 2.  DATA LOADING & FORMATTING
# =============================================================================

def load_and_group_posts(csv_path: str, max_comments: int = None) -> List[Dict[str, Any]]:
    """
    Load flat comment CSV and group rows by post (text_ha).

    Each returned dict represents one post:
        post_id         — the post text used as key
        post_text       — text_ha column (the Facebook post)
        comments        — list of comment texts from text column (capped at max_comments)
        majority_intent — most frequent Intent among the post's comments
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df            = df.dropna(subset=["text_ha", "text"])
    df["text"]    = df["text"].astype(str).str.strip()
    df["text_ha"] = df["text_ha"].astype(str).str.strip()
    df["Intent"]  = df["Intent"].fillna("Unknown").astype(str)

    posts = []
    for post_text, group in df.groupby("text_ha"):
        comments = group["text"].tolist() if max_comments is None else group["text"].tolist()[:max_comments]
        intents  = group["Intent"].tolist() if max_comments is None else group["Intent"].tolist()[:max_comments]
        if not comments:
            continue
        majority_intent = Counter(intents).most_common(1)[0][0]
        posts.append({
            "post_id":         post_text[:80],   # short key for display/logging
            "post_text":       post_text,
            "comments":        comments,
            "majority_intent": majority_intent,
        })

    print(f"  Loaded {len(posts)} posts from {csv_path}")
    _print_post_stats(posts)
    return posts


def _print_post_stats(posts: List[Dict]) -> None:
    intent_counts = Counter(p["majority_intent"] for p in posts)
    n_counts      = Counter(len(p["comments"])   for p in posts)
    total = len(posts)
    print("  Majority-intent distribution:")
    for name in ["Criticism", "Statement", "Inquiry", "Forward", "Appreciation"]:
        cnt = intent_counts.get(name, 0)
        print(f"    {name:<14} {cnt:>3}  ({100*cnt/total:.1f}%)")
    print(f"  Comment-count dist (after cap): {dict(sorted(n_counts.items()))}")


def format_post(post: Dict, tokenizer) -> str:
    """
    Convert one post dict into a ChatML string.
    Training target is the assistant turn (all comments joined by separator).
    """
    n = len(post["comments"])
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Given this Facebook post, generate {n} "
                f"comment{'s' if n != 1 else ''} that users might leave:\n"
                f"{post['post_text']}"
            ),
        },
        {
            "role": "assistant",
            "content": f"\n{SEPARATOR}\n".join(post["comments"]),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# =============================================================================
# 3.  STRATIFIED TRAIN / VAL SPLIT  (post-level, no leakage)
# =============================================================================

def stratified_split(
    posts: List[Dict],
    val_split: float,
    seed: int,
    tokenizer,
) -> Tuple[Dataset, Dataset]:
    """
    Split at the POST level stratified by majority_intent.

    Order:
      1. Compute majority intent per post        (done in load_and_group_posts)
      2. Stratify-split at post level            (StratifiedShuffleSplit)
      3. Apply chat template within each split   (comments stay with their post)

    No post ever appears in both train and val — no data leakage.
    """
    # Map intent to int for sklearn
    labels = [INTENT_MAP.get(p["majority_intent"], -1) for p in posts]
    unknown = [p["majority_intent"] for p, l in zip(posts, labels) if l == -1]
    if unknown:
        print(f"  WARNING: Unknown intents {set(unknown)} treated as separate class.")

    label_counts = Counter(labels)
    can_stratify = all(c >= 2 for c in label_counts.values())

    indices = np.arange(len(posts))

    if can_stratify:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_split, random_state=seed
        )
        train_idx, val_idx = next(splitter.split(indices, labels))
        print(f"  ✅ Stratified split — train: {len(train_idx)} posts, val: {len(val_idx)} posts")
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        cut = int(len(indices) * (1 - val_split))
        train_idx, val_idx = indices[:cut], indices[cut:]
        small = [INTENT_NAMES.get(k, str(k)) for k, v in label_counts.items() if v < 2]
        print(f"  ⚠️  Classes {small} have < 2 posts — using random split.")
        print(f"     Random split — train: {len(train_idx)}, val: {len(val_idx)}")

    train_posts = [posts[i] for i in train_idx]
    val_posts   = [posts[i] for i in val_idx]

    _print_split_stats("TRAIN", train_posts)
    _print_split_stats("VAL",   val_posts)

    train_ds = Dataset.from_list([
        {"text": format_post(p, tokenizer)} for p in train_posts
    ])
    val_ds = Dataset.from_list([
        {"text": format_post(p, tokenizer)} for p in val_posts
    ])
    return train_ds, val_ds


def _print_split_stats(name: str, posts: List[Dict]) -> None:
    ic = Counter(p["majority_intent"] for p in posts)
    total = len(posts)
    dist  = {n: ic.get(n, 0) for n in ["Criticism","Statement","Inquiry","Forward","Appreciation"]}
    print(f"\n  {name} ({total} posts): {dist}")


# =============================================================================
# 4.  MODEL LOADING  (Unsloth)
# =============================================================================

def load_model(model_name: str, max_seq_length: int, load_in_4bit: bool,
               lora_r: int, lora_alpha: int, lora_dropout: float):
    from unsloth import FastLanguageModel

    print(f"\n  Loading {model_name}  (4bit={load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit   = load_in_4bit,
        dtype          = None,   # auto bf16/fp16
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                          = lora_r,
        lora_alpha                 = lora_alpha,
        lora_dropout               = lora_dropout,
        target_modules             = LORA_TARGETS,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Base models have no chat template — set ChatML manually
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<|im_start|>assistant\n'}}"
            "{% endif %}"
        )
        print("  Chat template set manually (base model has none by default)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
    return model, tokenizer


# =============================================================================
# 5.  TRAINING
# =============================================================================

def train(model, tokenizer, train_ds: Dataset, val_ds: Dataset, args) -> None:
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    sft_cfg = SFTConfig(
        output_dir                    = args.output_dir,
        num_train_epochs              = args.num_train_epochs,
        per_device_train_batch_size   = args.per_device_train_batch_size,
        per_device_eval_batch_size    = args.per_device_train_batch_size,
        gradient_accumulation_steps   = args.gradient_accumulation_steps,
        learning_rate                 = args.learning_rate,
        warmup_ratio                  = args.warmup_ratio,
        lr_scheduler_type             = args.lr_scheduler_type,
        weight_decay                  = args.weight_decay,
        fp16                          = not is_bfloat16_supported(),
        bf16                          = is_bfloat16_supported(),
        max_seq_length                = args.max_seq_length,
        dataset_text_field            = "text",
        packing                       = False,
        logging_steps                 = args.logging_steps,
        eval_strategy                 = "steps",
        eval_steps                    = args.eval_steps,
        save_strategy                 = "steps",
        save_steps                    = args.save_steps,
        save_total_limit              = args.save_total_limit,
        load_best_model_at_end        = True,
        metric_for_best_model         = "eval_loss",
        greater_is_better             = False,
        report_to                     = "none",
        seed                          = args.seed,
        data_seed                     = args.seed,
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        args          = sft_cfg,
    )

    # Loss only on assistant (comment) tokens — not on instruction/system tokens
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )

    if torch.cuda.is_available():
        mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        total_mem = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {mem} / {total_mem} GB reserved")

    print("\n  Starting training...")
    stats = trainer.train()

    if torch.cuda.is_available():
        peak = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"  Peak VRAM: {peak} GB")

    runtime = stats.metrics.get("train_runtime", 0)
    print(f"  Train runtime : {runtime:.1f}s  ({runtime/60:.1f} min)")
    print(f"  Train loss    : {stats.metrics.get('train_loss', 'N/A'):.4f}")


# =============================================================================
# 6.  EVALUATION ON TEST SET
# =============================================================================

def evaluate_test_set(
    model,
    tokenizer,
    test_posts: List[Dict],
    output_dir: str,
    n_generate: int,
    eval_max_posts: int,
    word_counts: np.ndarray,      # training comment word-count distribution
    max_new_tokens_fallback: int,  # used only if word_counts is empty
    temperature: float,
    top_p: float,
) -> None:
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    # Qwen3VLProcessor wraps the real text tokenizer — unwrap it
    text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    eval_posts = test_posts[:eval_max_posts]
    print(f"\n  Generating comments for {len(eval_posts)} test posts (n = real comment count per post)...")
    print(f"  max_new_tokens: sampled per comment from training length distribution")
    print(f"  Tokenizer type: {type(tokenizer).__name__}")

    results = []
    for i, post in enumerate(eval_posts):
        # Use the actual number of real comments this post has in the test set
        n = len(post["comments"])

        # max_tokens budget: one sample per comment + separator overhead
        if len(word_counts) > 0:
            max_new_tokens = sum(sample_max_tokens(word_counts) for _ in range(n))
            max_new_tokens += (n - 1) * 4  # separator overhead
        else:
            max_new_tokens = max_new_tokens_fallback

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Given this Facebook post, generate {n} "
                    f"comment{'s' if n != 1 else ''} that users might leave:\n"
                    f"{post['post_text']}"
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Use the unwrapped text tokenizer to bypass Qwen3VLProcessor's vision path
        input_ids = text_tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                temperature    = temperature,
                top_p          = top_p,
                do_sample      = True,
                pad_token_id   = tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = text_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generated = [c.strip() for c in
                     re.split(rf"\n\s*{re.escape(SEPARATOR)}\s*\n", raw) if c.strip()]
        generated = generated[:n]  # never return more than requested

        results.append({
            "post_id":             post["post_id"],
            "post_text":           post["post_text"],
            "n_requested":         n,
            "n_generated":         len(generated),
            "max_new_tokens_used": max_new_tokens,
            "generated_comments":  generated,
            "reference_comments":  post["comments"],
        })
        print(f"  [{i+1}/{len(eval_posts)}] requested={n}, generated={len(generated)}  (max_tokens={max_new_tokens})")

    # Metrics
    all_gen = [c for r in results for c in r["generated_comments"]]
    d1 = _distinct_n(all_gen, 1)
    d2 = _distinct_n(all_gen, 2)

    gen_lens = [len(c.split()) for c in all_gen]
    ref_lens = [len(c.split()) for p in eval_posts for c in p["comments"]]
    correct  = sum(1 for r in results if r["n_generated"] == n_generate)

    summary = {
        "n_posts_evaluated":        len(eval_posts),
        "n_comments_requested":     n_generate,
        "distinct_1":               round(d1, 4),
        "distinct_2":               round(d2, 4),
        "avg_gen_length_words":     round(sum(gen_lens)/len(gen_lens), 2) if gen_lens else 0,
        "avg_ref_length_words":     round(sum(ref_lens)/len(ref_lens), 2) if ref_lens else 0,
        "correct_count_pct":        round(100 * correct / len(results), 1),
    }

    print(f"\n{'='*60}")
    print("TEST SET EVALUATION SUMMARY")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save JSON (full details)
    with open(out_path / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    # Save CSV — one row per generated comment
    csv_rows = []
    for r in results:
        for idx, gen_comment in enumerate(r["generated_comments"], start=1):
            csv_rows.append({
                "post_id":             r["post_id"],
                "post_text":           r["post_text"],
                "comment_index":       idx,
                "generated_comment":   gen_comment,
                "n_requested":         r["n_requested"],
                "n_generated":         r["n_generated"],
                "max_new_tokens_used": r["max_new_tokens_used"],
                "word_count":          len(gen_comment.split()),
            })
    pd.DataFrame(csv_rows).to_csv(out_path / "generated_comments_qwn.csv", index=False, encoding="utf-8")

    print(f"\n  Results saved → {out_path / 'eval_results.json'}")
    print(f"  CSV saved     → {out_path / 'generated_comments_qwn.csv'}")


def _distinct_n(texts: List[str], n: int) -> float:
    all_ngrams = []
    for t in texts:
        toks = t.lower().split()
        all_ngrams.extend(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0


# =============================================================================
# 6.  MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen3.5-4B-Base for comment generation")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", default=v, action="store_true")
        else:
            p.add_argument(f"--{k}", default=v, type=type(v) if v is not None else str)
    return p.parse_args()


def main():
    args = parse_args()

    # HuggingFace login (needed for gated models; Qwen is public but good practice)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("Logged in to HuggingFace Hub.")

    print("=" * 60)
    print(f"Model      : {args.model_name}")
    print(f"Train CSV  : {args.train_csv}")
    print(f"Test CSV   : {args.test_csv}")
    print(f"Output dir : {args.output_dir}")
    print(f"Epochs     : {args.num_train_epochs}  |  LR: {args.learning_rate}")
    print(f"Val split  : {args.val_split}  (stratified by Intent at post level)")
    print("=" * 60)

    # ── 1. Load model & tokenizer ────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model(
        model_name     = args.model_name,
        max_seq_length = args.max_seq_length,
        load_in_4bit   = True,
        lora_r         = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
    )

    # ── 2. Load data ─────────────────────────────────────────────────────────
    print("\n[2/5] Loading data...")
    base = Path(__file__).parent
    train_posts = load_and_group_posts(str(base / args.train_csv), max_comments=args.max_comments)
    test_posts  = load_and_group_posts(str(base / args.test_csv),  max_comments=None)  # all comments kept

    # Compute length distribution from training comments (used at inference)
    word_counts = compute_length_distribution(train_posts)

    # ── 3. Stratified train/val split (post-level) ───────────────────────────
    print("\n[3/5] Stratified train/val split (post-level, stratify by Intent)...")
    train_ds, val_ds = stratified_split(
        posts     = train_posts,
        val_split = args.val_split,
        seed      = args.seed,
        tokenizer = tokenizer,
    )

    # ── 4. Train ─────────────────────────────────────────────────────────────
    print("\n[4/5] Training...")
    train(model, tokenizer, train_ds, val_ds, args)

    # ── 5. Save ──────────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n  Model saved → {args.output_dir}")

    # ── 6. Evaluate on test set ──────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set...")
    evaluate_test_set(
        model                    = model,
        tokenizer                = tokenizer,
        test_posts               = test_posts,
        output_dir               = args.output_dir,
        n_generate               = args.n_generate,
        eval_max_posts           = min(args.eval_max_posts, len(test_posts)),
        word_counts              = word_counts,
        max_new_tokens_fallback  = args.max_new_tokens,
        temperature              = args.temperature,
        top_p                    = args.top_p,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
