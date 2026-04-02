"""
APPROACH 6: Multi-Task Loss Head Fine-tuning
Built on top of Approach 3 (Llama-3.2-3B + length distribution conditioning)

Changes vs v12:
- Adds SentimentHead, IntentHead, RelatednessHead on top of LLM hidden states
- Custom MultiTaskSFTTrainer overrides compute_loss to combine:
    L_total = L_CE + λ_s * L_sentiment + λ_i * L_intent + λ_r * L_relatedness
- Auxiliary heads are DROPPED at inference time (generation is identical to v12)
- Class weights (inverse frequency) applied to sentiment and intent CE losses
- Stratified train/eval/test split on Intent
- <PERSON> replaced with 'buurman' during training, reverted at generation

Expected CSV columns:
    text_ha    → housing association post
    text       → tenant comment
    Sentiment  → -1/0/1  (mapped to 0/1/2 internally)
    Intent     → Criticism/Statement/Inquiry/Forward/Appreciation
    Similarity → float in [0, 1] (cosine similarity from WP1)

Usage:
    python train_multitask_v13.py \
        --csv_path data/comments_with_labels.csv \
        --output_dir ./llama-multitask-v13 \
        --lambda_sentiment 0.05 \
        --lambda_intent 0.05 \
        --lambda_relatedness 0.05
"""

import argparse
import re
import warnings
import pickle
from typing import Dict, Optional, Tuple, Union
import bitsandbytes as bnb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from trl import SFTTrainer

warnings.filterwarnings("ignore")

EOC_TOKEN = "<EOC>"

# ================================
# LABEL CONSTANTS
# ================================

NUM_SENTIMENTS = 3
NUM_INTENTS = 5
LABEL_IGNORE_IDX = -100

# Single placeholder — consistent between training and revert
PERSON_PLACEHOLDER = 'buurman'


# ================================
# AUXILIARY HEAD DEFINITIONS
# ================================

class SentimentHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = NUM_SENTIMENTS, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IntentHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = NUM_INTENTS, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelatednessHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, post_hidden: torch.Tensor, comment_hidden: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([post_hidden, comment_hidden], dim=-1)
        return self.net(combined).squeeze(-1)


# ================================
# CUSTOM TRAINER
# ================================

class MultiTaskSFTTrainer(SFTTrainer):
    """
    Extends SFTTrainer with auxiliary sentiment / intent / relatedness heads.
    Class weights are computed from inverse frequency of WP1 training distribution.

    Class weights are computed dynamically from the actual train split counts
    (passed in via sentiment_class_counts and intent_class_counts tensors).
    This ensures weights reflect the true training distribution after stratified splitting.
    """

    def __init__(
        self,
        *args,
        sentiment_head: SentimentHead,
        intent_head: IntentHead,
        relatedness_head: RelatednessHead,
        lambda_sentiment: float = 0.05,
        lambda_intent: float = 0.05,
        lambda_relatedness: float = 0.05,
        sentiment_class_counts: Optional[torch.Tensor] = None,
        intent_class_counts: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        device = next(self.model.parameters()).device

        self.sentiment_head    = sentiment_head.to(device)
        self.intent_head       = intent_head.to(device)
        self.relatedness_head  = relatedness_head.to(device)
        self.lambda_sentiment  = lambda_sentiment
        self.lambda_intent     = lambda_intent
        self.lambda_relatedness = lambda_relatedness

        # ── Inverse-frequency class weights from TRAIN split counts ──
        # Counts are computed dynamically in train() from train_df after stratified split
        if sentiment_class_counts is None:
            raise ValueError("sentiment_class_counts must be provided (computed from train_df)")
        if intent_class_counts is None:
            raise ValueError("intent_class_counts must be provided (computed from train_df)")

        # Sentiment: Custom weights CE loss
        sentiment_weights = torch.tensor([3.0, 2.0, 0.3], dtype = torch.float)
        self.sentiment_weights = sentiment_weights.to(device)
        

        
        # Order: Criticism=0, Statement=1, Inquiry=2, Forward=3, Appreciation=4
        i_weights = torch.tensor([6.0, 2.0, 3.0, 4.0, 0.1], dtype=torch.float)
#                               Crit  Stmt  Inq  Fwd   App
        self.intent_weights = i_weights.to(device)
        
        
        print(f"\n{'='*60}\nCLASS WEIGHTS / TARGET DISTS (from train split)\n{'='*60}")
        print(f"Sentiment counts  (0=neg,1=neu,2=pos): {sentiment_class_counts.tolist()}")
        print(f"Sentiment p_target:                    {[round(x,4) for x in self.sentiment_weights.tolist()]}")
        print(f"Intent counts (0=Crit,1=Stmt,2=Inq,3=Fwd,4=App): {intent_class_counts.tolist()}")
        #print(f"Intent weights:                        {[round(x,4) for x in self.intent_target_dist.tolist()]}")
        print(f"Intent weights (custom):               {[round(x,4) for x in self.intent_weights.tolist()]}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_device(self):
        return next(self.model.parameters()).device

    def _mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1).float()
        summed = (hidden * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _split_post_comment(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        comment_mask = (labels != LABEL_IGNORE_IDX).float()
        post_mask = (attention_mask == 1).float() * (labels == LABEL_IGNORE_IDX).float()

        post_zero = (post_mask.sum(dim=1) == 0)
        if post_zero.any():
            seq_len = hidden.size(1)
            half = seq_len // 2
            fallback_post = torch.zeros_like(post_mask)
            fallback_post[:, :half] = attention_mask[:, :half].float()
            post_mask[post_zero] = fallback_post[post_zero]

        post_pool    = self._mean_pool(hidden, post_mask)
        comment_pool = self._mean_pool(hidden, comment_mask)
        return post_pool, comment_pool, post_mask, comment_mask
    
    def _prepare_dataset(self, dataset, tokenizer, packing, dataset_text_field, *args, **kwargs):
        label_cols = {}
        for col in ["sentiment_labels", "intent_labels", "relatedness_labels"]:
            if col in dataset.column_names:
                label_cols[col] = dataset[col]
        
        dataset = super()._prepare_dataset(
            dataset, tokenizer, packing, dataset_text_field, *args, **kwargs
        )
        
        for col, values in label_cols.items():
            if col not in dataset.column_names:
                dataset = dataset.add_column(col, values)
        
        return dataset
        
    def create_optimizer(self):
        aux_params = (
            list(self.intent_head.parameters()) +
            list(self.sentiment_head.parameters()) +
            list(self.relatedness_head.parameters())
        )
        aux_param_ids = {id(p) for p in aux_params}
    
        lm_params = [p for p in self.model.parameters()
                     if id(p) not in aux_param_ids]
    
        optimizer = bnb.optim.AdamW8bit([
            {"params": lm_params,  "lr": self.args.learning_rate},
            {"params": aux_params, "lr": 1e-3},
        ])
    
        self.optimizer = optimizer
        return optimizer
    # ------------------------------------------------------------------
    # compute_loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        device = self._get_device()

        sentiment_labels   = inputs.pop("sentiment_labels",   None)
        intent_labels      = inputs.pop("intent_labels",      None)
        relatedness_labels = inputs.pop("relatedness_labels", None)
        
        if self.state.global_step == 0:
          print(f"DEBUG sentiment_labels: {sentiment_labels}")
          print(f"DEBUG intent_labels: {intent_labels}")
          print(f"DEBUG input keys: {list(inputs.keys())}")

        outputs = model(**inputs, output_hidden_states=True)
        lm_loss = outputs.loss

        last_hidden = outputs.hidden_states[-1]

        attention_mask   = inputs.get("attention_mask",
                                      torch.ones(last_hidden.shape[:2], device=device))
        labels_for_split = inputs.get("labels",
                                      torch.full(last_hidden.shape[:2], LABEL_IGNORE_IDX, device=device))

        post_pool, comment_pool, _, _ = self._split_post_comment(
            last_hidden.float(), attention_mask, labels_for_split,
        )

        aux_loss = torch.tensor(0.0, device=device)
        # Add these right after aux_loss initialisation
        l_s = torch.tensor(0.0, device=device)
        l_i = torch.tensor(0.0, device=device)
        l_r = torch.tensor(0.0, device=device)

        # ── Sentiment (per-sample weighted CE) ──
        if sentiment_labels is not None and self.lambda_sentiment > 0:
            sentiment_labels = sentiment_labels.to(device)
            valid = sentiment_labels != LABEL_IGNORE_IDX
            if valid.any():
                s_logits = self.sentiment_head(comment_pool[valid])
                l_s = F.cross_entropy(
                    s_logits,
                    sentiment_labels[valid],
                    weight=self.sentiment_weights.to(s_logits.device),
                )
            aux_loss = aux_loss + self.lambda_sentiment * l_s
        
        # ── Intent (per-sample weighted CE) ──
        if intent_labels is not None and self.lambda_intent > 0:
            intent_labels = intent_labels.to(device)
            valid = intent_labels != LABEL_IGNORE_IDX
            if valid.any():
                i_logits = self.intent_head(comment_pool[valid])
                l_i = F.cross_entropy(
                    i_logits,
                    intent_labels[valid],
                    weight=self.intent_weights.to(i_logits.device),
                )
            aux_loss = aux_loss + self.lambda_intent * l_i

        # ── Relatedness (MSE) ──
        if relatedness_labels is not None and self.lambda_relatedness > 0:
            relatedness_labels = relatedness_labels.to(device).float()
            valid = relatedness_labels >= 0.0
            if valid.any():
                r_pred = self.relatedness_head(post_pool[valid], comment_pool[valid])
                l_r = F.mse_loss(r_pred, relatedness_labels[valid])
                aux_loss = aux_loss + self.lambda_relatedness * l_r
                
                
        if self.state.global_step % 10 == 0:
          print(
          f"[step {self.state.global_step}] "
          f"LM: {lm_loss.item():.4f} | "
          f"Intent CE: {l_i.item() if intent_labels is not None and valid.any() else 0.0:.4f} | "
          f"Intent weighted: {(self.lambda_intent * l_i).item() if intent_labels is not None and valid.any() else 0.0:.4f} | "
          f"Sentiment KL: {l_s.item() if sentiment_labels is not None else 0.0:.4f} | "
          f"Sentiment weighted: {(self.lambda_sentiment * l_s).item() if sentiment_labels is not None else 0.0:.4f} | "
          f"Relatedness MSE: {l_r.item() if relatedness_labels is not None else 0.0:.4f} | "
          f"Total: {(lm_loss + aux_loss).item():.4f}"
        )

        total_loss = lm_loss + aux_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss


# ================================
# LENGTH DISTRIBUTION UTILITIES
# ================================

def compute_and_save_length_distribution(df: pd.DataFrame, output_path: str):
    word_counts = df['comment'].apply(lambda x: len(str(x).split())).values
    distribution = {
        'word_counts': word_counts,
        'mean':    np.mean(word_counts),
        'median':  np.median(word_counts),
        'std':     np.std(word_counts),
        'percentiles': {
            'p25': np.percentile(word_counts, 25),
            'p50': np.percentile(word_counts, 50),
            'p75': np.percentile(word_counts, 75),
            'p90': np.percentile(word_counts, 90),
            'p95': np.percentile(word_counts, 95),
        }
    }
    with open(output_path, 'wb') as f:
        pickle.dump(distribution, f)
    print(f"\n{'='*70}\nLENGTH DISTRIBUTION COMPUTED\n{'='*70}")
    print(f"Mean: {distribution['mean']:.2f}  Median: {distribution['median']:.2f}  "
          f"Std: {distribution['std']:.2f}")
    print(f"Saved to: {output_path}\n")
    return distribution


def load_length_distribution(distribution_path: str):
    with open(distribution_path, 'rb') as f:
        return pickle.load(f)


def sample_length_category_and_tokens(word_counts: np.ndarray):
    target_words = np.random.choice(word_counts)
    if target_words <= 5:
        return "zeer kort", 20
    elif target_words <= 12:
        return "kort", 40
    elif target_words <= 25:
        return "gemiddeld", 70
    else:
        return "lang", min(int(target_words * 2.5), 250)


# ================================
# NaN CALLBACK
# ================================

class NaNCheckCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            if metrics['eval_loss'] != metrics['eval_loss']:
                print(f"\n⚠️  WARNING: eval_loss is NaN at step {state.global_step}")
                metrics['eval_loss'] = float('inf')
        return control


# ================================
# DATA PREPARATION
# ================================

def prepare_dataset(
    csv_path: str,
    train_size: float = 0.8,
    eval_size: float = 0.2,
    max_samples: Optional[int] = None,
) -> Tuple[DatasetDict, pd.DataFrame, pd.DataFrame]:
    """
    Load CSV, replace <PERSON> with 'buurman', map labels, stratified split on Intent.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"text_ha": "post", "text": "comment"})

    # ── Replace <PERSON> with neutral placeholder ──
    df['comment'] = df['comment'].str.replace('<PERSON>', PERSON_PLACEHOLDER, regex=False)
    print(f"✅ Replaced <PERSON> with '{PERSON_PLACEHOLDER}'")

    # ── Label mappings ──
    SENTIMENT_MAP = {-1: 0, 0: 1, 1: 2}
    INTENT_MAP = {
        "Criticism":    0,
        "Statement":    1,
        "Inquiry":      2,
        "Forward":      3,
        "Appreciation": 4,
    }

    # ── Ensure columns exist ──
    if "Sentiment" not in df.columns:
        print("⚠️  No 'Sentiment' column — sentiment head disabled.")
        df["Sentiment"] = LABEL_IGNORE_IDX
    if "Intent" not in df.columns:
        print("⚠️  No 'Intent' column — intent head disabled.")
        df["Intent"] = LABEL_IGNORE_IDX
    if "Similarity" not in df.columns:
        print("⚠️  No 'Similarity' column — relatedness head disabled.")
        df["Similarity"] = -1.0



    # Rename to internal names
    df = df.rename(columns={"Sentiment": "sentiment", "Intent": "intent", "Similarity": "relatedness"})

    # ── Map sentiment ──
    unmapped = ~df["sentiment"].isin(SENTIMENT_MAP.keys()) & (df["sentiment"] != LABEL_IGNORE_IDX)
    if unmapped.any():
        print(f"⚠️  {unmapped.sum()} rows unrecognised sentiment → ignored.")
        df.loc[unmapped, "sentiment"] = LABEL_IGNORE_IDX
    df["sentiment"] = df["sentiment"].map(
        lambda x: SENTIMENT_MAP.get(x, LABEL_IGNORE_IDX)
    ).astype(int)

    # ── Map intent ──
    unmapped = ~df["intent"].isin(INTENT_MAP.keys()) & (df["intent"] != LABEL_IGNORE_IDX)
    if unmapped.any():
        print(f"⚠️  {unmapped.sum()} rows unrecognised intent → ignored.")
        print(f"   Values: {df.loc[unmapped, 'intent'].unique().tolist()}")
        df.loc[unmapped, "intent"] = LABEL_IGNORE_IDX
    df["intent"] = df["intent"].map(
        lambda x: INTENT_MAP.get(x, LABEL_IGNORE_IDX) if isinstance(x, str) else x
    ).astype(int)

    # ── Relatedness ──
    df["relatedness"] = pd.to_numeric(df["relatedness"], errors="coerce").fillna(-1.0).astype(float)

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)

    # ── Stratified split on intent ──
    train_df, eval_df = train_test_split(
        df, test_size=eval_size, random_state=42, stratify=df["intent"],
    )

    train_df = train_df.reset_index(drop=True)
    eval_df  = eval_df.reset_index(drop=True)

    intent_names = {0: 'Criticism', 1: 'Statement', 2: 'Inquiry', 3: 'Forward', 4: 'Appreciation'}
    total = len(df)

    print(f"\n{'='*70}\nSTRATIFIED DATA SPLIT (on Intent)\n{'='*70}")
    print(f"Total: {total}  Train: {len(train_df)}  Eval: {len(eval_df)}")

    for split_name, split_df in [("Train", train_df), ("Eval", eval_df)]:
        dist = {intent_names[k]: v for k, v in
                split_df['intent'].value_counts().sort_index().items()}
        s_dist = split_df[split_df['sentiment'] != LABEL_IGNORE_IDX]['sentiment'] \
                     .value_counts().sort_index().to_dict()
        print(f"\n  {split_name}:")
        print(f"    Intent:    {dist}")
        print(f"    Sentiment: {s_dist}  (0=neg,1=neu,2=pos)")
        print(f"    Relatedness mean: "
              f"{split_df[split_df['relatedness'] >= 0]['relatedness'].mean():.3f}")
        lengths = split_df['comment'].astype(str).str.split().str.len()
        print(f"    Length — mean: {lengths.mean():.1f}  median: {lengths.median():.0f}  "
              f"std: {lengths.std():.1f}")

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "eval":  Dataset.from_pandas(eval_df),
    }), train_df


# ================================
# MODEL SETUP
# ================================

def setup_model_and_tokenizer(model_id: str):
    print(f"\n{'='*70}\nLOADING MODEL\n{'='*70}\nModel: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if EOC_TOKEN not in tokenizer.get_vocab():
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": [EOC_TOKEN]})
        print(f"Added {num_added} token(s). <EOC> ID: {tokenizer.convert_tokens_to_ids(EOC_TOKEN)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} → {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def setup_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )


# ================================
# DATASET FORMATTING
# ================================

SYSTEM_PROMPT = """Je bent een echte Nederlandse huurder die reageert op Facebook-berichten van je woningcorporatie.
    
Jouw reacties worden beoordeeld op drie belangrijke criteria:

1. SENTIMENT (emotionele toon):
   - Laat je emotie zien: blij, teleurgesteld, bezorgd, boos, neutraal, of een mix
   - Wees authentiek - niet elke reactie hoeft positief te zijn
   - Match je gevoel bij de inhoud van het bericht

2. INTENT (wat wil je bereiken) - kies één van deze vijf types:
   - KRITIEK: je bent ontevreden en uit dat direct
   - VERKLARING: je deelt een feit, mening of observatie zonder oordeel
   - VRAAG: je wil informatie of verduidelijking
   - DOORSTUREN: je tag iemand anders in het bericht
   - WAARDERING: je reageert positief of bedankt de woningcorporatie

3. RELATEERDHEID (aansluiting bij bericht):
   - Reageer direct op wat er in het bericht staat
   - Gebruik details uit het bericht in je reactie
   - Toon dat je het bericht gelezen en begrepen hebt

Schrijf natuurlijk Nederlands met:
- Informele taal en spreektaal ("ff", "gwn", "tis")
- Soms typefouten zoals bij echt typen
- Emoji's waar natuurlijk"""


def format_chat_template_with_eoc(example, tokenizer):
    post    = str(example["post"]).strip()
    comment = str(example["comment"]).strip()

    word_count = len(comment.split())
    if word_count <= 5:
        category = "zeer kort"
    elif word_count <= 12:
        category = "kort"
    elif word_count <= 25:
        category = "gemiddeld"
    else:
        category = "lang"

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Schrijf een {category} natuurlijke Facebookreactie op dit bericht:\n\n{post}"},
        {"role": "assistant", "content": f"{comment} {EOC_TOKEN}"},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return {
        "text":               text,
        "sentiment_labels":   int(example.get("sentiment",   LABEL_IGNORE_IDX)),
        "intent_labels":      int(example.get("intent",      LABEL_IGNORE_IDX)),
        "relatedness_labels": float(example.get("relatedness", -1.0)),
    }


# ================================
# CUSTOM DATA COLLATOR
# ================================

class MultiTaskDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        sentiment_labels   = [f.pop("sentiment_labels",   LABEL_IGNORE_IDX) for f in features]
        intent_labels      = [f.pop("intent_labels",      LABEL_IGNORE_IDX) for f in features]
        relatedness_labels = [f.pop("relatedness_labels", -1.0)             for f in features]
        
        # Remove text column — it's a string, can't be tensorized
        for f in features:
            f.pop("text", None)

        batch = super().__call__(features)

        batch["sentiment_labels"]   = torch.tensor(sentiment_labels,   dtype=torch.long)
        batch["intent_labels"]      = torch.tensor(intent_labels,      dtype=torch.long)
        batch["relatedness_labels"] = torch.tensor(relatedness_labels, dtype=torch.float)

        return batch


# ================================
# TEST SET GENERATION
# ================================

def generate_test_comments(
    model_path: str,
    test_df: pd.DataFrame,
    length_distribution: Dict,
    output_path: str,
    num_samples_per_post: int = 1,
):
    print(f"\n{'='*70}\nGENERATING TEST SET COMMENTS\n{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if EOC_TOKEN not in tokenizer.get_vocab():
        raise ValueError("<EOC> token not found in tokenizer!")
    eoc_token_id = tokenizer.convert_tokens_to_ids(EOC_TOKEN)

    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, device_map="auto", torch_dtype=torch.float16,
        )
        if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, model_path)
        print("✅ PEFT model loaded")
    except Exception as e:
        print(f"PEFT load failed ({e}), loading as full model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16)
        model.resize_token_embeddings(len(tokenizer))

    # Auxiliary heads are NOT loaded — inference is pure causal LM
    model.eval()

    word_counts = length_distribution['word_counts']
    results = []

    for idx, row in test_df.iterrows():
        post         = row['post']
        true_comment = row['comment']

        for sample_num in range(num_samples_per_post):
            length_category, max_tokens = sample_length_category_and_tokens(word_counts)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Schrijf een {length_category} natuurlijke "
                                               f"Facebookreactie op dit bericht:\n\n{post}"}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    eos_token_id=eoc_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False
            )
            generated_text = generated_text.replace(EOC_TOKEN, "").replace("<|eot_id|>", "").strip()

            # ── Revert placeholder → <PERSON> (case-insensitive) ──
            generated_text = re.sub(
                r'\b' + re.escape(PERSON_PLACEHOLDER) + r'\b',
                '<PERSON>', generated_text, flags=re.IGNORECASE
            )

            results.append({
                'post':                  post,
                'true_comment':          true_comment,
                'generated_comment':     generated_text,
                'length_category':       length_category,
                'max_tokens_allowed':    max_tokens,
                'generated_word_count':  len(generated_text.split()),
                'true_word_count':       len(true_comment.split()),
                'sample_num':            sample_num,
                'test_idx':              idx,
            })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(test_df)} posts...")

    results_df = pd.DataFrame(results)

    # Also revert placeholder in true_comment column (was replaced during prepare_dataset)
    results_df['true_comment'] = results_df['true_comment'].apply(
        lambda x: re.sub(
            r'\b' + re.escape(PERSON_PLACEHOLDER) + r'\b',
            '<PERSON>', str(x), flags=re.IGNORECASE
        )
    )

    results_df.to_csv(output_path, index=False)
    print(f"\nGenerated {len(results)} comments.")
    print(f"Mean generated length: {results_df['generated_word_count'].mean():.2f} words")
    print(f"Mean true length:      {results_df['true_word_count'].mean():.2f} words")
    print(f"Saved to: {output_path}")


# ================================
# TRAINING
# ================================

def train(
    model_id: str,
    csv_path: str,
    test_data_path: str,
    output_dir: str,
    max_samples: Optional[int],
    train_size: float,
    eval_size: float,
    num_epochs: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    max_seq_length: int,
    early_stopping_patience: int,
    lambda_sentiment: float,
    lambda_intent: float,
    lambda_relatedness: float,
):
    dataset, train_df = prepare_dataset(
        csv_path, train_size, eval_size, max_samples
    )
    
    # Load hand-labelled test set separately
    test_df = pd.read_csv(test_data_path)
    test_df = test_df.rename(columns={"text_ha": "post", "text": "comment"})
    test_df['comment'] = test_df['comment'].str.replace('<PERSON>', 'buurman', regex=True)
    print(f"Hand-labelled test set: {len(test_df)} samples")

    length_dist_path = output_dir + "_length_distribution.pkl"
    length_dist = compute_and_save_length_distribution(train_df, length_dist_path)

    model, tokenizer = setup_model_and_tokenizer(model_id)
    hidden_size = model.config.hidden_size
    print(f"\nHidden size: {hidden_size}")

    print(f"\n{'='*70}\nFORMATTING DATA\n{'='*70}")
    fmt = lambda x: format_chat_template_with_eoc(x, tokenizer)
    train_ds = dataset["train"].map(fmt, remove_columns=dataset["train"].column_names)
    eval_ds  = dataset["eval"].map(fmt,  remove_columns=dataset["eval"].column_names)

    # Sanity check
    sample = train_ds[0]["text"]
    assert "<|start_header_id|>assistant<|end_header_id|>" in sample, "❌ Missing assistant header!"
    assert EOC_TOKEN in sample, f"❌ Missing {EOC_TOKEN}!"
    eoc_id = tokenizer.convert_tokens_to_ids(EOC_TOKEN)
    assert eoc_id in tokenizer.encode(sample, add_special_tokens=False), \
        f"❌ {EOC_TOKEN} not tokenized!"
    print(f"✅ Sanity check passed. <EOC> ID: {eoc_id}")

    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = next(model.parameters()).device
    sentiment_head   = SentimentHead(hidden_size).to(device)
    intent_head      = IntentHead(hidden_size).to(device)
    relatedness_head = RelatednessHead(hidden_size).to(device)

    print(f"\n{'='*70}\nAUXILIARY HEADS\n{'='*70}")
    print(f"SentimentHead   params: {sum(p.numel() for p in sentiment_head.parameters()):,}")
    print(f"IntentHead      params: {sum(p.numel() for p in intent_head.parameters()):,}")
    print(f"RelatednessHead params: {sum(p.numel() for p in relatedness_head.parameters()):,}")
    print(f"Loss weights:  λ_s={lambda_sentiment}  λ_i={lambda_intent}  λ_r={lambda_relatedness}")

    collator = MultiTaskDataCollator(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        #optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        report_to="none",
        push_to_hub=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
        NaNCheckCallback(),
    ]

    # ── Compute class counts from train_df (used for inverse-frequency weights) ──
    sentiment_class_counts = torch.tensor(
        [(train_df['sentiment'] == i).sum() for i in range(NUM_SENTIMENTS)],
        dtype=torch.float
    )
    intent_class_counts = torch.tensor(
        [(train_df['intent'] == i).sum() for i in range(NUM_INTENTS)],
        dtype=torch.float
    )
    print(f"\n{'='*70}\nTRAIN SPLIT CLASS COUNTS\n{'='*70}")
    print(f"Sentiment: {sentiment_class_counts.tolist()}  (0=neg, 1=neu, 2=pos)")
    print(f"Intent:    {intent_class_counts.tolist()}  (0=Crit,1=Stmt,2=Inq,3=Fwd,4=App)")

    trainer = MultiTaskSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        callbacks=callbacks,
        sentiment_head=sentiment_head,
        intent_head=intent_head,
        relatedness_head=relatedness_head,
        lambda_sentiment=lambda_sentiment,
        lambda_intent=lambda_intent,
        lambda_relatedness=lambda_relatedness,
        sentiment_class_counts=sentiment_class_counts,
        intent_class_counts=intent_class_counts,
        
    )

    print(f"\n{'='*70}\nSTARTING TRAINING\n{'='*70}")
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}  Test: {len(test_df)}")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Model saved to: {output_dir}")

    heads_path = output_dir + "_aux_heads.pt"
    torch.save({
        "sentiment_head":    sentiment_head.state_dict(),
        "intent_head":       intent_head.state_dict(),
        "relatedness_head":  relatedness_head.state_dict(),
        "lambda_sentiment":  lambda_sentiment,
        "lambda_intent":     lambda_intent,
        "lambda_relatedness": lambda_relatedness,
    }, heads_path)
    print(f"✅ Auxiliary heads saved to: {heads_path}")

    test_output_path = output_dir + "_test_set_v13_generated.csv"
    generate_test_comments(
        model_path=output_dir,
        test_df=test_df,
        length_distribution=length_dist,
        output_path=test_output_path,
        num_samples_per_post=1,
    )

    print(f"\n{'='*70}\nALL DONE!\n{'='*70}")
    print(f"Model:               {output_dir}")
    print(f"Length distribution: {length_dist_path}")
    print(f"Auxiliary heads:     {heads_path}")
    print(f"Test results:        {test_output_path}")


# ================================
# CLI
# ================================

def main():
    parser = argparse.ArgumentParser("Approach 6: Multi-Task Loss Head Fine-tuning")
    parser.add_argument("--model_id",   type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--csv_path",  type=str, required=True)
    parser.add_argument("--test_data_path", type=str,  
                        default="/vast.mnt/home/20245179/Desktop/final_test_set.csv")       
    parser.add_argument("--output_dir",type=str, default="./llama-multitask-v40_CElosssentiment_CElossintent_seplrate")

    parser.add_argument("--max_samples",type=int,   default=None)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--eval_size",  type=float, default=0.2)

    parser.add_argument("--num_epochs",              type=int,   default=20)
    parser.add_argument("--batch_size",              type=int,   default=32)
    parser.add_argument("--grad_accum",              type=int,   default=4)
    parser.add_argument("--learning_rate",           type=float, default=2e-5)
    parser.add_argument("--max_seq_length",          type=int,   default=512)
    parser.add_argument("--early_stopping_patience", type=int,   default=3)

    parser.add_argument("--lambda_sentiment",   type=float, default=0.3)
    parser.add_argument("--lambda_intent",      type=float, default=0.5)
    parser.add_argument("--lambda_relatedness", type=float, default=0.1)

    args = parser.parse_args()
    train(
        model_id=args.model_id,
        csv_path=args.csv_path,
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        train_size=args.train_size,
        eval_size=args.eval_size,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        early_stopping_patience=args.early_stopping_patience,
        lambda_sentiment=args.lambda_sentiment,
        lambda_intent=args.lambda_intent,
        lambda_relatedness=args.lambda_relatedness,
    )


if __name__ == "__main__":
    main()