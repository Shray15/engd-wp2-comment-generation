"""
DEFINITIVE FIX: Fine-tune Gemma with <EOC> token + Early Stopping
- 70% train, 10% eval, 20% test split
- Auto-generate comments for test set with realistic length sampling
- NaN eval_loss resolved
- Early stopping when eval_loss increases (prevents overfitting)
"""

import argparse
import warnings
import pickle
from typing import Dict, Optional
import re
import pandas as pd
import numpy as np
import torch
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

names = ['buurman']

# ================================
# LENGTH DISTRIBUTION UTILITIES
# ================================

def compute_and_save_length_distribution(df: pd.DataFrame, output_path: str):
    word_counts = df['comment'].apply(lambda x: len(str(x).split())).values
    
    distribution = {
        'word_counts': word_counts,
        'mean': np.mean(word_counts),
        'median': np.median(word_counts),
        'std': np.std(word_counts),
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
    
    print(f"\n{'='*70}")
    print("LENGTH DISTRIBUTION COMPUTED")
    print(f"{'='*70}")
    print(f"Mean: {distribution['mean']:.2f} words")
    print(f"Median: {distribution['median']:.2f} words")
    print(f"Std: {distribution['std']:.2f} words")
    print(f"25th percentile: {distribution['percentiles']['p25']:.0f} words")
    print(f"75th percentile: {distribution['percentiles']['p75']:.0f} words")
    print(f"90th percentile: {distribution['percentiles']['p90']:.0f} words")
    print(f"Saved to: {output_path}\n")
    
    return distribution


def load_length_distribution(distribution_path: str):
    with open(distribution_path, 'rb') as f:
        dist = pickle.load(f)
    
    print(f"\n{'='*70}")
    print("LOADED LENGTH DISTRIBUTION")
    print(f"{'='*70}")
    print(f"Mean: {dist['mean']:.2f} words")
    print(f"Median: {dist['median']:.2f} words")
    print(f"Std: {dist['std']:.2f} words")
    print(f"25th percentile: {dist['percentiles']['p25']:.0f} words")
    print(f"50th percentile: {dist['percentiles']['p50']:.0f} words")
    print(f"75th percentile: {dist['percentiles']['p75']:.0f} words")
    print(f"90th percentile: {dist['percentiles']['p90']:.0f} words\n")
    
    return dist


def sample_length_category_and_tokens(word_counts: np.ndarray):
    target_words = np.random.choice(word_counts)
    
    if target_words <= 5:
        category = "zeer kort"
        max_tokens = 8
    elif target_words <= 12:
        category = "kort"
        max_tokens = 18
    elif target_words <= 25:
        category = "gemiddeld"
        max_tokens = 38
    else:
        category = "lang"
        max_tokens = min(int(target_words * 1.5), 150)
    
    return category, max_tokens


# ================================
# CUSTOM CALLBACK FOR NaN HANDLING
# ================================

class NaNCheckCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            eval_loss = metrics['eval_loss']
            if eval_loss != eval_loss:  # NaN check
                print(f"\n⚠️  WARNING: eval_loss is NaN at step {state.global_step}")
                print("   This won't affect training, but early stopping may not work properly.")
                metrics['eval_loss'] = float('inf')
        return control


# ================================
# DATA & MODEL SETUP
# ================================

def prepare_dataset(
    csv_path: str, 
    train_size: float = 0.8, 
    eval_size: float = 0.2, 
    max_samples: Optional[int] = None
) -> tuple[DatasetDict, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"text_ha": "post", "text": "comment"})
    df['comment'] = df['comment'].str.replace('<PERSON>', 'buurman', regex=False)

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)

    INTENT_MAP = {
        "Criticism": 0, "Statement": 1,
        "Inquiry": 2, "Forward": 3, "Appreciation": 4
    }
    df["Intent"] = df["Intent"].map(INTENT_MAP)

    total = len(df)

    train_df, eval_df = train_test_split(
        df, test_size=eval_size, random_state=42, stratify=df["Intent"]
    )

    train_df = train_df.reset_index(drop=True)
    eval_df  = eval_df.reset_index(drop=True)

    print(f"\n{'='*70}")
    print("DATA SPLIT")
    print(f"{'='*70}")
    print(f"Total samples: {total}")
    print(f"Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
    print(f"Eval:  {len(eval_df)} ({len(eval_df)/total*100:.1f}%)")

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "eval": Dataset.from_pandas(eval_df),
    }), train_df


def setup_model_and_tokenizer(model_id: str):
    """Load model in 4-bit and add <EOC> token."""
    print(f"\n{'='*70}")
    print("LOADING MODEL")
    print(f"{'='*70}")
    print(f"Model: {model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"

    # FIX: only add pad token here, resize happens AFTER model is loaded
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Add <EOC> token
    if EOC_TOKEN not in tokenizer.get_vocab():
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": [EOC_TOKEN]})
        print(f"Added {num_added} tokens. <EOC> ID: {tokenizer.convert_tokens_to_ids(EOC_TOKEN)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
    )

    # Resize embeddings AFTER model is loaded
    print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} → {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def format_chat_template_with_eoc(example, tokenizer):
    """Format with <EOC> at the end and length conditioning."""
    post = str(example["post"]).strip()
    comment = str(example["comment"]).strip()
    
    word_count = len(comment.split())
    category, _ = sample_length_category_and_tokens(np.array([word_count]))
    
    system_prompt = """Je bent een echte Nederlandse huurder die reageert op Facebook-berichten van je woningcorporatie.
    
Jouw reacties worden beoordeeld op drie belangrijke criteria:

1. SENTIMENT (emotionele toon):
   - Laat je emotie zien: blij, teleurgesteld, bezorgd, boos, neutraal, of een mix
   - Wees authentiek - niet elke reactie hoeft positief te zijn
   - Match je gevoel bij de inhoud van het bericht

2. INTENT (wat wil je bereiken) - kies één van deze vijf types:
   - KRITIEK: je bent ontevreden en uit dat direct
   - VERKLARING: je deelt een feit, mening of observatie zonder oordeel
   - VRAAG: je wil informatie of verduidelijking
   - DOORSTUREN: je tag iemand anders in het bericht (<PERSON>)
   - WAARDERING: je reageert positief of bedankt de woningcorporatie

3. RELATEERDHEID (aansluiting bij bericht):
   - Reageer direct op wat er in het bericht staat
   - Gebruik details uit het bericht in je reactie
   - Toon dat je het bericht gelezen en begrepen hebt

Schrijf natuurlijk Nederlands met:
- Informele taal en spreektaal ("ff", "gwn", "tis")
- Soms typefouten zoals bij echt typen
- Emoji's waar natuurlijk"""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Schrijf een {category} natuurlijke Facebookreactie op dit bericht:\n\n{post}",
        },
        {
            "role": "assistant",
            "content": f"{comment} {EOC_TOKEN}",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    return {"text": text}


# ================================
# GENERATION FUNCTION FOR TEST SET
# ================================

def generate_test_comments(
    model_path: str,
    test_df: pd.DataFrame,
    length_distribution: Dict,
    output_path: str,
    num_samples_per_post: int = 1,
):
    print(f"\n{'='*70}")
    print("GENERATING TEST SET COMMENTS")
    print(f"{'='*70}")
    
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    if EOC_TOKEN not in tokenizer.get_vocab():
        raise ValueError(f"<EOC> token not found in tokenizer!")
    
    eoc_token_id = tokenizer.convert_tokens_to_ids(EOC_TOKEN)
    print(f"<EOC> token ID: {eoc_token_id}")

    print(f"\nLoading PEFT/LoRA model...")
    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        print(f"Base model: {peft_config.base_model_name_or_path}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        base_vocab = base_model.get_input_embeddings().weight.shape[0]
        if base_vocab != len(tokenizer):
            print(f"Resizing base model embeddings: {base_vocab} → {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))
        
        model = PeftModel.from_pretrained(base_model, model_path)
        print("✅ PEFT model loaded successfully")
        
    except Exception as e:
        print(f"Failed to load as PEFT model: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))
    
    model.eval()
    print(f"<EOC> token ID: {eoc_token_id}")
    print(f"Generating {num_samples_per_post} comment(s) per post...")
    print(f"Using realistic length sampling from training distribution")
    
    results = []
    word_counts = length_distribution['word_counts']
    
    system_prompt = """Je bent een echte Nederlandse huurder die reageert op Facebook-berichten van je woningcorporatie.
    
Jouw reacties worden beoordeeld op drie belangrijke criteria:

1. SENTIMENT (emotionele toon):
   - Laat je emotie zien: blij, teleurgesteld, bezorgd, boos, neutraal, of een mix
   - Wees authentiek - niet elke reactie hoeft positief te zijn
   - Match je gevoel bij de inhoud van het bericht

2. INTENT (wat wil je bereiken) - kies één van deze vijf types:
   - KRITIEK: je bent ontevreden en uit dat direct
   - VERKLARING: je deelt een feit, mening of observatie zonder oordeel
   - VRAAG: je wil informatie of verduidelijking
   - DOORSTUREN: je tag iemand anders in het bericht (<PERSON>)
   - WAARDERING: je reageert positief of bedankt de woningcorporatie

3. RELATEERDHEID (aansluiting bij bericht):
   - Reageer direct op wat er in het bericht staat
   - Gebruik details uit het bericht in je reactie
   - Toon dat je het bericht gelezen en begrepen hebt

Schrijf natuurlijk Nederlands met:
- Informele taal en spreektaal ("ff", "gwn", "tis")
- Soms typefouten zoals bij echt typen
- Emoji's waar natuurlijk"""

    # Rename only if columns haven't been renamed yet
    if "text_ha" in test_df.columns:
        test_df = test_df.rename(columns={"text_ha": "post", "text": "comment"})

    for idx, row in test_df.iterrows():
        post = row['post']
        true_comment = row['comment']
        
        for sample_num in range(num_samples_per_post):
            length_category, max_tokens = sample_length_category_and_tokens(word_counts)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Schrijf een {length_category} natuurlijke Facebookreactie op dit bericht:\n\n{post}"}
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
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
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=False
            )
            
            # Clean up Gemma-specific tokens
            generated_text = generated_text.replace(EOC_TOKEN, "").strip()
            generated_text = generated_text.replace("<end_of_turn>", "").strip()
            
            generated_word_count = len(generated_text.split())
            
            results.append({
                'post': post,
                'true_comment': true_comment,
                'generated_comment': generated_text,
                'length_category': length_category,
                'max_tokens_allowed': max_tokens,
                'generated_word_count': generated_word_count,
                'true_word_count': len(true_comment.split()),
                'sample_num': sample_num,
                'test_idx': idx,
            })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} posts...")

    results_df = pd.DataFrame(results)
    
    results_df['true_comment'] = results_df['true_comment'].str.replace('buurman', "<PERSON>", regex=True)
    results_df['generated_comment'] = results_df['generated_comment'].str.replace('buurman', '<PERSON>', regex=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Generated {len(results)} comments for {len(test_df)} posts")
    print(f"\nLength Statistics:")
    print(f"  Mean generated length: {results_df['generated_word_count'].mean():.2f} words")
    print(f"  Mean true length: {results_df['true_word_count'].mean():.2f} words")
    print(f"  Median generated: {results_df['generated_word_count'].median():.0f} words")
    print(f"  Median true: {results_df['true_word_count'].median():.0f} words")
    print(f"\nLength Category Distribution:")
    print(results_df['length_category'].value_counts(normalize=True).sort_index())
    print(f"\nSaved to: {output_path}")


# ================================
# TRAINING WITH EARLY STOPPING
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
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.0,
):
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    dataset, train_df = prepare_dataset(
        csv_path,
        train_size=train_size,
        eval_size=eval_size,
        max_samples=max_samples
    )
    
    # Load hand-labelled test set separately
    test_df = pd.read_csv(test_data_path)
    test_df = test_df.rename(columns={"text_ha": "post", "text": "comment"})
    test_df['comment'] = test_df['comment'].str.replace('<PERSON>', 'buurman', regex=True)
    print(f"Hand-labelled test set: {len(test_df)} samples")
    
    # Compute and save length distribution from training data
    length_dist_path = output_dir + "_length_distribution.pkl"
    length_dist = compute_and_save_length_distribution(train_df, length_dist_path)

    # Model/tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_id)

    # Format datasets
    print("\n" + "="*70)
    print("FORMATTING DATA")
    print("="*70)

    train_ds = dataset["train"].map(
        lambda x: format_chat_template_with_eoc(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    eval_ds = dataset["eval"].map(
        lambda x: format_chat_template_with_eoc(x, tokenizer),
        remove_columns=dataset["eval"].column_names,
    )

    # Sanity check
    print("\n" + "="*70)
    print("SANITY CHECK")
    print("="*70)
    sample = train_ds[0]["text"]

    print("\nFIRST 600 chars:")
    print(sample[:600])

    print("\nLAST 300 chars:")
    print(sample[-300:])

    # Gemma uses <start_of_turn>model instead of Llama's header tokens
    assert "<start_of_turn>model" in sample or "assistant" in sample, "❌ Missing assistant turn!"
    assert EOC_TOKEN in sample, f"❌ Missing {EOC_TOKEN}!"

    eoc_id = tokenizer.convert_tokens_to_ids(EOC_TOKEN)
    sample_ids = tokenizer.encode(sample, add_special_tokens=False)
    assert eoc_id in sample_ids, f"❌ {EOC_TOKEN} not tokenized!"

    print(f"\n✅ All checks passed! {EOC_TOKEN} ID: {eoc_id}")

    # LoRA
    print("\n" + "="*70)
    print("SETTING UP LORA")
    print("="*70)
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data collator
    print("\n" + "="*70)
    print("SETTING UP COLLATOR")
    print("="*70)
    print("Using DataCollatorForLanguageModeling (avoids NaN eval issues)")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training args
    print("\n" + "="*70)
    print("TRAINING ARGUMENTS + EARLY STOPPING")
    print("="*70)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        # Learning
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,

        # Logging and evaluation
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,

        # Saving - CRITICAL for early stopping
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Precision - bfloat16 for Gemma
        fp16=False,
        bf16=True,

        # Optimization
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,

        # Misc
        report_to="none",
        push_to_hub=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    print(f"Learning rate: {learning_rate}")
    print(f"Epochs (max): {num_epochs}")
    print(f"Effective batch: {batch_size * grad_accum}")
    print(f"\nEARLY STOPPING: patience={early_stopping_patience}")

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        ),
        NaNCheckCallback(),
    ]

    trainer = SFTTrainer(
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
    )

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}, Test: {len(test_df)}")

    trainer.train()

    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"[OK] Model saved to: {output_dir}")
    
    print("\n" + "="*70)
    print("GENERATING TEST SET COMMENTS WITH LENGTH SAMPLING")
    print("="*70)
    
    test_output_path = output_dir + "_test_set_v30_generated.csv"
    
    generate_test_comments(
        model_path=output_dir,
        test_df=test_df,
        length_distribution=length_dist,
        output_path=test_output_path,
        num_samples_per_post=1,
    )
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"Model: {output_dir}")
    print(f"Length distribution: {length_dist_path}")
    print(f"Test results: {test_output_path}")


def main():
    parser = argparse.ArgumentParser("Fine-tune Gemma with Length Distribution Sampling")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str,   
                        default="/vast.mnt/home/20245179/Desktop/final_test_set.csv")
    parser.add_argument("--output_dir", type=str, default="./gemma-fb-comments-finetuned_v01")
    
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--eval_size", type=float, default=0.2)

    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

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
    )

if __name__ == "__main__":
    main()