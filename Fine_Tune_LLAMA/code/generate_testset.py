"""
Standalone Test Comment Generation Script
Based on Approach 6 (Multi-Task Loss Head Fine-tuning)

Loads a saved PEFT checkpoint and generates comments for a test set.
Auxiliary heads are NOT used at inference — pure causal LM generation.

Uses the original category-based length sampling (zeer kort / kort / gemiddeld / lang)
with improved token budgets to avoid truncation, plus:
- trim_to_complete_sentence() applied post-generation
- dual EOS: [eoc_token_id, tokenizer.eos_token_id]
- tqdm progress bar
- <|end_of_text|> cleanup alongside EOC and eot_id
- dtype= instead of deprecated torch_dtype=
- mean_resizing=False to suppress embedding resize warning

Usage:
    python generate_comments.py \
        --model_path ./llama-multitask-v40_CElosssentiment_CElossintent_seplrate \
        --test_data_path ./final_test_set.csv \
        --length_dist_path ./llama-multitask-v40_CElosssentiment_CElossintent_seplrate_length_distribution.pkl \
        --output_path ./generated_comments.csv \
        --num_samples 1
"""

import argparse
import re
import pickle
import warnings
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================
# CONSTANTS
# ================================

EOC_TOKEN          = "<EOC>"
PERSON_PLACEHOLDER = "buurman"

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
- Emoji's waar natuurlijk

Schrijf altijd een volledige, afgeronde reactie. Maak nooit een zin half af."""


# ================================
# LENGTH DISTRIBUTION
# ================================

def load_length_distribution(distribution_path: str) -> Dict:
    with open(distribution_path, "rb") as f:
        return pickle.load(f)


def sample_length_category_and_tokens(word_counts: np.ndarray) -> Tuple[str, int]:
    """
    Sample a length category and token budget from the training distribution.
    Token budgets use ~2.5 tokens/word headroom to avoid truncation in Dutch.

    Categories:
        zeer kort : ≤5 words   → 8 tokens
        kort      : ≤12 words  → 18 tokens
        gemiddeld : ≤25 words  → 38 tokens
        lang      : >25 words  → min(target * 1.5, 150) tokens
    """
    target_words = int(np.random.choice(word_counts))
    if target_words <= 5:
        return "zeer kort", 8
    elif target_words <= 12:
        return "kort", 18
    elif target_words <= 25:
        return "gemiddeld", 38
    else:
        return "lang", min(int(target_words * 1.5), 150)


# ================================
# POST-PROCESSING
# ================================

def trim_to_complete_sentence(text: str) -> str:
    """
    Trim generated text to the last complete sentence by finding the last
    sentence-ending punctuation mark. Handles Dutch text and emoji.
    Falls back to the original text if no sentence-ending punctuation found.
    """
    text = text.strip()
    if not text:
        return text
    match = re.search(
        r'^(.*['
        r'.!?…'
        r'\u00bb\u201d\u2019'        # », ", '
        r'\U0001F600-\U0001F64F'     # emoticons
        r'\U0001F300-\U0001F5FF'     # misc symbols
        r'\U0001F900-\U0001F9FF'     # supplemental symbols
        r'\u2764\u2665'              # ❤ ♥
        r'\U0001F44D\U0001F44E'      # ������ ������
        r'])',
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return text


# ================================
# MODEL LOADING
# ================================

def load_model_and_tokenizer(model_path: str):
    print(f"\n{'='*70}\nLOADING MODEL FROM: {model_path}\n{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if EOC_TOKEN not in tokenizer.get_vocab():
        raise ValueError(
            f"<EOC> token not found in tokenizer at {model_path}. "
            "Make sure you are loading from the correct checkpoint."
        )
    eoc_token_id = tokenizer.convert_tokens_to_ids(EOC_TOKEN)
    print(f"✅ <EOC> token found. ID: {eoc_token_id}")

    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model  = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map="auto",
            dtype=torch.float16,
        )
        if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            print(f"Resizing embeddings: "
                  f"{base_model.get_input_embeddings().weight.shape[0]} → {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model = PeftModel.from_pretrained(base_model, model_path)
        print("✅ PEFT model loaded successfully")
    except Exception as e:
        print(f"PEFT load failed ({e}), attempting full model load...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float16,
        )
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        print("✅ Full model loaded successfully")

    model.eval()
    return model, tokenizer, eoc_token_id


# ================================
# GENERATION
# ================================

def generate_comments(
    model_path: str,
    test_data_path: str,
    length_dist_path: str,
    output_path: str,
    num_samples_per_post: int = 1,
):
    # ── Load length distribution ──
    print(f"\nLoading length distribution from: {length_dist_path}")
    length_distribution = load_length_distribution(length_dist_path)
    word_counts = length_distribution["word_counts"]
    print(f"Length dist — mean: {length_distribution['mean']:.1f}  "
          f"median: {length_distribution['median']:.1f}  "
          f"std: {length_distribution['std']:.1f}")

    # ── Load test data ──
    print(f"\nLoading test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    if "text_ha" in test_df.columns:
        test_df = test_df.rename(columns={"text_ha": "post", "text": "comment"})
    test_df["comment"] = test_df["comment"].str.replace(
        "<PERSON>", PERSON_PLACEHOLDER, regex=False
    )
    print(f"✅ Test set loaded: {len(test_df)} samples")

    # ── Load model ──
    model, tokenizer, eoc_token_id = load_model_and_tokenizer(model_path)

    # ── Generate ──
    print(f"\n{'='*70}\nGENERATING COMMENTS\n{'='*70}")
    print(f"Samples per post : {num_samples_per_post}")
    print(f"Total to generate: {len(test_df) * num_samples_per_post}\n")

    results = []
    total   = len(test_df) * num_samples_per_post
    pbar    = tqdm(total=total, desc="Generating", unit="comment", dynamic_ncols=True)

    for idx, row in test_df.iterrows():
        #post         = str(row["post"]).strip()
        post = row["post"]
        #true_comment = str(row["comment"]).strip()
        true_comment = row["comment"]

        for sample_num in range(num_samples_per_post):
            length_category, max_tokens = sample_length_category_and_tokens(word_counts)

            pbar.set_postfix({
                "idx":  idx,
                "cat":  length_category,
                "toks": max_tokens,
            })

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Schrijf een {length_category} natuurlijke "
                        f"Facebookreactie op dit bericht:\n\n{post}"
                    ),
                },
            ]

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    eos_token_id=[eoc_token_id, tokenizer.eos_token_id],
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only newly generated tokens
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False,
            )

            # Clean up special tokens
            generated_text = (
                generated_text
                .replace(EOC_TOKEN, "")
                .replace("<|eot_id|>", "")
                .replace("<|end_of_text|>", "")
                .strip()
            )

            # Trim to last complete sentence
            generated_text = trim_to_complete_sentence(generated_text)

            # Revert placeholder → <PERSON>
            generated_text = re.sub(
                r"\b" + re.escape(PERSON_PLACEHOLDER) + r"\b",
                "<PERSON>",
                generated_text,
                flags=re.IGNORECASE,
            )

            results.append({
                "post":                 post,
                "true_comment":         true_comment,
                "generated_comment":    generated_text,
                "length_category":      length_category,
                "max_tokens_allowed":   max_tokens,
                "generated_word_count": len(generated_text.split()),
                "true_word_count":      len(true_comment.split()),
                "sample_num":           sample_num,
                "test_idx":             idx,
            })

            pbar.update(1)

    pbar.close()

    # ── Save ──
    results_df = pd.DataFrame(results)

    # Revert placeholder in true_comment column too
    results_df["true_comment"] = results_df["true_comment"].apply(
        lambda x: re.sub(
            r"\b" + re.escape(PERSON_PLACEHOLDER) + r"\b",
            "<PERSON>", str(x), flags=re.IGNORECASE,
        )
    )

    results_df.to_csv(output_path, index=False)

    # ── Summary ──
    incomplete = results_df["generated_comment"].apply(
        lambda x: bool(re.search(r"[a-zA-Z0-9]$", str(x).strip()))
    ).sum()

    print(f"\n{'='*70}\nGENERATION COMPLETE\n{'='*70}")
    print(f"Total generated      : {len(results_df)}")
    print(f"Mean generated words : {results_df['generated_word_count'].mean():.2f}")
    print(f"Mean true words      : {results_df['true_word_count'].mean():.2f}")
    print(f"Comments < 3 words   : {(results_df['generated_word_count'] < 3).sum()}")
    print(f"Still incomplete     : {incomplete} ({incomplete / len(results_df) * 100:.1f}%)")
    print(f"Saved to             : {output_path}")


# ================================
# CLI
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate test comments from a saved Approach 6 checkpoint."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved PEFT checkpoint directory",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to test CSV (expects text_ha/post and text/comment columns)",
    )
    parser.add_argument(
        "--length_dist_path",
        type=str,
        required=True,
        help="Path to length_distribution.pkl saved during training",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./generated_comments.csv",
        help="Where to save the output CSV (default: ./generated_comments.csv)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of generated comments per post (default: 1)",
    )

    args = parser.parse_args()

    generate_comments(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        length_dist_path=args.length_dist_path,
        output_path=args.output_path,
        num_samples_per_post=args.num_samples,
    )


if __name__ == "__main__":
    main()