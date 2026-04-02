"""
Standalone Test Set Generation Script
Loads a fine-tuned Gemma model and generates comments for the test set.

Fixes applied vs original:
- Dual + triple EOS: [eoc_token_id, tokenizer.eos_token_id, end_of_turn_id]
- Full Gemma template leakage cleanup (<start_of_turn>, \nuser\n, \nmodel\n)
- Dutch-only instruction added to system prompt
- tqdm progress bar
- Completeness summary at the end

Usage:
    python generate_test_comments.py \
        --model_path ./gemma-fb-comments-finetuned_v01 \
        --test_data_path /vast.mnt/home/20245179/Desktop/final_test_set.csv \
        --output_path ./generated_comments.csv
"""

import argparse
import re
import warnings
import pickle
from typing import Dict, Optional
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from tqdm import tqdm

warnings.filterwarnings("ignore")

EOC_TOKEN = "<EOC>"
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
   - DOORSTUREN: je tag iemand anders in het bericht (<PERSON>)
   - WAARDERING: je reageert positief of bedankt de woningcorporatie

3. RELATEERDHEID (aansluiting bij bericht):
   - Reageer direct op wat er in het bericht staat
   - Gebruik details uit het bericht in je reactie
   - Toon dat je het bericht gelezen en begrepen hebt

Schrijf natuurlijk Nederlands met:
- Informele taal en spreektaal ("ff", "gwn", "tis")
- Soms typefouten zoals bij echt typen
- Emoji's waar natuurlijk

Schrijf UITSLUITEND in het Nederlands. Gebruik geen andere talen.
Schrijf altijd een volledige, afgeronde reactie. Maak nooit een zin half af."""



# ================================
# POST-PROCESSING
# ================================

def clean_gemma_artefacts(text: str) -> str:
    """Remove Gemma chat template leakage tokens."""
    text = re.sub(r'<start_of_turn>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'<end_of_turn>.*',   '', text, flags=re.DOTALL)
    text = re.sub(r'\nuser\n.*',         '', text, flags=re.DOTALL)
    text = re.sub(r'\nmodel\n.*',        '', text, flags=re.DOTALL)
    return text.strip()


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

def strip_foreign_script(text: str) -> str:
    """Remove non-Latin/Dutch characters from generated text."""
    # Keep: basic Latin, Latin extended (Dutch accents), punctuation, digits, whitespace
    text = re.sub(r'[^\x00-\u024F\s\d\W]', '', text)
    # Clean up any double spaces left behind
    text = re.sub(r'  +', ' ', text).strip()
    return text
# ================================
# LENGTH SAMPLING
# ================================

def sample_length_category_and_tokens(word_counts: np.ndarray):
    """Category-based sampling with increased token budgets to reduce truncation."""
    target_words = np.random.choice(word_counts)
    if target_words <= 5:
        return "zeer kort", 13
    elif target_words <= 12:
        return "kort", 30
    elif target_words <= 25:
        return "gemiddeld", 64
    else:
        return "lang", min(int(target_words * 2.5), 250)


# ================================
# MODEL LOADING
# ================================

def load_model_and_tokenizer(model_path: str):
    print(f"\n{'='*70}\nLOADING MODEL\n{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if EOC_TOKEN not in tokenizer.get_vocab():
        raise ValueError("<EOC> token not found in tokenizer! Are you sure this is a fine-tuned model?")

    eoc_token_id    = tokenizer.convert_tokens_to_ids(EOC_TOKEN)
    end_of_turn_id  = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    print(f"<EOC> token ID:        {eoc_token_id}")
    print(f"<end_of_turn> token ID: {end_of_turn_id}")

    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        print(f"Base model: {peft_config.base_model_name_or_path}")

        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            print(f"Resizing embeddings: "
                  f"{base_model.get_input_embeddings().weight.shape[0]} → {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(base_model, model_path)
        print("✅ PEFT/LoRA model loaded successfully")

    except Exception as e:
        print(f"PEFT load failed ({e}), trying as full model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model.resize_token_embeddings(len(tokenizer))
        print("✅ Full model loaded successfully")

    model.eval()
    return model, tokenizer, eoc_token_id, end_of_turn_id


# ================================
# GENERATION
# ================================

def generate_comments(
    model_path: str,
    test_data_path: str,
    output_path: str,
    length_dist_path: Optional[str] = None,
    num_samples_per_post: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
):
    # ── Load test data ──
    print(f"\n{'='*70}\nLOADING TEST DATA\n{'='*70}")
    test_df = pd.read_csv(test_data_path)
    if "text_ha" in test_df.columns:
        test_df = test_df.rename(columns={"text_ha": "post", "text": "comment"})
    test_df["comment"] = test_df["comment"].str.replace("<PERSON>", PERSON_PLACEHOLDER, regex=False)
    print(f"Test set: {len(test_df)} samples")

    # ── Load length distribution ──
    if length_dist_path:
        print(f"Loading length distribution from: {length_dist_path}")
        with open(length_dist_path, "rb") as f:
            length_dist = pickle.load(f)
        word_counts = length_dist["word_counts"]
        print(f"Mean length: {length_dist['mean']:.1f} words")
    else:
        print("No length distribution provided — using uniform sampling (1-50 words)")
        word_counts = np.arange(1, 50)

    # ── Load model ──
    model, tokenizer, eoc_token_id, end_of_turn_id = load_model_and_tokenizer(model_path)

    # ── Generate ──
    print(f"\n{'='*70}\nGENERATING COMMENTS\n{'='*70}")
    print(f"Posts: {len(test_df)}  |  Samples per post: {num_samples_per_post}")
    print(f"Temperature: {temperature}  top_p: {top_p}  top_k: {top_k}\n")

    results = []
    total = len(test_df) * num_samples_per_post
    pbar  = tqdm(total=total, desc="Generating", unit="comment", dynamic_ncols=True)

    for idx, row in test_df.iterrows():
        post         = row["post"]
        true_comment = row["comment"]

        for sample_num in range(num_samples_per_post):
            length_category, max_tokens = sample_length_category_and_tokens(word_counts)

            pbar.set_postfix({
                "idx": idx,
                "cat": length_category,
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
                    eos_token_id=[eoc_token_id, tokenizer.eos_token_id, end_of_turn_id],
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False,
            )

            # Clean EOC and Gemma template artefacts
            generated_text = generated_text.replace(EOC_TOKEN, "").strip()
            generated_text = clean_gemma_artefacts(generated_text)
            generated_text = clean_gemma_artefacts(generated_text)
            generated_text = strip_foreign_script(generated_text)   # ← add here


            # Trim to last complete sentence
            generated_text = trim_to_complete_sentence(generated_text)

            # Revert placeholder → <PERSON>
            generated_text = re.sub(
                r"\b" + re.escape(PERSON_PLACEHOLDER) + r"\b",
                "<PERSON>", generated_text, flags=re.IGNORECASE,
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

    # ── Save results ──
    results_df = pd.DataFrame(results)

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
    print(f"Median generated     : {results_df['generated_word_count'].median():.0f}")
    print(f"Median true          : {results_df['true_word_count'].median():.0f}")
    print(f"Comments < 3 words   : {(results_df['generated_word_count'] < 3).sum()}")
    print(f"Still incomplete     : {incomplete} ({incomplete / len(results_df) * 100:.1f}%)")
    print(f"\nLength category distribution:")
    print(results_df["length_category"].value_counts(normalize=True).sort_index())
    print(f"\nSaved to: {output_path}")


# ================================
# CLI
# ================================

def main():
    parser = argparse.ArgumentParser("Generate test set comments from fine-tuned Gemma model")

    parser.add_argument("--model_path",      type=str, required=True,
                        help="Path to fine-tuned model directory")
    parser.add_argument("--test_data_path",  type=str,
                        default="/vast.mnt/home/20245179/Desktop/final_test_set.csv",
                        help="Path to test set CSV")
    parser.add_argument("--output_path",     type=str,
                        default="./multiloss_generated_testsetv2.csv",
                        help="Where to save generated comments")
    parser.add_argument("--length_dist_path", type=str, default=None,
                        help="Path to length distribution .pkl file (optional)")
    parser.add_argument("--num_samples",     type=int,   default=1)
    parser.add_argument("--temperature",     type=float, default=0.7)
    parser.add_argument("--top_p",           type=float, default=0.9)
    parser.add_argument("--top_k",           type=int,   default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    args = parser.parse_args()

    generate_comments(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        output_path=args.output_path,
        length_dist_path=args.length_dist_path,
        num_samples_per_post=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()