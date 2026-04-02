import re
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import nltk

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")


import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass(frozen=True)
class WordLenBin:
    lo: int      # inclusive
    hi: int      # inclusive
    p: float     # probability mass

def word_lengths(texts: List[str]) -> np.ndarray:
    return np.array([len(str(t).split()) for t in texts], dtype=int)

def make_fixed_width_bins(
    texts: List[str],
    bin_width: int = 5,
    min_len: int = 1,
    max_len: Optional[int] = None,
    add_overflow_bin: bool = True
) -> List[WordLenBin]:
    """
    Equal-width word-count bins: [min_len..min_len+W-1], [min_len+W..min_len+2W-1], ...
    Probabilities are empirical (fraction of real comments in each bin).

    If add_overflow_bin=True and max_len is set, the last bin is [max_len+1 .. observed_max].
    If max_len is None, bins go up to observed_max and overflow bin is not needed.
    """
    x = word_lengths(texts)
    x = x[x >= 0]
    if len(x) == 0:
        raise ValueError("No texts/lengths provided.")
    if bin_width < 1:
        raise ValueError("bin_width must be >= 1")

    observed_max = int(x.max())
    if max_len is None:
        max_len = observed_max

    # Build equal-width edges up to max_len
    # Example: min=1, width=5 -> bins: 1-5, 6-10, 11-15, ...
    edges = list(range(min_len, max_len + bin_width, bin_width))
    # Ensure last edge covers max_len
    if edges[-1] <= max_len:
        edges.append(edges[-1] + bin_width)

    bins: List[WordLenBin] = []
    n = len(x)

    # Fixed-width bins up to max_len
    for start in edges[:-1]:
        lo = start
        hi = start + bin_width - 1
        # Clip the final fixed bin to max_len (optional, keeps geometry consistent except last cap)
        if hi > max_len:
            hi = max_len
        if lo > hi:
            continue

        count = int(((x >= lo) & (x <= hi)).sum())
        p = count / n
        # Keep empty bins if you want a complete grid; otherwise skip.
        bins.append(WordLenBin(lo=lo, hi=hi, p=p))

    # Optional overflow tail bin to catch lengths > max_len
    if add_overflow_bin and observed_max > max_len:
        lo = max_len + 1
        hi = observed_max
        count = int((x >= lo).sum())
        p = count / n
        bins.append(WordLenBin(lo=lo, hi=hi, p=p))

    # Renormalize (in case you clipped / overflow logic)
    total = sum(b.p for b in bins)
    if total == 0:
        raise ValueError("All bins have zero mass; check min_len/max_len/bin_width.")
    bins = [WordLenBin(b.lo, b.hi, b.p / total) for b in bins]

    return bins

def sample_word_bin(bins: List[WordLenBin], seed: Optional[int] = None) -> WordLenBin:
    rng = np.random.default_rng(seed)
    ps = np.array([b.p for b in bins], dtype=float)
    ps = ps / ps.sum()
    idx = rng.choice(len(bins), p=ps)
    return bins[int(idx)]

def bins_as_dict(bins: List[WordLenBin]) -> Dict[str, float]:
    return {f"{b.lo}-{b.hi}": b.p for b in bins}



# ----------------------------------
# 2) Improved Prompt builder + extraction
# ----------------------------------
def extract_comment(text: str) -> Optional[str]:
    """
    Extract comment from model output with multiple fallback strategies.
    """
    # Strategy 1: Extract text between quotes (most specific)
    quote_pattern = re.compile(r'"([^"]+)"', re.DOTALL)
    matches = quote_pattern.findall(text)
    if matches:
        return matches[-1].strip()
    
    # Strategy 2: Look for REACTIE: prefix
    reactie_pattern = re.compile(r'REACTIE:\s*(.+?)(?:\n|$)', re.DOTALL | re.IGNORECASE)
    match = reactie_pattern.search(text)
    if match:
        return match.group(1).strip().strip('"')
    
    # Strategy 3: Take first 1-3 sentences if they look like a comment
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        first_line = lines[0].strip('"').strip()
        # Check if it's not instruction-like text
        if not any(word in first_line.lower() for word in ['hier is', 'voorbeeldbericht', 'reactie:', 'belangrijk']):
            return first_line
    
    return None


def build_chat_prompt(
    post_text: str,
    chosen_bin: WordLenBin,
    example_post: str,
    example_comments: List[str],
) -> List[Dict[str, str]]:
    """
    Build a proper chat-formatted prompt for Llama models.
    Returns a list of message dicts for apply_chat_template.
    """
    system_message = """Je bent een Nederlandstalige Facebookgebruiker die reageert op berichten van een woningcorporatie. Je schrijft korte, natuurlijke reacties zoals echte bewoners dat zouden doen."""
    
    user_message = f"""Schrijf ÉÉN natuurlijke Facebookreactie in het Nederlands op het volgende bericht:

BERICHT:
{post_text}

VOORBEELDEN van het stijl (niet kopiëren, alleen inspiratie):
- "{example_comments[0]}"
- "{example_comments[1]}"
- "{example_comments[2]}"

REGELS:
- Gebruik {chosen_bin.lo}-{chosen_bin.hi} woorden
- Schrijf 1-3 zinnen
- Informele, directe toon
- ALLEEN de reactie, geen uitleg
- Zet de reactie tussen aanhalingstekens

REACTIE:"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]


# ----------------------------
# 3) Row-level Dataset
# ----------------------------
class PromptDataset(Dataset):
    def __init__(self, messages_list: List[List[Dict]], post_texts: List[str], bins: List[WordLenBin]):
        self.messages_list = messages_list
        self.post_texts = post_texts
        self.bins = bins

    def __len__(self):
        return len(self.messages_list)

    def __getitem__(self, idx):
        return {
            "messages": self.messages_list[idx],
            "post": self.post_texts[idx],
            "bin": self.bins[idx],
            "idx": idx,
        }


# ----------------------------
# 4) Batched generation function
# ----------------------------
def calculate_max_tokens_for_words(target_words: int, safety_multiplier: float = 2.0) -> int:
    """
    Estimate max_new_tokens needed for a target word count.
    Increased safety multiplier for Dutch text.
    """
    estimated_tokens = int(target_words * safety_multiplier)
    return max(30, min(estimated_tokens, 150))


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    messages_batch: List[List[Dict]],
    chosen_bins: List[WordLenBin],
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> List[str]:
    """
    Generate completions using proper chat template.
    """
    # Apply chat template to each message list
    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in messages_batch
    ]
    
    max_target_words = max(bin.hi for bin in chosen_bins)
    max_new_tokens = calculate_max_tokens_for_words(max_target_words)

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,  # Reduced since we're generating short comments
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    gen_out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the new tokens (remove prompt)
    prompt_len = enc["input_ids"].shape[1]
    decoded_completions = tokenizer.batch_decode(
        gen_out[:, prompt_len:], 
        skip_special_tokens=True
    )

    return decoded_completions


# ----------------------------
# 5) Main
# ----------------------------
def main():
    # ---- Load data
    df = pd.read_csv(r"/vast.mnt/home/20245179/Desktop/final_test_set.csv")
    nltk.download("punkt", quiet=True)

    df["count of sentences"] = df["text"].apply(lambda x: len(nltk.sent_tokenize(str(x))))

    dataframe = df.groupby(["text_ha"]).agg({
        "text_ha": "first",
        "text": lambda x: len(x),
        "count of sentences": lambda x: list(x)
    }).reset_index(drop=True)

    # ---- Build bins from real comments
    real_comments = df["text"].tolist()
    bins = make_fixed_width_bins(
    texts=real_comments,
    bin_width=5,      # choose 3, 5, 10 depending on granularity you want
    min_len=1,
    max_len=None,      # optional cap; tail goes to overflow bin if add_overflow_bin=True
    add_overflow_bin=True
    )


    # ---- Load model/tokenizer
    model_id = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Decoder-only models should use left padding for batched generation
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # bfloat16 instead of float16
    )
    model.eval()

    # ---- Examples
    example_post = (
        "'Schoonmaken. Opruimen. Foxhol winterklaar maken. Doe je mee?' Dat was het motto om de huurders in Foxhol aan te moedigen om in actie te komen. "
        "In Midden-Groningen houden we jaarlijks een actiedag waarin de leefbaarheid, in een wijk of dorp waar we veel woningen bezitten, centraal staat. "
        "Dit keer was Foxhol aan de beurt. Drie containers met een inhoud van veertig kuub lagen aan het eind van de middag vol met grof vuil! "
        "Tijdens de opruiming werden de broodjes hamburger liefdevol in ontvangst genomen door de bewoners, gepaard met koffie en thee. Al met al een succesvolle dag!"
    )
    example_comments = [
        "Mag men ook wel eens in Musselkanaal doen....top actie…",
        "Zorg eerst maar Dat .. West indischekade .. opgeruimd en netjes word .. dat zijn in totaal 3 straten 3 flatten Suc6 ..  ik steek me poten niet uit als jullie niet eens op of om kijken",
        "Best wel jammer, oud jaar komt er weer aan. Je had er lekker warm bij kunnen zitten met al dat hout en matrassen"
    ]

    # ---- Build generation data
    rng = np.random.default_rng(42)

    messages_list = []
    post_texts = []
    chosen_bins_list = []
    meta_post_index = []
    
    
    for i in tqdm(range(len(dataframe)), desc="Building prompts"):
        post = dataframe.loc[i, "text_ha"]
        n_comments = int(dataframe.loc[i, "text"])

        for _ in range(n_comments):
            chosen_bin = sample_word_bin(bins, rng)
            messages = build_chat_prompt(post, chosen_bin, example_post, example_comments)
            
            messages_list.append(messages)
            post_texts.append(post)
            chosen_bins_list.append(chosen_bin)
            meta_post_index.append(i)

    # ---- Batched generation with retry
    batch_size = 4  # Reduced for 3B model
    max_attempts = 3
    
    results = [None] * len(messages_list)
    pending = list(range(len(messages_list)))

    # Adjusted sampling parameters for better instruction following
    temperature = 0.7  # Lower temperature for more focused outputs
    top_p = 0.9
    top_k = 50
    repetition_penalty = 1.1

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break

        print(f"\n{'='*60}")
        print(f"Attempt {attempt}/{max_attempts} - Pending: {len(pending)} items")
        print(f"{'='*60}\n")

        # Create batch data for pending items
        pending_messages = [messages_list[idx] for idx in pending]
        pending_bins = [chosen_bins_list[idx] for idx in pending]

        # Process in batches
        new_pending = []
        
        for batch_start in tqdm(range(0, len(pending), batch_size), desc=f"Generating (attempt {attempt})"):
            batch_end = min(batch_start + batch_size, len(pending))
            
            batch_messages = pending_messages[batch_start:batch_end]
            batch_bins = pending_bins[batch_start:batch_end]
            batch_indices = pending[batch_start:batch_end]
            
            # Increase temperature slightly on retries
            current_temp = temperature + (attempt - 1) * 0.1
            
            completions = generate_batch(
                model=model,
                tokenizer=tokenizer,
                messages_batch=batch_messages,
                chosen_bins=batch_bins,
                temperature=current_temp,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            
            for completion, orig_idx in zip(completions, batch_indices):
                print(f"\n--- Index {orig_idx} ---")
                print(f"RAW OUTPUT:\n{completion}\n")
                
                comment = extract_comment(completion)
                print(f"EXTRACTED: {comment}\n")
                
                if comment and len(comment.split()) >= 3:  # At least 3 words
                    results[orig_idx] = comment
                else:
                    new_pending.append(orig_idx)
                    print(f"⚠️  Failed extraction or too short\n")

        pending = new_pending

    # Mark failures
    for idx in pending:
        results[idx] = "No match found"

    # ---- Build output
    out = pd.DataFrame({
        "Post": post_texts,
        "Comments": results,
        "group_post_index": meta_post_index,
        "target_word_range": [f"{b.lo}-{b.hi}" for b in chosen_bins_list]
    })

    output_file = "gemma_few_shot_testset.csv"
    out.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Saved: {output_file}")
    print(f"Total rows: {len(out)}")
    print(f"Successful: {(out['Comments'] != 'No match found').sum()}")
    print(f"Failed: {(out['Comments'] == 'No match found').sum()}")
    print(f"Success rate: {(out['Comments'] != 'No match found').sum() / len(out) * 100:.1f}%")


if __name__ == "__main__":
    main()