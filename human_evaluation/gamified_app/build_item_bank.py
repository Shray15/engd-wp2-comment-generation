"""
build_item_bank.py — Curate the static item bank for the gamified human-eval app.

Reuses the anonymization/filtering logic from human_evaluation/code/subsets.py, then
samples a fresh, balanced pool of posts that were NOT already used in the original
80-item study (human_evaluation/subsets/master_with_conditions.csv), so repeat
annotators don't just see the same items again.

Each post contributes exactly one comment — either its real tenant comment or its
LLM-generated one, never both (blind single-item design, no post repeats).

Run once, offline:
    python build_item_bank.py

Output: item_bank.json — [{item_id, post_id, post, comment, condition}, ...]
"""

import re
import json
import pandas as pd

SOURCE_CSV = r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\Fine Tune gemma\objective\objective_generated_commentsv2.csv"
OUTPUT_JSON = r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\human_evaluation\gamified_app\item_bank.json"

BANK_SIZE = 80  # even: split 40 real / 40 synthetic
RANDOM_STATE = 7  # different seed from the original study's 42, deliberately

# ── Anonymization filters (copied from human_evaluation/code/subsets.py) ──────

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
    return re.sub(
        r'Stichting Woonbedrijf|ACTI-UM|Actium|actium|#Actium|#actium'
        r'|Woonbedrijf|woonbedrijf|#Woonbedrijf|#woonbedrijf'
        r'|#vestide|#Vestide|Vestide'
        r'|#SWS\.Hhvl|#SWS\.HHVL'
        r'|WSLeusden|Woonwaarts|Bindkracht10|Nijestee'
        r'|De Woonschakel|Woonschakel|365Zon|Amendex'
        r'|Hemink Groep B\.V\.|Hemink Groep'
        r'|Cognitum|BAM Wonen|Donker Groep|WIJeindhoven'
        r'|KleurrijkWonen|De Kernen|Van Wijnen'
        r'|Groenen Bouw|Van Santvoort|Groenrijk'
        r'|Heijmans|@Heijmans|Wooninc|ZOwonen'
        r'|Woningstichting Den Helder|Lunet'
        r'|1Twente|Tubantia|Caspar de Haan|@caspardehaan\.nl'
        r'|@levgroep|@cordaadwelzijn'
        r'|@luzac\.eindhoven|@koffiehuisjeeindhoven'
        r'|Be More You|Barabaz',
        '<ORG>',
        str(text)
    )


def replace_name(text):
    if pd.isna(text):
        return text
    return re.sub(
        r'Roy Beijnsberger|Henny Zink|Marion'
        r'|Annelotte|Marjan Maat|Wilma Wouters|Marco Karman'
        r'|Anja|Marijke|Michael',
        '<PERSON>',
        str(text)
    )


def replace_social_handles(text):
    if pd.isna(text):
        return text
    return re.sub(r'@\w+', '<ORG>', str(text))


def replace_email(text):
    if pd.isna(text):
        return text
    text = re.sub(r'\w+@<ORG>\.\w+', '<EMAIL>', str(text))
    return re.sub(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
        '<EMAIL>',
        text
    )


def replace_url(text):
    if pd.isna(text):
        return text
    return re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE).sub('<URL>', str(text))


def replace_phone(text):
    if pd.isna(text):
        return text
    return re.sub(
        r'(\+31|0031|0)[\s\-]?'
        r'(\d[\s\-]?){8,9}',
        '<PHONE>',
        str(text)
    )


def is_only_url_or_org(text, min_words=5):
    if pd.isna(text):
        return True
    cleaned = re.sub(r'<URL>|<ORG>|<PERSON>|<PHONE>', '', str(text)).strip()
    cleaned = re.sub(r'[^\w\s]', '', cleaned).strip()
    words = [w for w in cleaned.split() if w]
    return len(words) < min_words


def anonymize(text):
    text = replace_url(text)
    text = replace_email(text)
    text = replace_org(text)
    text = replace_name(text)
    text = replace_phone(text)
    text = replace_social_handles(text)
    return text


def main():
    data = pd.read_csv(SOURCE_CSV)

    data = data[~data['true_comment'].apply(is_only_person_tags)]
    data = data[~data['generated_comment'].apply(is_only_person_tags)]
    data = data[data['true_comment'].apply(is_meaningful)]
    data = data[data['generated_comment'].apply(is_meaningful)]

    for col in ['post', 'true_comment', 'generated_comment']:
        data[col] = data[col].apply(anonymize)

    before = len(data)
    data = data[~data['post'].apply(is_only_url_or_org)]
    print(f"After removing tag-only posts: {len(data)} rows (removed {before - len(data)})")

    data = data.drop_duplicates(subset=['post'], keep='first').reset_index(drop=True)
    print(f"Available posts after filtering: {len(data)}")

    # Note: the filtered pool (~83 posts) overlaps heavily with the 80 posts already
    # used in the original human_evaluation study (human_evaluation/subsets/) — there
    # are only ~8 completely fresh posts under these filters. Overlap is accepted here
    # (different context, different/mostly new players, months later) rather than
    # shrinking the bank to 8 posts.

    assert len(data) >= BANK_SIZE, f"Need at least {BANK_SIZE} posts, only have {len(data)}"

    sample = data.sample(n=BANK_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    sample['post_id'] = range(1, BANK_SIZE + 1)

    half = BANK_SIZE // 2
    real_items = sample.iloc[:half][['post_id', 'post', 'true_comment']].copy()
    real_items.columns = ['post_id', 'post', 'comment']
    real_items['condition'] = 'real'
    real_items = real_items.reset_index(drop=True)

    syn_items = sample.iloc[half:][['post_id', 'post', 'generated_comment']].copy()
    syn_items.columns = ['post_id', 'post', 'comment']
    syn_items['condition'] = 'synthetic'
    syn_items = syn_items.reset_index(drop=True)

    # Interleave real/synthetic (real, synth, real, synth, ...) so any contiguous
    # window of item_ids is balanced — needed for build_assignments.py's
    # sliding-window overlap design to hand out balanced sets per person.
    interleaved = []
    for i in range(half):
        interleaved.append(real_items.iloc[i])
        interleaved.append(syn_items.iloc[i])
    bank = pd.DataFrame(interleaved).reset_index(drop=True)
    bank['item_id'] = range(1, len(bank) + 1)
    bank = bank[['item_id', 'post_id', 'post', 'comment', 'condition']]

    print(f"\nBank size: {len(bank)} ({(bank['condition'] == 'real').sum()} real, "
          f"{(bank['condition'] == 'synthetic').sum()} synthetic)")

    records = bank.to_dict(orient='records')
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved item bank to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
