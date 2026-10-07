"""
build_assignments.py — Admin tool: pre-assign overlapping item sets to named
participants for a facilitated Rate & Reveal session (e.g. a company lunch demo).

True circular sliding-window design, same spirit as human_evaluation/code/subsets.py's
original 7-annotator study, but the ring is sized to the actual headcount (2-20) instead
of a fixed participant count, so it always closes cleanly:

  - Each participant rates WINDOW items (4 pairs = 8 items, ~8 min at ~1 min/item).
  - The ring is n_users * STRIDE items long. Consecutive participants' windows overlap
    by WINDOW - STRIDE items, wrapping circularly (participant n_users-1 overlaps back
    into participant 0 — a real circle, like the original study's annotator chain).
  - With just 2 participants, the ring is exactly WINDOW items long, so both people
    land on the *same* items — full overlap, which is what you want when there are only
    two of you to compare.
  - Every item in the ring gets exactly WINDOW / STRIDE = 2 raters, for any headcount
    2-20 — no orphaned single-rater items, no manual "circular patch" needed (unlike
    subsets.py, whose 7*10 didn't tile 80 evenly).

  Why pairs, not raw items: item_bank.json interleaves real/synthetic items in pairs
  (item i real, item i+1 synthetic — see build_item_bank.py). Sliding a window across
  raw items can wrap mid-pair, silently breaking the intended real/synthetic balance for
  whichever participant's window straddles the seam (verified: with odd headcounts this
  actually happens). Operating in whole pairs makes that structurally impossible — every
  pair contributes exactly one real + one synthetic item, no matter how it wraps.

Edit PARTICIPANTS below to the real names/IDs before each event (2-20 names), then run:

    python build_assignments.py

Output: assignments.json — {user_id: [item_id, ...]} in each person's shuffled
play order. Also prints a sanity-check summary (items/person, real/synth split,
raters/item) so you can catch a problem before the event starts.
"""

import json
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
ITEM_BANK_PATH = os.path.join(_HERE, "item_bank.json")
OUTPUT_JSON = os.path.join(_HERE, "assignments.json")

# ── Edit before each event: 2-20 names ─────────────────────────────────
PARTICIPANTS = [
    "alex", "bo",
]
WINDOW_PAIRS = 4   # pairs per participant -> 8 items, ~8 min at ~1 min/item
STRIDE_PAIRS = 2   # overlap step in pairs (WINDOW/STRIDE=2 -> 2 raters/item, always)
MIN_PARTICIPANTS = 2
MAX_PARTICIPANTS = 20

# Which pair-slice of the bank the ring starts from — rotate this between repeat events
# (with the same people) so they don't see identical items every time. None picks a
# fresh random offset each run and prints it so you can pin it via this constant if you
# ever need to reproduce an exact assignment.
START_OFFSET_PAIRS = None


def build_assignments(participants, bank, window_pairs=WINDOW_PAIRS, stride_pairs=STRIDE_PAIRS,
                       start_offset_pairs=None):
    n_users = len(participants)
    bank_pairs = len(bank) // 2

    assert MIN_PARTICIPANTS <= n_users <= MAX_PARTICIPANTS, (
        f"Expected {MIN_PARTICIPANTS}-{MAX_PARTICIPANTS} participants, got {n_users}."
    )
    assert len(bank) % 2 == 0, "item_bank.json must have an even number of items (real/synthetic pairs)."

    ring_pairs = n_users * stride_pairs  # circle length, scales with headcount
    assert window_pairs <= ring_pairs, (
        f"window_pairs ({window_pairs}) can't exceed the ring ({ring_pairs} pairs for "
        f"{n_users} participants at stride {stride_pairs}) — would repeat an item within "
        f"one person's own set. Need at least {-(-window_pairs // stride_pairs)} participants "
        f"at this stride."
    )
    assert ring_pairs <= bank_pairs, (
        f"{n_users} participants need a ring of {ring_pairs} pairs ({ring_pairs*2} items), "
        f"but the bank only has {bank_pairs} pairs ({len(bank)} items). Shrink STRIDE_PAIRS, "
        f"grow the bank, or invite fewer people."
    )

    if start_offset_pairs is None:
        start_offset_pairs = random.randint(0, bank_pairs - ring_pairs)
    assert 0 <= start_offset_pairs <= bank_pairs - ring_pairs, "start_offset_pairs out of range for this ring."

    assignments = {}
    coverage = {item["item_id"]: 0 for item in bank}
    summary = []

    for idx, user_id in enumerate(participants):
        start = idx * stride_pairs
        pair_positions = [(start + k) % ring_pairs for k in range(window_pairs)]
        abs_pairs = [start_offset_pairs + p for p in pair_positions]

        block = []
        for p in abs_pairs:
            block.append(bank[2 * p])       # real half of the pair
            block.append(bank[2 * p + 1])   # synthetic half of the pair

        real_count = sum(1 for i in block if i["condition"] == "real")
        synth_count = len(block) - real_count

        rng = random.Random(idx)  # deterministic per-user shuffle
        rng.shuffle(block)

        assignments[user_id] = [i["item_id"] for i in block]
        for i in block:
            coverage[i["item_id"]] += 1

        summary.append((user_id, [p + 1 for p in sorted(abs_pairs)], real_count, synth_count))

    return assignments, coverage, start_offset_pairs, ring_pairs, summary


def main():
    with open(ITEM_BANK_PATH, encoding="utf-8") as f:
        bank = json.load(f)

    assignments, coverage, offset_pairs, ring_pairs, summary = build_assignments(
        PARTICIPANTS, bank, WINDOW_PAIRS, STRIDE_PAIRS, START_OFFSET_PAIRS
    )

    for user_id, pairs, real_count, synth_count in summary:
        print(f"{user_id:>10}: pairs {pairs} | real={real_count}, synthetic={synth_count}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(assignments, f, ensure_ascii=False, indent=2)

    used = {iid: c for iid, c in coverage.items() if c > 0}
    coverage_counts = {}
    for c in used.values():
        coverage_counts[c] = coverage_counts.get(c, 0) + 1

    n_users = len(PARTICIPANTS)
    window_items = WINDOW_PAIRS * 2
    print(f"\n=== SANITY CHECKS ===")
    print(f"Participants: {n_users}, items/person: {window_items}, bank size: {len(bank)}")
    print(f"Ring: {ring_pairs} pairs ({ring_pairs*2} items), starting at pair {offset_pairs+1}")
    print(f"Total ratings if everyone finishes: {n_users * window_items}")
    print(f"Raters per used item (expect all = {WINDOW_PAIRS // STRIDE_PAIRS}): {coverage_counts}")
    print(f"Bank items untouched this run: {len(bank) - len(used)}")
    if n_users == 2:
        same = set(assignments[PARTICIPANTS[0]]) == set(assignments[PARTICIPANTS[1]])
        print(f"Full overlap for the 2-person case: {same}")

    print(f"\nSaved assignments to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
