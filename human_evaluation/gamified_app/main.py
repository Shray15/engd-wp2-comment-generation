"""
WoonPraat Gamified Human Evaluation — FastAPI backend.

Serves a static, pre-curated item bank (item_bank.json, no model/GPU at runtime) and
two play modes:
  - Rate & Reveal : rate fluency/relevance/humanness (1-5). No reveal — the condition
                    is never sent back to the client, so the player never learns
                    whether a given item was real or AI-generated.
  - Spot the Fake : guess real vs synthetic, then reveal correct/incorrect

Rate & Reveal: before playing, the player is shown a shuffled list of candidate posts
(post text only — condition stays hidden) and picks up to RATE_MAX_PICKS of them; the
offered list and its order are randomized fresh every session.

Spot the Fake: no picker — always a fixed SPOT_ROUNDS-item random session (balanced
real/synthetic), since guessing real-vs-fake works better as a quick, repeatable round
than a manually curated one.

Every session logs to results.xlsx, one sheet per mode, appended in place. Anonymous —
only a session_id, no identity.

Run with:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import random
import threading
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openpyxl import Workbook, load_workbook

_HERE = os.path.dirname(os.path.abspath(__file__))
ITEM_BANK_PATH = os.path.join(_HERE, "item_bank.json")
ASSIGNMENTS_PATH = os.path.join(_HERE, "assignments.json")
RESULTS_PATH = os.path.join(_HERE, "results.xlsx")

RATE_MAX_PICKS = 5   # cap on how many posts a player can pick for Rate & Reveal
SPOT_ROUNDS = 10      # fixed session length for Spot the Fake (no picker)
RATE_SHEET = "rate_and_reveal"
SPOT_SHEET = "spot_the_fake"
RATE_HEADER = ["user_id", "session_id", "timestamp", "item_id", "post_id", "condition",
               "fluency", "relevance", "humanness", "item_order"]
SPOT_HEADER = ["session_id", "timestamp", "item_id", "post_id", "condition",
               "user_guess", "correct", "item_order"]

with open(ITEM_BANK_PATH, encoding="utf-8") as f:
    ITEM_BANK = json.load(f)

ITEMS_BY_ID = {item["item_id"]: item for item in ITEM_BANK}

# Optional admin-curated assignments for a facilitated Rate & Reveal session
# (see build_assignments.py): {user_id: [item_id, ...]}. Absent by default —
# free-pick mode keeps working with no assignments.json present.
if os.path.exists(ASSIGNMENTS_PATH):
    with open(ASSIGNMENTS_PATH, encoding="utf-8") as f:
        ASSIGNMENTS = json.load(f)
else:
    ASSIGNMENTS = {}

# In-memory session store: session_id -> {mode, items: [item_id...], answered: {item_id}}
SESSIONS = {}

_excel_lock = threading.Lock()


def _ensure_workbook():
    if os.path.exists(RESULTS_PATH):
        return
    wb = Workbook()
    ws = wb.active
    ws.title = RATE_SHEET
    ws.append(RATE_HEADER)
    ws2 = wb.create_sheet(SPOT_SHEET)
    ws2.append(SPOT_HEADER)
    wb.save(RESULTS_PATH)


def _row_exists(ws, header, dedupe_cols, row_dict):
    idxs = [header.index(c) for c in dedupe_cols]
    for row in ws.iter_rows(min_row=2, values_only=True):
        if all(row[i] == row_dict[col] for i, col in zip(idxs, dedupe_cols)):
            return True
    return False


def _append_row(sheet_name, header, row_dict, dedupe_cols=None):
    """
    Append a row, optionally skipping if one already matches on dedupe_cols
    (only checked when the first dedupe column's value is truthy — e.g. a
    known user_id). Guards against double-logging the same assigned item if
    a participant's browser refreshes mid-session. Returns True if appended.
    """
    with _excel_lock:
        _ensure_workbook()
        wb = load_workbook(RESULTS_PATH)
        if sheet_name not in wb.sheetnames:
            ws = wb.create_sheet(sheet_name)
            ws.append(header)
        else:
            ws = wb[sheet_name]
        if dedupe_cols and row_dict.get(dedupe_cols[0]) and _row_exists(ws, header, dedupe_cols, row_dict):
            return False
        ws.append([row_dict[col] for col in header])
        wb.save(RESULTS_PATH)
        return True


def _random_items(n):
    """n balanced random items (real/synthetic split as evenly as possible)."""
    reals = [i for i in ITEM_BANK if i["condition"] == "real"]
    synths = [i for i in ITEM_BANK if i["condition"] == "synthetic"]
    half = n // 2
    extra = n % 2
    real_count, synth_count = half + extra, half
    chosen = random.sample(reals, real_count) + random.sample(synths, synth_count)
    random.shuffle(chosen)
    return [i["item_id"] for i in chosen]


# ── FastAPI app ─────────────────────────────────────────────────────────

app = FastAPI(title="WoonPraat Gamified Human Eval")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(_HERE, "static", "index.html"),
                         headers={"Cache-Control": "no-store"})


@app.get("/items/candidates")
def get_candidates():
    """
    Shuffled list of every post in the bank, post text only (no comment, no
    condition) — shown before Rate & Reveal starts so the player can pick which
    posts to include in their session. Freshly shuffled on every call. Not used by
    Spot the Fake, which always runs a fixed random SPOT_ROUNDS-item session.
    """
    items = list(ITEM_BANK)
    random.shuffle(items)
    return [{"item_id": i["item_id"], "post_id": i["post_id"], "post": i["post"]}
            for i in items]


@app.get("/assignments/users")
def get_assignment_users():
    """
    User IDs with a pre-curated Rate & Reveal set (see build_assignments.py),
    for a facilitated session's dropdown. Empty list if no assignments.json
    exists — the frontend falls back to free-pick mode in that case.
    """
    return sorted(ASSIGNMENTS.keys())


class StartSessionRequest(BaseModel):
    mode: str  # "rate" or "spot"
    item_ids: list[int] | None = None  # rate-mode only: player's picks from /items/candidates
    user_id: str | None = None  # rate-mode only: pre-assigned participant (see build_assignments.py)


@app.post("/session/start")
def start_session(req: StartSessionRequest):
    if req.mode not in ("rate", "spot"):
        raise HTTPException(status_code=422, detail="mode must be 'rate' or 'spot'")

    if req.mode == "spot":
        # No picker for Spot the Fake — always a fixed-length random session.
        item_ids = _random_items(SPOT_ROUNDS)
    elif req.user_id is not None:
        if req.user_id not in ASSIGNMENTS:
            raise HTTPException(status_code=422, detail=f"Unknown user_id {req.user_id!r}")
        item_ids = list(ASSIGNMENTS[req.user_id])
    elif req.item_ids is None:
        item_ids = _random_items(RATE_MAX_PICKS)
    else:
        seen = set()
        item_ids = []
        for iid in req.item_ids:
            if iid not in ITEMS_BY_ID:
                raise HTTPException(status_code=422, detail=f"Unknown item_id {iid}")
            if iid not in seen:
                seen.add(iid)
                item_ids.append(iid)
        if len(item_ids) > RATE_MAX_PICKS:
            item_ids = item_ids[:RATE_MAX_PICKS]
        if not item_ids:
            raise HTTPException(status_code=422, detail="Select at least one post")

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "mode": req.mode,
        "items": item_ids,
        "answered": set(),
        "user_id": req.user_id,
    }

    return {
        "session_id": session_id,
        "mode": req.mode,
        "total_items": len(item_ids),
    }


@app.get("/session/{session_id}/item/{index}")
def get_item(session_id: str, index: int):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    items = session["items"]
    if index < 0 or index >= len(items):
        raise HTTPException(status_code=404, detail="Item index out of range")

    item = ITEMS_BY_ID[items[index]]
    return {
        "index": index,
        "total_items": len(items),
        "item_id": item["item_id"],
        "post": item["post"],
        "comment": item["comment"],
        # condition intentionally withheld until submit
    }


class RateSubmission(BaseModel):
    session_id: str
    index: int
    fluency: int
    relevance: int
    humanness: int


@app.post("/session/rate")
def submit_rating(sub: RateSubmission):
    session = SESSIONS.get(sub.session_id)
    if session is None or session["mode"] != "rate":
        raise HTTPException(status_code=404, detail="Unknown rate-mode session")
    if not (0 <= sub.index < len(session["items"])):
        raise HTTPException(status_code=404, detail="Item index out of range")
    for field, value in [("fluency", sub.fluency), ("relevance", sub.relevance),
                          ("humanness", sub.humanness)]:
        if value not in (1, 2, 3, 4, 5):
            raise HTTPException(status_code=422, detail=f"{field} must be 1-5")

    item_id = session["items"][sub.index]
    if item_id in session["answered"]:
        raise HTTPException(status_code=409, detail="Item already answered")
    session["answered"].add(item_id)

    item = ITEMS_BY_ID[item_id]
    _append_row(RATE_SHEET, RATE_HEADER, {
        "user_id": session.get("user_id") or "",
        "session_id": sub.session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "item_id": item["item_id"],
        "post_id": item["post_id"],
        "condition": item["condition"],
        "fluency": sub.fluency,
        "relevance": sub.relevance,
        "humanness": sub.humanness,
        "item_order": sub.index + 1,
    }, dedupe_cols=["user_id", "item_id"])

    # Deliberately no condition in the response — Rate & Reveal no longer reveals
    # real/AI after each item, so the client is never told which one it was.
    return {"ok": True}


class SpotSubmission(BaseModel):
    session_id: str
    index: int
    guess: str  # "real" or "synthetic"


@app.post("/session/spot")
def submit_guess(sub: SpotSubmission):
    session = SESSIONS.get(sub.session_id)
    if session is None or session["mode"] != "spot":
        raise HTTPException(status_code=404, detail="Unknown spot-mode session")
    if not (0 <= sub.index < len(session["items"])):
        raise HTTPException(status_code=404, detail="Item index out of range")
    if sub.guess not in ("real", "synthetic"):
        raise HTTPException(status_code=422, detail="guess must be 'real' or 'synthetic'")

    item_id = session["items"][sub.index]
    if item_id in session["answered"]:
        raise HTTPException(status_code=409, detail="Item already answered")
    session["answered"].add(item_id)

    item = ITEMS_BY_ID[item_id]
    correct = (sub.guess == item["condition"])
    _append_row(SPOT_SHEET, SPOT_HEADER, {
        "session_id": sub.session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "item_id": item["item_id"],
        "post_id": item["post_id"],
        "condition": item["condition"],
        "user_guess": sub.guess,
        "correct": correct,
        "item_order": sub.index + 1,
    })

    return {"condition": item["condition"], "correct": correct}


app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading as _threading

    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    print("Starting WoonPraat Gamified Human Eval server...")
    print("Opening browser at http://localhost:8000")
    _threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
