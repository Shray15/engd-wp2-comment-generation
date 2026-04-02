import re
import html
import unicodedata



def preprocess(text: str) -> str:
    """
    Lightweight, model-friendly normalization for cased transformers.
    Keeps case, punctuation, emojis, and diacritics.
    """
    _URL_RE   = re.compile(r'https?://\S+|www\.\S+', re.I)
    _EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
    _USER_RE  = re.compile(r'(?<=\s)@\w+')
    _NUM_RE   = re.compile(r'\b\d+(?:[.,]\d+)?\b')  # integers & decimals
    if text is None:
        return ""
    t = str(text)

    # 1) Unescape HTML entities and strip tags
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)

    # 2) Unicode normalize (compatibility + composition)
    t = unicodedata.normalize("NFKC", t)

    # 3) Standardize volatile tokens (keep signal, avoid overfitting)
    t = _URL_RE.sub("<URL>", t)
    t = _EMAIL_RE.sub("<EMAIL>", t)
    t = _USER_RE.sub("<USER>", t)
    t = _NUM_RE.sub("<NUMBER>", t)

    # 4) Whitespace tidy & mild punctuation cleanup (keep ?!.,;:)
    t = re.sub(r"[^\S\r\n]+", " ", t)           # collapse spaces
    t = re.sub(r"\s+([?.!,;:])", r"\1", t)      # no space before punctuation
    t = re.sub(r"([?.!,;:]){3,}", r"\1\1\1", t) # clamp runs like "!!!!!" -> "!!!"

    # 5) Final trim
    return t.strip()
