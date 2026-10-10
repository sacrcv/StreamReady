"""
Answer matching shared by the evaluation tasks.

A correct answer is encoded four different ways across the evaluated tasks, so
scoring needs four corresponding paths:

* **Multiple choice** -- compare the predicted option letter. Models rarely emit a
  bare letter, so the prediction is stripped of the usual preambles ("The best
  answer is", "(B)") before the letter is read. This mirrors the parser used in
  the reference OVO-Bench evaluation.
* **Counting** -- OVO-Bench REC and ProReady-QA REC want an integer; "3",
  "3 times" and "three" must all match a gold of 3.
* **Yes/No** -- OVO-Bench SSR and CRR are binary, and the official scorer accepts
  a bare "Y"/"N" as well as the full word.
* **Open-ended** -- ProReady-QA's free-form answers, and any open-ended source
  added later. Where a source's official protocol is GPT judging, this module can
  only do normalised text matching, so the number it produces is a lower bound
  and is labelled as such by the task.
"""

import re
import string
from typing import List, Optional

CHOICE_LETTERS = "ABCDEFGH"

_ARTICLES = {"a", "an", "the"}
_PUNC = str.maketrans("", "", string.punctuation)

_ANSWER_PREFIXES = (
    "the best answer is", "the correct answer is", "the best option is",
    "the correct option is", "the answer is", "best answer:", "best option:",
    "answer:", "option:", "the answer", "my answer is",
)

_NUM_WORDS = {
    "zero": 0, "no": 0, "none": 0, "once": 1, "one": 1, "twice": 2, "two": 2,
    "thrice": 3, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    t = str(text).lower().strip().translate(_PUNC)
    return " ".join(w for w in t.split() if w not in _ARTICLES)


def extract_choice_letter(prediction: str, num_options: int = 4) -> str:
    """Read the chosen option letter out of a free-form prediction.

    Returns "" when no letter can be identified, which the caller scores as wrong
    rather than silently guessing.
    """
    s = str(prediction).strip()
    low = s.lower()
    for prefix in _ANSWER_PREFIXES:
        idx = low.find(prefix)
        if idx != -1:
            s = s[idx + len(prefix):].strip()
            low = s.lower()
    s = s.lstrip(" :*-\u2014").strip()
    valid = CHOICE_LETTERS[:max(1, min(num_options, len(CHOICE_LETTERS)))]

    # "(B)" or "B." or "B)" or a leading bare letter.
    m = re.match(rf"^\(?([{valid}])\)?[\.\):,]?(\s|$)", s)
    if m:
        return m.group(1)
    if s and s[0].upper() in valid and (len(s) == 1 or not s[1].isalnum()):
        return s[0].upper()
    # Otherwise the first standalone valid letter anywhere in the text.
    m = re.search(rf"\b([{valid}])\b", s)
    if m:
        return m.group(1).upper()
    return ""


def first_number(text: str) -> Optional[float]:
    """First integer/decimal in the text, or a spelled-out number word."""
    m = re.search(r"-?\d+(?:\.\d+)?", str(text))
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    for word, val in _NUM_WORDS.items():
        if re.search(rf"\b{word}\b", str(text), re.I):
            return float(val)
    return None


def yes_no(text: str) -> Optional[bool]:
    """Interpret a prediction as Yes/No, tolerating a bare Y/N."""
    s = str(text).strip()
    if not s:
        return None
    if re.match(r"^\s*y(es)?\b", s, re.I):
        return True
    if re.match(r"^\s*n(o)?\b", s, re.I):
        return False
    low = normalize(s)
    if "yes" in low.split()[:4]:
        return True
    if "no" in low.split()[:4] or "not" in low.split()[:4]:
        return False
    return None


def match_multiple_choice(prediction: str, gold_letter: str,
                          options: Optional[List[str]] = None) -> bool:
    gold = str(gold_letter).strip()[:1].upper()
    if not gold:
        return False
    n = len(options) if options else 4
    pred = extract_choice_letter(prediction, n)
    if pred and pred == gold:
        return True
    # Models sometimes answer with the option's text instead of its letter.
    if options:
        idx = CHOICE_LETTERS.find(gold)
        if 0 <= idx < len(options):
            gold_text = normalize(options[idx])
            if gold_text and normalize(prediction) == gold_text:
                return True
    return False


def match_open_ended(prediction: str, gold: str, task: str = "") -> bool:
    """Normalised text match, with numeric and Yes/No special cases.

    Not a substitute for the GPT judging that some open-ended protocols use; it
    is a deterministic lower bound.
    """
    p, g = normalize(prediction), normalize(gold)
    if not g:
        return False
    if p == g:
        return True

    # Counting tasks: compare numerically.
    if task.upper() == "REC" or (first_number(g) is not None and len(g.split()) <= 3):
        pn, gn = first_number(p), first_number(g)
        if pn is not None and gn is not None:
            return abs(pn - gn) < 1e-6

    # Binary tasks.
    if g in ("yes", "no"):
        pb = yes_no(prediction)
        return pb is not None and pb == (g == "yes")

    # A short gold phrase contained in a longer generated answer.
    if len(g.split()) <= 4 and g in p:
        return True
    return False


def score_answer(prediction: str, gold_text: str, gold_letter: str = "",
                 options: Optional[List[str]] = None, task: str = "") -> bool:
    """Dispatch to the right matcher for this question."""
    if options and gold_letter:
        return match_multiple_choice(prediction, gold_letter, options)
    return match_open_ended(prediction, gold_text, task)
