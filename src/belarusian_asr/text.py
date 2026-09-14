# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Text normalization for scoring, and character and word error rates."""

from __future__ import annotations

import re
import unicodedata

_APOSTROPHES = {ord("\N{RIGHT SINGLE QUOTATION MARK}"): "'", ord("\N{MODIFIER LETTER APOSTROPHE}"): "'"}
_STRESS = "\N{COMBINING ACUTE ACCENT}"


def normalize(text: str) -> str:
    """Lower case, one apostrophe, no stress marks, no punctuation, single spaces."""
    text = unicodedata.normalize("NFC", text.replace(_STRESS, "")).lower().translate(_APOSTROPHES)
    return " ".join(re.sub(r"[^\w' ]+", " ", text).split())


def has_digits(text: str) -> bool:
    return bool(re.search(r"\d", text))


def edit_distance(a, b) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def errors(reference: str, hypothesis: str) -> dict[str, int]:
    """Character and word edits against the reference, after normalize(); rates are errors / length."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    return {
        "chars": len(ref),
        "char_errors": edit_distance(ref, hyp),
        "words": len(ref.split()),
        "word_errors": edit_distance(ref.split(), hyp.split()),
    }


def rates(rows: list[dict[str, int]]) -> dict[str, float]:
    """Corpus CER and WER: total edits over total length, not a mean of per-clip rates."""
    chars, words = sum(r["chars"] for r in rows), sum(r["words"] for r in rows)
    return {
        "cer": round(sum(r["char_errors"] for r in rows) / chars, 4) if chars else 0.0,
        "wer": round(sum(r["word_errors"] for r in rows) / words, 4) if words else 0.0,
    }
