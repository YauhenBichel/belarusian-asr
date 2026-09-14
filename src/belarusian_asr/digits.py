# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Numbers spoken as words in the transcript, written as digits (inverse text normalization).

Spoken numbers come out of the model as words ("у тысяча дзевяцьсот шэсцьдзесят трэцім годзе") while written
Belarusian, and the FLEURS references, use digits ("у 1963 годзе"). This finds runs of number words that form one
valid number (hundreds, then tens or teens, then units, under decreasing scales) and writes them as digits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

THOUSAND, MILLION, BILLION = 10**3, 10**6, 10**9

# Every case form, all genders. Variants the model or writers use are listed too; the corpus check reports forms
# that never occur in real Belarusian text.
CARDINALS: dict[int, str] = {
    0: "нуль нуля нулю нулём нулі",
    1: "адзін адна адно адны аднаго адной аднае аднаму адну адным адных аднымі",
    2: "два дзве двух дзвюх двум дзвюм двума дзвюма",
    3: "тры трох тром трыма",
    4: "чатыры чатырох чатыром чатырма",
    5: "пяць пяці пяццю",
    6: "шэсць шасці шасцю",
    7: "сем сямі сямю",
    8: "восем васьмі васьмю",
    9: "дзевяць дзевяці дзевяццю",
    10: "дзесяць дзесяці дзесяццю",
    40: "сорак сарака",
    50: "пяцьдзясят пяцьдзесят пяцідзесяці пяццюдзесяццю",
    60: "шэсцьдзясят шэсцьдзесят шасцідзесяці шасцюдзесяццю",
    70: "семдзесят сямідзесяці сямюдзесяццю",
    80: "восемдзесят васьмідзесяці васьмюдзесяццю",
    90: "дзевяноста",
    100: "сто ста",
    200: "дзвесце двухсот двумстам двумастамі двухстах",
    300: "трыста трохсот тромстам трымастамі трохстах",
    400: "чатырыста чатырохсот чатыромстам чатырмастамі чатырохстах",
    THOUSAND: "тысяча тысячы тысячу тысячай тысячаю тысяч тысячам тысячамі тысячах",
    MILLION: "мільён мільёна мільёну мільёнам мільёне мільёны мільёнаў мільёнамі мільёнах",
    BILLION: "мільярд мільярда мільярду мільярдам мільярдзе мільярды мільярдаў мільярдамі мільярдах",
}
# 11-19, 20, 30: stem + ь / і / цю.
for value, stem in {11: "адзінаццац", 12: "дванаццац", 13: "трынаццац", 14: "чатырнаццац", 15: "пятнаццац",
                    16: "шаснаццац", 17: "сямнаццац", 18: "васямнаццац", 19: "дзевятнаццац", 20: "дваццац",
                    30: "трыццац"}.items():
    CARDINALS[value] = f"{stem}ь {stem}і {stem}цю"
# 500-900: nominative, then the genitive and instrumental stems of the unit with сот / стам / стамі / стах.
HUNDREDS = {
    500: ("пяцьсот", "пяці", "пяццю"), 600: ("шэсцьсот", "шасці", "шасцю"), 700: ("сямсот", "сямі", "сямю"),
    800: ("васямсот", "васьмі", "васьмю"), 900: ("дзевяцьсот", "дзевяці", "дзевяццю"),
}
for value, (nominative, genitive, instrumental) in HUNDREDS.items():
    CARDINALS[value] = f"{nominative} {genitive}сот {genitive}стам {instrumental}стамі {genitive}стах"
CARDINALS[700] += " семсот"  # the model's spelling

HARD_ENDINGS = "ы ая ае ое ыя ага ога ай ой аму ому ым ую ых ымі ою аю".split()
VELAR_ENDINGS = "і ая ое ае ія ога ага ой ай ому аму ім ую іх імі ою аю".split()
ORDINAL_STEMS = {
    1: "перш", 4: "чацвёрт", 5: "пят", 6: "шост", 7: "сём", 8: "восьм", 9: "дзявят", 10: "дзясят",
    11: "адзінаццат", 12: "дванаццат", 13: "трынаццат", 14: "чатырнаццат", 15: "пятнаццат", 16: "шаснаццат",
    17: "сямнаццат", 18: "васямнаццат", 19: "дзевятнаццат", 20: "дваццат", 30: "трыццат", 40: "саракав",
    50: "пяцідзясят", 60: "шасцідзясят", 70: "сямідзясят", 80: "васьмідзясят", 90: "дзевяност", 100: "сот",
    200: "двухсот", 300: "трохсот", 400: "чатырохсот", 500: "пяцісот", 600: "шасцісот", 700: "сямісот",
    800: "васьмісот", 900: "дзевяцісот", THOUSAND: "тысячн", MILLION: "мільённ",
}
ORDINAL_SPECIAL = {2: [f"друг{e}" for e in VELAR_ENDINGS],
                   3: "трэці трэцяя трэцяе трэція трэцяга трэцяй трэцяму трэцім трэцюю трэціх трэцімі".split()}
# Compound adjectives: "трохгадовы" -> "3-гадовы". Built on the genitive-like combining form of the number.
COMBINING = {1: "адна", 2: "двух", 3: "трох", 4: "чатырох", 5: "пяці", 6: "шасці", 7: "сямі", 8: "васьмі",
             9: "дзевяці", 10: "дзесяці", 11: "адзінаццаці", 12: "дванаццаці", 13: "трынаццаці", 14: "чатырнаццаці",
             15: "пятнаццаці", 16: "шаснаццаці", 17: "сямнаццаці", 18: "васямнаццаці", 19: "дзевятнаццаці",
             20: "дваццаці", 30: "трыццаці", 40: "сарака", 50: "пяцідзесяці", 60: "шасцідзесяці",
             70: "сямідзесяці", 80: "васьмідзесяці", 90: "дзевяноста", 100: "ста", THOUSAND: "тысяча"}
COMPOUND_ROOTS = ["гадов", "градусн", "працэнтн", "кіламетров", "мятров", "павярхов"]
# Plural "першыя", "другіх" mostly mean "the first ones", "other": not numbers at the end of a run.
PLURAL_ENDINGS = {"ыя", "ых", "ымі", "ія", "іх", "імі"}
MONTHS = set(("студзеня лютага сакавіка красавіка мая чэрвеня "
              "ліпеня жніўня верасня кастрычніка лістапада снежня").split())


@dataclass(frozen=True)
class Word:
    value: int
    kind: str  # unit, teen, ten, hundred, scale
    ending: str = ""  # "ordinal", or the compound suffix such as "гадовы"


def kind_of(value: int) -> str:
    if value >= THOUSAND:
        return "scale"
    if value >= 100:
        return "hundred"
    if value >= 20 or value == 10:
        return "ten"
    if value >= 11:
        return "teen"
    return "unit"


def build_lexicon() -> dict[str, Word]:
    lexicon: dict[str, Word] = {}
    for value, forms in CARDINALS.items():
        for form in forms.split():
            lexicon[form] = Word(value, kind_of(value))
    for value, stem in ORDINAL_STEMS.items():
        endings = VELAR_ENDINGS if stem[-1] in "гкх" else HARD_ENDINGS
        for ending in endings:
            if value == 1 and ending in PLURAL_ENDINGS:
                continue
            lexicon.setdefault(stem + ending, Word(value, kind_of(value), "ordinal"))
    for value, forms in ORDINAL_SPECIAL.items():
        for form in forms:
            if value == 2 and form.endswith(tuple(PLURAL_ENDINGS)):
                continue
            lexicon.setdefault(form, Word(value, kind_of(value), "ordinal"))
    for value, head in COMBINING.items():
        for root in COMPOUND_ROOTS:
            for ending in HARD_ENDINGS:
                lexicon.setdefault(head + root + ending, Word(value, kind_of(value), root + ending))
    return lexicon


LEXICON = build_lexicon()
LEVEL = {"unit": 1, "teen": 2, "ten": 2, "hundred": 3}
TOKEN = re.compile(r"[\w'\N{RIGHT SINGLE QUOTATION MARK}]+")


def parse_run(words: list[Word]) -> list[tuple[int, int, Word]]:
    """Split a run of number words into numbers: (token count, value, last word) for each."""
    out, i = [], 0
    while i < len(words):
        total, group, level, scale_cap, j = 0, 0, 4, BILLION * 1000, i
        while j < len(words):
            w = words[j]
            if w.kind == "scale":
                if w.value >= scale_cap:
                    break
                total += (group or 1) * w.value
                group, level, scale_cap = 0, 4, w.value
            else:
                if LEVEL[w.kind] >= level:
                    break
                group += w.value
                level = 1 if w.kind in ("teen", "unit") else LEVEL[w.kind]
            j += 1
            if w.ending:
                break
        out.append((j - i, total + group, words[j - 1]))
        i = j
    return out


def to_digits(text: str, min_value: int = 11, lone_scale_words: bool = True, single_ordinals: bool = False) -> str:
    """Write runs of number words as digits; everything else is copied unchanged.

    A number of one word stays a word when its value is below min_value ("адна з", "два дні"), when it is a bare
    scale word and lone_scale_words is set ("тысячы людзей"), or when it is an ordinal and single_ordinals is not
    set ("соты" is also "honeycombs"). Compound adjectives always convert ("трохгадовы" -> "3-гадовы"), and so does
    an ordinal right before a month ("шостага кастрычніка" -> "6 кастрычніка"). Numbers from 10 000 up are grouped
    by thousands with a space, as Belarusian writing does ("400 000"); years and other four-digit numbers are not.
    """
    out, pos, run = [], 0, []  # run: (match, Word) for adjacent number words, separated only by spaces

    def flush(following: str | None) -> None:
        nonlocal pos
        k = 0
        parts = parse_run([w for _, w in run])
        for n, (count, value, word) in enumerate(parts):
            first, last = run[k][0], run[k + count - 1][0]
            compound = word.ending not in ("", "ordinal")
            date = word.ending == "ordinal" and n == len(parts) - 1 and following in MONTHS
            keep = count == 1 and not compound and not date and (
                value < min_value or (word.kind == "scale" and lone_scale_words)
                or (word.ending == "ordinal" and not single_ordinals))
            out.append(text[pos:first.start()])
            if keep:
                out.append(text[first.start():last.end()])
            else:
                digits = f"{value:,}".replace(",", " ") if value >= 10_000 else str(value)
                out.append(f"{digits}-{word.ending}" if compound else digits)
            pos = last.end()
            k += count
        run.clear()

    for m in TOKEN.finditer(text):
        word = LEXICON.get(m.group().lower())
        if run and (word is None or text[run[-1][0].end():m.start()].strip()):
            gap = text[run[-1][0].end():m.start()]
            flush(m.group().lower() if not gap.strip() else None)
        if word is not None:
            run.append((m, word))
    if run:
        flush(None)
    out.append(text[pos:])
    return "".join(out)
