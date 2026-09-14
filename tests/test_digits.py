# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Numbers spoken as words, written as digits."""

import re
import unittest

import numpy as np

from belarusian_asr import Transcriber
from belarusian_asr.digits import LEXICON, to_digits


class Digits(unittest.TestCase):
    def test_number_words_become_digits(self):
        for spoken, written in [
            ("у тысяча дзевяцьсот шэсцьдзесят трэцім годзе", "у 1963 годзе"),
            ("ад пяцідзесяці да двухсот копій", "ад 50 да 200 копій"),
            ("У дзве тысячы другім годзе", "У 2002 годзе"),
            ("пяцідзесяці трохгадовы кома", "53-гадовы кома"),
            ("дваццаць пяць чалавек", "25 чалавек"),
            ("сто дваццаць тры тысячы чатырыста пяцьдзясят шэсць", "123 456"),
            ("каля чатырохсот тысяч выпадкаў", "каля 400 000 выпадкаў"),
            ("шостага кастрычніка тысяча семсот восемдзесят дзявятага года", "6 кастрычніка 1789 года"),
            ("у дваццаць першым стагоддзі", "у 21 стагоддзі"),
        ]:
            with self.subTest(spoken):
                self.assertEqual(to_digits(spoken), written)

    def test_text_without_a_written_number_is_unchanged(self):
        for text in [
            "адна з іх",  # one word below 11
            "два тры",  # two numbers of one word each
            "тысячы людзей",  # a bare scale word
            "соты раз",  # a lone ordinal; "соты" is also honeycombs
            "у дзевятнаццатым стагоддзі",
            "тысяча мільён",  # scales out of order are two separate words
            "Мы згодны з заявай Алімпійскага камітэта.",
        ]:
            with self.subTest(text):
                self.assertEqual(to_digits(text), text)

    def test_plural_first_and_second_are_not_numbers(self):
        self.assertEqual(to_digits("каля сарака другіх зняволеных"), "каля 40 другіх зняволеных")

    def test_punctuation_and_case_are_kept_and_a_comma_ends_a_number(self):
        self.assertEqual(to_digits("Цягам ночы, дваццаць пяць разоў."), "Цягам ночы, 25 разоў.")
        self.assertEqual(to_digits("дваццаць, пяць"), "20, пяць")
        self.assertEqual(to_digits("пятнаццаць пяць"), "15 пяць")

    def test_options(self):
        self.assertEqual(to_digits("два дні", min_value=0), "2 дні")
        self.assertEqual(to_digits("тысячы людзей", lone_scale_words=False), "1000 людзей")
        self.assertEqual(to_digits("соты раз", single_ordinals=True), "100 раз")

    def test_lexicon_holds_lowercase_belarusian_words_only(self):
        for form in LEXICON:
            self.assertRegex(form, re.compile(r"^[абвгдеёжзійклмнопрстуўфхцчшыьэюя]+$"))


class TranscriberWritesDigits(unittest.TestCase):
    class Model:
        def recognize(self, samples, sample_rate=16000):
            return " У дзве тысячы другім годзе "

    def test_digits_by_default_and_words_on_request(self):
        samples = np.zeros(16000, dtype=np.float32)
        self.assertEqual(Transcriber(model=self.Model()).transcribe(samples, 16000).text, "У 2002 годзе")
        self.assertEqual(Transcriber(model=self.Model(), digits=False).transcribe(samples, 16000).text,
                         "У дзве тысячы другім годзе")


if __name__ == "__main__":
    unittest.main()
