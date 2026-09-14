# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Everything that runs without the model: text, hashes and downloads, audio, the transcriber's routing, the benchmark."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from belarusian_asr import Transcriber, bench, models, text
from belarusian_asr.audio import AudioError, load


def wav_bytes(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * rate), dtype=np.float32), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class FakeModel:
    def __init__(self):
        self.calls = []

    def recognize(self, samples, sample_rate=16000):
        self.calls.append(len(samples) / sample_rate)
        return " Прывітанне, свет "

    def with_vad(self, vad, **options):
        self.vad_options = options
        return FakeSegmenter()


class FakeSegmenter:
    class S:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    def recognize(self, samples, sample_rate=16000):
        return [self.S(0.0, 14.9, "першы кавалак"), self.S(15.3, 29.0, " "), self.S(29.4, 41.0, "другі кавалак")]


class Text(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(text.normalize("Сем" + chr(0x2019) + "і, «Эн-Эйч-Кэй»: хва" + chr(0x301) + "лямі!"), "сем'і эн эйч кэй хвалямі")

    def test_errors_and_corpus_rates(self):
        row = text.errors("Мае сябры прыедуць.", "мае сябры прыедзе")
        self.assertEqual((row["words"], row["word_errors"]), (3, 1))
        self.assertEqual(text.rates([row, {"chars": 10, "char_errors": 0, "words": 1, "word_errors": 0}])["wer"], 0.25)
        self.assertTrue(text.has_digits("4892 м") and not text.has_digits("чатыры"))


class Downloads(unittest.TestCase):
    def artifact(self, payload: bytes, digest: str | None = None):
        return models.Artifact("x", "owner/repo", "a" * 40, {"f.bin": digest or hashlib.sha256(payload).hexdigest()},
                               "MIT", "https://example.org")

    def test_verified_download_and_reuse(self):
        payload, calls = b"model bytes", []

        def opener(url):
            calls.append(url)
            return io.BytesIO(payload)

        with tempfile.TemporaryDirectory() as root:
            folder = models.fetch(self.artifact(payload), Path(root), opener=opener)
            self.assertEqual((folder / "f.bin").read_bytes(), payload)
            models.fetch(self.artifact(payload), Path(root), opener=opener)
        self.assertEqual(calls, ["https://huggingface.co/owner/repo/resolve/" + "a" * 40 + "/f.bin"])

    def test_a_file_that_does_not_match_is_refused_and_removed(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(models.ChecksumError):
                models.fetch(self.artifact(b"good", digest="0" * 64), Path(root), opener=lambda url: io.BytesIO(b"evil"))
            self.assertEqual(list(Path(root).rglob("f.bin*")), [])

    def test_pinned_artifacts_are_complete(self):
        for artifact in (models.FASTCONFORMER, models.SILERO_VAD):
            self.assertEqual(len(artifact.revision), 40)
            self.assertTrue(all(len(h) == 64 for h in artifact.files.values()))


class Audio(unittest.TestCase):
    def test_mono_float(self):
        samples, rate = load(wav_bytes(0.5))
        self.assertEqual((samples.dtype, rate, len(samples)), (np.float32, 16000, 8000))

    def test_refusals(self):
        with self.assertRaises(AudioError):
            load(b"not audio at all")
        with self.assertRaises(AudioError):
            load(wav_bytes(0.1, rate=12345))


class Routing(unittest.TestCase):
    def test_short_audio_goes_whole(self):
        model = FakeModel()
        result = Transcriber(model=model).transcribe(wav_bytes(12))
        self.assertEqual((result.text, result.duration, model.calls), ("Прывітанне, свет", 12.0, [12.0]))

    def test_long_audio_is_cut_by_vad_and_empty_segments_dropped(self):
        model = FakeModel()
        result = Transcriber(model=model, vad_model=object()).transcribe(wav_bytes(41))
        self.assertEqual(result.text, "першы кавалак другі кавалак")
        self.assertEqual([s.end for s in result.segments], [14.9, 41.0])
        self.assertEqual(model.calls, [])
        self.assertEqual(model.vad_options["max_speech_duration_s"], 15.0)

    def test_samples_need_a_rate(self):
        with self.assertRaises(ValueError):
            Transcriber(model=FakeModel()).transcribe(np.zeros(10, dtype=np.float32))


class Bench(unittest.TestCase):
    def test_git_blob_id(self):
        self.assertEqual(bench.git_blob_sha1(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")

    def test_one_recording_per_sentence_and_summary(self):
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root)
            for name in ("a1.wav", "a2.wav", "b1.wav"):
                (audio / name).write_bytes(b"")
            rows = [["1", "a1.wav", "Два словы", "", "", "160000", "F"], ["1", "a2.wav", "Два словы", "", "", "160000", "M"],
                    ["2", "b1.wav", "4892 м", "", "", "400000", "F"], ["3", "missing.wav", "x", "", "", "1", "F"]]
            clips = bench.one_per_sentence(rows, audio)
        self.assertEqual([c["id"] for c in clips], ["1", "2"])
        clips[0]["sys"] = {"hypothesis": "", "seconds_taken": 1.0, **text.errors("Два словы", "два словы")}
        clips[1]["sys"] = {"hypothesis": "", "seconds_taken": 2.0, **text.errors("4892 м", "чатыры тысячы")}
        summary = bench.summarize(clips, "sys")
        self.assertEqual(summary["no_digits"], {"clips": 1, "cer": 0.0, "wer": 0.0})
        self.assertEqual(sorted(summary["by_length"]), ["10-15s", "25s+"])
        self.assertEqual(summary["real_time_factor"], round(3.0 / 35.0, 3))
        self.assertIn("| sys |", bench.markdown({"dataset": "d", "clips": 2, "audio_seconds_mean": 17.5,
                                                 "systems": {"sys": summary}}))


class Resume(unittest.TestCase):
    def test_a_stopped_run_resumes_without_scoring_clips_again(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            clips = lambda: [{"id": str(i), "path": Path(f"{i}.wav"), "reference": "мае сябры", "seconds": 5.0} for i in range(3)]
            calls = []

            def first(path):
                calls.append(path.name)
                if path.name == "2.wav":
                    raise KeyboardInterrupt  # stopped in the middle of the third clip
                return "мае сябры"

            with self.assertRaises(KeyboardInterrupt):
                bench.score(clips(), "test", out, {"sys": first}, log=lambda line: None)
            self.assertEqual(len((out / "fleurs-be-test.partial.jsonl").read_text(encoding="utf-8").splitlines()), 2)

            def second(path):
                calls.append("again " + path.name)
                return "мае"

            report = bench.score(clips(), "test", out, {"sys": second}, log=lambda line: None)
            self.assertEqual(calls, ["0.wav", "1.wav", "2.wav", "again 2.wav"])
            self.assertEqual(report["clips"], 3)
            self.assertTrue((out / "fleurs-be-test.json").exists() and (out / "fleurs-be-test.md").exists())


if __name__ == "__main__":
    unittest.main()
