# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""The HTTP server with a fake transcriber and a fake whisper.cpp fallback (needs the [server] extra)."""

import io
import unittest

import httpx
import numpy as np
import soundfile as sf
from starlette.testclient import TestClient

from belarusian_asr import Result, Segment
from belarusian_asr.audio import AudioError
from belarusian_asr.server import create_app, is_belarusian


def wav() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class FakeTranscriber:
    def __init__(self):
        self.calls = 0

    def transcribe(self, data):
        self.calls += 1
        if data.startswith(b"bad"):
            raise AudioError("cannot read the audio")
        return Result("Мае сябры прыедуць", 2.5, [Segment(0.0, 2.5, "Мае сябры прыедуць")])


class Server(unittest.TestCase):
    def setUp(self):
        self.engine, self.forwarded = FakeTranscriber(), []

        def whisper(request: httpx.Request) -> httpx.Response:
            self.forwarded.append(request)
            return httpx.Response(200, json={"text": "hello"})

        fallback = httpx.AsyncClient(transport=httpx.MockTransport(whisper))
        self.client = TestClient(create_app(self.engine, fallback_url="http://whisper.test", client=fallback))
        self.plain = TestClient(create_app(self.engine))

    def post(self, client, path="/v1/audio/transcriptions", data=None, content=None):
        return client.post(path, files={"file": ("a.wav", content if content is not None else wav(), "audio/wav")},
                           data=data or {})

    def test_health(self):
        body = self.client.get("/health").json()
        self.assertEqual((body["status"], body["fallback"]), ("ok", True))

    def test_belarusian_in_both_dialects_and_formats(self):
        for path in ("/v1/audio/transcriptions", "/inference"):
            r = self.post(self.client, path, {"language": "be-BY"})
            self.assertEqual((r.status_code, r.json()), (200, {"text": "Мае сябры прыедуць"}))
        r = self.post(self.client, data={"response_format": "text"})
        self.assertEqual(r.text, "Мае сябры прыедуць\n")
        r = self.post(self.client, data={"response_format": "verbose_json", "language": "BE"})
        self.assertEqual((r.json()["language"], r.json()["duration"], r.json()["segments"][0]["end"]), ("belarusian", 2.5, 2.5))
        self.assertEqual(self.forwarded, [])

    def test_other_languages_go_to_the_fallback_or_are_refused(self):
        r = self.post(self.client, "/inference", {"language": "en", "response_format": "json"})
        self.assertEqual((r.status_code, r.json()), (200, {"text": "hello"}))
        self.assertEqual(len(self.forwarded), 1)
        self.assertIn(b'name="language"', self.forwarded[0].content)
        self.assertIn("fallback", r.headers["x-belarusian-asr-route"])
        r = self.post(self.plain, data={"language": "uk"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("Belarusian only", r.json()["error"]["message"])

    def test_refusals(self):
        self.assertEqual(self.post(self.client, data={"response_format": "srt"}).status_code, 400)
        self.assertEqual(self.post(self.client, content=b"").status_code, 400)
        self.assertEqual(self.post(self.client, content=b"bad audio").status_code, 400)
        self.assertEqual(self.client.post("/inference", data={"language": "be"}).status_code, 400)
        small = TestClient(create_app(self.engine, max_bytes=100))
        self.assertEqual(self.post(small).status_code, 413)

    def test_language_detection(self):
        self.assertTrue(all(is_belarusian(v) for v in (None, "", "be", "BE", "be_BY", "bel", "Belarusian")))
        self.assertFalse(any(is_belarusian(v) for v in ("ru", "uk", "en-US")))


if __name__ == "__main__":
    unittest.main()
