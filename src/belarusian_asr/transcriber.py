# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Belarusian speech to text on the CPU.

Clips up to `whole_up_to_s` seconds (default 20) go to the model in one piece: on FLEURS that was better than
cutting them (CER 2.6 % against 3.4 % on references without digits). Longer audio, which the model cannot take,
is cut into speech segments of at most `max_segment_s` seconds by Silero VAD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import audio as audio_io
from .models import FASTCONFORMER, SILERO_VAD, fetch


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Result:
    text: str
    duration: float
    segments: list[Segment] = field(default_factory=list)


class Transcriber:
    def __init__(
        self,
        cache: Path | None = None,
        whole_up_to_s: float = 20.0,
        max_segment_s: float = 15.0,
        threads: int | None = None,
        model: Any = None,
        vad_model: Any = None,
    ) -> None:
        self.cache, self.whole_up_to_s, self.max_segment_s, self.threads = cache, whole_up_to_s, max_segment_s, threads
        self._model, self._segmenter, self._vad_model = model, None, vad_model

    def _session_options(self) -> Any:
        if not self.threads:
            return None
        import onnxruntime as rt

        options = rt.SessionOptions()
        options.intra_op_num_threads = self.threads
        return options

    @property
    def model(self) -> Any:
        if self._model is None:
            import onnx_asr

            self._model = onnx_asr.load_model("nemo-conformer-ctc", fetch(FASTCONFORMER, self.cache),
                                              sess_options=self._session_options())
        return self._model

    def _segments_of(self, samples: np.ndarray, rate: int) -> list[Segment]:
        if self._segmenter is None:
            vad = self._vad_model
            if vad is None:
                import onnx_asr

                vad = onnx_asr.load_vad("silero", fetch(SILERO_VAD, self.cache))
            self._segmenter = self.model.with_vad(vad, max_speech_duration_s=self.max_segment_s,
                                                  min_silence_duration_ms=150, speech_pad_ms=80)
        return [Segment(round(float(s.start), 2), round(float(s.end), 2), s.text.strip())
                for s in self._segmenter.recognize(samples, sample_rate=rate)]

    def transcribe(self, source: str | Path | bytes | np.ndarray, sample_rate: int | None = None) -> Result:
        """Text of the speech in `source`: a path, the bytes of an audio file, or samples with sample_rate."""
        if isinstance(source, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate is required with samples")
            samples, rate = source.astype(np.float32), sample_rate
        else:
            samples, rate = audio_io.load(source)
        duration = round(len(samples) / rate, 2)
        if duration <= self.whole_up_to_s:
            text = str(self.model.recognize(samples, sample_rate=rate)).strip()
            return Result(text, duration, [Segment(0.0, duration, text)])
        segments = [s for s in self._segments_of(samples, rate) if s.text]
        return Result(" ".join(s.text for s in segments), duration, segments)
