# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Audio in, 32-bit float mono out. WAV, FLAC, OGG and MP3 through libsndfile; no ffmpeg needed."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf

# The rates onnx-asr resamples from.
SUPPORTED_RATES = (8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000)


class AudioError(ValueError):
    pass


def load(source: str | Path | bytes) -> tuple[np.ndarray, int]:
    """(samples, sample rate): mono float32. Raises AudioError with a reason a user can act on."""
    try:
        data, rate = sf.read(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source),
                             dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
        raise AudioError(f"cannot read the audio ({exc}); WAV, FLAC, OGG and MP3 are supported") from None
    if data.size == 0:
        raise AudioError("the audio is empty")
    if rate not in SUPPORTED_RATES:
        raise AudioError(f"sample rate {rate} Hz is not supported; convert with: ffmpeg -i in -ar 16000 out.wav")
    return data.mean(axis=1).astype(np.float32), int(rate)
