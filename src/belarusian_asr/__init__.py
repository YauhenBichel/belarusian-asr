# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Belarusian speech recognition on the CPU: NVIDIA's Belarusian FastConformer through ONNX Runtime.

    from belarusian_asr import Transcriber
    Transcriber().transcribe("clip.wav").text
"""

from .transcriber import Result, Segment, Transcriber

__version__ = "0.1.0"
__all__ = ["Result", "Segment", "Transcriber", "__version__"]
