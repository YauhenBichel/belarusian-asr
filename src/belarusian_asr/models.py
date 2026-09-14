# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""The model files, pinned by revision and SHA-256, and a download that refuses anything else.

Nothing is redistributed here: the files are fetched from Hugging Face on first use into a cache folder
(BELARUSIAN_ASR_CACHE, default ~/.cache/belarusian-asr) and checked before they are used.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    name: str
    repo: str
    revision: str
    files: dict[str, str]  # file name -> SHA-256
    license: str
    source: str

    def url(self, file: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{file}"


# NVIDIA NeMo FastConformer-Hybrid Large (Belarusian, punctuation and capitalization), trained on Mozilla Common
# Voice 12. ONNX export of its CTC branch by OpenVoiceOS for onnx-asr.
FASTCONFORMER = Artifact(
    name="stt_be_fastconformer_hybrid_large_pc_onnx",
    repo="OpenVoiceOS/stt_be_fastconformer_hybrid_large_pc_onnx",
    revision="3b8ee6e9287481a49c013ad5b09828b74f1b4c8c",
    files={
        "model.onnx": "dc68a87c0f7baf0151b4313fdef5910d22dbc2a60c388db53067bb11833f4813",
        "vocab.txt": "4f269128aa00ae362a159a42e90c00dbb252ac4b35646bc056061eff255e0c54",
        "config.json": "e807f8692efbb65fec1e6356c55e58e9ce023d67f49b0dc36608b1d9d4b59480",
    },
    license="CC-BY-4.0",
    source="https://huggingface.co/nvidia/stt_be_fastconformer_hybrid_large_pc",
)

# Silero VAD, to cut audio longer than the model takes into speech segments.
SILERO_VAD = Artifact(
    name="silero-vad-onnx",
    repo="istupakov/silero-vad-onnx",
    revision="b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
    files={
        "silero_vad.onnx": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        "config.json": "1094039d370c82889582ba739a3d1caac5754c8b3a17a66a534200c9f72086e2",
    },
    license="MIT",
    source="https://github.com/snakers4/silero-vad",
)


class ChecksumError(RuntimeError):
    pass


def cache_dir() -> Path:
    return Path(os.environ.get("BELARUSIAN_ASR_CACHE") or Path.home() / ".cache" / "belarusian-asr")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(artifact: Artifact, root: Path | None = None, opener=urllib.request.urlopen) -> Path:
    """The artifact's folder, with every file present and matching its SHA-256. Downloads what is missing."""
    folder = (root or cache_dir()) / artifact.name / artifact.revision
    folder.mkdir(parents=True, exist_ok=True)
    for file, expected in artifact.files.items():
        target = folder / file
        if target.exists() and sha256_of(target) == expected:
            continue
        partial = target.with_name(target.name + ".part")
        with opener(artifact.url(file)) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        actual = sha256_of(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise ChecksumError(f"{artifact.repo}/{file}@{artifact.revision[:8]}: SHA-256 {actual}, expected {expected}")
        partial.replace(target)
    return folder
