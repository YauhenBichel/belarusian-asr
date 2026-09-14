# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""A reproducible benchmark on FLEURS Belarusian (google/fleurs, be_by, CC-BY-4.0), pinned by revision and hash.

FLEURS reads every sentence two or three times; one recording per sentence is scored, so no sentence counts twice.
Scores are corpus CER and WER after normalize(), reported three ways: all clips, clips whose reference has no
digits (FLEURS writes "4892 м"; a model that says the number in words is right but scored wrong), and by length.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Callable

from .text import errors, has_digits, rates

FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_URL = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REVISION}/data/be_by"
SPLITS = {
    "dev": {"tsv_blob": "3d9f5764e745d7e629c55bb46e998d122e80cb22",
            "tar_sha256": "a1477879b97012680bb9b087dd60475bc67ca502629a60c849da7cd1a1e6d42d"},
    "test": {"tsv_blob": "8b2871d9f13ce3f1f679dbb1aacf9b7f931b1baa",
             "tar_sha256": "16f14328bc329497e9746e03da6d9b0fa907c66f32c317ee0c80faa370d7900d"},
}
LENGTHS = ((0, 10), (10, 15), (15, 20), (20, 25), (25, 1000))


def git_blob_sha1(data: bytes) -> str:
    """The id git (and Hugging Face) give a file that is not stored in LFS."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _download(url: str, target: Path, opener=urllib.request.urlopen) -> None:
    partial = target.with_name(target.name + ".part")
    with opener(url) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.replace(target)


def fetch_split(split: str, root: Path) -> tuple[list[list[str]], Path]:
    """(tsv rows, audio folder) of a FLEURS be_by split, downloaded once and verified."""
    spec, folder = SPLITS[split], root / FLEURS_REVISION / split
    folder.mkdir(parents=True, exist_ok=True)
    tsv, tar = folder / f"{split}.tsv", folder / f"{split}.tar.gz"
    if not tsv.exists() or git_blob_sha1(tsv.read_bytes()) != spec["tsv_blob"]:
        _download(f"{FLEURS_URL}/{split}.tsv", tsv)
        if git_blob_sha1(tsv.read_bytes()) != spec["tsv_blob"]:
            raise RuntimeError(f"FLEURS {split}.tsv does not match its git blob id")
    audio = folder / "audio"
    if not audio.exists():
        if not tar.exists():
            _download(f"{FLEURS_URL}/audio/{split}.tar.gz", tar)
        digest = hashlib.sha256()
        with tar.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != spec["tar_sha256"]:
            raise RuntimeError(f"FLEURS {split}.tar.gz does not match its SHA-256")
        with tarfile.open(tar) as archive:
            archive.extractall(audio, filter="data")  # no absolute paths, links or devices
    rows = [line.rstrip("\n").split("\t") for line in tsv.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, audio


def one_per_sentence(rows: list[list[str]], audio: Path) -> list[dict]:
    """The first available recording of each sentence id, in file order: id, path, reference, seconds."""
    wavs = {p.name: p for p in audio.rglob("*.wav")}
    seen, out = set(), []
    for row in rows:
        if len(row) < 6 or row[0] in seen or row[1] not in wavs:
            continue
        seen.add(row[0])
        out.append({"id": row[0], "path": wavs[row[1]], "reference": row[2], "seconds": int(row[5]) / 16000})
    return out


def summarize(clips: list[dict], system: str) -> dict:
    scored = [c for c in clips if system in c]
    no_digits = [c for c in scored if not has_digits(c["reference"])]
    by_length = {}
    for lo, hi in LENGTHS:
        part = [c[system] for c in scored if lo <= c["seconds"] < hi]
        if part:
            by_length[f"{lo}-{hi}s" if hi < 1000 else f"{lo}s+"] = {"clips": len(part), **rates(part)}
    seconds = sum(c[system]["seconds_taken"] for c in scored)
    audio = sum(c["seconds"] for c in scored)
    return {
        "clips": len(scored),
        "all": rates([c[system] for c in scored]),
        "no_digits": {"clips": len(no_digits), **rates([c[system] for c in no_digits])},
        "by_length": by_length,
        "seconds_per_clip": round(seconds / len(scored), 2) if scored else 0.0,
        "real_time_factor": round(seconds / audio, 3) if audio else 0.0,
    }


def whisper_cpp(cli: str, model: str, threads: int = 12) -> Callable[[Path], str]:
    def run(path: Path) -> str:
        out = subprocess.run([cli, "-m", model, "-l", "be", "-t", str(threads), "-nt", "-np", "-f", str(path)],
                             capture_output=True, text=True, timeout=1800, check=False)
        return out.stdout.strip()

    return run


def run(split: str, limit: int | None, out_dir: Path, systems: dict[str, Callable[[Path], str]],
        data_root: Path, log=print) -> dict:
    rows, audio = fetch_split(split, data_root)
    clips = one_per_sentence(rows, audio)[:limit]
    return score(clips, split, out_dir, systems, log)


def score(clips: list[dict], split: str, out_dir: Path, systems: dict[str, Callable[[Path], str]], log=print) -> dict:
    """Score every clip with every system, then write the reports.

    Each scored clip is appended to fleurs-be-<split>.partial.jsonl as it finishes, and a run that stops (a signal, a
    reboot) picks up from there: clips already scored by every system are not scored again.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"fleurs-be-{split}.partial.jsonl"
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            saved = json.loads(line)
            if all(name in saved for name in systems):
                done[saved["id"]] = saved
    with checkpoint.open("a", encoding="utf-8") as journal:
        for n, clip in enumerate(clips, 1):
            if clip["id"] in done:
                clip.update({name: done[clip["id"]][name] for name in systems})
                continue
            for name, transcribe in systems.items():
                started = time.perf_counter()
                hypothesis = transcribe(clip["path"])
                clip[name] = {"hypothesis": hypothesis, "seconds_taken": round(time.perf_counter() - started, 3),
                              **errors(clip["reference"], hypothesis)}
            journal.write(json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in clip.items()},
                                     ensure_ascii=False) + "\n")
            journal.flush()
            log(f"{n}/{len(clips)} {clip['id']}")
    if done:
        log(f"resumed: {len(done)} clips were already scored")
    report = {
        "dataset": f"google/fleurs be_by {split} @ {FLEURS_REVISION}",
        "clips": len(clips),
        "audio_seconds_mean": round(sum(c["seconds"] for c in clips) / len(clips), 1) if clips else 0.0,
        "systems": {name: summarize(clips, name) for name in systems},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [{k: (str(v) if isinstance(v, Path) else v) for k, v in c.items()} for c in clips]
    (out_dir / f"fleurs-be-{split}.json").write_text(
        json.dumps({"summary": report, "clips": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / f"fleurs-be-{split}.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(report: dict) -> str:
    lines = [f"Dataset: {report['dataset']}, {report['clips']} sentences, "
             f"{report['audio_seconds_mean']} s mean.", "",
             "| system | CER | WER | CER, no digits | WER, no digits | s per clip | real-time factor |",
             "|---|---|---|---|---|---|---|"]
    for name, s in report["systems"].items():
        lines.append(f"| {name} | {s['all']['cer']:.2%} | {s['all']['wer']:.2%} | {s['no_digits']['cer']:.2%} "
                     f"| {s['no_digits']['wer']:.2%} | {s['seconds_per_clip']} | {s['real_time_factor']} |")
    lengths = sorted({k for s in report["systems"].values() for k in s["by_length"]},
                     key=lambda k: int(re.match(r"\d+", k).group()))
    lines += ["", "CER by clip length:", "", "| system | " + " | ".join(lengths) + " |",
              "|---|" + "---|" * len(lengths)]
    for name, s in report["systems"].items():
        cells = [f"{s['by_length'][k]['cer']:.2%} ({s['by_length'][k]['clips']})" if k in s["by_length"] else "-"
                 for k in lengths]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
