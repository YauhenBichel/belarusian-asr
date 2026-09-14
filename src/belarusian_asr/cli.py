# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""belarusian-asr transcribe | serve | download | bench"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="belarusian-asr", description="Belarusian speech recognition on the CPU.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--threads", type=int, help="CPU threads for the model (default: ONNX Runtime's choice)")
    parser.add_argument("--words", action="store_true", help="keep numbers as spoken words instead of digits")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="print the text of audio files (WAV, FLAC, OGG, MP3)")
    t.add_argument("files", nargs="+", type=Path)
    t.add_argument("--json", action="store_true", help="one JSON object per file, with segments")

    s = sub.add_parser("serve", help="HTTP: OpenAI /v1/audio/transcriptions and whisper.cpp /inference")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=11805)
    s.add_argument("--fallback", help="a whisper.cpp server for other languages, e.g. http://127.0.0.1:11806")

    sub.add_parser("download", help="fetch and verify the model files now (they are otherwise fetched on first use)")

    b = sub.add_parser("bench", help="score on FLEURS Belarusian (downloads the split, pinned and verified)")
    b.add_argument("--split", choices=["dev", "test"], default="test")
    b.add_argument("--limit", type=int, help="first N sentences")
    b.add_argument("--out", type=Path, default=Path("bench"))
    b.add_argument("--data", type=Path, default=Path.home() / ".cache" / "belarusian-asr" / "fleurs")
    b.add_argument("--whisper-cli", help="also score whisper.cpp: path to whisper-cli")
    b.add_argument("--whisper-model", help="with --whisper-cli: the ggml model file")
    b.add_argument("--whisper-threads", type=int, default=12)

    args = parser.parse_args(argv)
    from .transcriber import Transcriber

    if args.command == "download":
        from .models import FASTCONFORMER, SILERO_VAD, fetch

        for artifact in (FASTCONFORMER, SILERO_VAD):
            print(f"{artifact.repo}@{artifact.revision[:8]} ({artifact.license}): {fetch(artifact)}")
        return 0

    if args.command == "transcribe":
        engine, status = Transcriber(threads=args.threads, digits=not args.words), 0
        for path in args.files:
            try:
                result = engine.transcribe(path)
            except (OSError, ValueError) as exc:
                print(f"{path}: {exc}", file=sys.stderr)
                status = 1
                continue
            if args.json:
                print(json.dumps({"file": str(path), "text": result.text, "duration": result.duration,
                                  "segments": [s.__dict__ for s in result.segments]}, ensure_ascii=False))
            else:
                print(result.text if len(args.files) == 1 else f"{path}: {result.text}")
        return status

    if args.command == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(Transcriber(threads=args.threads, digits=not args.words), fallback_url=args.fallback),
                    host=args.host, port=args.port, log_level="info")
        return 0

    from . import bench

    engine = Transcriber(threads=args.threads, digits=not args.words)
    systems = {"belarusian-asr (FastConformer)": lambda path: engine.transcribe(path).text}
    if args.whisper_cli:
        if not args.whisper_model:
            parser.error("--whisper-cli needs --whisper-model")
        systems["whisper.cpp large-v3" if "large-v3" in args.whisper_model else "whisper.cpp"] = bench.whisper_cpp(
            args.whisper_cli, args.whisper_model, args.whisper_threads)
    report = bench.run(args.split, args.limit, args.out, systems, args.data,
                       log=lambda line: print(line, file=sys.stderr, flush=True))
    print(bench.markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
