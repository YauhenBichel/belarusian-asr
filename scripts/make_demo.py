#!/usr/bin/env python3
# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Build docs/demo/results.json and the sample clips from `belarusian-asr bench` output.

    python scripts/make_demo.py --full bench/full/fleurs-be-test.json --compare bench/vs-whisper/fleurs-be-test.json

Samples: clips from the comparison run with no digits in the reference, 4-12 seconds, chosen to show both the usual
case and a hard one. They are FLEURS recordings (CC BY 4.0), written unmodified in content as Ogg Vorbis to keep the
page small.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "docs" / "demo"


def pick(clips: list[dict], systems: list[str], count: int) -> list[dict]:
    usable = [c for c in clips if not re.search(r"\d", c["reference"]) and 4 <= c["seconds"] <= 12
              and all(s in c for s in systems)]
    usable.sort(key=lambda c: c[systems[0]]["char_errors"] / max(1, c[systems[0]]["chars"]))
    if len(usable) <= count:
        return usable
    step = (len(usable) - 1) / (count - 1)
    return [usable[round(i * step)] for i in range(count)]  # from the best to the hardest, evenly


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--compare", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    full = json.loads(args.full.read_text(encoding="utf-8"))
    compare = json.loads(args.compare.read_text(encoding="utf-8"))
    systems = list(compare["summary"]["systems"])
    audio_dir = DEMO / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for n, clip in enumerate(pick(compare["clips"], systems, args.samples), 1):
        data, rate = sf.read(clip["path"], dtype="float32")
        name = f"fleurs-{clip['id']}.ogg"
        sf.write(audio_dir / name, data, rate, format="OGG", subtype="VORBIS")
        samples.append({
            "id": clip["id"], "audio": f"audio/{name}", "seconds": round(clip["seconds"], 1),
            "reference": clip["reference"],
            "outputs": {s: {"text": clip[s]["hypothesis"],
                            "cer": round(clip[s]["char_errors"] / max(1, clip[s]["chars"]), 3)} for s in systems},
        })

    out = {"full": full["summary"], "compare": compare["summary"], "samples": samples}
    (DEMO / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(samples)} samples, {DEMO / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
