# belarusian-asr

[![CI](https://github.com/YauhenBichel/belarusian-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/YauhenBichel/belarusian-asr/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE)
[![Demo](https://img.shields.io/badge/demo-benchmark%20and%20samples-c8313e)](https://yauhenbichel.github.io/belarusian-asr/)

**Belarusian speech recognition that runs on a CPU**, with a server that speaks OpenAI's and whisper.cpp's
transcription APIs, and a benchmark you can re-run.

**Беларускае распазнаванне маўлення на звычайным працэсары**: сервер з API OpenAI і whisper.cpp і бэнчмарк,
які можна паўтарыць.

It packages NVIDIA's Belarusian FastConformer (trained on Common Voice) for ONNX Runtime: no PyTorch, no NeMo, no GPU.
The model files are downloaded on first use, pinned by revision and checked by SHA-256.

## Why

Asked for Belarusian, general speech models do badly:

- Whisper large-v3 got about one character in ten wrong on FLEURS Belarusian, and more than four words in ten;
- Voxtral Mini, a multilingual audio LLM, answered Belarusian speech in English or Russian;
- the models that do know Belarusian live inside NeMo or large research toolkits.

## Accuracy

FLEURS Belarusian, measured on a 32-core CPU. One recording per sentence. CER and WER are corpus rates after
lower-casing and removing punctuation.

**FLEURS dev, the first 100 sentences** (15.4 s of audio each on average):

| system | CER | WER | CER, references without digits (84) | WER, without digits | seconds per clip |
|---|---|---|---|---|---|
| **belarusian-asr** (FastConformer, ONNX, CPU) | **6.7 %** | **14.7 %** | **2.6 %** | **9.9 %** | **0.27** |
| whisper.cpp large-v3, CPU, 12 threads | 10.5 % | 44.6 % | 10.1 % | 44.2 % | 8.2 |
| Voxtral Mini 3B (5 of those clips) | 89.6 % | 99.9 % | | | 6.0 |

"Without digits": FLEURS writes numbers as digits ("4892 м") and this model says them in words ("чатыры тысячы
восемсот дзевяноста два"), which is right but scores as wrong. The full test split is in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). Accuracy drops on clips longer than 20 seconds (see there); longer audio is
cut into speech segments automatically.

## Install

```bash
pip install belarusian-asr              # the library and the command line
pip install "belarusian-asr[server]"    # also the HTTP server
belarusian-asr download                 # optional: fetch and verify the model now (about 460 MB)
```

Python 3.11 or newer, on Linux, macOS or Windows.

## Use

```bash
belarusian-asr transcribe clip.wav          # WAV, FLAC, OGG or MP3
belarusian-asr transcribe *.ogg --json      # one JSON object per file, with segments
```

```python
from belarusian_asr import Transcriber

asr = Transcriber()                          # loads once; reuse it
result = asr.transcribe("clip.wav")
print(result.text, result.duration)
for segment in result.segments:              # one segment, or several for audio over 20 s
    print(segment.start, segment.end, segment.text)
```

## Server

```bash
belarusian-asr serve --port 11805
curl -s 127.0.0.1:11805/v1/audio/transcriptions -F file=@clip.wav             # OpenAI
curl -s 127.0.0.1:11805/inference -F file=@clip.wav -F response_format=json   # whisper.cpp
```

- `response_format`: `json`, `text` or `verbose_json` (with segments and duration).
- `language`: empty or Belarusian (`be`, `bel`, `be-BY`) is transcribed here. For any other language, pass
  `--fallback http://127.0.0.1:11806` to forward it to a whisper.cpp server; without one, the request is refused,
  because this model writes Belarusian whatever was said.
- One decode at a time, 25 MB upload limit, binds to 127.0.0.1. There is no authentication: put a gateway in front
  before exposing it.

## Benchmark

```bash
belarusian-asr bench --split test --out bench
belarusian-asr bench --split test --limit 200 --whisper-cli ./whisper-cli --whisper-model ggml-large-v3.bin
```

Downloads FLEURS Belarusian at a pinned revision (the audio checked by SHA-256), scores one recording per sentence,
and writes `bench/fleurs-be-<split>.json` (every hypothesis) and `.md` (the tables).

## Limits

- Trained on read speech (Common Voice). Conversational, noisy or far-field audio will be worse than these numbers.
- Numbers come out as words, names of foreign places are often misspelled, and there are no timestamps finer than
  segments.
- Belarusian only: a clip in Russian or Ukrainian comes back as Belarusian-looking text. Route other languages to a
  multilingual model (`--fallback`).

## Licence

The code is Apache-2.0. The model is **CC BY 4.0** (NVIDIA, [stt_be_fastconformer_hybrid_large_pc](https://huggingface.co/nvidia/stt_be_fastconformer_hybrid_large_pc),
ONNX export by [OpenVoiceOS](https://huggingface.co/OpenVoiceOS/stt_be_fastconformer_hybrid_large_pc_onnx)), and Silero
VAD is MIT: this package downloads them and does not redistribute them. If you ship an application built on it,
credit NVIDIA as CC BY 4.0 requires. FLEURS (benchmark and demo clips) is CC BY 4.0. Details in [NOTICE](NOTICE).

Built on [onnx-asr](https://github.com/istupakov/onnx-asr) (MIT) and [ONNX Runtime](https://onnxruntime.ai/) (MIT).

## Tests

```bash
pip install -e ".[server]"
python -m unittest discover -s tests       # no model and no network needed
```

CI also downloads the real model and transcribes a FLEURS clip, through the command line and the server.
