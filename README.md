# belarusian-asr

[![CI](https://github.com/YauhenBichel/belarusian-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/YauhenBichel/belarusian-asr/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/belarusian-asr.svg)](https://pypi.org/project/belarusian-asr/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![language: Belarusian](https://img.shields.io/badge/language-беларуская-c8313e)
[![Demo](https://img.shields.io/badge/demo-listen%20and%20compare-c8313e)](https://yauhenbichel.github.io/belarusian-asr/)

Open **Belarusian speech recognition** (ASR, speech-to-text) that runs on an ordinary CPU: NVIDIA's Belarusian
FastConformer packaged for ONNX Runtime, a server that speaks OpenAI's and whisper.cpp's transcription APIs, and a
benchmark anyone can re-run.

**Беларускае распазнаванне маўлення, якое працуе на звычайным працэсары.** Адкрыты пакет, сервер з API OpenAI і
whisper.cpp і бэнчмарк, які можна паўтарыць.

## Demo

**[Listen and compare on the demo page](https://yauhenbichel.github.io/belarusian-asr/)**: six FLEURS recordings, the
reference, belarusian-asr and whisper.cpp large-v3 side by side, with the differences marked — from the easiest clip
to the hardest, including one where Whisper does better.

Two of them:

| | |
|---|---|
| [▶ FLEURS 1894](https://yauhenbichel.github.io/belarusian-asr/audio/fleurs-1894.ogg) (9.7 s) | |
| reference | Кары ўяўляе сабой страву з мяса ці гародніны, якая прыпраўлена травамі і спецыямі. |
| **belarusian-asr** — CER 0.0 % | Кары ўяўляе сабой страву з мяса ці гародніны, якая прыпраўлена травамі і спецыямі. |
| whisper.cpp large-v3 — CER 7.5 % | Кары уяуляе сабой страву з мяса ці гародніны, якая приправляна травамі і спеціямі. |
| [▶ FLEURS 1860](https://yauhenbichel.github.io/belarusian-asr/audio/fleurs-1860.ogg) (9.6 s) | |
| reference | Па звестках з канцылярыі губернатара, дзевятнаццаць з агульнай колькасці пацярпелых былі афіцэрамі паліцыі. |
| **belarusian-asr** — CER 3.8 % | Па звестках з канцылярыяй губернатара, у дзевятнаццаць з агульнай колькасці пацярпелых былі афіцэрамі паліцыі. |
| whisper.cpp large-v3 — CER 21.9 % | Па звездках з канцеляры губернатора, 19 загульной колькасті патярпелых былі афіцэрымі паліцыі. |

## Why

Asked for Belarusian, the speech models most people reach for do badly:

- **Whisper large-v3** gets about one character in ten and **more than four words in ten** wrong on FLEURS Belarusian;
- **Voxtral Mini**, a multilingual audio LLM, answered Belarusian speech in English or Russian;
- the models that do know Belarusian live inside NeMo or large research toolkits, with PyTorch and often a GPU.

This project started when a Belarusian recording sent through a local AI gateway came back translated into English.

## Use

Python 3.11+, Linux, macOS or Windows. No GPU, no PyTorch.

```bash
pip install belarusian-asr                  # the library and the command line
belarusian-asr transcribe clip.wav          # WAV, FLAC, OGG or MP3; the model (~460 MB) downloads on first use
```

```python
from belarusian_asr import Transcriber

asr = Transcriber()                          # load once, reuse
result = asr.transcribe("clip.wav")
print(result.text)                           # also result.duration and result.segments
```

**As a server** — any OpenAI or whisper.cpp client works:

```bash
pip install "belarusian-asr[server]"
belarusian-asr serve --port 11805
curl -s 127.0.0.1:11805/v1/audio/transcriptions -F file=@clip.wav             # OpenAI
curl -s 127.0.0.1:11805/inference -F file=@clip.wav -F response_format=json   # whisper.cpp
```

`response_format` is `json`, `text` or `verbose_json` (segments and duration). A `language` other than Belarusian goes to
a whisper.cpp server given with `--fallback`, or is refused: this model writes Belarusian whatever was said.

**Speed:** 0.16 s for a 15.0-second clip on a 16-core CPU with 12 threads —
about 44 times faster than whisper.cpp large-v3 on the same machine.

## How it works

- **Model:** [stt_be_fastconformer_hybrid_large_pc](https://huggingface.co/nvidia/stt_be_fastconformer_hybrid_large_pc)
  by NVIDIA (CC BY 4.0), a 115 M-parameter FastConformer trained on Mozilla Common Voice Belarusian, in the
  [ONNX export by OpenVoiceOS](https://huggingface.co/OpenVoiceOS/stt_be_fastconformer_hybrid_large_pc_onnx), run by
  [onnx-asr](https://github.com/istupakov/onnx-asr) and ONNX Runtime.
- **Long audio:** clips up to 20 s are decoded whole (that measured better); longer audio is cut into speech segments of
  at most 15 s by [Silero VAD](https://github.com/snakers4/silero-vad).
- **Supply chain:** the model files are fetched on first use at a pinned revision and checked by SHA-256 before they are
  loaded; nothing is redistributed.
- **Benchmark:** `belarusian-asr bench` downloads FLEURS Belarusian at a pinned revision, scores one recording per
  sentence and resumes if it is stopped.

## Accuracy

FLEURS Belarusian test split, one recording per sentence, on an AMD Ryzen AI Max+ 395 (16 cores) with 12 threads per
system. The first 200 test sentences, both systems on the same clips:

| system | CER | WER | CER, no digits | WER, no digits | s per clip |
|---|---|---|---|---|---|
| **belarusian-asr** | **6.3 %** | **14.1 %** | **2.9 %** | **10.6 %** | **0.16** |
| whisper.cpp large-v3 | 10.4 % | 43.1 % | 10.6 % | 43.9 % | 7.09 |

All 349 test sentences: CER 6.1 %, and 2.8 % on the 280 without digits.
"No digits" leaves out references with numbers: FLEURS writes "4892 м", this model says the number in words, which is
right but scores as wrong. By clip length, method and how to reproduce: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Roadmap

1. Real-world audio: an evaluation set of phone, video and conversational Belarusian, with native speakers checking
   transcripts — FLEURS is read speech in quiet rooms.
2. Numbers and dates written as digits (inverse text normalisation), as an option.
3. Word timestamps, and streaming for live captions.
4. A Hugging Face Space where anyone can try their own recording.

Contributions, especially recordings and corrections from Belarusian speakers, are welcome:
[CONTRIBUTING.md](CONTRIBUTING.md).

## Contributors

<!-- readme: contributors,bots/- -start -->
<p align="center">
  <a href="https://github.com/YauhenBichel" title="Yauhen Bichel" aria-label="Yauhen Bichel"><img src=".github/faces/YauhenBichel.svg" width="87" height="99" alt="Yauhen Bichel" /></a>
</p>
<!-- readme: contributors,bots/- -end -->

## Licences

Code: Apache-2.0 ([LICENSE](LICENSE)). The model, downloaded from its authors at first use, is **CC BY 4.0** (NVIDIA):
an application built on it must credit NVIDIA. Silero VAD is MIT. FLEURS — the benchmark data and the demo clips — is
CC BY 4.0 (Conneau et al., 2022). Details in [NOTICE](NOTICE); how to cite: [CITATION.cff](CITATION.cff).

## Disclaimer

Provided as is, without warranty. Transcripts can be wrong, especially for names, numbers and noisy audio. Do not use
them where a mistake could cause harm; this is not a medical, legal or accessibility-certified product.
