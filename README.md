# belarusian-asr

[![CI](https://github.com/YauhenBichel/belarusian-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/YauhenBichel/belarusian-asr/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/belarusian-asr.svg)](https://pypi.org/project/belarusian-asr/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![language: Belarusian](https://img.shields.io/badge/language-беларуская-c8313e)
[![Demo](https://img.shields.io/badge/demo-listen%20and%20compare-c8313e)](https://yauhenbichel.github.io/belarusian-asr/)

**Belarusian speech to text.** It runs on a normal computer: no GPU needed.

**Беларускае маўленне ў тэкст.** Працуе на звычайным камп'ютары, без відэакарты.

## Better than Whisper for Belarusian

We tested both on the same 200 Belarusian recordings (FLEURS test set), on the same computer.

| | Wrong letters | Wrong words | Time per recording |
|---|---|---|---|
| **belarusian-asr** | **4.7 %** | **12.9 %** | **0.19 s** |
| Whisper large-v3 | 10.4 % | 43.1 % | 7.09 s |

- **2.2 times fewer wrong letters.**
- **3.3 times fewer wrong words.** Whisper gets more than four words in ten wrong; belarusian-asr gets about one in eight.
- **37 times faster.**

On the 163 sentences without numbers the gap is bigger: 3.1 % wrong letters against 10.6 %, and 10.8 % wrong words
against 43.9 % (4.1 times fewer). All the numbers: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Demo

**[Listen and compare](https://yauhenbichel.github.io/belarusian-asr/)**: six recordings, with what was said, what
belarusian-asr wrote and what Whisper wrote. The mistakes are marked. One of the six is a recording where Whisper does
better.

Two of them:

| | |
|---|---|
| [▶ Listen: FLEURS 1894](https://yauhenbichel.github.io/belarusian-asr/audio/fleurs-1894.ogg) (9.7 s) | |
| What was said | Кары ўяўляе сабой страву з мяса ці гародніны, якая прыпраўлена травамі і спецыямі. |
| **belarusian-asr** (0.0 % wrong letters) | Кары ўяўляе сабой страву з мяса ці гародніны, якая прыпраўлена травамі і спецыямі. |
| Whisper large-v3 (7.5 % wrong letters) | Кары уяуляе сабой страву з мяса ці гародніны, якая приправляна травамі і спеціямі. |
| [▶ Listen: FLEURS 1742](https://yauhenbichel.github.io/belarusian-asr/audio/fleurs-1742.ogg) (10.2 s) | |
| What was said | Назва краіны Ганконг пазычана ў вострава Ганконг, які з'яўляецца цэнтрам прыцягнення для многіх турыстаў. |
| **belarusian-asr** (3.9 % wrong letters) | Назва краіны ганко пазычана ў вострава ганко, які з'яўляецца цэнтрам прыцягнення для многіх турыстаў. |
| Whisper large-v3 (15.5 % wrong letters) | Назва країны Гангкок пазычана ў острова Гангкок, які зляуляюцца центрам прытягнення для многих турыстав. |

## Why

Popular speech models are bad at Belarusian:

- **Whisper large-v3** gets more than four words in ten wrong.
- **Voxtral Mini** did not write Belarusian at all. It translated the speech into English or Russian.
- The good Belarusian models are hidden inside big research toolkits that need PyTorch and often a GPU.

This project started when a Belarusian recording came back from an AI tool translated into English.

## Use

You need Python 3.11 or newer, on Linux, macOS or Windows.

```bash
pip install belarusian-asr
belarusian-asr transcribe clip.wav          # WAV, FLAC, OGG or MP3
```

The model (about 460 MB) downloads the first time.

Numbers come out as digits, the way Belarusian is written: "у 1963 годзе", "400 000", "53-гадовы". To keep them as
spoken words, use `belarusian-asr --words transcribe clip.wav` or `Transcriber(digits=False)`.

In Python:

```python
from belarusian_asr import Transcriber

asr = Transcriber()                          # load it once and reuse it
print(asr.transcribe("clip.wav").text)
```

As a server, for any app that talks to OpenAI or whisper.cpp:

```bash
pip install "belarusian-asr[server]"
belarusian-asr serve --port 11805
curl -s 127.0.0.1:11805/v1/audio/transcriptions -F file=@clip.wav             # OpenAI style
curl -s 127.0.0.1:11805/inference -F file=@clip.wav -F response_format=json   # whisper.cpp style
```

It understands only Belarusian. For other languages, point `--fallback` to a whisper.cpp server.

## How it works

- **The model** is NVIDIA's Belarusian speech model,
  [stt_be_fastconformer_hybrid_large_pc](https://huggingface.co/nvidia/stt_be_fastconformer_hybrid_large_pc)
  (CC BY 4.0). It learned from Mozilla Common Voice recordings. We run the
  [ONNX version made by OpenVoiceOS](https://huggingface.co/OpenVoiceOS/stt_be_fastconformer_hybrid_large_pc_onnx)
  with [onnx-asr](https://github.com/istupakov/onnx-asr) and ONNX Runtime, so there is no PyTorch.
- **Long recordings** (over 20 seconds) are cut at pauses with [Silero VAD](https://github.com/snakers4/silero-vad).
- **Numbers:** the model says numbers in words. Our own converter (`belarusian_asr.digits`) knows every case form of
  Belarusian numbers and writes them as digits. Single small numbers stay words ("адна з", "два дні"). No existing
  tool did this for Belarusian.
- **Safety:** the model files are checked against fixed SHA-256 hashes before they are used.
- **Testing:** `belarusian-asr bench` downloads the FLEURS Belarusian test set and repeats the comparison above.

## Roadmap

1. Test on real life audio: phone calls, videos and conversations, checked by Belarusian speakers.
2. Even better accuracy: we are measuring bigger models. A Belarusian language model did not help on the test set.
3. Word timings and live captions.
4. A Hugging Face Space where you can try your own recording.

Help from Belarusian speakers is very welcome: [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributors

<!-- readme: contributors,bots/- -start -->
<p align="center">
  <a href="https://github.com/YauhenBichel" title="Yauhen Bichel" aria-label="Yauhen Bichel"><img src=".github/faces/YauhenBichel.svg" width="87" height="99" alt="Yauhen Bichel" /></a>
</p>
<!-- readme: contributors,bots/- -end -->

## Licences

The code is Apache-2.0 ([LICENSE](LICENSE)). The model is CC BY 4.0 by NVIDIA; if you build an app with it, credit
NVIDIA. Silero VAD is MIT. The FLEURS test recordings and the demo clips are CC BY 4.0 (Conneau et al., 2022). More in
[NOTICE](NOTICE). To cite this project: [CITATION.cff](CITATION.cff).

## Disclaimer

No warranty. The text can be wrong, most often with names, numbers and noisy audio. Do not use it where a mistake could
hurt someone. It is not a medical, legal or accessibility-certified product.
