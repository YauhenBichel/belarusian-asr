# Benchmarks

All numbers are from `belarusian-asr bench`, on AMD Ryzen AI Max+ 395 (16 cores, 32 threads, AVX-512), Linux, ONNX Runtime 1.30, 12 threads per system.

## Method

- **Data:** FLEURS Belarusian (`google/fleurs`, `be_by`), CC BY 4.0, pinned to revision `70bb2e84`; the audio archives are checked by SHA-256 and the transcript files by their git blob id.
- **One recording per sentence:** FLEURS reads each sentence two or three times; only the first available recording is scored, so no sentence counts twice.
- **Scores:** corpus CER and WER (total edits over total length), after lower-casing, removing punctuation and stress marks, and unifying apostrophes.
- **No digits:** FLEURS writes numbers as digits ("4892 м"). belarusian-asr says them in words ("чатыры тысячы восемсот дзевяноста два"), which is right but scored as wrong, so every table also shows the sentences whose reference has no digits.
- **Speed:** wall-clock seconds per clip, including audio decoding; real-time factor is processing time over audio duration.

## FLEURS Belarusian test, all 349 sentences

15.2 s of audio per sentence on average; 280 of them have no digits.

| system | CER | WER | CER, no digits | WER, no digits | s per clip | real-time factor |
|---|---|---|---|---|---|---|
| **belarusian-asr (FastConformer)** | **6.13 %** | **14.16 %** | **2.80 %** | **10.62 %** | **0.16** | **0.01** |

CER by clip length (sentences in brackets):

| system | 0-10s | 10-15s | 15-20s | 20-25s | 25s+ |
|---|---|---|---|---|---|
| belarusian-asr (FastConformer) | 4.40 % (67) | 5.67 % (122) | 5.33 % (108) | 8.99 % (35) | 8.31 % (17) |

## Against whisper.cpp large-v3: the first 200 test sentences

Same clips for both systems, 15.0 s each on average, 163 without digits. whisper.cpp v1.9.4, `ggml-large-v3.bin`, `-l be`, CPU build.

| system | CER | WER | CER, no digits | WER, no digits | s per clip | real-time factor |
|---|---|---|---|---|---|---|
| **belarusian-asr (FastConformer)** | **6.34 %** | **14.08 %** | **2.89 %** | **10.64 %** | **0.16** | **0.011** |
| whisper.cpp large-v3 | 10.45 % | 43.06 % | 10.64 % | 43.89 % | 7.09 | 0.474 |

CER by clip length:

| system | 0-10s | 10-15s | 15-20s | 20-25s | 25s+ |
|---|---|---|---|---|---|
| belarusian-asr (FastConformer) | 3.89 % (40) | 6.90 % (73) | 4.61 % (56) | 10.50 % (19) | 7.38 % (12) |
| whisper.cpp large-v3 | 9.66 % (40) | 9.63 % (73) | 8.89 % (56) | 9.26 % (19) | 20.03 % (12) |

## Earlier: Voxtral Mini 3B

On 5 FLEURS dev sentences through an OpenAI-compatible gateway, Voxtral Mini 3B (a multilingual audio LLM) answered Belarusian speech in English or Russian: CER 89.6 %, WER 99.9 %.

## Reproduce

```bash
pip install "belarusian-asr[server]"
belarusian-asr --threads 12 bench --split test --out bench/full
belarusian-asr --threads 12 bench --split test --limit 200 --out bench/vs-whisper \
  --whisper-cli ./build/bin/whisper-cli --whisper-model ggml-large-v3.bin --whisper-threads 12
```

Every hypothesis is in `bench/*/fleurs-be-test.json`; a stopped run resumes from its checkpoint.

## Limits of these numbers

- FLEURS is read speech in quiet conditions. Conversation, noise and far-field audio will score worse.
- The model was trained on Common Voice, which is also read speech; FLEURS was not part of its training.
- Long clips are rare in FLEURS (17 over 25 s in the test split), so the long-clip rates are noisy.
