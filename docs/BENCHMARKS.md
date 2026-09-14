# Benchmarks

All numbers are from `belarusian-asr bench`, on AMD Ryzen AI Max+ 395 (16 cores, 32 threads, AVX-512), Linux, ONNX Runtime 1.30, 12 threads per system.

## Method

- **Data:** FLEURS Belarusian (`google/fleurs`, `be_by`), CC BY 4.0, pinned to revision `70bb2e84`; the audio archives are checked by SHA-256 and the transcript files by their git blob id.
- **One recording per sentence:** FLEURS reads each sentence two or three times; only the first available recording is scored, so no sentence counts twice.
- **Scores:** corpus CER and WER (total edits over total length), after lower-casing, removing punctuation and stress marks, and unifying apostrophes.
- **Numbers:** FLEURS mostly writes numbers as digits ("4892 м"). The model says them in words ("чатыры тысячы восемсот дзевяноста два"); belarusian-asr writes those as digits (`belarusian_asr.digits`), which is what these tables score. FLEURS is not consistent: some references keep numbers as words ("восемдзесят працэнтаў"), and a number written differently from the reference counts as wrong either way. So every table also shows the sentences whose reference has no digits.
- **Speed:** wall-clock seconds per clip, including audio decoding; real-time factor is processing time over audio duration.

## FLEURS Belarusian test, all 349 sentences

15.2 s of audio per sentence on average; 280 of them have no digits.

| system | CER | WER | CER, no digits | WER, no digits | s per clip | real-time factor |
|---|---|---|---|---|---|---|
| **belarusian-asr (FastConformer)** | **4.75 %** | **13.19 %** | **2.96 %** | **10.76 %** | **0.20** | **0.013** |

CER by clip length (sentences in brackets):

| system | 0-10s | 10-15s | 15-20s | 20-25s | 25s+ |
|---|---|---|---|---|---|
| belarusian-asr (FastConformer) | 3.34 % (67) | 4.44 % (122) | 4.14 % (108) | 6.57 % (35) | 6.94 % (17) |

Before numbers were written as digits (0.1.0 drafts): CER 6.13 %, WER 14.16 %; without digits 2.80 % and 10.62 %.

## Against whisper.cpp large-v3: the first 200 test sentences

Same clips for both systems, 15.0 s each on average, 163 without digits. whisper.cpp v1.9.4, `ggml-large-v3.bin`, `-l be`, CPU build.

| system | CER | WER | CER, no digits | WER, no digits | s per clip | real-time factor |
|---|---|---|---|---|---|---|
| **belarusian-asr (FastConformer)** | **4.70 %** | **12.94 %** | **3.10 %** | **10.78 %** | **0.19** | **0.013** |
| whisper.cpp large-v3 | 10.45 % | 43.06 % | 10.64 % | 43.89 % | 7.09 | 0.474 |

CER by clip length:

| system | 0-10s | 10-15s | 15-20s | 20-25s | 25s+ |
|---|---|---|---|---|---|
| belarusian-asr (FastConformer) | 4.22 % (40) | 4.99 % (73) | 3.57 % (56) | 6.36 % (19) | 5.79 % (12) |

whisper.cpp's hypotheses and times are from its run for 0.1.0; its output does not depend on belarusian-asr. Before
numbers were written as digits, belarusian-asr scored CER 6.34 % and WER 14.08 % here (2.89 % and 10.64 % without
digits). Numbers as digits help the 37 sentences with digits (CER 19.0 % to 10.6 %) and cost a little on the others,
where FLEURS sometimes writes numbers as words.
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
