# Hugging Face post

Written for huggingface.co/posts, which caps a post at about 2,000 characters. The numbers come from
docs/BENCHMARKS.md (FLEURS Belarusian test, the first 200 sentences, references without digits); re-check them there
whenever the post is reused, and publish the package to PyPI before posting, since the post tells people to
`pip install` it.

---

**Belarusian speech recognition: Whisper large-v3 gets four words in ten wrong. A small model on a CPU gets one in ten.**

I sent a Belarusian recording through my local AI gateway and it came back in English. Voxtral had translated it. Whisper large-v3 does transcribe Belarusian, but on FLEURS it gets about 10.6 % of characters and 43.9 % of words wrong.

Meanwhile NVIDIA trained a Belarusian FastConformer on Common Voice ([stt_be_fastconformer_hybrid_large_pc](https://huggingface.co/nvidia/stt_be_fastconformer_hybrid_large_pc), CC BY 4.0) — 115 M parameters, hidden inside NeMo. So I packaged it: **belarusian-asr**, ONNX Runtime only, no PyTorch, no GPU.

FLEURS Belarusian test, same 200 sentences, same CPU:

```
                     CER     WER    s/clip
belarusian-asr       2.9 %   10.6 %  0.16
whisper large-v3    10.6 %   43.9 %  7.09
```

(references without digits: FLEURS writes numbers as digits, this model says them in words)

```bash
pip install "belarusian-asr[server]"
belarusian-asr transcribe clip.wav
belarusian-asr serve   # OpenAI + whisper.cpp compatible API
```

The benchmark is one command, pinned and reproducible, and the demo page lets you listen to the clips and see both transcripts side by side — including the one where Whisper wins.

Honest limits: read speech only so far, numbers come out as words, Belarusian only. What I need most is real-world audio — phone calls, videos, conversation — and native speakers to tell me where it fails.

Калі вы размаўляеце па-беларуску — паспрабуйце і напішыце, дзе памыляецца.

https://github.com/YauhenBichel/belarusian-asr
https://yauhenbichel.github.io/belarusian-asr/
