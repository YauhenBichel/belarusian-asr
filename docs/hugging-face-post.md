# Hugging Face post

For huggingface.co/posts, which allows about 2,000 characters. The numbers come from docs/BENCHMARKS.md (FLEURS
Belarusian test set, the first 200 recordings, sentences without numbers). Check them there before reusing the post,
and publish the package to PyPI first, because the post says `pip install`.

---

**belarusian-asr: Belarusian speech to text with 4 times fewer wrong words than Whisper large-v3**

Whisper large-v3 gets more than four words in ten wrong in Belarusian. Voxtral Mini did not even write Belarusian — it translated my recording into English.

So I packaged NVIDIA's Belarusian speech model ([stt_be_fastconformer_hybrid_large_pc](https://huggingface.co/nvidia/stt_be_fastconformer_hybrid_large_pc), CC BY 4.0) into a small Python package. It runs on a normal CPU. No GPU, no PyTorch.

Same 200 recordings from the FLEURS Belarusian test set, same computer:

```
                   wrong letters   wrong words   time
belarusian-asr        2.9 %          10.6 %     0.16 s
Whisper large-v3     10.6 %          43.9 %     7.09 s
```

That is 3.7 times fewer wrong letters, 4.1 times fewer wrong words, and 44 times faster.

```bash
pip install belarusian-asr
belarusian-asr transcribe clip.wav
```

It also has a server that works with OpenAI and whisper.cpp clients, and one command that repeats the whole test.

On the demo page you can listen to the recordings and see both texts side by side, with the mistakes marked. I also show the one recording where Whisper does better.

What it cannot do yet: it was tested on read speech, it writes numbers as words, and it knows only Belarusian. I need real life recordings — calls, videos, conversations — and Belarusian speakers to tell me where it fails.

Калі вы размаўляеце па-беларуску — паспрабуйце і напішыце, дзе памыляецца.

https://github.com/YauhenBichel/belarusian-asr
https://yauhenbichel.github.io/belarusian-asr/
