# Changelog

## 0.1.0 (2026-09-14)

- `Transcriber`: Belarusian speech to text on the CPU with NVIDIA's Belarusian FastConformer (CC BY 4.0) through
  ONNX Runtime; WAV, FLAC, OGG and MP3; clips over 20 s cut by Silero VAD.
- Numbers are written as digits ("у 1963 годзе", "400 000", "53-гадовы"): `belarusian_asr.digits` reads every case
  form of Belarusian cardinals and ordinals. `Transcriber(digits=False)` and `belarusian-asr --words` keep the words.
- Model files pinned by revision and checked by SHA-256.
- `belarusian-asr transcribe | serve | download | bench`.
- Server: OpenAI `/v1/audio/transcriptions` and whisper.cpp `/inference` (json, text, verbose_json), a whisper.cpp
  fallback for other languages.
- FLEURS Belarusian benchmark, pinned and resumable. Test split: CER 4.8 %, 3.0 % without digits; against
  whisper.cpp large-v3 on 200 sentences, 4.7 % against 10.4 % (3.1 % against 10.6 % without digits).
- Demo page with six FLEURS clips side by side with whisper.cpp.
