# Changelog

## 0.1.0 (2026-09-14)

- `Transcriber`: Belarusian speech to text on the CPU with NVIDIA's Belarusian FastConformer (CC BY 4.0) through
  ONNX Runtime; WAV, FLAC, OGG and MP3; clips over 20 s cut by Silero VAD.
- Model files pinned by revision and checked by SHA-256.
- `belarusian-asr transcribe | serve | download | bench`.
- Server: OpenAI `/v1/audio/transcriptions` and whisper.cpp `/inference` (json, text, verbose_json), a whisper.cpp
  fallback for other languages.
- FLEURS Belarusian benchmark, pinned and resumable. Test split: CER 6.1 %, 2.8 % without
  digits; against whisper.cpp large-v3 on 200 sentences, 2.9 % against 10.6 %.
- Demo page with six FLEURS clips side by side with whisper.cpp.
