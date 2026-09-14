# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""An HTTP server that speaks two transcription dialects.

  GET  /health
  POST /v1/audio/transcriptions   OpenAI: multipart `file`, `language`, `response_format` (json, text, verbose_json)
  POST /inference                 whisper.cpp's server: the same fields, so a client of whisper.cpp can switch

Belarusian (no `language`, or be / bel / belarusian, any case, with or without a region) is transcribed here. Any other
language goes to `fallback_url`, a whisper.cpp server, when one is set; otherwise the request is refused, because
this model knows only Belarusian and would answer in Belarusian letters whatever was said.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from . import __version__
from .audio import AudioError
from .models import FASTCONFORMER
from .transcriber import Transcriber

MAX_BYTES = 25 * 1024 * 1024
FORMATS = ("json", "text", "verbose_json")
BELARUSIAN = {"be", "bel", "belarusian"}


def language_code(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip().lower().replace("_", "-").split("-")[0]


def is_belarusian(value: str | None) -> bool:
    code = language_code(value)
    return code is None or code in BELARUSIAN


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}}, status_code=status)


def create_app(
    transcriber: Transcriber | None = None,
    fallback_url: str | None = None,
    max_bytes: int = MAX_BYTES,
    client: httpx.AsyncClient | None = None,
) -> Starlette:
    engine = transcriber or Transcriber()
    lock = asyncio.Semaphore(1)  # one decode at a time: predictable CPU use and latency
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok", "version": __version__, "model": FASTCONFORMER.source,
                             "fallback": bool(fallback_url)})

    async def forward(form: Any, upload: UploadFile, data: bytes) -> Response:
        fields = {k: v for k, v in form.multi_items() if isinstance(v, str)}
        files = {"file": (upload.filename or "audio", data, upload.content_type or "application/octet-stream")}
        try:
            up = await http.post(fallback_url.rstrip("/") + "/inference", data=fields, files=files)
        except httpx.HTTPError as exc:
            return error(502, f"fallback transcription server unavailable: {type(exc).__name__}")
        out = Response(up.content, status_code=up.status_code, media_type=up.headers.get("content-type"))
        out.headers["x-belarusian-asr-route"] = f"fallback: language '{language_code(fields.get('language'))}'"
        return out

    async def transcribe(request: Request) -> Response:
        try:
            form = await request.form(max_files=1, max_part_size=max_bytes)
        except Exception as exc:  # starlette raises several kinds for malformed multipart
            return error(400, f"expected multipart/form-data with a 'file' part: {exc}")
        try:
            return await handle(form)
        finally:
            await form.close()  # the upload's spooled temporary file

    async def handle(form: Any) -> Response:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            return error(400, "multipart field 'file' (the audio) is required")
        data = await upload.read()
        if not data:
            return error(400, "the audio file is empty")
        if len(data) > max_bytes:
            return error(413, f"audio file over {max_bytes // (1024 * 1024)} MB")
        response_format = str(form.get("response_format") or "json")
        if response_format not in FORMATS:
            return error(400, f"response_format '{response_format}' is not supported; use json, text or verbose_json")
        language = form.get("language") if isinstance(form.get("language"), str) else None
        if not is_belarusian(language):
            if fallback_url:
                return await forward(form, upload, data)
            return error(400, f"this server transcribes Belarusian only; language '{language_code(language)}' "
                              "needs a fallback server (belarusian-asr serve --fallback URL)")
        async with lock:
            try:
                result = await run_in_threadpool(engine.transcribe, data)
            except AudioError as exc:
                return error(400, str(exc))
        headers = {"x-belarusian-asr-route": "fastconformer"}
        if response_format == "text":
            return PlainTextResponse(result.text + "\n", headers=headers)
        if response_format == "verbose_json":
            segments = [{"id": i, "start": s.start, "end": s.end, "text": s.text} for i, s in enumerate(result.segments)]
            return JSONResponse({"task": "transcribe", "language": "belarusian", "duration": result.duration,
                                 "text": result.text, "segments": segments}, headers=headers)
        return JSONResponse({"text": result.text}, headers=headers)

    return Starlette(routes=[
        Route("/health", health, methods=["GET"]),
        Route("/v1/audio/transcriptions", transcribe, methods=["POST"]),
        Route("/inference", transcribe, methods=["POST"]),
    ])
