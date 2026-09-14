# Security

This package downloads model files from Hugging Face, reads audio files, and, with `belarusian-asr serve`, runs an
HTTP server. The server binds to 127.0.0.1 unless told otherwise and has no authentication: put it behind something
that has, before you expose it.

## Reporting a problem

Please report vulnerabilities privately through GitHub's "Report a vulnerability" button on the Security tab of
this repository, rather than in a public issue. You can expect an acknowledgement within a week.

## What counts

- A crafted audio file or request that causes code execution, or reads or writes outside the process.
- A way to make the package load a model file other than the pinned revision whose SHA-256 it checks.
- A request that makes the server forward to anywhere but the configured fallback.

A wrong transcription is a bug, not a vulnerability: please open a normal issue, ideally with the audio (if you may
share it) and what was said.
