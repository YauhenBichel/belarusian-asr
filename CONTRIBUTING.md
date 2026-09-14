# Contributing

Native speakers are the best reviewers of a speech recognizer: a wrong ending, a dropped `ў` or a name heard
wrongly is easy to hear and hard for a benchmark to explain.

## Reporting a wrong transcription

Open an issue with what was said, what came back, and, if you may share it, the audio. Say where the audio came from
(phone, microphone, video) and how long it is: the model is strongest on clips under 15 seconds.

## Changing the code

```bash
pip install -e ".[server]"
python -m unittest discover -s tests
```

The unit tests need no model and no network. Every behaviour change needs a test.

A change that could move accuracy comes with a benchmark run, before and after, on the same split and limit:

```bash
belarusian-asr bench --split test --limit 200 --out bench
```

and the table from `bench/fleurs-be-test.md` in the pull request.

## Model files

Model files are pinned by revision and SHA-256 in `src/belarusian_asr/models.py`. Changing a pin needs the new hashes
and a benchmark run showing the change is not worse.
