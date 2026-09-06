# UVT

Active project: `/Users/teotim/Documents/голос`. Work only in this existing checkout. Python 3.10+, aiohttp server, native PySide6 GUI, vanilla JavaScript Tampermonkey userscript.

`src/uvt/app.py` runs live capture → VAD → STT → translation → TTS. `dub.py` renders files; `server.py` serializes browser jobs and serves progressive clips; `server_workspace.py` provides bounded uploads and exports; `src/uvt/web/` is the file workspace. `server_settings.py` owns validated persistent dashboard defaults.

M4 default: Parakeet MLX v3 → TranslateGemma 4B 4bit → Piper RU/UK. `setup_local.py` pins artifacts and checks cache offline. Jobs do not download weights. Heavy natural profile adds Demucs/Qwen/F5. `local-meeting` uses local subtitles and requires an explicitly configured capture device.

Use `.venv/bin/python -m pytest` with `PYTHONPATH=src`; ffmpeg tests can use `scripts/with-local-runtime.sh`. Ordinary CLI media commands automatically validate/repair the existing compatible x265 library through `media_runtime.py` (process environment only).

Scope, verification and follow-up: `TODO_LOCAL_M4.md`. Research: `docs/LOCAL_MODELS_M4.md`. Measured model results: `artifacts/benchmark/BENCHMARK.md`. Risks: model residency/cancellation, full-duration audio buffers, browser auth/origin, download cache and media export. Do not infer GPU speed from dummy-engine tests.
