# UVT

Active project: `/Users/teotim/Documents/голос`. Work only in this existing checkout. Python 3.10+, aiohttp server, native PySide6 GUI, vanilla JavaScript Tampermonkey userscript.

`src/uvt/app.py` runs live capture → VAD → STT → translation → TTS. `dub.py` renders files; `server.py` serializes browser jobs and serves progressive clips; `server_workspace.py` provides bounded uploads and exports; `src/uvt/web/` is the file workspace. `server_settings.py` owns validated persistent dashboard defaults. `workspace_video.py` retains video originals and completed manifests, synchronizes playback with clean dubbing in the workspace, and exports the chosen audio mix.

M4 default: Parakeet MLX v3 → TranslateGemma 4B 4bit → Piper RU/UK. `setup_local.py` pins artifacts and checks cache offline. Jobs do not download weights. Heavy natural profile adds Demucs/Qwen/F5. Selectable local-hymt/local-moss/local-nemotron add pinned Hy-MT2 MLX, MOSS ONNX and Nemotron MLX; weights stay in ignored .models/. MOSS supports Russian but not Ukrainian. `local-meeting` uses local subtitles and requires an explicitly configured capture device.

Use `.venv/bin/python -m pytest` with `PYTHONPATH=src`; ffmpeg tests can use `scripts/with-local-runtime.sh`. Ordinary CLI media commands automatically validate/repair the existing compatible x265 library through `media_runtime.py` (process environment only).

Scope, verification and follow-up: `TODO_LOCAL_M4.md`. Research: `docs/LOCAL_MODELS_M4.md`. Measured model results: `artifacts/benchmark/BENCHMARK.md`. Risks: model residency/cancellation, full-duration audio buffers, browser auth/origin, download cache and media export. Do not infer GPU speed from dummy-engine tests.

Global gender voice pairs are scoped by TTS engine and target language in server settings. voice_references.py stores private F5/MOSS samples; MOSS exposes Adam/Nathan/Ava/Bella. Offline HyMT/MLX/OpenAI translation receives bounded before/after source context; live remains past-only. Shared reference TTS serializes each replica role without duplicating weights. Verification: artifacts/verification/voice-selection/.

Userscript 0.18.3: all visible video players are eligible again (preview filtering explicitly reverted by user); idle controls fully fade and stop hit-testing, movement wakes them. Profile/language/voice overrides are video-local and inherit server defaults; they apply to the next job. Browser fixtures: tests/userscript_main_player.cjs and tests/userscript_video_settings.cjs.

Userscript 0.18.5 uses GM_xmlhttpRequest/GM.xmlHttpRequest for UVT JSON and compressed audio (localhost-only @connect by default), abortable requests and no automatic POST retry; completed audio uses a revocable Blob URL. Installer update is required for new grants. See tests/userscript_transport.cjs and tests/userscript_completed_audio.cjs. Installed extension inventory is unavailable through current browser tool policy.

MOSS uses bounded same-reference recovery on output limits; fatal_tts/user_message stops the current dub when exhausted. Tests: test_tts_moss_onnx.py and test_dub_moss_failure.py. Real recovery smoke waits for idle user jobs; never run a second local model alongside active dubbing.

Per-video profile metadata must explicitly use settings_mode=override, including when global settings are saved; empty local voice_id is a string, not null. See tests/userscript_video_settings.cjs and the per-request server compatibility regression tests.

media_discovery.py also handles bunkr.cr/f pages via dedicated literal config and the pinned public CDN signer, within the existing discovery deadline. No page JS execution or browser cookies. Tests: tests/test_bunkr_discovery.py; live verification was URL discovery + HEAD200, not full media/dubbing.
