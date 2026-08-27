"""Сервер браузерной кнопки: задача из локального файла → готовая дорожка."""
import asyncio
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from uvt.config import AppConfig, load_config

aiohttp = pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.target_lang = "ru"
    cfg.vad.engine = "energy"
    cfg.latency.preset = "ultra"
    cfg.stt.engine = "dummy"
    cfg.translation.engine = "dummy"
    cfg.tts.engine = "dummy"
    return cfg


def _install_voice_stubs(cfg: AppConfig, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    cfg.tts.voice_dir = str(root)
    for model_name in dict(getattr(cfg.tts, "voice_models", {}) or {}).values():
        model = root / str(model_name)
        model.write_bytes(b"voice")
        Path(f"{model}.json").write_text(
            '{"audio": {"sample_rate": 22050}}', encoding="utf-8"
        )


def test_nllb_is_reported_as_private_local_translation():
    from uvt.server import DubServer

    cfg = _cfg()
    cfg.stt.engine = "mlx-whisper"
    cfg.translation.engine = "nllb-ct2"
    cfg.tts.engine = "piper"
    meta = DubServer(cfg)._metadata()

    assert meta["profile"]["kind"] == "local"
    assert meta["privacy"]["data_leaves_device"] is False


def test_route_metadata_names_actual_profile_and_port():
    from uvt.server import DubServer

    server = DubServer(
        _cfg(),
        route_label="GPT",
        profile_name="cloud-fast",
        listen_port=8766,
    )

    meta = server._metadata()
    assert meta["route"] == {
        "label": "GPT",
        "profile": "cloud-fast",
        "port": 8766,
    }
    assert meta["profile"]["name"] == "cloud-fast"


def test_local_server_exposes_allowlisted_profiles_and_named_voices(monkeypatch, tmp_path):
    from uvt.server import DubServer

    monkeypatch.setenv("UVT_CACHE", str(tmp_path))
    profiles = {
        name: load_config(str(ROOT / "profiles" / f"{name}.yaml"))
        for name in ("local-fast", "local-balanced", "local-quality")
    }
    for cfg in profiles.values():
        _install_voice_stubs(cfg, tmp_path / "voices")
    server = DubServer(
        profiles["local-balanced"],
        profile_name="local-balanced",
        selectable_profiles=profiles,
    )

    meta = server._metadata_for_request(
        {"profile_id": "local-quality", "target_lang": "uk"}
    )

    assert meta["api_version"] == 2
    assert meta["capabilities"]["profile_selection"] is True
    assert meta["capabilities"]["tts_preview"] is True
    assert meta["profile"]["name"] == "local-quality"
    assert meta["profile"]["engines"]["stt"] == "mlx-whisper"
    assert {item["id"] for item in meta["profiles"]} == {
        "local-fast", "local-balanced", "local-quality"
    }
    assert {item["id"] for item in meta["voices"]} == {
        "ru_RU-dmitri-medium",
        "ru_RU-irina-medium",
        "uk_UA-mykyta-high",
        "uk_UA-tetiana-high",
    }
    assert "voice_dir" not in repr(meta)
    assert "/Users/" not in repr(meta)


def test_profile_and_exact_voice_are_part_of_job_cache_key(monkeypatch, tmp_path):
    from uvt.server import DubServer

    monkeypatch.setenv("UVT_CACHE", str(tmp_path))
    profiles = {
        name: load_config(str(ROOT / "profiles" / f"{name}.yaml"))
        for name in ("local-fast", "local-balanced", "local-quality")
    }
    for cfg in profiles.values():
        _install_voice_stubs(cfg, tmp_path / "voices")
    server = DubServer(
        profiles["local-balanced"],
        profile_name="local-balanced",
        selectable_profiles=profiles,
    )
    base = {
        "page_url": "https://example.test/watch/1",
        "target_lang": "ru",
        "voice_gender": "female",
    }

    balanced = server._cache_key({**base, "profile_id": "local-balanced"})
    fast = server._cache_key({**base, "profile_id": "local-fast"})
    irina = server._cache_key(
        {**base, "profile_id": "local-balanced", "voice_id": "ru_RU-irina-medium"}
    )

    assert balanced != fast
    assert balanced != irina
    with pytest.raises(ValueError, match="недоступен"):
        server._cache_key({**base, "profile_id": "../../private-profile"})
    with pytest.raises(ValueError, match="не подходит"):
        server._cache_key(
            {**base, "target_lang": "uk", "voice_id": "ru_RU-irina-medium"}
        )


def test_profile_default_exact_voice_is_preserved(monkeypatch, tmp_path):
    from uvt.server import DubServer

    monkeypatch.setenv("UVT_CACHE", str(tmp_path))
    cfg = load_config(str(ROOT / "profiles" / "local-balanced.yaml"))
    _install_voice_stubs(cfg, tmp_path / "voices")
    cfg.tts.voice_id = "ru_RU-irina-medium"

    selected, _profile = DubServer(
        cfg, profile_name="local-balanced"
    )._config_for_request({"target_lang": "ru"})

    assert selected.tts.voice_id == "ru_RU-irina-medium"
    assert selected.tts.voice_gender == "female"


def test_local_piper_rejects_unsupported_target_before_job(monkeypatch, tmp_path):
    from uvt.server import DubServer

    monkeypatch.setenv("UVT_CACHE", str(tmp_path))
    cfg = load_config(str(ROOT / "profiles" / "local-balanced.yaml"))
    _install_voice_stubs(cfg, tmp_path / "voices")
    server = DubServer(cfg, profile_name="local-balanced")

    meta = server._metadata_for_request({"target_lang": "de"})
    assert meta["capabilities"]["voice_selection"] is False
    assert meta["capabilities"]["tts_preview"] is False
    assert meta["limits"]["local_tts_languages"] == ["ru", "uk"]
    with pytest.raises(ValueError, match="Piper-голоса"):
        server._cache_key(
            {
                "page_url": "https://example.test/watch/unsupported",
                "target_lang": "de",
                "voice_gender": "female",
            }
        )


def test_profile_catalog_disables_missing_optional_models(monkeypatch, tmp_path):
    from uvt.server import DubServer

    monkeypatch.setenv("UVT_CACHE", str(tmp_path))
    profiles = {
        name: load_config(str(ROOT / "profiles" / f"{name}.yaml"))
        for name in ("local-fast", "local-balanced", "local-quality")
    }
    for cfg in profiles.values():
        _install_voice_stubs(cfg, tmp_path / "voices")
    server = DubServer(
        profiles["local-balanced"],
        profile_name="local-balanced",
        selectable_profiles=profiles,
    )
    server._record_local_preflight(
        {
            "ready": False,
            "models": {
                "parakeet": {"ready": True},
                "nllb": {"ready": False},
                "translategemma": {"ready": True},
                "whisper": {"ready": False},
            },
            "piper": {"ready": True},
        }
    )

    installed = {item["id"]: item["installed"] for item in server._profile_catalog()}
    assert installed == {
        "local-fast": False,
        "local-balanced": True,
        "local-quality": False,
    }
    with pytest.raises(ValueError, match="setup-mac-local --preset fast"):
        server._cache_key(
            {
                "page_url": "https://example.test/watch/missing-fast",
                "profile_id": "local-fast",
                "target_lang": "ru",
                "voice_gender": "male",
            }
        )


async def test_prepared_stt_reused_only_for_matching_profile(monkeypatch, tmp_path):
    from uvt.server import DubServer

    monkeypatch.setenv("UVT_CACHE", str(tmp_path))
    profiles = {
        name: load_config(str(ROOT / "profiles" / f"{name}.yaml"))
        for name in ("local-fast", "local-balanced", "local-quality")
    }

    class FakeSTT:
        def __init__(self):
            self.closed = 0

        async def close(self):
            self.closed += 1

    server = DubServer(
        profiles["local-balanced"],
        profile_name="local-balanced",
        selectable_profiles=profiles,
    )
    shared = FakeSTT()
    server._prepared_stt = shared
    assert await server.take_prepared_stt(profiles["local-fast"]) is shared
    assert shared.closed == 0

    different = FakeSTT()
    server._prepared_stt = different
    assert await server.take_prepared_stt(profiles["local-quality"]) is None
    assert different.closed == 1


async def test_local_piper_preview_returns_cached_wav(monkeypatch, tmp_path):
    import uvt.server as server_module
    from uvt.server import DubServer

    monkeypatch.setenv("UVT_CACHE", str(tmp_path))
    cfg = _cfg()
    cfg.tts.engine = "piper"
    cfg.tts.voice_gender = "female"
    calls: list[str] = []

    class FakeTTS:
        async def warmup(self):
            calls.append("warmup")

        async def synthesize(self, text, language):
            calls.append(f"synthesize:{language}:{len(text)}")
            return np.array([0.0, 0.25, -0.25], dtype=np.float32), 22050

        async def close(self):
            calls.append("close")

    monkeypatch.setattr(server_module.registry, "create", lambda *_args: FakeTTS())
    server = DubServer(cfg, profile_name="local-balanced")
    client = TestClient(TestServer(server.app()))
    await client.start_server()
    payload = {
        "text": "Так звучит локальный голос.",
        "target_lang": "ru",
        "voice_gender": "female",
    }
    try:
        first = await client.post("/tts/preview", json=payload)
        assert first.status == 200
        assert first.headers["Content-Type"].startswith("audio/wav")
        assert (await first.read()).startswith(b"RIFF")

        second = await client.post("/tts/preview", json=payload)
        assert second.status == 200
        assert (await second.read()).startswith(b"RIFF")
        assert calls == ["warmup", "synthesize:ru:27", "close"]

        automatic = await client.post(
            "/tts/preview",
            json={**payload, "voice_gender": "auto"},
        )
        assert automatic.status == 422
    finally:
        await client.close()


async def test_local_piper_preview_times_out_and_releases_lock(monkeypatch, tmp_path):
    import uvt.server as server_module
    from uvt.server import DubServer

    monkeypatch.setenv("UVT_CACHE", str(tmp_path))
    monkeypatch.setattr(server_module, "_PREVIEW_TIMEOUT_S", 0.01)
    cfg = _cfg()
    cfg.tts.engine = "piper"
    cfg.tts.voice_gender = "female"
    calls: list[str] = []

    class StuckTTS:
        async def warmup(self):
            calls.append("warmup")

        async def synthesize(self, _text, _language):
            calls.append("synthesize")
            await asyncio.Event().wait()

        async def close(self):
            calls.append("close")

    monkeypatch.setattr(server_module.registry, "create", lambda *_args: StuckTTS())
    server = DubServer(cfg, profile_name="local-balanced")
    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        response = await client.post(
            "/tts/preview",
            json={
                "text": "Проверка таймаута.",
                "target_lang": "ru",
                "voice_gender": "female",
            },
        )
        assert response.status == 504
        assert "не ответил" in await response.text()
        assert calls == ["warmup", "synthesize", "close"]
        assert server._lock.locked() is False
    finally:
        await client.close()


async def test_local_request_waits_for_startup_preflight():
    from uvt.server import DubServer

    cfg = _cfg()
    cfg.stt.engine = "parakeet-mlx"
    server = DubServer(cfg, profile_name="local-balanced")
    pending = asyncio.create_task(asyncio.Event().wait())
    server._prepare_stt_task = pending
    try:
        with pytest.raises(ValueError, match="ещё проверяет локальные модели"):
            server._cache_key({"page_url": "https://example.test/watch/startup"})
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    server._prepare_stt_task = None
    server._model_readiness = {
        "status": "error",
        "phase": "preflight",
        "detail": "offline cache damaged",
    }
    with pytest.raises(ValueError, match="не прошёл локальную проверку"):
        server._cache_key({"page_url": "https://example.test/watch/failed-startup"})


async def test_network_server_fails_closed_without_api_token(monkeypatch):
    from uvt.server import run_server

    monkeypatch.delenv("UVT_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="требует UVT_API_TOKEN"):
        await run_server(_cfg(), host="0.0.0.0", port=18765)


async def test_local_route_preflights_and_preloads_stt(monkeypatch):
    import uvt.server as server_module
    import uvt.setup_local as setup_local
    from uvt.server import DubServer

    cfg = _cfg()
    cfg.stt.engine = "parakeet-mlx"
    cfg.translation.engine = "translategemma-mlx"
    cfg.tts.engine = "piper"
    calls: list[str] = []

    class FakeSTT:
        async def warmup(self):
            calls.append("warmup")

        async def close(self):
            calls.append("close")

    fake = FakeSTT()
    monkeypatch.setattr(server_module, "create_stt_engine", lambda _cfg: fake)
    monkeypatch.setattr(setup_local, "preset_for_profile", lambda _cfg: "balanced")
    monkeypatch.setattr(
        setup_local,
        "preflight_mac_local",
        lambda _preset: {"ready": True, "preset": "balanced"},
    )

    server = DubServer(cfg, profile_name="local-balanced")
    assert server._metadata()["model_readiness"]["status"] == "pending"
    await server.prepare_local_models()
    assert calls == ["warmup"]
    assert server._metadata()["model_readiness"]["status"] == "ready"

    assert await server.take_prepared_stt() is fake
    assert server._metadata()["model_readiness"]["status"] == "in-use"
    await fake.close()
    assert calls == ["warmup", "close"]


def test_source_cache_key_ignores_tracking_but_keeps_page_parameters():
    from uvt.server import _source_cache_key

    clean = _source_cache_key(
        {"page_url": "https://example.test/watch?id=42&chapter=3"}
    )
    tracked = _source_cache_key(
        {
            "page_url": (
                "https://EXAMPLE.test/watch?id=42&utm_source=ad&chapter=3"
                "&fbclid=tracking#player"
            )
        }
    )
    different_video = _source_cache_key(
        {"page_url": "https://example.test/watch?id=43&chapter=3"}
    )

    assert tracked == clean
    assert different_video != clean


async def test_downloaded_source_cache_is_shared_between_personal_routes(
    monkeypatch, tmp_path
):
    import uvt.server as server_module
    from uvt.server import DubServer

    downloads: list[str] = []

    async def fake_download(url, dest_dir, referer=None, out_name="media.m4a", **_kwargs):
        downloads.append(url)
        output = dest_dir / out_name
        output.write_bytes(b"shared source audio")
        return output

    monkeypatch.setattr(server_module, "_download_media", fake_download)
    free = DubServer(_cfg(), route_label="Free", profile_name="free-quality")
    cloud = DubServer(_cfg(), route_label="GPT", profile_name="cloud-fast")
    free.audio_dir = tmp_path
    cloud.audio_dir = tmp_path
    first_workdir = tmp_path / "first"
    second_workdir = tmp_path / "second"
    first_workdir.mkdir()
    second_workdir.mkdir()
    first_request = {
        "page_url": "https://example.test/watch/42?utm_source=one",
        "media_url": "https://cdn.example.test/signed-one.m3u8",
    }
    second_request = {
        "page_url": "https://example.test/watch/42?utm_source=two",
        "media_url": "https://cdn.example.test/signed-two.m3u8",
    }

    first = await free._resolve_cached_source(first_request, first_workdir)
    updates: list[tuple[float | None, str]] = []
    second = await cloud._resolve_cached_source(
        second_request,
        second_workdir,
        progress=lambda fraction, detail: updates.append((fraction, detail)),
    )

    assert downloads == [first_request["media_url"]]
    assert first == second
    assert first.parent == tmp_path / "sources"
    assert updates == [
        (1.0, "исходный звук взят из общего кэша — повторно не скачиваю")
    ]


async def test_dub_job_from_file(tmp_path):
    rate = 16000
    t = np.arange(rate) / rate
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    silence = np.zeros(rate // 2, dtype=np.float32)
    src = tmp_path / "in.wav"
    sf.write(src, np.concatenate([silence, tone, silence, silence]), rate)

    from uvt.server import DubServer

    server = DubServer(_cfg())
    server.audio_dir = tmp_path  # не пишем в общий кэш из тестов
    assert server._safe_endpoint("http://localhost:not-a-port/v1") is None
    # Auto speaker/timbre selection and an explicit male request must not
    # alias to one browser cache entry.
    assert server._cache_key({"file": str(src)}) != server._cache_key(
        {"file": str(src), "voice_gender": "male"}
    )

    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        meta_response = await client.get("/meta")
        assert meta_response.status == 200
        assert meta_response.headers["Access-Control-Allow-Origin"] == "*"
        meta = await meta_response.json()
        assert meta["mode"] == "batch"
        assert meta["capabilities"]["live_translation"] is False
        assert meta["profile"]["engines"] == {
            "stt": "dummy", "translation": "dummy", "tts": "dummy",
        }
        assert meta["privacy"]["data_leaves_device"] is False

        resp = await client.post("/dub", json={"file": str(src), "target_lang": "de"})
        assert resp.status == 200
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
        created = await resp.json()
        job_id = created["id"]
        # Старые поля id/job_url остались, новый контракт не выдаёт batch за live.
        assert created["job_url"] == f"/job/{job_id}"
        assert created["mode"] == "batch"
        assert created["is_live"] is False
        assert created["meta"]["mode"] == "batch"
        assert created["meta"]["profile"]["target_lang"] == "de"
        assert created["status"] == "queued"
        assert created["stage"] == "queue"
        assert created["queue_position"] == 1
        assert created["timing"]["started_at"] is None

        info = None
        for _ in range(200):
            info = await (await client.get(f"/job/{job_id}")).json()
            if info["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.05)
        assert info is not None and info["status"] == "done", info

        # target_lang из запроса дошёл до перевода
        assert info["entries"][0]["translated"] == "HELLO 1 [de]"
        assert info["progress"] == 1.0
        assert info["stage"] == "done"
        assert info["is_live"] is False
        assert info["queue_position"] is None
        assert info["timing"]["finished_at"] is not None
        assert info["timing"]["elapsed_seconds"] >= 0

        audio = await client.get(info["audio_url"])
        assert audio.status == 200
        body = await audio.read()
        assert len(body) > 1000  # настоящий m4a, не пустышка

        # неизвестная задача → 404, но с CORS-заголовком
        missing = await client.get("/job/nope")
        assert missing.status == 404
        assert missing.headers["Access-Control-Allow-Origin"] == "*"
    finally:
        await client.close()


async def test_optional_api_token_protects_jobs_and_audio(monkeypatch, tmp_path):
    """Remote personal deployments can opt into a token without changing localhost DX."""
    monkeypatch.setenv("UVT_API_TOKEN", "personal-secret")
    from uvt.server import DubServer

    server = DubServer(_cfg())
    server.audio_dir = tmp_path
    (tmp_path / "ready.m4a").write_bytes(b"not-a-real-m4a")
    server._audio_access_tokens["ready"] = "audio-secret"
    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        denied = await client.get("/meta")
        assert denied.status == 401
        assert denied.headers["Access-Control-Allow-Origin"] == "*"

        allowed = await client.get("/meta", headers={"X-UVT-Token": "personal-secret"})
        assert allowed.status == 200

        no_audio_token = await client.get("/audio/ready.m4a")
        assert no_audio_token.status == 401
        audio = await client.get("/audio/ready.m4a?access=audio-secret")
        assert audio.status == 200
    finally:
        await client.close()


async def test_restricted_or_html_media_url_gets_actionable_error(monkeypatch, tmp_path):
    """Do not hide a browser denial page behind an opaque ffmpeg exit code."""
    import uvt.server as server_module

    async def failed_process(cmd, *_args, **_kwargs):
        # ffmpeg can leave a partial destination behind on an HTTP denial.
        # It must not later be confused with a successful yt-dlp fallback.
        Path(cmd[-1]).write_bytes(b"partial")
        raise RuntimeError("ffmpeg завершился с ошибкой (код 183)")

    monkeypatch.setattr(server_module, "_run_process", failed_process)
    with pytest.raises(RuntimeError, match="не обходит такие ограничения"):
        await server_module._download_media(
            "https://example.test/private-video.mp4", tmp_path, referer="https://example.test/page"
        )
    assert not (tmp_path / "media.m4a").exists()


async def test_browser_media_sources_precede_page_and_rank_resources(monkeypatch, tmp_path):
    """Audio/HLS from the browser beats a muxed player MP4 before page extraction."""
    import uvt.server as server_module
    from uvt.server import DubServer

    server = DubServer(_cfg())
    direct = "https://cdn.example.test/current-video.mp4"
    manifest = "https://cdn.example.test/hls/master.m3u8"
    video = "https://cdn.example.test/video-1080.mp4"
    attempts: list[tuple[str, str]] = []

    async def fake_media(url, dest_dir, referer=None, out_name="media.m4a"):
        attempts.append(("media", url))
        if url != manifest:
            raise RuntimeError("not the usable stream")
        path = dest_dir / out_name
        path.write_bytes(b"audio")
        return path

    async def unexpected_page(*_args, **_kwargs):
        attempts.append(("page", "unexpected"))
        raise AssertionError("page extractor must run only after browser candidates")

    monkeypatch.setattr(server_module, "_download_media", fake_media)
    monkeypatch.setattr(server_module, "_download_page", unexpected_page)
    resolved = await server._resolve_source(
        {
            "page_url": "https://site.example.test/watch/42",
            "media_url": direct,
            # Deliberately put video before manifest: a manifest must still
            # win over the current muxed MP4, avoiding a full video download.
            "media_candidates": [video, manifest, direct],
        },
        tmp_path,
    )

    assert resolved.is_file()
    assert attempts == [("media", manifest)]


def test_browser_candidate_order_prefers_audio_then_hls_over_muxed_video():
    from uvt.server import _browser_media_candidates

    direct = "https://cdn.example.test/current-video.mp4"
    video = "https://cdn.example.test/video-1080.mp4"
    manifest = "https://cdn.example.test/hls/master.m3u8"
    audio = "https://cdn.example.test/audio/original.m4a"

    assert _browser_media_candidates(
        {"media_url": direct, "media_candidates": [video, manifest, audio]}
    ) == [audio, manifest, direct, video]


def test_browser_candidate_cap_keeps_current_src_as_last_fallback():
    from uvt.server import _browser_media_candidates

    direct = "https://cdn.example.test/current-video.mp4"
    manifests = [f"https://cdn.example.test/hls/stream-{index}.m3u8" for index in range(6)]

    candidates = _browser_media_candidates(
        {"media_url": direct, "media_candidates": manifests}
    )

    assert candidates == [*manifests[:5], direct]


async def test_page_ytdlp_is_fallback_after_all_browser_media_fail(monkeypatch, tmp_path):
    import uvt.server as server_module
    from uvt.server import DubServer

    server = DubServer(_cfg())
    direct = "https://cdn.example.test/current.mp4"
    manifest = "https://cdn.example.test/master.m3u8"
    calls: list[str] = []

    async def failed_media(url, *_args, **_kwargs):
        calls.append(f"media:{url}")
        raise RuntimeError("expired browser token")

    async def page_fallback(url, dest_dir):
        calls.append(f"page:{url}")
        path = dest_dir / "from-page.webm"
        path.write_bytes(b"audio")
        return path

    monkeypatch.setattr(server_module, "_download_media", failed_media)
    monkeypatch.setattr(server_module, "_download_page", page_fallback)
    resolved = await server._resolve_source(
        {
            "page_url": "https://site.example.test/watch/7",
            "media_url": direct,
            "media_candidates": [manifest],
        },
        tmp_path,
    )

    assert resolved.name == "from-page.webm"
    assert calls == [
        f"media:{manifest}",
        f"media:{direct}",
        "page:https://site.example.test/watch/7",
    ]


async def test_page_download_uses_audio_first_ytdlp_selector(monkeypatch, tmp_path):
    """No lowest-quality video fallback while a standalone audio format exists."""
    import uvt.dub as dub_module
    import uvt.server as server_module

    captured: list[str] = []

    async def fake_process(cmd, *_args, **_kwargs):
        captured.extend(cmd)
        (tmp_path / "source.webm").write_bytes(b"audio")

    monkeypatch.setattr(dub_module, "_find_ytdlp", lambda: "yt-dlp")
    monkeypatch.setattr(
        dub_module,
        "_ytdlp_js_args",
        lambda: ["--js-runtimes", "node:/test/node"],
    )
    monkeypatch.setattr(server_module, "_run_process", fake_process)
    result = await server_module._download_page("https://site.example.test/watch/99", tmp_path)

    assert result.name == "source.webm"
    assert captured[captured.index("-f") + 1] == server_module._YT_DLP_AUDIO_SELECTOR
    assert "--no-playlist" in captured
    assert "--ignore-config" in captured
    assert captured[captured.index("--js-runtimes") + 1] == "node:/test/node"
    assert "--merge-output-format" not in captured


async def test_youtube_403_retries_with_compatible_android_client(monkeypatch, tmp_path):
    import uvt.dub as dub_module
    import uvt.server as server_module

    calls: list[list[str]] = []

    async def fake_process(cmd, *_args, on_line=None, **_kwargs):
        calls.append(cmd)
        assert on_line is not None
        if len(calls) == 1:
            (tmp_path / "partial.webm.part").write_bytes(b"partial")
            on_line("ERROR: unable to download video data: HTTP Error 403: Forbidden")
            raise RuntimeError("yt-dlp завершился с ошибкой (код 1)")
        (tmp_path / "source.mp4").write_bytes(b"audio")

    monkeypatch.setattr(dub_module, "_find_ytdlp", lambda: "yt-dlp")
    monkeypatch.setattr(dub_module, "_ytdlp_js_args", lambda: [])
    monkeypatch.setattr(server_module, "_run_process", fake_process)

    result = await server_module._download_page(
        "https://www.youtube.com/watch?v=public-video",
        tmp_path,
        progress=lambda *_args: None,
    )

    assert result.name == "source.mp4"
    assert len(calls) == 2
    assert calls[1][calls[1].index("--extractor-args") + 1] == (
        "youtube:player_client=android"
    )
    assert not (tmp_path / "partial.webm.part").exists()


async def test_page_download_surfaces_original_ytdlp_error(monkeypatch, tmp_path):
    import uvt.dub as dub_module
    import uvt.server as server_module

    async def failed_process(_cmd, *_args, on_line=None, **_kwargs):
        assert on_line is not None
        on_line("WARNING: JavaScript runtime is unavailable")
        on_line("ERROR: [youtube] video is unavailable")
        raise RuntimeError("yt-dlp завершился с ошибкой (код 1)")

    monkeypatch.setattr(dub_module, "_find_ytdlp", lambda: "yt-dlp")
    monkeypatch.setattr(dub_module, "_ytdlp_js_args", lambda: [])
    monkeypatch.setattr(server_module, "_run_process", failed_process)

    with pytest.raises(RuntimeError, match="video is unavailable"):
        await server_module._download_page(
            "https://site.example.test/watch/unavailable",
            tmp_path,
            progress=lambda *_args: None,
        )


async def test_direct_media_download_parses_ffmpeg_progress(monkeypatch, tmp_path):
    import uvt.server as server_module

    captured: list[str] = []
    updates: list[tuple[float | None, str]] = []

    async def fake_process(cmd, *_args, on_line=None, **_kwargs):
        captured.extend(cmd)
        assert on_line is not None
        on_line("out_time=00:00:05.000000")
        on_line("progress=continue")
        Path(cmd[-1]).write_bytes(b"a" * 10_001)

    monkeypatch.setattr(server_module, "_run_process", fake_process)
    result = await server_module._download_media(
        "https://cdn.example.test/media.m3u8",
        tmp_path,
        duration_hint=10.0,
        progress=lambda fraction, detail: updates.append((fraction, detail)),
    )

    assert result.is_file()
    assert ["-progress", "pipe:1", "-nostats"] == captured[
        captured.index("-progress") : captured.index("-progress") + 3
    ]
    assert updates[0][0] == 0.0
    assert any(fraction == pytest.approx(0.5) for fraction, _detail in updates)
    assert updates[-1] == (1.0, "исходный звук получен")


def test_download_progress_is_visible_in_job_payload_and_render_stays_monotonic():
    from uvt.server import DubServer, Job

    server = DubServer(_cfg())
    job = Job(id="progress")
    server._set_download_progress(job, 0.5, "поток 1 из 1: получаю исходный звук: 0:05")
    payload = server._job_payload(job)

    assert payload["stage"] == "download"
    assert payload["stage_progress"] == pytest.approx(0.5)
    assert payload["detail"].startswith("поток 1")
    assert payload["progress"] == pytest.approx(0.04)

    server._set_render_progress(job, 0, 100)
    assert job.stage == "transcribe"
    assert job.progress == pytest.approx(0.08)


def test_download_progress_logs_only_at_five_percent_buckets(caplog):
    from uvt.server import DubServer, Job

    server = DubServer(_cfg())
    job = Job(id="terminal-progress")
    caplog.set_level("INFO", logger="uvt.server")
    for fraction in (0.01, 0.049, 0.051, 0.099, 0.10):
        server._set_download_progress(job, fraction, "получаю исходный звук")

    messages = [record.getMessage() for record in caplog.records if "получение звука" in record.getMessage()]
    assert len(messages) == 3
    assert "0%" in messages[0]
    assert "5%" in messages[1]
    assert "10%" in messages[2]


def test_download_progress_without_duration_logs_elapsed_periodically(caplog, monkeypatch):
    import uvt.server as server_module
    from uvt.server import DubServer, Job

    server = DubServer(_cfg())
    job = Job(id="elapsed-progress")
    caplog.set_level("INFO", logger="uvt.server")
    ticks = iter((100.0, 108.0, 115.0))
    monkeypatch.setattr(server_module.time, "monotonic", lambda: next(ticks))

    for elapsed in ("0:05", "0:13", "0:20"):
        server._set_download_progress(job, None, f"получаю исходный звук: {elapsed}")

    messages = [record.getMessage() for record in caplog.records if "получение звука" in record.getMessage()]
    assert len(messages) == 2
    assert all("получаю исходный звук" in message for message in messages)
    assert job.progress == 0.0
    assert job.stage_progress == 0.0


def test_ytdlp_progress_parser_handles_marker_without_terminal_formatting():
    from uvt.server import _yt_dlp_progress_parser

    updates: list[tuple[float | None, str]] = []
    parser = _yt_dlp_progress_parser(lambda fraction, detail: updates.append((fraction, detail)))
    parser("UVT_PROGRESS: 42.5%")

    assert len(updates) == 1
    assert updates[0][0] == pytest.approx(0.425)
    assert updates[0][1] == "скачиваю звук со страницы: 42%"


async def test_ready_clips_are_published_during_the_job(tmp_path):
    """Прогрессивный дубляж: реплики доступны отдельными файлами по таймкодам."""
    rate = 16000
    t = np.arange(rate) / rate
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    silence = np.zeros(rate // 2, dtype=np.float32)
    src = tmp_path / "in.wav"
    sf.write(src, np.concatenate([silence, tone, silence, tone, silence]), rate)

    from uvt.server import DubServer

    server = DubServer(_cfg())
    server.audio_dir = tmp_path
    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        created = await (await client.post("/dub", json={"file": str(src)})).json()
        job_id = created["id"]
        info = None
        for _ in range(200):
            info = await (await client.get(f"/job/{job_id}")).json()
            if info["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.05)
        assert info is not None and info["status"] == "done", info

        clips = info["clips"]
        assert len(clips) == len(info["entries"]), "публикуется каждая озвученная реплика"
        for clip in clips:
            # Тайминг для планировщика в браузере
            assert clip["duration"] > 0
            assert clip["at"] >= 0
            assert clip["at"] <= clip["source_start"]
            assert clip["translated"] and clip["original"]
            assert clip["speaker"]
            # Реплика реально отдаётся отдельным файлом
            audio = await client.get(clip["url"])
            assert audio.status == 200
            assert audio.headers["Content-Type"] == "audio/wav"
            assert len(await audio.read()) > 100

        # Реплики идут в порядке таймкодов — иначе браузеру пришлось бы сортировать
        assert [clip["at"] for clip in clips] == sorted(clip["at"] for clip in clips)
    finally:
        await client.close()


async def test_clip_urls_respect_the_api_token(monkeypatch, tmp_path):
    monkeypatch.setenv("UVT_API_TOKEN", "personal-secret")
    from uvt.server import DubServer

    server = DubServer(_cfg())
    server.audio_dir = tmp_path
    clips_dir = tmp_path / "clips" / "job42"
    clips_dir.mkdir(parents=True)
    (clips_dir / "0.wav").write_bytes(b"RIFF....WAVEfmt ")
    server._audio_access_tokens["job42"] = "clip-secret"

    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        denied = await client.get("/clip/job42/0.wav")
        assert denied.status == 401
        allowed = await client.get("/clip/job42/0.wav?access=clip-secret")
        assert allowed.status == 200
        # Обход каталогов невозможен
        escaped = await client.get("/clip/job42/..%2F..%2Fready.m4a?access=clip-secret")
        assert escaped.status in (400, 404)
    finally:
        await client.close()
