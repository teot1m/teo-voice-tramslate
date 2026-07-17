"""Сервер браузерной кнопки: задача из локального файла → готовая дорожка."""
import asyncio
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from uvt.config import AppConfig

aiohttp = pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")


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
    monkeypatch.setattr(server_module, "_run_process", fake_process)
    result = await server_module._download_page("https://site.example.test/watch/99", tmp_path)

    assert result.name == "source.webm"
    assert captured[captured.index("-f") + 1] == server_module._YT_DLP_AUDIO_SELECTOR
    assert "--no-playlist" in captured
    assert "--merge-output-format" not in captured


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
