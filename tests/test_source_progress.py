import pytest

from uvt.server import _yt_dlp_progress_parser


def test_extractor_phases_and_unknown_total_are_not_fake_percentages():
    updates, diagnostics = [], []
    receive = _yt_dlp_progress_parser(lambda fraction, detail: updates.append((fraction, detail)), diagnostics)
    receive("[Site] id: Downloading pc webpage")
    receive("[Site] id: Downloading m3u8 information")
    receive("[Site] id: Downloading JSON metadata")
    receive("UVT_PROGRESS:NA;UVT_BYTES:1048576")
    assert len(updates) == 4 and all(value is None for value, _ in updates)
    assert "1.0 МБ" in updates[-1][1]
    receive("UVT_PROGRESS:25%;UVT_BYTES:1048576")
    receive("WARNING: retrying a fragment")
    assert updates[-1][0] == .25
    assert diagnostics[-1] == "WARNING: retrying a fragment"


def test_parser_uses_latest_update_if_legacy_reader_bundles_carriage_returns():
    updates = []
    receive = _yt_dlp_progress_parser(lambda fraction, detail: updates.append(fraction))
    receive("UVT_PROGRESS:1%\rUVT_PROGRESS:55%")
    assert updates == [.55]


async def test_page_command_enables_streamed_progress_and_bounded_retries(monkeypatch, tmp_path):
    import uvt.server as module
    import uvt.dub as dub

    monkeypatch.setattr(dub, "_find_ytdlp", lambda: "yt-dlp")
    monkeypatch.setattr(dub, "_ytdlp_js_args", lambda: [])
    updates = []

    async def run(cmd, timeout, what, on_line):
        assert "--newline" in cmd and "--progress" in cmd
        assert cmd[cmd.index("--socket-timeout") + 1] == "15"
        assert cmd[cmd.index("--retries") + 1] == "2"
        assert cmd[cmd.index("--fragment-retries") + 1] == "2"
        assert cmd[cmd.index("--extractor-retries") + 1] == "1"
        assert cmd[-2:] == ["--", "https://example.test/video"]
        assert "UVT_BYTES:" in cmd[cmd.index("--progress-template") + 1]
        on_line("[Site] id: Downloading webpage")
        on_line("UVT_PROGRESS:75%;UVT_BYTES:100")
        (tmp_path / "audio.m4a").write_bytes(b"fake audio")

    monkeypatch.setattr(module, "_run_process", run)
    result = await module._download_page("https://example.test/video", tmp_path, progress=lambda fraction, detail: updates.append((fraction, detail)))
    assert result.name == "audio.m4a"
    assert any(fraction == .75 for fraction, _ in updates)
    assert updates[-1][0] == 1


async def test_connection_timeout_keeps_actual_reason(monkeypatch, tmp_path):
    import uvt.server as module
    import uvt.dub as dub
    import uvt.source_download as source

    monkeypatch.setattr(dub, "_find_ytdlp", lambda: "yt-dlp")
    monkeypatch.setattr(dub, "_ytdlp_js_args", lambda: [])

    async def timeout(*args, **kwargs):
        raise source.SourceDownloadTimeout("сайт не начал передачу видео за 120 с")

    monkeypatch.setattr(source, "run_source_download", timeout)
    with pytest.raises(RuntimeError, match="не начал передачу видео за 120 с") as error:
        await module._download_page("https://example.test/video", tmp_path)
    assert "Unsupported URL" not in str(error.value)


async def test_job_retains_failed_source_stage(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    from uvt.config import AppConfig
    from uvt.server import DubServer, Job

    monkeypatch.setenv("UVT_CACHE", str(tmp_path))
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.stt.engine = cfg.tts.engine = cfg.translation.engine = "dummy"
    server = DubServer(cfg)
    monkeypatch.setattr(server, "_resolve_cached_source", AsyncMock(side_effect=RuntimeError("HTTP 403")))
    job = Job(id="source-error")
    await server._run_job(job, {}, cfg, "configured")
    assert job.status == "error"
    assert server._job_payload(job)["failed_stage"] == "download"


def test_external_ffmpeg_progress_is_visible_without_inventing_percentage():
    updates = []
    receive = _yt_dlp_progress_parser(lambda fraction, detail: updates.append((fraction, detail)))
    receive("size=100KiB time=00:00:05.72 bitrate=128kbits/s")
    assert updates[0][0] is None
    assert "00:00:05.72" in updates[0][1]
