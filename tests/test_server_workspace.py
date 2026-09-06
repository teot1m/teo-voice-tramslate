"""Workspace API: real local media with dummy models, bounded upload errors."""
from __future__ import annotations

import asyncio
import io
import json
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from uvt.config import AppConfig
from uvt import server_workspace as workspace


@pytest.fixture(autouse=True)
def isolated_workspace(monkeypatch, tmp_path):
    monkeypatch.delenv("UVT_API_TOKEN", raising=False)
    monkeypatch.setenv("UVT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(workspace.shutil, "disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3))


def _cfg():
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.target_lang = "ru"
    cfg.vad.engine = "energy"
    cfg.vad.min_speech_ms = 100
    cfg.latency.preset = "ultra"
    cfg.stt.engine = cfg.translation.engine = cfg.tts.engine = "dummy"
    return cfg


@asynccontextmanager
async def client_for(tmp_path, token=""):
    from uvt.server import DubServer
    server = DubServer(_cfg())
    server.api_token = token
    server.audio_dir = tmp_path / "audio"
    server.audio_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        yield server, client
    finally:
        for task in server._tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*server._tasks.values(), return_exceptions=True)
        await client.close()


def wav_bytes(stereo=False):
    rate = 8000
    t = np.arange(5200) / rate
    left = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    left[:2800] = 0
    samples = left
    if stereo:
        right = (0.12 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
        right[:2800] = 0
        samples = np.column_stack([left, right])
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def upload_form(data=None, *, filename="meeting.wav", options=None):
    form = FormData()
    if data is not None:
        form.add_field("file", data, filename=filename, content_type="application/octet-stream")
    if options is not None:
        form.add_field("options", options if isinstance(options, str) else json.dumps(options), content_type="application/json")
    return form


async def wait_terminal(server, job_id):
    task = server._tasks.get(job_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=15)
    await asyncio.sleep(0)  # uploaded source is removed by the task callback
    return server._job_payload(server.jobs[job_id])


def probe(path, entry):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", entry, "-of", "json", str(path),
    ], capture_output=True, check=True, timeout=5)
    return json.loads(result.stdout)


requires_media = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires local FFmpeg")


@requires_media
@pytest.mark.parametrize("stereo", [False, True])
async def test_upload_real_wav_finishes_and_exports_subtitles_audio(tmp_path, stereo):
    data = wav_bytes(stereo)
    if not stereo:
        assert 10000 < len(data) < 12000
    async with client_for(tmp_path) as (server, client):
        response = await client.post("/workspace/upload", data=upload_form(data, options={
            "target_lang": "de", "source_name": "caller cannot replace filename",
            "file": "/private/do-not-open.wav", "page_url": "http://do-not-fetch.invalid/",
        }))
        assert response.status == 200, await response.text()
        created = await response.json()
        info = await wait_terminal(server, created["id"])
        assert info["status"] == "done", info
        assert info["entries"][0]["translated"] == "HELLO 1 [de]"
        assert info["source_name"] == "meeting.wav"
        assert not list((server.audio_dir / "uploads").iterdir())
        for kind in ("srt", "vtt", "json", "txt"):
            result = await client.get(info["downloads"][kind])
            assert result.status == 200
            assert "attachment" in result.headers["Content-Disposition"]
            content = await result.text()
            assert "HELLO 1 [de]" in content
            if kind == "srt":
                assert " --> " in content
            if kind == "vtt":
                assert content.startswith("WEBVTT")
            if kind == "json":
                assert json.loads(content)[0]["target_lang"] == "de"
        result = await client.get(info["downloads"]["m4a"])
        assert result.status == 200
        output = tmp_path / "downloaded.m4a"
        output.write_bytes(await result.read())
        assert output.stat().st_size > 1000
        metadata = probe(output, "stream=channels")
        assert metadata["streams"][0]["channels"] == 2  # Offline mix renders stereo.
        if stereo:
            decoded = subprocess.run([
                "ffmpeg", "-v", "error", "-i", str(output), "-f", "f32le", "-ar", "8000", "pipe:1",
            ], capture_output=True, check=True, timeout=5)
            samples = np.frombuffer(decoded.stdout, dtype=np.float32).reshape(-1, 2)
            assert np.mean(np.abs(samples[:, 0] - samples[:, 1])) > 0.001
        jobs = await (await client.get("/workspace/jobs")).json()
        assert jobs["jobs"][0]["id"] == created["id"]
        assert "entries" not in jobs["jobs"][0]


@requires_media
async def test_upload_tiny_video_remuxes_original_video_and_two_audio_tracks(tmp_path):
    source_audio = tmp_path / "tone.wav"
    source_audio.write_bytes(wav_bytes())
    source_video = tmp_path / "fixture.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=32x32:r=10:d=0.65",
        "-i", str(source_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "mpeg4",
        "-c:a", "pcm_s16le", "-shortest", str(source_video),
    ], check=True, capture_output=True, timeout=8)
    async with client_for(tmp_path) as (server, client):
        response = await client.post("/workspace/upload", data=upload_form(source_video.read_bytes(), filename="clip.mkv", options={"export_video": True}))
        assert response.status == 200, await response.text()
        info = await wait_terminal(server, (await response.json())["id"])
        assert info["status"] == "done", info
        result = await client.get(info["downloads"]["mkv"])
        assert result.status == 200
        output = tmp_path / "translated.mkv"
        output.write_bytes(await result.read())
        streams = probe(output, "stream=codec_type,codec_name")["streams"]
        assert [s["codec_name"] for s in streams if s["codec_type"] == "video"] == ["mpeg4"]
        assert len([s for s in streams if s["codec_type"] == "audio"]) == 2
        assert len(list((server.audio_dir / "uploads").iterdir())) == 1  # Original retained for playback/export.


@requires_media
async def test_cancel_uploaded_job_removes_owned_input(monkeypatch, tmp_path):
    import uvt.server as server_module
    started = asyncio.Event()
    async def slow_render(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(server_module, "render_dub_track", slow_render)
    async with client_for(tmp_path) as (server, client):
        response = await client.post("/workspace/upload", data=upload_form(wav_bytes()))
        assert response.status == 200, await response.text()
        job_id = (await response.json())["id"]
        await asyncio.wait_for(started.wait(), timeout=5)
        assert len(list((server.audio_dir / "uploads").iterdir())) == 1
        task = server._tasks[job_id]
        result = await client.post(f"/job/{job_id}/cancel")
        assert result.status == 200
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        assert server.jobs[job_id].status == "cancelled"
        assert not list((server.audio_dir / "uploads").iterdir())


async def test_workspace_shell_is_public_but_api_and_downloads_are_authenticated(tmp_path):
    from uvt.server import Job
    async with client_for(tmp_path, token="test-api-token") as (server, client):
        for path in ("/workspace", "/workspace/app.js", "/workspace/style.css", "/workspace/userscript.user.js"):
            response = await client.get(path)
            assert response.status == 200, (path, await response.text())
        assert (await client.get("/workspace/jobs")).status == 401
        assert (await client.post("/workspace/upload", data=upload_form(wav_bytes()))).status == 401
        headers = {"X-UVT-Token": "test-api-token"}
        assert (await client.get("/workspace/jobs", headers=headers)).status == 200
        for jid, token in (("ready", "first-job-token"), ("other", "other-job-token")):
            server.jobs[jid] = Job(id=jid, status="done")
            server._audio_access_tokens[jid] = token
            (server.audio_dir / f"{jid}.m4a").write_bytes(b"scoped download")
        assert (await client.get("/download/ready/m4a")).status == 401
        assert (await client.get("/download/ready/m4a?access=other-job-token")).status == 401
        assert (await client.get("/download/ready/m4a?access=first-job-token")).status == 200
        assert (await client.get("/download/ready/m4a", headers=headers)).status == 200
        assert (await client.get("/download/other/m4a?access=first-job-token")).status == 401
        assert (await client.get("/download/ready/srt?access=first-job-token")).status == 200


async def test_foreign_origin_cannot_access_jobs_or_upload_without_api_token(tmp_path):
    async with client_for(tmp_path) as (_, client):
        headers = {"Origin": "https://outside.invalid"}
        assert (await client.get("/workspace/jobs", headers=headers)).status == 403
        assert (await client.post("/workspace/upload", headers=headers, data=upload_form(wav_bytes()))).status == 403


@pytest.mark.parametrize("body,filename,options,expected", [
    (b"", "empty.wav", None, 400),
    (b"bad", "file.exe", None, 415),
    (b"bad", "file.wav", "{broken", 400),
    (b"bad", "file.wav", "[]", 400),
    (None, "file.wav", "{}", 400),
])
async def test_invalid_upload_requests_return_4xx_and_clean_files(tmp_path, body, filename, options, expected):
    async with client_for(tmp_path) as (server, client):
        response = await client.post("/workspace/upload", data=upload_form(body, filename=filename, options=options))
        assert response.status == expected, await response.text()
        assert not server.jobs
        assert not list((server.audio_dir / "uploads").glob("*"))


async def test_missing_multipart_boundary_is_bad_request(tmp_path):
    async with client_for(tmp_path) as (server, client):
        response = await client.post("/workspace/upload", data=b"broken", headers={"Content-Type": "multipart/form-data"})
        assert response.status == 400
        assert not server.jobs


async def test_oversized_file_is_rejected_during_streaming_and_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace, "MAX_UPLOAD_BYTES", 1024)
    async with client_for(tmp_path) as (server, client):
        response = await client.post("/workspace/upload", data=upload_form(b"x" * 2048))
        assert response.status == 413, await response.text()
        assert not server.jobs
        assert not list((server.audio_dir / "uploads").iterdir())


@requires_media
@pytest.mark.parametrize("body", [b"not a media file", b"#EXTM3U\n#EXTINF:1,\nfile:///does-not-exist.wav\n"])
async def test_invalid_content_and_playlist_masquerading_as_mp4_are_rejected(tmp_path, body):
    async with client_for(tmp_path) as (server, client):
        response = await client.post("/workspace/upload", data=upload_form(body, filename="clip.mp4"))
        assert response.status == 422, await response.text()
        assert not server.jobs
        assert not list((server.audio_dir / "uploads").iterdir())


class FakeProbe:
    def __init__(self, output=b"", *, complete=True, returncode=0):
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(output)
        if complete:
            self.stdout.feed_eof()
        self.returncode = returncode if complete else None
        self.killed = False
    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stdout.feed_eof()
    async def wait(self):
        return self.returncode


def metadata(*, streams=None, duration="1", format_name="wav"):
    return json.dumps({"format": {"duration": duration, "format_name": format_name}, "streams": streams if streams is not None else [{"codec_type": "audio"}]}).encode()


async def test_probe_uses_protocol_and_format_allowlists(monkeypatch, tmp_path):
    commands = []
    async def create(*args, **kwargs):
        commands.append(args)
        return FakeProbe(metadata())
    monkeypatch.setattr(workspace.asyncio, "create_subprocess_exec", create)
    assert await workspace.validate_media(tmp_path / "file.wav") is False
    command = commands[0]
    assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
    allowed = set(command[command.index("-format_whitelist") + 1].split(","))
    assert "wav" in allowed and "mov" in allowed and "matroska" in allowed
    assert not allowed.intersection({"hls", "dash", "concat"})


@pytest.mark.parametrize("payload", [
    metadata(streams=[{"codec_type": "video"}]), metadata(duration="nan"),
    metadata(duration="0"), metadata(duration="5401"), metadata(format_name="hls"),
    b"[]", b"null", b"{broken", b'{"format": {}, "streams": null}',
])
async def test_probe_rejects_missing_audio_invalid_metadata_and_duration(monkeypatch, tmp_path, payload):
    async def create(*args, **kwargs):
        return FakeProbe(payload)
    monkeypatch.setattr(workspace.asyncio, "create_subprocess_exec", create)
    with pytest.raises(web.HTTPUnprocessableEntity):
        await workspace.validate_media(tmp_path / "file.wav")


async def test_probe_timeout_and_output_limit_kill_child(monkeypatch, tmp_path):
    child = FakeProbe(complete=False)
    async def create(*args, **kwargs):
        return child
    monkeypatch.setattr(workspace.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(workspace, "PROBE_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(web.HTTPUnprocessableEntity) as exc:
        await workspace.validate_media(tmp_path / "file.wav")
    assert "слишком много времени" in exc.value.text
    assert child.killed
    child = FakeProbe(b"0123456789", complete=False)
    monkeypatch.setattr(workspace, "MAX_PROBE_BYTES", 8)
    with pytest.raises(web.HTTPUnprocessableEntity) as exc:
        await workspace.validate_media(tmp_path / "file.wav")
    assert "слишком большой" in exc.value.text
    assert child.killed


async def test_probe_missing_dependency_has_friendly_service_unavailable(monkeypatch, tmp_path):
    async def missing(*args, **kwargs):
        raise FileNotFoundError("ffprobe")
    monkeypatch.setattr(workspace.asyncio, "create_subprocess_exec", missing)
    with pytest.raises(web.HTTPServiceUnavailable) as exc:
        await workspace.validate_media(tmp_path / "file.wav")
    assert "ffprobe" in exc.value.text


async def test_probe_cancellation_reaps_child(monkeypatch, tmp_path):
    child = FakeProbe(complete=False)
    started = asyncio.Event()
    async def create(*args, **kwargs):
        started.set()
        return child
    monkeypatch.setattr(workspace.asyncio, "create_subprocess_exec", create)
    task = asyncio.create_task(workspace.validate_media(tmp_path / "file.wav"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert child.killed


@pytest.mark.parametrize("value,expected", [
    ("C:\\private\\meeting.wav", "meeting.wav"), ("../../home/video.mp4", "video.mp4"),
    ("test\n\r\x00name.wav", "testname.wav"), ("x" * 300 + ".wav", "x" * 180),
])
def test_upload_source_name_does_not_expose_paths_or_controls(value, expected):
    assert workspace.upload_source_name(value) == expected
