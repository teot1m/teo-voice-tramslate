"""Neutral video fixtures exercise playback, retained originals and restart safety."""
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import time

import pytest

from test_server_workspace import client_for, upload_form, wav_bytes, wait_terminal, probe, requires_media, isolated_workspace
from uvt.server import DubServer, Job
from uvt.workspace_video import VideoResults


def tiny_video(tmp_path):
    path = tmp_path / "neutral.mp4"
    audio = tmp_path / "neutral.wav"
    audio.write_bytes(wav_bytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=navy:s=64x48:r=10:d=0.7", "-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True, capture_output=True, timeout=8)
    return path


async def video_finished(server, jid):
    task = server._video_results.tasks.get(jid)
    if task:
        await asyncio.wait_for(asyncio.shield(task), timeout=15)
    return server._job_payload(server.jobs[jid])


@requires_media
async def test_video_upload_player_original_translated_and_restart(tmp_path):
    original = tiny_video(tmp_path)
    async with client_for(tmp_path, token="local-secret") as (server, client):
        response = await client.post("/workspace/upload", data=upload_form(original.read_bytes(), filename="neutral.mp4"), headers={"X-UVT-Token": "local-secret"})
        assert response.status == 200, await response.text()
        jid = (await response.json())["id"]
        await wait_terminal(server, jid)
        info = await video_finished(server, jid)
        assert info["video"]["status"] == "ready", info["video"]
        assert info["video"]["independent_audio"] is True
        assert info["video"]["resolution"] == "64×48"
        assert {"original", "translated"}.issubset(info["downloads"])
        response = await client.get(info["downloads"]["original"])
        assert response.status == 200
        assert await response.read() == original.read_bytes()
        response = await client.get(info["video"]["url"], headers={"Range": "bytes=0-63"})
        assert response.status == 206
        assert len(await response.read()) == 64
        assert response.headers["Content-Disposition"].startswith("inline")
        response = await client.get(f"/download/{jid}/video")
        assert response.status == 401
        response = await client.get(info["downloads"]["translated"])
        mixed = tmp_path / "mixed.mp4"
        mixed.write_bytes(await response.read())
        streams = probe(mixed, "stream=codec_type,codec_name")["streams"]
        assert [(s["codec_type"], s["codec_name"]) for s in streams] == [("video", "h264"), ("audio", "aac")]
        manifest = server.audio_dir / "video-results" / jid / "result.json"
        saved = manifest.read_text()
        assert "local-secret" not in saved and "?access=" not in saved
        assert manifest.stat().st_mode & 0o777 == 0o600
        previous_token = info["video"]["url"].split("access=")[1]
        server.jobs.clear()
        server._audio_access_tokens.clear()
        restored = VideoResults(server)
        server._video_results = restored
        assert jid in server.jobs
        restored_info = server._job_payload(server.jobs[jid])
        assert restored_info["status"] == "done"
        assert restored_info["video"]["url"].split("access=")[1] != previous_token
        assert restored_info["entries"] == info["entries"]
        assert restored.path(server.jobs[jid], "original").is_file()


@requires_media
async def test_finished_url_result_can_add_video_without_translating_again(tmp_path, monkeypatch):
    original = tiny_video(tmp_path)
    async with client_for(tmp_path) as (server, client):
        job = Job(id="123456abcdef", status="done", source_name="Neutral fixture")
        server.jobs[job.id] = job
        shutil.copyfile(original, server.audio_dir / f"{job.id}.m4a")
        video = server._video_results
        video.record_request(job, {"page_url": "https://example.org/neutral-video"})
        downloads = []
        async def download(current, directory):
            downloads.append(current.id)
            output = directory / "original.mp4"
            shutil.copyfile(original, output)
            return output
        monkeypatch.setattr(video, "_download", download)
        response = await client.post(f"/workspace/job/{job.id}/video", json={"original_volume": 0.25, "translation_volume": 0.7})
        assert response.status == 202
        info = await video_finished(server, job.id)
        assert info["video"]["status"] == "ready", info["video"]
        assert info["status"] == "done"
        assert downloads == [job.id]
        assert info["video"]["original_volume"] == 0.25
        response = await client.post(f"/workspace/job/{job.id}/video", json={"original_volume": 0, "translation_volume": 0.5})
        assert response.status == 202
        info = await video_finished(server, job.id)
        assert downloads == [job.id]  # Same retained original, new mix only.
        assert info["video"]["translation_volume"] == 0.5


@pytest.mark.parametrize("value", [-1, 2, float("inf"), "0.2", None])
async def test_video_volume_limits(tmp_path, value):
    async with client_for(tmp_path) as (server, client):
        job = Job(id="123456abcdef", status="done")
        server.jobs[job.id] = job
        server._video_results.record_request(job, {"page_url": "https://example.org/neutral"})
        response = await client.post(f"/workspace/job/{job.id}/video", json={"original_volume": value})
        assert response.status == 400
        assert not server._video_results.tasks


async def test_cancel_video_keeps_completed_translation_and_stops_worker(tmp_path, monkeypatch):
    async with client_for(tmp_path) as (server, client):
        job = Job(id="123456abcdef", status="done")
        server.jobs[job.id] = job
        video = server._video_results
        video.record_request(job, {"page_url": "https://example.org/neutral"})
        started, stopped = asyncio.Event(), asyncio.Event()
        async def download(current, directory):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        monkeypatch.setattr(video, "_download", download)
        response = await client.post(f"/workspace/job/{job.id}/video", json={})
        assert response.status == 202
        await asyncio.wait_for(started.wait(), 3)
        response = await client.post(f"/workspace/job/{job.id}/video/cancel")
        assert response.status == 200
        assert stopped.is_set()
        assert job.status == "done"
        assert video.records[job.id]["status"] == "cancelled"
        assert not list(video.directory(job.id).glob("download-*"))


async def test_video_disk_bound_cancels_child(tmp_path, monkeypatch):
    async with client_for(tmp_path) as (server, _client):
        directory = tmp_path / "bounded"
        directory.mkdir()
        stopped = asyncio.Event()
        async def run(*args, **kwargs):
            (directory / "huge.part").write_bytes(b"x" * 4096)
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        monkeypatch.setattr("uvt.server._run_process", run)
        with pytest.raises(RuntimeError, match="больше 2 ГБ"):
            await server._video_results._bounded_process(["fake"], directory, max_bytes=1024)
        assert stopped.is_set()


async def test_video_routes_require_dashboard_and_token(tmp_path):
    async with client_for(tmp_path) as (server, client):
        job = Job(id="123456abcdef", status="done")
        server.jobs[job.id] = job
        server._video_results.record_request(job, {"page_url": "https://example.org/neutral"})
        response = await client.post(f"/workspace/job/{job.id}/video", json={})
        assert response.status == 202
        response = await client.post(f"/workspace/job/{job.id}/video", json={}, headers={"Origin": "https://external.example"})
        assert response.status == 403


async def test_restore_ignores_external_source_and_regenerates_safe_downloads(tmp_path):
    async with client_for(tmp_path) as (server, _client):
        jid = "123456abcdef"
        directory = server.audio_dir / "video-results" / jid
        directory.mkdir(parents=True)
        (server.audio_dir / f"{jid}.m4a").write_bytes(b"audio")
        foreign = tmp_path / "private-video.mp4"
        foreign.write_bytes(b"private")
        (directory / "result.json").write_text(json.dumps({"version": 1, "job": {"id": jid, "status": "done", "downloads": {"original": "https://untrusted.example/"}}, "request": {}, "source": str(foreign), "has_video": True}))
        video = VideoResults(server)
        assert video.records[jid]["source"] is None
        assert video.path(server.jobs[jid], "original") is None
        assert "original" not in server.jobs[jid].downloads


@requires_media
async def test_cached_url_video_is_retained_without_download(tmp_path):
    original = tiny_video(tmp_path)
    async with client_for(tmp_path) as (server, _client):
        job = Job(id="123456abcdef", status="running")
        video = server._video_results
        video.record_request(job, {"page_url": "https://example.org/neutral"})
        await video.retain_source(job, original, {"page_url": "https://example.org/neutral"})
        retained = video.path(job, "original")
        assert retained and retained.is_relative_to(server.audio_dir)
        assert retained.read_bytes() == original.read_bytes()
        original.unlink()
        assert retained.is_file()  # Original cache expiry does not remove this result.
        video.discard(job)
        assert not retained.exists()


def test_player_ignores_interrupted_play_but_reports_real_autoplay_failure():
    if not shutil.which("node"):
        pytest.skip("Node is needed to exercise the browser event handler")
    source = (Path(__file__).parents[1] / "src/uvt/web/workspace.js").read_text()
    function = source[source.index("  function syncTranslation("):source.index("  function updateVolumes(")]
    script = """
    let failure = 'AbortError';
    const errorBox = {textContent: ''};
    const video = {getAttribute: ()=> 'source', playbackRate: 1, currentTime: 3, paused: false, seeking: false};
    const audio = {getAttribute: ()=> 'audio', currentTime: 3, play: ()=> Promise.reject({name:failure})};
    const $ = name => ({'result-video':video,'result-audio':audio,'video-error':errorBox})[name];
    """ + function + """
    (async()=>{
      syncTranslation(true); await Promise.resolve();
      if(errorBox.textContent) throw new Error('Interrupted play must not report failure');
      failure = 'NotAllowedError'; syncTranslation(true); await Promise.resolve();
      if(!errorBox.textContent) throw new Error('Autoplay denial must be reported');
      console.log('ok');
    })().catch(error=>{console.error(error); process.exitCode=1;});
    """
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


async def test_video_queue_is_bounded(tmp_path):
    async with client_for(tmp_path) as (server, client):
        video = server._video_results
        job = Job(id="123456abcdef", status="done")
        server.jobs[job.id] = job
        video.record_request(job, {"page_url": "https://example.org/neutral"})
        for index in range(8):
            video.tasks[f"{index:012x}"] = asyncio.create_task(asyncio.Event().wait())
        response = await client.post(f"/workspace/job/{job.id}/video", json={})
        assert response.status == 429
        assert len(video.tasks) == 8


@requires_media
async def test_result_manifest_disk_error_does_not_discard_completed_audio(tmp_path, monkeypatch):
    async with client_for(tmp_path) as (server, client):
        def cannot_persist(job):
            raise OSError("No space left")
        monkeypatch.setattr(server._video_results, "persist", cannot_persist)
        response = await client.post("/workspace/upload", data=upload_form(wav_bytes()))
        jid = (await response.json())["id"]
        info = await wait_terminal(server, jid)
        assert info["status"] == "done"
        result = await client.get(info["downloads"]["m4a"])
        assert result.status == 200
        assert len(await result.read()) > 1000


async def test_real_video_download_builds_ytdlp_command_and_reports_progress(tmp_path, monkeypatch):
    """Exercise lazy imports and all download wrappers; replace subprocess only."""
    monkeypatch.setattr("uvt.dub._find_ytdlp", lambda: "/fake/venv/bin/yt-dlp")
    monkeypatch.setattr("uvt.dub._ytdlp_js_args", lambda: ["--js-runtimes", "node:/fake/node"])
    async with client_for(tmp_path) as (server, _client):
        job = Job(id="123456abcdef", status="done")
        video = server._video_results
        video.record_request(job, {"page_url": "https://example.org/neutral"})
        directory = tmp_path / "video-download"
        directory.mkdir()
        calls = []
        async def run(command, timeout_s, what, *, on_line=None):
            calls.append(command)
            assert timeout_s == 1800
            assert command[0] == "/fake/venv/bin/yt-dlp"
            assert command[command.index("--js-runtimes") + 1] == "node:/fake/node"
            assert command[command.index("--max-filesize") + 1] == str(2 * 1024**3)
            assert command[-2:] == ["--", "https://example.org/neutral"]
            on_line("UVT_PROGRESS:50.0%;UVT_BYTES:4096")
            output = Path(command[command.index("-o") + 1].replace("%(ext)s", "mp4"))
            output.write_bytes(b"neutral video fixture")
        monkeypatch.setattr("uvt.server._run_process", run)
        result = await video._download(job, directory)
        assert result == directory / "original.mp4"
        assert result.read_bytes() == b"neutral video fixture"
        assert len(calls) == 1
        assert video.records[job.id]["progress"] == 0.5


async def test_real_video_download_falls_back_from_page_extractor_to_discovered_stream(tmp_path, monkeypatch):
    monkeypatch.setattr("uvt.dub._find_ytdlp", lambda: "/fake/yt-dlp")
    async with client_for(tmp_path) as (server, _client):
        job = Job(id="123456abcdef", status="done")
        video = server._video_results
        video.record_request(job, {"page_url": "https://example.org/neutral"})
        directory = tmp_path / "fallback-download"
        directory.mkdir()
        calls, discovered = [], []
        async def discover(page_url):
            discovered.append(page_url)
            return ["https://cdn.example.org/neutral.mp4"]
        async def run(command, timeout_s, what, **kwargs):
            calls.append(command)
            if command[0] == "/fake/yt-dlp":
                (directory / "original.part").write_bytes(b"incomplete")
                raise RuntimeError("Unsupported URL")
            assert command[0] == "ffmpeg"
            assert not (directory / "original.part").exists()
            assert command[command.index("-i") + 1] == "https://cdn.example.org/neutral.mp4"
            assert command[command.index("-headers") + 1] == "Referer: https://example.org/neutral\r\n"
            assert command[command.index("-rw_timeout") + 1] == "15000000"
            Path(command[-1]).write_bytes(b"fallback video fixture")
        monkeypatch.setattr("uvt.server._run_process", run)
        monkeypatch.setattr("uvt.media_discovery.discover_page_media", discover)
        result = await video._download(job, directory)
        assert result.read_bytes() == b"fallback video fixture"
        assert len(calls) == 2
        assert discovered == ["https://example.org/neutral"]


async def test_real_video_download_accepts_browser_stream_without_page(tmp_path, monkeypatch):
    monkeypatch.setattr("uvt.dub._find_ytdlp", lambda: None)
    async with client_for(tmp_path) as (server, _client):
        job = Job(id="123456abcdef", status="done")
        video = server._video_results
        video.record_request(job, {"media_url": "https://cdn.example.org/neutral.mp4"})
        directory = tmp_path / "browser-download"
        directory.mkdir()
        async def run(command, timeout_s, what, **kwargs):
            assert command[0] == "ffmpeg"
            assert "-headers" not in command
            assert command[command.index("-i") + 1] == "https://cdn.example.org/neutral.mp4"
            Path(command[-1]).write_bytes(b"browser video fixture")
        monkeypatch.setattr("uvt.server._run_process", run)
        result = await video._download(job, directory)
        assert result.read_bytes() == b"browser video fixture"
