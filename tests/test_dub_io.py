"""Blocking media operations must not freeze job status or outlive temp files."""

import asyncio
import threading

import pytest

from uvt.dub import _run_blocking


async def _wait_for_thread(event):
    async def wait():
        while not event.is_set():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait(), timeout=2)


async def test_media_io_keeps_event_loop_responsive():
    started, release = threading.Event(), threading.Event()

    def blocking_read():
        started.set()
        assert release.wait(2)
        return "decoded audio"

    task = asyncio.create_task(_run_blocking(blocking_read))
    try:
        # This coroutine must run while the media worker is still busy.
        await _wait_for_thread(started)
        assert not task.done()
    finally:
        release.set()
    assert await task == "decoded audio"


@pytest.mark.parametrize("worker_fails", [False, True])
async def test_media_cancellation_drains_worker_before_temp_cleanup(worker_fails):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocking_write():
        started.set()
        try:
            assert release.wait(2)
            if worker_fails:
                raise OSError("media write failed")
        finally:
            finished.set()

    task = asyncio.create_task(_run_blocking(blocking_write))
    try:
        await _wait_for_thread(started)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # Repeated cancel must not abandon the worker either.
        await asyncio.sleep(0)
        assert not task.done()
        assert not finished.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


async def test_media_error_preserves_failure_for_job_status():
    def failing_read():
        raise OSError("cannot decode media")

    with pytest.raises(OSError, match="cannot decode media"):
        await _run_blocking(failing_read)


async def test_media_cancellation_signals_cooperative_worker_before_draining():
    started, stopped, finished = threading.Event(), threading.Event(), threading.Event()

    def worker():
        started.set()
        assert stopped.wait(2), "cancellation never reached the native worker"
        finished.set()

    task = asyncio.create_task(_run_blocking(worker, _on_cancel=stopped.set))
    await _wait_for_thread(started)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert stopped.is_set() and finished.is_set()


async def test_separation_reports_stages_on_event_loop_thread(tmp_path, monkeypatch):
    import numpy as np
    import uvt.dub as dub_module
    from uvt.config import AppConfig
    from uvt.separate import SeparatedAudio

    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.tts.engine = "dummy"
    cfg.separation.enabled = True
    source = tmp_path / "source.wav"
    source.write_bytes(b"mocked decode")
    monkeypatch.setattr(dub_module, "_decode_file", lambda *_args: np.zeros((4410, 2), dtype=np.float32))
    updates = []
    owner = threading.get_ident()

    def separate(samples, rate, *, progress, cancel_event, **kwargs):
        assert threading.get_ident() != owner
        assert not cancel_event.is_set()
        progress(0.05, 0.1)
        progress(0.1, 0.1)
        return SeparatedAudio(samples, samples.copy(), rate)

    class EndOfProbe(Exception):
        pass

    async def transcribe(*args, **kwargs):
        await asyncio.sleep(0)
        assert [item[0] for item in updates] == ["decode", "separate", "separate", "separate"]
        assert updates[-1][1] == 1
        assert all(item[3] == owner for item in updates)
        raise EndOfProbe

    monkeypatch.setattr(dub_module, "separate_speech", separate)
    monkeypatch.setattr(dub_module, "_transcribe_all", transcribe)
    with pytest.raises(EndOfProbe):
        await dub_module.render_dub_track(cfg, source, on_stage=lambda stage, fraction, detail: updates.append((stage, fraction, detail, threading.get_ident())))


async def test_render_cancellation_stops_separation_without_starting_stt(tmp_path, monkeypatch):
    import numpy as np
    import uvt.dub as dub_module
    from uvt.config import AppConfig
    from uvt.separate import SeparationCancelled

    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.tts.engine = "dummy"
    cfg.separation.enabled = True
    source = tmp_path / "source.wav"
    source.write_bytes(b"mocked decode")
    monkeypatch.setattr(dub_module, "_decode_file", lambda *_args: np.zeros((4410, 2), dtype=np.float32))
    started, finished = threading.Event(), threading.Event()

    def separate(*args, cancel_event, **kwargs):
        started.set()
        assert cancel_event.wait(2)
        finished.set()
        raise SeparationCancelled

    async def transcribe(*args, **kwargs):
        pytest.fail("cancelled separation must never start recognition")

    monkeypatch.setattr(dub_module, "separate_speech", separate)
    monkeypatch.setattr(dub_module, "_transcribe_all", transcribe)
    task = asyncio.create_task(dub_module.render_dub_track(cfg, source))
    await _wait_for_thread(started)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert finished.is_set()
