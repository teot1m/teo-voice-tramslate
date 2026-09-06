import asyncio

import pytest

from uvt.source_download import run_source_download, SourceDownloadTimeout


async def test_extractor_activity_does_not_hide_connection_timeout():
    stopped = asyncio.Event()
    updates = []

    async def runner(cmd, timeout, what, on_line):
        try:
            while True:
                on_line("WARNING: retrying webpage")
                await asyncio.sleep(.002)
        finally:
            stopped.set()

    with pytest.raises(SourceDownloadTimeout, match="не начал передачу"):
        await run_source_download(runner, [], lambda line: None, lambda fraction, detail: updates.append((fraction, detail)), startup_timeout=.03, heartbeat_seconds=.005)
    assert stopped.is_set()
    assert updates and all(fraction is None for fraction, _ in updates)


async def test_active_transfer_can_outlive_page_discovery_deadline():
    async def runner(cmd, timeout, what, on_line):
        assert timeout == 1800
        for percent in range(10, 61, 10):
            on_line(f"UVT_PROGRESS:{percent}%")
            await asyncio.sleep(.007)

    await run_source_download(runner, [], lambda line: None, startup_timeout=.015, idle_timeout=.03, heartbeat_seconds=.005)


async def test_stalled_transfer_has_its_own_deadline_and_keeps_last_percent():
    stopped = asyncio.Event()
    updates = []

    async def runner(cmd, timeout, what, on_line):
        try:
            on_line("UVT_PROGRESS:25.0%")
            await asyncio.Event().wait()
        finally:
            stopped.set()

    with pytest.raises(SourceDownloadTimeout, match="нет новых данных"):
        await run_source_download(runner, [], lambda line: None, lambda fraction, detail: updates.append(fraction), startup_timeout=.2, idle_timeout=.025, heartbeat_seconds=.005)
    assert stopped.is_set()
    assert updates and set(updates) == {.25}


async def test_unknown_total_progress_still_counts_as_transfer_activity():
    async def runner(cmd, timeout, what, on_line):
        for size in range(100, 701, 100):
            on_line(f"UVT_PROGRESS:NA;UVT_BYTES:{size}")
            await asyncio.sleep(.005)

    await run_source_download(runner, [], lambda line: None, startup_timeout=.01, idle_timeout=.02, heartbeat_seconds=.005)


async def test_source_cancellation_drains_runner_even_if_cancelled_twice():
    started, cleaning, release, finished = (asyncio.Event() for _ in range(4))

    async def runner(cmd, timeout, what, on_line):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            finished.set()

    task = asyncio.create_task(run_source_download(runner, [], lambda line: None))
    await started.wait()
    task.cancel()
    await cleaning.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


async def test_original_extractor_error_is_preserved():
    async def runner(*args, **kwargs):
        raise RuntimeError("HTTP 403")

    with pytest.raises(RuntimeError, match="HTTP 403"):
        await run_source_download(runner, [], lambda line: None)


async def test_external_ffmpeg_transfer_does_not_hit_page_discovery_timeout():
    async def runner(cmd, timeout, what, on_line):
        for second in range(1, 7):
            on_line(f"size=100KiB time=00:00:0{second}.00 bitrate=128kbits/s")
            await asyncio.sleep(.007)

    await run_source_download(runner, [], lambda line: None, startup_timeout=.015, idle_timeout=.03, heartbeat_seconds=.005)


async def test_repeated_ffmpeg_timestamp_does_not_hide_stalled_transfer():
    async def runner(cmd, timeout, what, on_line):
        while True:
            on_line("size=100KiB time=00:00:01.00 bitrate=128kbits/s")
            await asyncio.sleep(.002)

    with pytest.raises(SourceDownloadTimeout, match="нет новых данных"):
        await run_source_download(runner, [], lambda line: None, startup_timeout=.02, idle_timeout=.025, heartbeat_seconds=.005)
