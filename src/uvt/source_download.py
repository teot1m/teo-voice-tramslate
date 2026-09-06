"""Bounded page discovery and stalled-transfer detection for a URL job."""

from __future__ import annotations

import asyncio
import re
import time

SOURCE_START_TIMEOUT = 120.0
SOURCE_IDLE_TIMEOUT = 90.0
SOURCE_HEARTBEAT_SECONDS = 10.0
_PERCENT = re.compile(r"UVT_PROGRESS:\s*([0-9]+(?:[.,][0-9]+)?)%")

_FFMPEG_TIME = re.compile(r"(?:^|\s)time=(\d{2}:\d{2}:\d{2}(?:\.\d+)?)")


def ffmpeg_progress_time(line):
    match = _FFMPEG_TIME.search(line)
    return match[1] if match else None


class SourceDownloadTimeout(RuntimeError):
    pass


async def run_source_download(
    runner, cmd, on_line, progress=None, *,
    startup_timeout=SOURCE_START_TIMEOUT,
    idle_timeout=SOURCE_IDLE_TIMEOUT,
    heartbeat_seconds=SOURCE_HEARTBEAT_SECONDS,
):
    """Keep the 30-minute transfer budget, but bound connection and idle time.

    Download markers and advancing native FFmpeg timestamps count as transfer
    liveness. Extractor warnings/retries never extend the discovery deadline. The runner owns
    subprocess teardown and must finish it when its task is cancelled.
    """
    started = time.monotonic()
    last_transfer = None
    last_fraction = None
    last_ffmpeg_time = None

    def receive(line):
        nonlocal last_transfer, last_fraction, last_ffmpeg_time
        if "UVT_PROGRESS:" in line:
            last_transfer = time.monotonic()
            match = _PERCENT.search(line)
            if match:
                last_fraction = min(.99, max(0, float(match[1].replace(",", ".")) / 100))
        # Some HLS formats use yt-dlp's external FFmpeg downloader, which
        # emits native time= stats before the final UVT marker.
        stamp = ffmpeg_progress_time(line)
        if stamp is not None and stamp != last_ffmpeg_time:
            last_ffmpeg_time = stamp
            last_transfer = time.monotonic()
        on_line(line)

    task = asyncio.create_task(runner(cmd, 1800, "yt-dlp", on_line=receive))
    try:
        while not task.done():
            now = time.monotonic()
            limit = startup_timeout if last_transfer is None else idle_timeout
            elapsed = now - (started if last_transfer is None else last_transfer)
            remaining = limit - elapsed
            if remaining <= 0:
                if last_transfer is None:
                    raise SourceDownloadTimeout(
                        f"сайт не начал передачу видео за {startup_timeout:g} с. "
                        "Откройте видео в Chrome и используйте кнопку UVT на плеере "
                        "либо загрузите локальный файл"
                    )
                raise SourceDownloadTimeout(
                    f"загрузка остановилась: нет новых данных от загрузчика {idle_timeout:g} с. "
                    "Повторите позже или используйте локальный файл"
                )
            try:
                await asyncio.wait_for(asyncio.shield(task), min(heartbeat_seconds, remaining))
            except asyncio.TimeoutError:
                if task.done():
                    return task.result()
                if progress is not None:
                    seconds = int(time.monotonic() - started)
                    if last_transfer is None:
                        progress(None, f"получаю сведения о видео: прошло {seconds} с; передача файла ещё не началась")
                    elif time.monotonic() - last_transfer >= heartbeat_seconds:
                        progress(last_fraction, "жду следующую порцию данных от сайта…")
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            # Repeated clicks on Cancel must not abandon a child process.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            try:
                task.result()
            except BaseException:
                pass
