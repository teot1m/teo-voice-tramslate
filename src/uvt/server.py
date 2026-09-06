"""Локальный сервер браузерной кнопки (uvt serve) — движок для userscript.

Userscript (browser/uvt.user.js) вешает кнопку на любой <video> на странице.
По нажатию он присылает сюда адрес страницы и прямую ссылку на медиапоток;
сервер скачивает звук, готовит дублированную дорожку (render_dub_track) и
отдаёт её как .m4a — скрипт проигрывает её синхронно с видео, приглушив
оригинал. Тот же UX, что у voice-over-translation, но перевод локальный/ваш.

API (JSON, CORS открыт):
  GET  /meta → честное описание batch-режима, активных движков и приватности
  POST /dub {page_url?, media_url?, file?, target_lang?} → {id, job_url, ...}
  GET  /job/{id} → {status, stage, progress, queue_position, timing, ...}
  POST /job/{id}/approve {approved} → решение по job в статусе awaiting_approval:
       облачный STT/перевод не справился и ждёт, переходить ли на локальный резерв
  GET  /audio/{id}.m4a → дорожка (поддерживает Range — перемотка работает)

Сервер слушает только 127.0.0.1 и обрабатывает по одной задаче за раз
(Whisper не стоит параллелить на одном CPU/GPU).
"""
from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import hmac
import html
import io
import ipaddress
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import webbrowser
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

from uvt.config import AppConfig, load_config
from uvt import registry
from uvt.dub import CLIP_LEAD_S, _run_blocking, render_dub_track
from uvt.fallback import ApprovalGate, create_stt_engine
from uvt.interfaces import STTEngine
from uvt.server_settings import (
    PARAKEET_LANGUAGE_IDS,
    NEMOTRON_LANGUAGE_IDS,
    ServerSettingsStore,
    SettingsConflictError,
    apply_settings,
    effective_settings,
    normalize_settings,
    local_voice_languages,
    route_key as settings_route_key,
    settings_catalog,
    settings_kind,
)

log = logging.getLogger("uvt.server")

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Не скачиваем видеопоток, пока сайт отдаёт отдельное аудио. Первый вариант
# сохраняет платформенную пометку original, остальные — совместимые fallback
# для сайтов, где metadata неполная или есть только muxed-файл.
_YT_DLP_AUDIO_SELECTOR = (
    "bestaudio[format_note*=original][acodec!=none]"
    "/bestaudio[acodec!=none]/bestaudio/worst[acodec!=none]"
    "/best[acodec!=none]/best"
)
_YT_DLP_YOUTUBE_ANDROID_ARGS = ["--extractor-args", "youtube:player_client=android"]
_LOCAL_PROFILE_LABELS = {
    "local-fast": "Быстро",
    "local-balanced": "Сбалансированный",
    "local-quality": "Качество",
    "local-natural": "Живые голоса · медленно",
    "local-hymt": "Hy-MT2 · быстрый + Piper",
    "local-moss": "Hy-MT2 + MOSS · живые голоса",
    "local-nemotron": "Nemotron + Hy-MT2 + Piper",
}
_VOICE_LABELS = {
    "ru_RU-dmitri-medium": "Дмитрий",
    "ru_RU-irina-medium": "Ирина",
    "uk_UA-mykyta-high": "Микита",
    "uk_UA-tetiana-high": "Тетяна",
}
_VOICE_GENDERS = {"auto", "male", "female"}
_PREVIEW_MAX_CHARS = 240
_PREVIEW_TIMEOUT_S = 30.0
_BROWSER_OPEN_TIMEOUT_S = 5.0

# Download is a short, separate phase before render_dub_track's 0–100% work.
# Keeping it in a small prefix makes the externally visible progress monotonic.
_DOWNLOAD_PROGRESS_SHARE = 0.08
_ProgressCallback = Callable[[float | None, str], None]
_MAX_BROWSER_MEDIA_CANDIDATES = 6
_SOURCE_CACHE_MAX_AGE_DAYS = 7.0
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
}


def _is_loopback_host(host: str) -> bool:
    """Accept localhost and every textual spelling of an IP loopback address."""
    normalized = host.strip().lower()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _dashboard_url(host: str, port: int) -> str:
    """Return a browser-friendly URL for a locally bound dashboard."""
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("порт UVT должен быть от 1 до 65535")
    normalized = host.strip()
    if normalized in {"", "0.0.0.0"}:
        browser_host = "127.0.0.1"
    elif normalized in {"::", "[::]"}:
        browser_host = "::1"
    else:
        browser_host = normalized
    if ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    return f"http://{browser_host}:{port}/"


def _configured_dashboard_url(env_name: str, fallback: str) -> tuple[str, bool]:
    """Validate an optional public route URL used by the cross-route dashboard."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return fallback, False
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{env_name}: некорректный URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            f"{env_name}: укажите только origin, например https://free.uvt.example"
        )
    if parsed.scheme != "https" and not _is_loopback_host(parsed.hostname):
        raise ValueError(f"{env_name}: для сетевого адреса требуется https://")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    default_port = 443 if parsed.scheme == "https" else 80
    port_suffix = f":{port}" if port and port != default_port else ""
    return f"{parsed.scheme}://{host}{port_suffix}/", True


async def _open_dashboard_in_browser(url: str) -> None:
    """Best-effort browser launch; never stop an otherwise healthy server."""
    try:
        opened = await asyncio.wait_for(
            asyncio.to_thread(webbrowser.open_new_tab, url),
            timeout=_BROWSER_OPEN_TIMEOUT_S,
        )
    except TimeoutError:
        log.warning(
            "браузер не ответил за %.0f с; панель UVT доступна по адресу %s",
            _BROWSER_OPEN_TIMEOUT_S,
            url,
        )
        return
    except Exception as exc:  # noqa: BLE001 - OS browser integration varies
        log.warning("не удалось открыть панель UVT в браузере: %s", exc)
        return
    if not opened:
        log.warning("браузер не подтвердил открытие панели UVT: %s", url)
    else:
        log.info("панель UVT открыта в браузере: %s", url)


def _dashboard_open_block_reason(host: str) -> str | None:
    """Explain why an explicit browser-open request is unsafe in this session."""
    if not _is_loopback_host(host):
        return "сервер слушает не loopback-адрес"

    truthy = {"1", "true", "yes", "on"}
    if os.environ.get("UVT_HEADLESS", "").strip().lower() in truthy:
        return "задан UVT_HEADLESS"
    if os.environ.get("CI", "").strip().lower() in truthy:
        return "запуск в CI без интерактивного браузера"
    if any(
        os.environ.get(name, "").strip()
        for name in ("SSH_CONNECTION", "SSH_TTY", "INVOCATION_ID", "JOURNAL_STREAM")
    ):
        return "удалённая или systemd-сессия без локального браузера"
    if (
        sys.platform != "darwin"
        and os.name != "nt"
        and not os.environ.get("DISPLAY", "").strip()
        and not os.environ.get("WAYLAND_DISPLAY", "").strip()
    ):
        return "графическая сессия не обнаружена"
    return None


# Three personal servers live in one event loop and share the on-disk source
# cache. The keyed lock ensures that two routes cannot download the same media
# concurrently before the first atomic cache write is complete.
_SOURCE_CACHE_LOCKS: dict[str, asyncio.Lock] = {}

# Это именно стадии подготовки готовой дорожки. Они не означают потоковый
# перевод: браузер получает результат только после завершения всей задачи.
_STAGE_DETAILS = {
    "queue": "ожидает свободный обработчик…",
    "download": "получаю исходный звук…",
    "decode": "декодирую исходный звук…",
    "separate": "отделяю речь от фона…",
    "transcribe": "распознаю речь…",
    "translate": "перевожу реплики…",
    "synthesize": "озвучиваю перевод…",
    "mix": "собираю аудиодорожку…",
    "done": "дорожка готова",
    "error": "подготовка не удалась",
    "cancelled": "отменено пользователем",
}


def _cache_dir() -> Path:
    root = os.environ.get("UVT_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")), "uvt"
    )
    path = Path(root).expanduser() / "serve"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canonical_source_url(value: str) -> str:
    """Drop tracking-only URL parts without losing parameters needed by a page."""
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return value.strip()
    if not parsed.scheme or not parsed.netloc:
        return value.strip()
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_KEYS
        ],
        doseq=True,
    )
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", query, "")
    )


def _source_cache_key(data: dict) -> str | None:
    """Stable shared key for a browser source; local files need no copied cache."""
    raw = data.get("page_url") or data.get("media_url")
    if not isinstance(raw, str) or not raw.strip():
        return None
    identity = _canonical_source_url(raw)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


async def _kill_process(proc: asyncio.subprocess.Process) -> None:
    """Terminate this invocation, including children in its owned POSIX group."""
    group = getattr(proc, "_uvt_process_group", None)

    def stop(force: bool = False) -> None:
        try:
            if group is not None and os.name == "posix":
                # Only _run_process marks a group that it created with
                # start_new_session. Never signal an inherited process group.
                os.killpg(group, signal.SIGKILL if force else signal.SIGTERM)
            elif proc.returncode is None:
                proc.kill() if force else proc.terminate()
        except ProcessLookupError:
            pass

    stop()
    try:
        if proc.returncode is None:
            await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        pass
    finally:
        # The leader may already have exited while ffmpeg/node still holds a
        # pipe open. Killing its owned group is required in that case as well.
        stop(force=True)
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            transport.close()
    try:
        await asyncio.wait_for(proc.wait(), timeout=2)
    except asyncio.TimeoutError:
        log.warning("не удалось подтвердить завершение подпроцесса за время очистки")


async def _read_process_lines(stream, on_line: Callable[[str], None]) -> None:
    """Drain CR/LF progress with bounded memory, including oversized lines."""
    pending = bytearray()
    max_line_bytes = 16 * 1024

    def emit() -> None:
        if not pending:
            return
        text = pending.decode("utf-8", errors="replace").strip()
        pending.clear()
        if not text:
            return
        try:
            on_line(text)
        except Exception:  # noqa: BLE001 - progress must never break downloading
            log.debug("не удалось разобрать прогресс подпроцесса", exc_info=True)

    while True:
        chunk = await stream.read(16 * 1024)
        if not chunk:
            emit()
            return
        start = 0
        for index, byte in enumerate(chunk):
            if byte not in (10, 13):
                continue
            pending.extend(chunk[start:index][:max(0, max_line_bytes - len(pending))])
            emit()
            start = index + 1
        # Keep the beginning of a diagnostic but continue consuming every byte
        # of an arbitrarily long line, so child output can never fill the pipe.
        pending.extend(chunk[start:][:max(0, max_line_bytes - len(pending))])


async def _run_process(
    cmd: list[str],
    timeout_s: float,
    what: str,
    *,
    on_line: Callable[[str], None] | None = None,
) -> None:
    """Run a bounded process; timeout covers both execution and pipe draining."""
    pipe = asyncio.subprocess.PIPE if on_line is not None else None
    kwargs = {"stdout": pipe, "stderr": pipe}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
    if os.name == "posix" and getattr(proc, "pid", None) is not None:
        proc._uvt_process_group = proc.pid
    readers: list[asyncio.Task] = []
    if on_line is not None:
        if proc.stdout is not None:
            readers.append(asyncio.create_task(_read_process_lines(proc.stdout, on_line)))
        if proc.stderr is not None:
            readers.append(asyncio.create_task(_read_process_lines(proc.stderr, on_line)))
    process_wait = asyncio.create_task(proc.wait())

    async def finished() -> int:
        # A reader failure must be visible immediately, while a live child can
        # still be terminated, rather than being hidden after proc.wait().
        results = await asyncio.gather(process_wait, *readers)
        return results[0]

    async def cleanup() -> None:
        try:
            await _kill_process(proc)
        finally:
            for task in [process_wait, *readers]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(process_wait, *readers, return_exceptions=True)

    async def drain_cleanup() -> None:
        # Repeated UI/API cancellation must not interrupt termination or leave
        # ffmpeg/yt-dlp descendants running after the job says it is cancelled.
        task = asyncio.create_task(cleanup())
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            task.result()
        except Exception:
            log.exception("ошибка завершения подпроцесса %s", what)

    try:
        code = await asyncio.wait_for(finished(), timeout=timeout_s)
    except asyncio.TimeoutError:
        await drain_cleanup()
        limit = f"{timeout_s:.0f} с" if timeout_s < 120 else f"{timeout_s / 60:.0f} мин"
        raise RuntimeError(f"{what} не уложился в {limit} — прерван") from None
    except asyncio.CancelledError:
        await drain_cleanup()
        raise
    except Exception:
        await drain_cleanup()
        raise
    if code != 0:
        raise RuntimeError(f"{what} завершился с ошибкой (код {code})")


def _seconds_from_clock(value: str) -> float | None:
    """Parse ffmpeg's ``HH:MM:SS.micro`` progress value without locale state."""
    try:
        hours, minutes, seconds = value.strip().split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        return None


def _ffmpeg_progress_parser(
    duration_hint: float | None, progress: _ProgressCallback
) -> Callable[[str], None]:
    """Turn ffmpeg ``-progress`` lines into a bounded source-download update."""
    last_seconds = -1.0

    def on_line(line: str) -> None:
        nonlocal last_seconds
        if not line.startswith("out_time="):
            return
        seconds = _seconds_from_clock(line.partition("=")[2])
        if seconds is None or seconds <= last_seconds:
            return
        last_seconds = seconds
        elapsed = _format_elapsed(seconds)
        if duration_hint and duration_hint > 0:
            # ffmpeg can report a final timestamp fractionally above duration;
            # reserve 100% for an actually successful process exit.
            fraction: float | None = min(0.99, max(0.0, seconds / duration_hint))
            progress(fraction, f"получаю исходный звук: {elapsed}")
        else:
            progress(None, f"получаю исходный звук: {elapsed}")

    return on_line


def _yt_dlp_progress_parser(
    progress: _ProgressCallback, diagnostics: list[str] | None = None,
) -> Callable[[str], None]:
    """Expose extractor phases, byte-only progress, and a bounded error tail."""
    from uvt.source_download import ffmpeg_progress_time

    pattern = re.compile(r"UVT_PROGRESS:\s*([0-9]+(?:[.,][0-9]+)?)%")
    byte_pattern = re.compile(r"UVT_BYTES:(\d+)")
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    last_fraction = None

    def on_line(line: str) -> None:
        nonlocal last_fraction
        matches = list(pattern.finditer(line))
        if matches:
            percent = min(99.0, max(0.0, float(matches[-1][1].replace(",", "."))))
            last_fraction = percent / 100.0
            progress(last_fraction, f"скачиваю звук со страницы: {percent:.0f}%")
            return
        if "UVT_PROGRESS:" in line:
            size = byte_pattern.search(line)
            detail = f"получено {int(size[1]) / 1024**2:.1f} МБ; размер файла неизвестен" if size else "получаю звук; размер файла неизвестен"
            progress(last_fraction, detail)
            return
        stamp = ffmpeg_progress_time(line)
        if stamp is not None:
            progress(last_fraction, f"получаю звук: обработано {stamp}; полный размер неизвестен")
            return
        clean = ansi.sub("", line).strip()
        if not clean:
            return
        if diagnostics is not None:
            diagnostics.append(clean[:800])
            del diagnostics[:-8]
        lowered = clean.lower()
        detail = None
        if "downloading" in lowered and "webpage" in lowered:
            detail = "получаю страницу видео…"
        elif "downloading" in lowered and any(kind in lowered for kind in ("m3u8", "mpd", "manifest")):
            detail = "проверяю доступные медиапотоки…"
        elif "downloading" in lowered and any(kind in lowered for kind in ("json", "metadata", "api")):
            detail = "получаю сведения о видео…"
        elif "retrying" in lowered or "retry " in lowered:
            detail = "сайт не ответил; выполняю ограниченную повторную попытку…"
        if detail is not None:
            progress(last_fraction, detail)

    return on_line


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


async def _download_media(
    url: str,
    dest_dir: Path,
    referer: str | None = None,
    out_name: str = "media.m4a",
    *,
    duration_hint: float | None = None,
    progress: _ProgressCallback | None = None,
) -> Path:
    """Скачивает только звук прямого медиапотока (mp4/m3u8/webm) через ffmpeg.

    Browser resource URLs are only hints. A URL that looks like ``.mp4`` can
    still return an HTML denial page, an expired token or a login wall. UVT
    deliberately does not try to replay browser cookies or bypass restrictions.
    """
    out = dest_dir / out_name
    log.info("скачиваю поток через ffmpeg: %.120s…", url)
    cmd = ["ffmpeg", "-v", "error", "-y", "-user_agent", _UA]
    if progress is not None:
        progress(0.0, "подключаюсь к исходному звуку…")
        cmd += ["-progress", "pipe:1", "-nostats"]
    if referer:
        # многие CDN отдают поток только со ссылающейся страницы
        cmd += ["-headers", f"Referer: {referer}\r\n"]
    cmd += ["-i", url, "-vn", "-acodec", "aac", "-b:a", "192k", str(out)]
    try:
        if progress is None:
            await _run_process(cmd, 900, "ffmpeg")
        else:
            await _run_process(
                cmd,
                900,
                "ffmpeg",
                on_line=_ffmpeg_progress_parser(duration_hint, progress),
            )
    except RuntimeError as exc:
        # Не даём частичному M4A победить позднее в page fallback по размеру.
        out.unlink(missing_ok=True)
        raise RuntimeError(
            "прямая ссылка не отдала доступный медиафайл: сайт мог потребовать "
            "авторизацию/куки, ограничить скачивание или вернуть страницу вместо видео. "
            "UVT не обходит такие ограничения. Используйте законно сохранённый "
            "локальный файл (`uvt dub <файл>`) либо публичную ссылку, которую "
            "поддерживает yt-dlp."
        ) from exc
    if not out.is_file() or out.stat().st_size < 10_000:
        out.unlink(missing_ok=True)
        raise RuntimeError(
            "поток скачался пустым или слишком коротким: сайт мог требовать "
            "авторизацию/куки либо не отдать медиа по этой ссылке. UVT не "
            "обходит доступ — используйте законно сохранённый локальный файл "
            "или публичный поддерживаемый источник."
        )
    log.info("поток скачан: %.1f МБ", out.stat().st_size / 1e6)
    if progress is not None:
        progress(1.0, "исходный звук получен")
    return out


def _probe_duration(path: Path) -> float | None:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


async def _download_page(
    page_url: str, dest_dir: Path, *, progress: _ProgressCallback | None = None
) -> Path:
    """Скачивает ролик по адресу страницы через yt-dlp (асинхронно, убиваемо)."""
    from uvt.dub import _find_ytdlp, _ytdlp_js_args
    from uvt.source_download import run_source_download, SourceDownloadTimeout

    ytdlp = _find_ytdlp()
    if ytdlp is None:
        raise RuntimeError("для ссылок нужен yt-dlp: pip install yt-dlp")
    log.info("скачиваю ролик через yt-dlp…")
    if progress is not None:
        progress(0.0, "подключаюсь к странице через yt-dlp…")
    def build_command(extra_args: list[str] | None = None) -> list[str]:
        cmd = [
            ytdlp,
            "--ignore-config",
            *_ytdlp_js_args(),
            *(extra_args or []),
            # Серверу нужен только звук: сначала отдельная original-дорожка,
            # затем любой audio-only формат. Если у сайта только muxed-видео,
            # берём наименьший аудио-содержащий вариант.
            "--no-playlist",
            "--no-color",
            "--newline", "--progress",
            "--socket-timeout", "15",
            "--retries", "2", "--fragment-retries", "2", "--extractor-retries", "1",
            "--retry-sleep", "2",
            "-f", _YT_DLP_AUDIO_SELECTOR,
            "--progress-delta", "3",
        ]
        cmd += ["--progress-template", "download:UVT_PROGRESS:%(progress._percent_str)s;UVT_BYTES:%(progress.downloaded_bytes)s"]
        cmd += ["-o", str(dest_dir / "%(title).80s.%(ext)s"), "--", page_url]
        return cmd

    async def run_attempt(
        diagnostics: list[str], extra_args: list[str] | None = None
    ) -> None:
        callback = progress or (lambda _fraction, _detail: None)
        await run_source_download(
            _run_process, build_command(extra_args),
            _yt_dlp_progress_parser(callback, diagnostics), progress,
        )

    files_before = set(dest_dir.iterdir())
    diagnostics: list[str] = []
    failure: RuntimeError | None = None
    try:
        await run_attempt(diagnostics)
    except RuntimeError as exc:
        failure = exc

    diagnostic_text = "\n".join(diagnostics).lower()
    host = (urlsplit(page_url).hostname or "").lower()
    is_youtube = (
        host in {"youtu.be", "youtube.com", "youtube-nocookie.com"}
        or host.endswith(".youtube.com")
        or host.endswith(".youtube-nocookie.com")
    )
    retry_android = failure is not None and is_youtube and any(
        marker in diagnostic_text
        for marker in ("http error 403", "unable to download video data", "sabr")
    )
    if retry_android:
        for path in dest_dir.iterdir():
            if path not in files_before and (path.is_file() or path.is_symlink()):
                path.unlink(missing_ok=True)
        log.info(
            "YouTube не отдал отдельный аудиопоток — повторяю через совместимый android-клиент"
        )
        if progress is not None:
            progress(0.0, "YouTube не отдал отдельный звук; пробую совместимый поток…")
        diagnostics = []
        try:
            await run_attempt(diagnostics, _YT_DLP_YOUTUBE_ANDROID_ARGS)
            failure = None
        except RuntimeError as exc:
            failure = exc

    if failure is not None:
        if isinstance(failure, SourceDownloadTimeout):
            raise RuntimeError(f"Не удалось получить источник: {failure}") from None
        # Some ordinary HTML5/Playerjs sites have public media URLs but no
        # dedicated yt-dlp extractor. Read only their explicit player config.
        if "unsupported url" in "\n".join(diagnostics).lower():
            import httpx
            from uvt.media_discovery import discover_page_media

            if progress is not None:
                progress(0.0, "yt-dlp не знает этот сайт; проверяю ссылки видеоплеера…")
            try:
                page_candidates = await discover_page_media(page_url)
            except (OSError, ValueError, httpx.HTTPError):
                page_candidates = []
            for index, candidate in enumerate(page_candidates):
                try:
                    return await _download_media(
                        candidate, dest_dir, referer=page_url,
                        out_name=f"page_media_{index}.m4a", progress=progress,
                    )
                except RuntimeError:
                    log.info("публичный поток плеера %d недоступен; пробую следующий", index + 1)
        reason = next(
            (line for line in reversed(diagnostics) if "ERROR:" in line),
            str(failure),
        )
        detail = f" Причина yt-dlp: {reason}" if reason else ""
        raise RuntimeError(
            f"yt-dlp не поддержал или не смог скачать {page_url}.{detail} UVT не обходит "
            "авторизацию, DRM и ограничения сайта; используйте законно сохранённый "
            "локальный файл или публичную ссылку поддерживаемого сервиса."
        ) from failure
    files = sorted(
        (path for path in dest_dir.iterdir() if path.is_file()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not files:
        raise RuntimeError("yt-dlp ничего не скачал")
    if progress is not None:
        progress(1.0, "исходный звук получен")
    return files[0]


def _rank_candidate(url: str) -> int:
    """Отдаёт приоритет аудио и HLS/DASH перед muxed-видео.

    ``video.currentSrc`` нередко является большим MP4, хотя userscript уже
    заметил аудио-ресурс или master manifest. Для UVT достаточно звука, поэтому
    сначала пробуем такие кандидаты; muxed-файл остаётся штатным fallback.
    """
    lowered = url.lower()
    score = 0
    # Самый дешёвый безопасный вариант — явная аудиодорожка.
    if re.search(r"\.(?:m4a|mp3|aac|opus|ogg)(?:[?#]|$)|audio|/aud", lowered):
        score -= 6
    # HLS/DASH позволяет ffmpeg выбрать аудио из manifest, не скачивая
    # заведомо приоритетный video currentSrc первым.
    if re.search(r"\.m3u8(?:[?#]|$)|\.mpd(?:[?#]|$)|/(?:hls|dash)/", lowered):
        score -= 3
    if re.search(r"master|playlist|manifest", lowered):
        score -= 2
    # Обычный MP4/WebM на <video> почти всегда muxed. Не отбрасываем его —
    # сайт может не отдать manifest, — только переносим после audio/stream.
    if re.search(r"\.(?:mp4|webm|mov|mkv)(?:[?#]|$)", lowered):
        score += 2
    if re.search(r"(2160|1440|1080|720|480|360|240)p?|av1|h26[45]|hevc|vp9|video", lowered):
        score += 1
    return score


def _browser_media_candidates(data: dict) -> list[str]:
    """Собрать browser-provided источники в безопасном и быстром порядке.

    Сначала текущий ``media_url`` выбранного плеера, затем сетевые подсказки
    audio/HLS/DASH перед прочими MP4/WebM. yt-dlp запускается последним.
    """
    raw_primary = data.get("media_url")
    primary = raw_primary.strip() if isinstance(raw_primary, str) else None
    candidates: list[str] = []
    if primary:
        candidates.append(primary)

    extras = data.get("media_candidates")
    if isinstance(extras, (list, tuple)):
        for extra in extras:
            if isinstance(extra, str) and extra.strip() and extra.strip() not in candidates:
                candidates.append(extra.strip())

    if not candidates:
        return []
    # Python sort is stable, поэтому равные по типу ресурсы остаются в порядке,
    # в котором их увидел плеер/браузер.
    # currentSrc belongs to the selected video. Resource-timing hints may
    # belong to an ad player; prioritizing every HLS URL can dub an advert or
    # spend minutes attempting unrelated streams before the real MP4.
    ordered = sorted(candidates, key=_rank_candidate)
    if primary in ordered:
        ordered.remove(primary)
        ordered.insert(0, primary)
    return ordered[:_MAX_BROWSER_MEDIA_CANDIDATES]


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | awaiting_approval | done | error | cancelled
    # queue | download | transcribe | translate | synthesize | mix | terminal
    stage: str = "queue"
    stage_progress: float = 0.0
    detail: str = ""
    progress: float = 0.0
    audio_url: str | None = None
    downloads: dict[str, str] = field(default_factory=dict)
    source_name: str = ""
    failed_stage: str = ""
    entries: list = field(default_factory=list)
    # Реплики, готовые ещё до конца обработки: браузер проигрывает их сразу,
    # не дожидаясь полной дорожки (прогрессивный дубляж).
    clips: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    stage_started_at: float = field(default_factory=time.time)
    # Заполнено только пока status == awaiting_approval: облачный STT/перевод
    # не справился, и job ждёт /job/{id}/approve, прежде чем уйти на локальный
    # резерв (или упасть, если пользователь откажет).
    approval_kind: str | None = None
    approval_cause: str | None = None
    profile_name: str = ""
    engines: dict[str, str] = field(default_factory=dict)


class DubServer:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        route_label: str = "UVT",
        profile_name: str = "configured",
        listen_port: int | None = None,
        selectable_profiles: dict[str, AppConfig] | None = None,
        dashboard_routes: list[dict[str, object]] | None = None,
        settings_store: ServerSettingsStore | None = None,
        settings_key: str | None = None,
    ) -> None:
        self._base_cfg = cfg.model_copy(deep=True)
        self.route_label = route_label
        self._base_profile_name = profile_name
        self.listen_port = listen_port
        self.dashboard_routes = [dict(item) for item in (dashboard_routes or [])]
        self._profile_configs = {
            str(name): profile.model_copy(deep=True)
            for name, profile in (selectable_profiles or {}).items()
        }
        self._profile_configs.setdefault(profile_name, cfg.model_copy(deep=True))
        self.settings_store = settings_store or ServerSettingsStore.memory()
        self.settings_key = settings_key or settings_route_key(route_label)
        self._settings_kind = settings_kind(
            self._base_cfg, selectable_profiles=bool(selectable_profiles)
        )
        settings_entry = self.settings_store.get_entry(self.settings_key)
        self._settings_revision = int(settings_entry["revision"])
        self._settings_saved = bool(settings_entry["saved"])
        self._settings_load_error = self.settings_store.load_error
        persisted = settings_entry.get("settings")
        if isinstance(persisted, dict):
            try:
                persisted = normalize_settings(
                    persisted,
                    current=effective_settings(
                        self._base_cfg,
                        kind=self._settings_kind,
                        profile_name=self._base_profile_name,
                    ),
                    kind=self._settings_kind,
                    profile_ids=set(self._profile_configs),
                    local_voices=self._settings_voice_catalog(persisted),
                    voice_engine=self._settings_voice_engine(persisted),
                    voice_catalogs=self._settings_voice_catalogs(),
                )
            except ValueError as exc:
                self._settings_saved = False
                self._settings_load_error = f"сохранённые настройки не применены: {exc}"
                persisted = None
        self.cfg, self.profile_name = apply_settings(
            self._base_cfg,
            persisted if isinstance(persisted, dict) else None,
            kind=self._settings_kind,
            base_profile_name=self._base_profile_name,
            profiles=self._profile_configs,
        )
        if self._settings_saved and isinstance(persisted, dict):
            try:
                self._validate_piper_voice_support(self.cfg)
                self._validate_stt_language_support(self.cfg)
            except ValueError as exc:
                self.cfg = self._base_cfg.model_copy(deep=True)
                self.profile_name = self._base_profile_name
                self._settings_saved = False
                self._settings_load_error = (
                    f"сохранённые настройки не применены: {exc}"
                )
        # Пустое значение сохраняет localhost DX без обязательной настройки.
        # На удалённом личном сервере задайте UVT_API_TOKEN: тогда API нельзя
        # вызвать с чужой страницы без токена из userscript.
        self.api_token = os.environ.get("UVT_API_TOKEN", "").strip()
        self.jobs: dict[str, Job] = {}
        self._job_configs: dict[str, AppConfig] = {}
        self.audio_dir = _cache_dir()
        self._lock = asyncio.Lock()
        self._settings_lock = asyncio.Lock()
        # (источник, языки) → id задачи: повторное нажатие кнопки не пересчитывает
        self._job_cache: dict[tuple, str] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        # run_server обновляет это поле, если пользователь сознательно выбрал
        # не-loopback host; /meta не должен обещать localhost в таком случае.
        self.listen_host = "127.0.0.1"
        # Без истории мы не придумываем ETA. После нескольких готовых задач
        # используем только среднее локальной сессии и помечаем его как оценку.
        self._completed_job_seconds: list[float] = []
        # Терминал получает те же изменения, что /job, но без строки на каждый
        # ffmpeg tick: ключ — job id, значение — последний 5%-bucket.
        self._download_log_buckets: dict[str, int] = {}
        # У потока без известной длительности нет честного процента. В этом
        # случае выводим elapsed-время не чаще раза в 15 секунд.
        self._download_last_log_at: dict[str, float] = {}
        # Открытые запросы апрува на переход к локальному резерву, по job id.
        self._approval_gates: dict[str, ApprovalGate] = {}
        self._preview_cache: dict[tuple[str, ...], bytes] = {}
        self._provider_voice_cache: tuple[float, list[dict[str, str]]] | None = None
        # Shipped Apple-Silicon routes preload only their first heavy stage.
        # Translation remains lazy so two GPU models are never resident during
        # the same dubbing stage on a 16 GB machine.
        local_stt = str(self.cfg.stt.engine) in {"parakeet-mlx", "mlx-whisper", "nemotron-mlx"}
        self._model_readiness: dict[str, object] = {
            "status": "pending" if local_stt else "not-applicable",
            "detail": (
                "локальная модель распознавания ещё не загружена"
                if local_stt
                else "для этого маршрута локальный MLX STT не используется"
            ),
        }
        self._prepared_stt: STTEngine | None = None
        self._prepare_stt_task: asyncio.Task | None = None
        self._prepare_stt_lock = asyncio.Lock()
        self._local_preflight_done = False
        self._profile_setup_readiness: dict[str, dict[str, object]] = {}
        # Аудио запрашивается тегом <audio>, куда нельзя положить заголовок.
        # Поэтому готовая дорожка получает отдельный непредсказуемый токен;
        # основной API-токен в URL дорожки не попадает.
        self._audio_access_tokens: dict[str, str] = {}
        self._cleanup_audio_cache()

    def _record_local_preflight(self, report: dict[str, object]) -> None:
        """Derive per-profile installed state from one offline ``all`` check."""
        from uvt.setup_local import LOCAL_SETUP_PRESETS, preset_for_profile

        model_statuses = dict(report.get("models", {}) or {})
        piper_status = dict(report.get("piper", {}) or {})
        report_ready = bool(report.get("ready"))
        piper_ready = bool(piper_status.get("ready", report_ready))
        for name, cfg in self._profile_configs.items():
            preset_name = preset_for_profile(cfg)
            if preset_name is None:
                continue
            preset = LOCAL_SETUP_PRESETS[preset_name]
            required = preset.model_keys
            if model_statuses:
                missing = [
                    key
                    for key in required
                    if not bool(dict(model_statuses.get(key, {}) or {}).get("ready"))
                ]
                if preset.requires_piper and not piper_ready:
                    missing.append("piper")
                installed = not missing
            else:
                # Small plugin/test reports may only expose the aggregate bit.
                installed = report_ready
                missing = [] if installed else [preset_name]
            self._profile_setup_readiness[name] = {
                "installed": installed,
                "preset": preset_name,
                "missing": missing,
                "detail": (
                    "модели установлены"
                    if installed
                    else "не установлено: " + ", ".join(missing)
                ),
            }

    def _require_profile_ready(self, profile_name: str) -> None:
        readiness = self._profile_setup_readiness.get(profile_name)
        prepare_task = self._prepare_stt_task
        if readiness is None and prepare_task is not None and not prepare_task.done():
            raise ValueError(
                f"профиль '{profile_name}' ещё проверяет локальные модели; "
                "дождитесь статуса 'готов' и повторите"
            )
        readiness_status = str(self._model_readiness.get("status") or "")
        readiness_phase = str(self._model_readiness.get("phase") or "")
        if readiness is None and readiness_status == "error":
            raise ValueError(
                f"профиль '{profile_name}' не прошёл локальную проверку: "
                f"{self._model_readiness.get('detail') or 'неизвестная ошибка'}"
            )
        if readiness is not None and readiness.get("installed") is False:
            preset = str(readiness.get("preset") or profile_name.removeprefix("local-"))
            raise ValueError(
                f"профиль '{profile_name}' не подготовлен: "
                f"{readiness.get('detail')}; выполните "
                f"uvt setup-mac-local --preset {preset} и перезапустите сервер"
            )
        selected = self._profile_configs.get(profile_name)
        if (
            readiness_status == "error"
            and readiness_phase == "preload"
            and selected is not None
            and self._stt_signature(selected) == self._stt_signature(self.cfg)
        ):
            raise ValueError(
                f"профиль '{profile_name}' не загрузил локальное распознавание: "
                f"{self._model_readiness.get('detail') or 'неизвестная ошибка'}"
            )

    async def prepare_local_models(self) -> None:
        """Offline-preflight the route and keep its first STT model warm."""
        if str(self.cfg.stt.engine) not in {"parakeet-mlx", "mlx-whisper", "nemotron-mlx"}:
            return

        async with self._prepare_stt_lock:
            if self._prepared_stt is not None:
                return
            self._model_readiness = {
                "status": "checking",
                "detail": "проверяю pinned-модели и локальные голоса без сети",
            }

            error_phase = "preflight"
            try:
                from uvt.setup_local import (
                    format_setup_report,
                    preflight_mac_local,
                    preset_for_profile,
                )

                if not self._local_preflight_done:
                    preset = preset_for_profile(self.cfg)
                    if preset is not None:
                        selectable_presets = {
                            preset_for_profile(profile)
                            for profile in self._profile_configs.values()
                        }
                        check_preset = "all" if len(selectable_presets - {None}) > 1 else preset
                        report = await asyncio.to_thread(
                            preflight_mac_local, check_preset
                        )
                        self._record_local_preflight(report)
                        self._local_preflight_done = True
                        current = self._profile_setup_readiness.get(self.profile_name)
                        if current is not None and current.get("installed") is False:
                            raise RuntimeError(str(current.get("detail")))
                        if current is None and not report.get("ready"):
                            raise RuntimeError(format_setup_report(report))
                    self._local_preflight_done = True

                error_phase = "preload"
                self._model_readiness = {
                    "status": "loading",
                    "detail": f"загружаю {self.cfg.stt.engine} в Metal",
                }
                engine = create_stt_engine(self.cfg)
                try:
                    await engine.warmup()
                except BaseException:
                    await engine.close()
                    raise
            except asyncio.CancelledError:
                self._model_readiness = {
                    "status": "cancelled",
                    "detail": "подготовка локальной модели остановлена",
                }
                raise
            except Exception as exc:  # noqa: BLE001 - visible startup readiness
                detail = " ".join(str(exc).split())
                self._model_readiness = {
                    "status": "error",
                    "phase": error_phase,
                    "detail": detail,
                }
                log.error(
                    "локальный маршрут %s/%s не готов: %s",
                    self.route_label,
                    self.profile_name,
                    detail,
                )
                return

            self._prepared_stt = engine
            self._model_readiness = {
                "status": "ready",
                "detail": f"{self.cfg.stt.engine} загружен; остальные модели включатся по этапам",
            }
            log.info(
                "локальный маршрут %s/%s готов: %s",
                self.route_label,
                self.profile_name,
                self._model_readiness["detail"],
            )

    def schedule_local_model_prepare(self) -> None:
        """Re-warm STT while the server is idle after a completed job."""
        if str(self.cfg.stt.engine) not in {"parakeet-mlx", "mlx-whisper", "nemotron-mlx"}:
            return
        if self._prepared_stt is not None:
            return
        if self._model_readiness.get("status") == "error":
            return
        if any(
            job.status in {"queued", "running", "awaiting_approval"}
            for job in self.jobs.values()
        ):
            return
        if self._prepare_stt_task is not None and not self._prepare_stt_task.done():
            return
        self._prepare_stt_task = asyncio.create_task(
            self.prepare_local_models(),
            name=f"uvt-preload-{self.profile_name}",
        )

    @staticmethod
    def _stt_signature(cfg: AppConfig) -> tuple[str, str, str]:
        return (
            str(cfg.stt.engine),
            str(getattr(cfg.stt, "model", "") or ""),
            str(getattr(cfg.stt, "revision", "") or ""),
        )

    async def take_prepared_stt(self, cfg: AppConfig | None = None) -> STTEngine | None:
        """Transfer the idle preloaded STT engine to one render stage."""
        task = self._prepare_stt_task
        if task is not None and not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - prepare records its own error
                pass
        async with self._prepare_stt_lock:
            engine = self._prepared_stt
            requested_cfg = cfg or self.cfg
            if engine is not None and self._stt_signature(requested_cfg) != self._stt_signature(self.cfg):
                # M4/16 GB: do not retain the default Parakeet while a quality
                # job loads Whisper. The default model is warmed again later.
                self._prepared_stt = None
                await engine.close()
                self._model_readiness = {
                    "status": "on-demand",
                    "detail": f"{requested_cfg.stt.engine} загрузится для выбранного профиля",
                }
                return None
            if engine is not None:
                self._prepared_stt = None
                self._model_readiness = {
                    "status": "in-use",
                    "detail": f"{requested_cfg.stt.engine} распознаёт текущую задачу",
                }
            return engine

    async def close_prepared_models(self) -> None:
        """Drain startup worker and release an idle preloaded model on shutdown."""
        task = self._prepare_stt_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        async with self._prepare_stt_lock:
            engine = self._prepared_stt
            self._prepared_stt = None
            if engine is not None:
                await engine.close()

    def _clip_token(self, job: Job) -> str:
        """Токен доступа к дорожке задачи, выданный заранее.

        Клипы отдаются во время обработки, поэтому токен нельзя создавать в
        самом конце: он нужен уже на первой готовой реплике. Итоговая дорожка
        затем использует этот же токен.
        """
        token = self._audio_access_tokens.get(job.id)
        if not token:
            token = uuid.uuid4().hex
            self._audio_access_tokens[job.id] = token
        return token

    def _store_clip(self, job: Job, clip, clips_dir: Path) -> None:
        """Сохраняет готовую реплику и публикует её в статусе задачи."""
        import soundfile as sf

        clips_dir.mkdir(parents=True, exist_ok=True)
        path = clips_dir / f"{clip.index}.wav"
        temporary = clips_dir / f".{clip.index}.wav.tmp"
        # Формат задаётся явно: по расширению .tmp его не определить, а
        # переименование нужно, чтобы браузер не забрал недописанный файл.
        sf.write(
            temporary, clip.samples, clip.sample_rate, subtype="PCM_16", format="WAV"
        )
        os.replace(temporary, path)

        url = f"/clip/{job.id}/{clip.index}.wav"
        if self.api_token:
            url = f"{url}?access={self._clip_token(job)}"
        duration = len(clip.samples) / max(clip.sample_rate, 1)
        # ``at`` — то же положение, что и в итоговой дорожке: озвучка стартует
        # чуть раньше оригинала, так закадровый звучит синхроннее.
        item = {
            "index": clip.index,
            "at": round(max(0.0, clip.source_start - CLIP_LEAD_S), 3),
            "duration": round(duration, 3),
            "source_start": round(clip.source_start, 3),
            "source_end": round(clip.source_end, 3),
            "original": clip.original,
            "translated": clip.translated,
            "voice_style": clip.voice_style,
            "speaker": clip.speaker,
            "url": url,
        }
        # Реплики готовятся параллельно и приходят в произвольном порядке, а
        # список отдаётся наружу — держим его упорядоченным по таймкоду.
        position = len(job.clips)
        while position > 0 and job.clips[position - 1]["at"] > item["at"]:
            position -= 1
        job.clips.insert(position, item)
        job.updated_at = time.time()

    def _drop_clips(self, job_id: str) -> None:
        clips_dir = self.audio_dir / "clips" / job_id
        if not clips_dir.is_dir():
            return
        for file in clips_dir.iterdir():
            try:
                file.unlink()
            except OSError:
                continue
        try:
            clips_dir.rmdir()
        except OSError:
            pass

    def _cleanup_audio_cache(self, max_age_days: float = 7.0) -> None:
        """Remove old rendered tracks and shared browser sources on startup."""
        cutoff = time.time() - max_age_days * 86400
        video_results = getattr(self, "_video_results", None)
        if video_results:
            video_results.prune(cutoff)
        removed = 0
        for file in (*self.audio_dir.glob("*.m4a"), *self.audio_dir.glob("*.mkv")):
            try:
                if file.stat().st_mtime < cutoff:
                    file.unlink()
                    self._audio_access_tokens.pop(file.stem, None)
                    removed += 1
            except OSError:
                continue
        if removed:
            log.info("кэш дорожек: удалено %d старых файлов", removed)
        # Interrupted uploads are task-owned temporary data; clean only stale
        # files at startup so another active route cannot lose its upload.
        uploads = self.audio_dir / "uploads"
        if uploads.is_dir():
            for uploaded in uploads.iterdir():
                try:
                    if uploaded.is_file() and uploaded.stat().st_mtime < cutoff:
                        uploaded.unlink()
                except OSError:
                    continue
        source_removed = 0
        source_dir = self.audio_dir / "sources"
        if source_dir.is_dir():
            for file in source_dir.iterdir():
                try:
                    if file.is_file() and file.stat().st_mtime < cutoff:
                        file.unlink()
                        source_removed += 1
                except OSError:
                    continue
        if source_removed:
            log.info("кэш исходного звука: удалено %d старых файлов", source_removed)
        clips_root = self.audio_dir / "clips"
        if clips_root.is_dir():
            clip_removed = 0
            for directory in clips_root.iterdir():
                try:
                    if not directory.is_dir() or directory.stat().st_mtime >= cutoff:
                        continue
                except OSError:
                    continue
                self._drop_clips(directory.name)
                clip_removed += 1
            if clip_removed:
                log.info("кэш реплик: удалено %d старых задач", clip_removed)

    def _cached_source(self, cache_key: str) -> Path | None:
        source_dir = self.audio_dir / "sources"
        if not source_dir.is_dir():
            return None
        for path in source_dir.glob(f"{cache_key}.*"):
            try:
                if path.is_file() and path.stat().st_size > 0:
                    os.utime(path, None)
                    return path
            except OSError:
                continue
        return None

    def _store_cached_source(self, cache_key: str, source: Path) -> Path:
        """Atomically persist one downloaded source for every personal route."""
        source_dir = self.audio_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            suffix = ".media"
        destination = source_dir / f"{cache_key}{suffix}"
        temporary = source_dir / f".{cache_key}-{uuid.uuid4().hex}.part"
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _settings_voice_catalog(self, payload: dict[str, object]) -> list[dict[str, object]]:
        profile_id = str(payload.get("profile_id") or getattr(self, "profile_name", self._base_profile_name))
        profile = self._profile_configs.get(profile_id, self._base_cfg)
        return self._voice_catalog(profile)

    def _settings_voice_engine(self, payload: dict[str, object]) -> str:
        if self._settings_kind != "local":
            return str(self._base_cfg.tts.engine)
        profile_id = str(payload.get("profile_id") or getattr(self, "profile_name", self._base_profile_name))
        return str(self._profile_configs.get(profile_id, self._base_cfg).tts.engine)

    def _settings_voice_catalogs(self) -> dict[str, list[dict[str, object]]]:
        result: dict[str, dict[str, dict[str, object]]] = {}
        for cfg in self._profile_configs.values():
            bucket = result.setdefault(str(cfg.tts.engine), {})
            for voice in self._voice_catalog(cfg):
                bucket[str(voice["id"])] = voice
        return {engine: list(voices.values()) for engine, voices in result.items()}

    @staticmethod
    def _voice_catalog(cfg: AppConfig) -> list[dict[str, object]]:
        if str(cfg.tts.engine) == "f5":
            from uvt.voice_references import catalog
            return [{"id": "", "label": "Голос оригинала", "language": "ru", "languages": ["ru", "uk"], "gender": "auto", "engine": "f5", "installed": True}] + catalog()
        if str(cfg.tts.engine) == "moss-onnx":
            from uvt.engines.tts_moss_onnx import BUILTIN_VOICES, SUPPORTED_LANGUAGES, _local_path
            from uvt.voice_references import catalog as reference_catalog
            model = _local_path(getattr(cfg.tts, "model_path", None), ".models/moss-tts")
            codec = _local_path(getattr(cfg.tts, "codec_path", None), ".models/moss-codec")
            installed = all(path.is_file() for path in (
                model / "browser_poc_manifest.json", model / "tokenizer.model",
                model / "moss_tts_global_shared.data", model / "moss_tts_local_shared.data",
                codec / "codec_browser_onnx_meta.json",
                codec / "moss_audio_tokenizer_encode.onnx",
                codec / "moss_audio_tokenizer_decode_full.onnx",
            ))
            return [{"id": voice, "label": voice, "language": "ru",
                     "languages": sorted(SUPPORTED_LANGUAGES), "gender": gender,
                     "quality": "nano", "engine": "moss-onnx", "installed": installed}
                    for voice, gender in BUILTIN_VOICES.items()] + [dict(voice, engine="moss-onnx", installed=installed) for voice in reference_catalog() if set(voice.get("languages", [])) & SUPPORTED_LANGUAGES]
        if str(cfg.tts.engine) in {"f5", "indextts"}:
            # These profiles clone the source; a leftover Piper model catalog
            # must not pretend that fixed Piper voice ids control this engine.
            return [{"id": "", "label": "Голос оригинала", "language": "ru",
                     "languages": ["ru", "uk"], "gender": "auto",
                     "quality": "reference", "engine": str(cfg.tts.engine), "installed": True}]
        voice_models = dict(getattr(cfg.tts, "voice_models", {}) or {})
        voice_dir = Path(
            str(getattr(cfg.tts, "voice_dir", "") or "~/.local/share/uvt/piper")
        ).expanduser()
        voices: list[dict[str, object]] = []
        for raw_key, raw_model in voice_models.items():
            key = str(raw_key).lower()
            if ":" not in key:
                continue
            language, gender = key.split(":", 1)
            if gender not in {"male", "female"}:
                continue
            model_path = Path(str(raw_model)).expanduser()
            if not model_path.is_absolute():
                model_path = voice_dir / model_path
            voice_id = model_path.stem
            quality = voice_id.rsplit("-", 1)[-1]
            voices.append(
                {
                    "id": voice_id,
                    "label": _VOICE_LABELS.get(voice_id, voice_id),
                    "language": language,
                    "gender": gender,
                    "quality": quality,
                    "engine": str(cfg.tts.engine),
                    "installed": model_path.is_file() and Path(f"{model_path}.json").is_file(),
                }
            )
        return sorted(voices, key=lambda item: (str(item["language"]), str(item["gender"])))

    @classmethod
    def _compatible_voices(cls, cfg: AppConfig) -> list[dict[str, object]]:
        target_root = str(cfg.target_lang or "").replace("_", "-").split("-", 1)[0].lower()
        return [
            voice
            for voice in cls._voice_catalog(cfg)
            if target_root in local_voice_languages(voice) and voice["installed"] is True
        ]

    @classmethod
    def _validate_piper_voice_support(cls, cfg: AppConfig) -> None:
        if str(cfg.tts.engine) == "moss-onnx":
            from uvt.engines.tts_moss_onnx import SUPPORTED_LANGUAGES
            target = str(cfg.target_lang).replace("_", "-").split("-", 1)[0].lower()
            if target not in SUPPORTED_LANGUAGES:
                raise ValueError("MOSS-TTS не поддерживает этот язык; для украинского выберите Piper")
            requested = str(getattr(cfg.tts, "voice_id", "") or "")
            if requested and requested not in {"Adam", "Bella"}:
                raise ValueError(f"голос '{requested}' недоступен в MOSS-TTS")
            return
        if str(cfg.tts.engine) != "piper":
            return
        catalog = cls._voice_catalog(cfg)
        # Legacy/custom single-model Piper configs do not have the structured
        # language:gender catalog and keep their existing runtime validation.
        if not catalog:
            return
        compatible = cls._compatible_voices(cfg)
        requested_voice = str(getattr(cfg.tts, "voice_id", "") or "")
        if requested_voice:
            if not any(voice["id"] == requested_voice for voice in compatible):
                raise ValueError(f"голос '{requested_voice}' не установлен")
            return
        gender = str(getattr(cfg.tts, "voice_gender", "auto") or "auto")
        required = {"male", "female"} if gender == "auto" else {gender}
        available = {str(voice["gender"]) for voice in compatible}
        missing = sorted(required - available)
        if missing:
            target = str(cfg.target_lang or "").lower()
            raise ValueError(
                f"для языка '{target}' нет установленного Piper-голоса "
                f"({', '.join(missing)}); добавьте его в tts.voice_models "
                "или выберите GPT/ElevenLabs"
            )

    @staticmethod
    def _validate_stt_language_support(cfg: AppConfig) -> None:
        if str(cfg.stt.engine) == "nemotron-mlx":
            source = str(cfg.source_lang or "auto").lower().replace("_", "-").split("-", 1)[0]
            if source not in NEMOTRON_LANGUAGE_IDS:
                raise ValueError("Nemotron не поддерживает выбранный язык оригинала")
            return
        if str(cfg.stt.engine) != "parakeet-mlx":
            return
        source = (
            str(cfg.source_lang or "auto")
            .replace("_", "-")
            .split("-", 1)[0]
            .lower()
        )
        if source not in PARAKEET_LANGUAGE_IDS:
            raise ValueError(
                "Parakeet не поддерживает выбранный язык; выберите один из 25 "
                "европейских языков или профиль «Качество» (Whisper)"
            )

    def _config_for_request(self, data: dict) -> tuple[AppConfig, str]:
        mode = str(data.get("settings_mode") or "legacy").strip().lower()
        if mode not in {"legacy", "server", "override"}:
            raise ValueError("settings_mode должен быть server или override")
        if mode == "server" or (mode == "legacy" and self._settings_saved):
            return self.cfg.model_copy(deep=True), self.profile_name
        requested_profile = str(data.get("profile_id") or data.get("profile") or self.profile_name)
        if self._settings_kind != "local":
            requested_profile = self.profile_name
        if requested_profile not in self._profile_configs:
            raise ValueError(f"профиль '{requested_profile}' недоступен; выберите: {', '.join(self._profile_configs)}")
        if self._settings_kind == "generic":
            cfg = self.cfg.model_copy(deep=True)
            for field in ("source_lang", "target_lang"):
                if data.get(field):
                    setattr(cfg, field, str(data[field]))
            gender = str(data.get("voice_gender") or cfg.tts.voice_gender or "auto").lower()
            if gender not in _VOICE_GENDERS:
                raise ValueError("voice_gender должен быть auto, male или female")
            cfg.tts.voice_gender = gender
            return cfg, requested_profile
        names = {"source_lang", "target_lang", "voice_gender", "male_voice_id", "female_voice_id", "voice_pairs"}
        names |= {"voice_id", "profile_id"} if self._settings_kind == "local" else {"stt_model", "translation_model", "tts_model", "tts_voice"}
        fields = {key: data[key] for key in names if key in data}
        if self._settings_kind == "local":
            fields["profile_id"] = requested_profile
        normalized = normalize_settings(
            fields,
            current=effective_settings(self.cfg, kind=self._settings_kind, profile_name=self.profile_name),
            kind=self._settings_kind, profile_ids=set(self._profile_configs),
            local_voices=self._settings_voice_catalog(fields),
            voice_engine=self._settings_voice_engine(fields),
            voice_catalogs=self._settings_voice_catalogs(),
            validate_target_availability=False,
        )
        cfg, profile_name = apply_settings(
            self._base_cfg, normalized, kind=self._settings_kind,
            base_profile_name=self._base_profile_name, profiles=self._profile_configs,
        )
        return cfg, profile_name

    def _profile_catalog(self) -> list[dict[str, object]]:
        catalog: list[dict[str, object]] = []
        for name, cfg in self._profile_configs.items():
            engines = {
                "stt": str(cfg.stt.engine),
                "translation": str(cfg.translation.engine),
                "tts": str(cfg.tts.engine),
            }
            catalog.append(
                {
                    "id": name,
                    "label": _LOCAL_PROFILE_LABELS.get(name, name),
                    "engines": engines,
                    "voices": self._voice_catalog(cfg),
                    "target_languages": sorted({language
                        for voice in self._voice_catalog(cfg)
                        for language in local_voice_languages(voice)}),
                    "loaded": (
                        self._stt_signature(cfg) == self._stt_signature(self.cfg)
                        and self._model_readiness.get("status") == "ready"
                    ),
                    "stt_loaded": (
                        self._stt_signature(cfg) == self._stt_signature(self.cfg)
                        and self._model_readiness.get("status") == "ready"
                    ),
                    "installed": self._profile_setup_readiness.get(name, {}).get(
                        "installed"
                    ),
                    "setup_preset": self._profile_setup_readiness.get(name, {}).get(
                        "preset"
                    ),
                    "source_languages": (
                        sorted(PARAKEET_LANGUAGE_IDS)
                        if str(cfg.stt.engine) == "parakeet-mlx"
                        else sorted(NEMOTRON_LANGUAGE_IDS)
                        if str(cfg.stt.engine) == "nemotron-mlx"
                        else []
                    ),
                }
            )
        return catalog

    def _has_active_jobs(self) -> bool:
        return self._lock.locked() or any(
            job.status in {"queued", "running", "awaiting_approval"}
            for job in self.jobs.values()
        )

    def _provider_status(self, cfg: AppConfig | None = None) -> list[dict[str, object]]:
        cfg = cfg or self.cfg
        providers: list[dict[str, object]] = []
        seen: set[str] = set()
        sections: list[tuple[str, object]] = []
        if str(cfg.stt.engine) == "openai-compatible" and not self._is_local_endpoint(
            str(getattr(cfg.stt, "base_url", "") or "")
        ):
            sections.append(("OpenAI STT", cfg.stt))
        if (
            str(cfg.translation.engine) == "openai-compatible"
            and not self._is_local_endpoint(
                str(getattr(cfg.translation, "base_url", "") or "")
            )
        ):
            sections.append(("OpenAI / GPT", cfg.translation))
        if str(cfg.tts.engine) in {"openai", "elevenlabs"}:
            sections.append(
                (
                    "OpenAI TTS"
                    if str(cfg.tts.engine) == "openai"
                    else "ElevenLabs",
                    cfg.tts,
                )
            )
        for label, section in sections:
            env_name = str(getattr(section, "api_key_env", "") or "").strip()
            if not env_name or env_name in seen:
                continue
            seen.add(env_name)
            providers.append(
                {
                    "label": label,
                    "env": env_name,
                    "configured": bool(os.environ.get(env_name, "").strip()),
                }
            )
        return providers

    def _settings_payload(self) -> dict[str, object]:
        voices = self._voice_catalog(self.cfg) if self._settings_kind == "local" else []
        profiles = self._profile_catalog() if self._settings_kind == "local" else []
        effective = effective_settings(
            self.cfg, kind=self._settings_kind, profile_name=self.profile_name
        )
        defaults = effective_settings(
            self._base_cfg,
            kind=self._settings_kind,
            profile_name=self._base_profile_name,
        )
        return {
            "api_version": 1,
            "route": {
                "id": self.settings_key,
                "label": self.route_label,
                "kind": self._settings_kind,
                "profile": self.profile_name,
                "port": self.listen_port,
            },
            "revision": self._settings_revision,
            "saved": self._settings_saved,
            "effective": effective,
            "defaults": defaults,
            "catalog": settings_catalog(
                self.cfg,
                kind=self._settings_kind,
                profiles=profiles,
                voices=voices,
            ),
            "engines": {
                "stt": str(self.cfg.stt.engine),
                "translation": str(self.cfg.translation.engine),
                "tts": str(self.cfg.tts.engine),
            },
            "provider_status": self._provider_status(),
            "can_save": not self._has_active_jobs(),
            "warning": (
                "файл сохранённых настроек не применён; подробности в терминале"
                if self._settings_load_error
                else None
            ),
            "notice": "изменения применятся к следующему переводу",
        }

    def _dashboard_request_allowed(self, request) -> bool:
        if self.api_token:
            return self._request_has_api_token(request)
        remote = str(request.remote or "").strip()
        if remote and not _is_loopback_host(remote):
            return False
        origin = request.headers.get("Origin")
        if not origin:
            return True
        try:
            parsed = urlparse(origin)
        except ValueError:
            return False
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and _is_loopback_host(str(parsed.hostname))
        )

    async def _apply_effective_settings(
        self, settings: dict[str, object] | None
    ) -> None:
        cfg, profile_name = apply_settings(
            self._base_cfg,
            settings,
            kind=self._settings_kind,
            base_profile_name=self._base_profile_name,
            profiles=self._profile_configs,
        )
        self._validate_piper_voice_support(cfg)
        self._validate_stt_language_support(cfg)
        stt_changed = self._stt_signature(cfg) != self._stt_signature(self.cfg)
        if stt_changed:
            try:
                await self.close_prepared_models()
            except Exception as exc:  # noqa: BLE001 - settings must stay disk/runtime consistent
                log.warning(
                    "не удалось корректно закрыть прежнюю STT-модель: %s",
                    type(exc).__name__,
                )
            self._prepare_stt_task = None
        self.cfg = cfg
        self.profile_name = profile_name
        self._preview_cache.clear()
        if stt_changed and str(cfg.stt.engine) in {"parakeet-mlx", "mlx-whisper", "nemotron-mlx"}:
            self._model_readiness = {
                "status": "pending",
                "detail": f"готовлю {cfg.stt.engine} для новых задач",
            }
            self.schedule_local_model_prepare()

    @staticmethod
    def _expected_revision(payload: dict[str, object]) -> int:
        value = payload.get("revision")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("передайте целочисленную revision из GET /settings")
        return value

    def _cache_key(
        self,
        data: dict,
        resolved: tuple[AppConfig, str] | None = None,
    ) -> tuple:
        source = data.get("page_url") or data.get("media_url") or data.get("file") or ""
        # Отсутствующий выбор должен значить текущий auto/manual маршрут
        # профиля, а не старый неявный male. Иначе запрос без voice_gender
        # мог получить из кэша дорожку, созданную с явным male voice.
        cfg, profile_name = resolved or self._config_for_request(data)
        self._require_profile_ready(profile_name)
        self._validate_piper_voice_support(cfg)
        self._validate_stt_language_support(cfg)
        return (
            str(source),
            str(data.get("mix_original") is True),
            str(data.get("export_video") is True),
            str(cfg.source_lang),
            str(cfg.target_lang),
            str(cfg.stt.engine),
            str(getattr(cfg.stt, "model", "") or ""),
            str(cfg.translation.engine),
            str(getattr(cfg.translation, "file_context_lines", 3)),
            str(getattr(cfg.translation, "model", "") or ""),
            str(cfg.tts.engine),
            str(getattr(cfg.tts, "model", "") or ""),
            str(getattr(cfg.tts, "voice", "") or ""),
            str(cfg.tts.voice_gender),
            str(cfg.tts.voice_id or ""),
            str(getattr(cfg.tts, "male_voice_id", "") or ""),
            str(getattr(cfg.tts, "female_voice_id", "") or ""),
            profile_name,
        )

    def _request_has_api_token(self, request) -> bool:
        """Проверяет opt-in токен без утечки его значения в логах."""
        if not self.api_token:
            return True
        supplied = request.headers.get("X-UVT-Token", "")
        return bool(supplied) and hmac.compare_digest(supplied, self.api_token)

    # --- видимый контракт batch-задачи ---

    @staticmethod
    def _clamp_progress(value: float) -> float:
        return min(1.0, max(0.0, value))

    def _set_stage(
        self,
        job: Job,
        stage: str,
        *,
        detail: str | None = None,
        stage_progress: float | None = None,
    ) -> None:
        """Обновляет этап одним местом, чтобы браузер не видел ложный live-статус."""
        now = time.time()
        if job.stage != stage:
            job.stage = stage
            job.stage_started_at = now
            job.stage_progress = 0.0
        if stage_progress is not None:
            job.stage_progress = round(self._clamp_progress(stage_progress), 3)
        job.detail = detail if detail is not None else _STAGE_DETAILS[stage]
        job.updated_at = now

    def _set_download_progress(
        self, job: Job, fraction: float | None, detail: str
    ) -> None:
        """Publish actual source-download state for /job polling.

        ``stage_progress`` is the true downloader percentage when known.
        ``progress`` retains a small monotonic overall slice for older browser
        clients that only render the legacy field.
        """
        stage_fraction = self._clamp_progress(fraction) if fraction is not None else 0.0
        job.progress = round(_DOWNLOAD_PROGRESS_SHARE * stage_fraction, 3)
        self._set_stage(
            job,
            "download",
            detail=detail,
            stage_progress=stage_fraction,
        )
        if fraction is not None:
            bucket = min(20, int(stage_fraction * 20))
            if self._download_log_buckets.get(job.id) != bucket:
                self._download_log_buckets[job.id] = bucket
                self._download_last_log_at[job.id] = time.monotonic()
                log.info("задача %s: получение звука %d%% — %s", job.id, bucket * 5, detail)
        else:
            now = time.monotonic()
            last_logged_at = self._download_last_log_at.get(job.id)
            if last_logged_at is None or now - last_logged_at >= 15:
                self._download_last_log_at[job.id] = now
                log.info("задача %s: получение звука — %s", job.id, detail)

    def _set_preprocess_progress(self, job: Job, stage: str, fraction: float, detail: str) -> None:
        if job.status != "running" or stage not in {"decode", "separate"}:
            return
        if stage == "separate":
            overall = _DOWNLOAD_PROGRESS_SHARE + (1 - _DOWNLOAD_PROGRESS_SHARE) * 0.15 * self._clamp_progress(fraction)
            job.progress = max(job.progress, round(overall, 3))
        self._set_stage(job, stage, detail=detail, stage_progress=fraction)

    def _set_render_progress(self, job: Job, done: int, total: int, *, preprocess_share: float = 0.0) -> None:
        """Переводит существующий progress render_dub_track в понятные этапы.

        Внутренний batch-конвейер уже сообщает 0–70 % для STT, 70–85 % для
        перевода, 85–97 % для TTS и финальный 100 % после сборки. Не меняем его
        API: только даём этой информации имена для браузера.
        """
        progress = self._clamp_progress(done / max(total, 1))
        overall = round(_DOWNLOAD_PROGRESS_SHARE + (1.0 - _DOWNLOAD_PROGRESS_SHARE) * (preprocess_share + (1 - preprocess_share) * progress), 3)
        if progress < 0.70:
            job.progress = overall
            self._set_stage(job, "transcribe", stage_progress=progress / 0.70)
        elif progress < 0.85:
            job.progress = overall
            self._set_stage(job, "translate", stage_progress=(progress - 0.70) / 0.15)
        elif progress < 0.97:
            job.progress = overall
            self._set_stage(job, "synthesize", stage_progress=(progress - 0.85) / 0.12)
        else:
            # mix начинается внутри render_dub_track после последнего TTS-progress;
            # упаковка WAV → M4A ниже остаётся в том же честном финальном этапе.
            # Не показываем 100 %, пока ffmpeg ещё не отдал конечную M4A.
            job.progress = round(
                _DOWNLOAD_PROGRESS_SHARE + (1.0 - _DOWNLOAD_PROGRESS_SHARE) * (preprocess_share + (1 - preprocess_share) * min(progress, 0.985)),
                3,
            )
            self._set_stage(
                job, "mix", stage_progress=min((progress - 0.97) / 0.03, 0.75)
            )

    def _request_approval(self, job: Job, kind: str, cause: str) -> None:
        """Переводит job в awaiting_approval; ждёт /job/{id}/approve без таймаута."""
        label = "распознавание" if kind == "stt" else "перевод"
        job.status = "awaiting_approval"
        job.approval_kind = kind
        job.approval_cause = cause
        job.detail = (
            f"облачное {label} не справилось ({cause}) — нужно решение: "
            "перейти на локальный резерв?"
        )
        job.updated_at = time.time()
        log.warning("задача %s: жду решения об апруве локального резерва (%s)", job.id, kind)

    @staticmethod
    def _is_local_endpoint(value: str) -> bool:
        try:
            host = (urlparse(value).hostname or "").lower()
        except ValueError:
            return False
        return host in {"localhost", "127.0.0.1", "::1"}

    @staticmethod
    def _safe_endpoint(value: object) -> str | None:
        """Показывает origin, но никогда query/учётные данные из конфигурации."""
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = urlparse(value)
        except ValueError:
            return None
        if not parsed.scheme or not parsed.hostname:
            return None
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port_number = parsed.port
        except ValueError:
            return None
        port = f":{port_number}" if port_number else ""
        return f"{parsed.scheme}://{host}{port}"

    def _engine_metadata(self, kind: str, section: object) -> dict:
        engine = str(getattr(section, "engine", "unknown") or "unknown")
        # base_url есть во всех конфигурациях STT/translation, но задействован
        # только OpenAI-совместимым движком. Не приписываем local Whisper
        # посторонний облачный endpoint из дефолтов.
        endpoint = (
            self._safe_endpoint(getattr(section, "base_url", None))
            if engine == "openai-compatible"
            else None
        )
        local_engines = {
            "stt": {"faster-whisper", "mlx-whisper", "parakeet-mlx", "nemotron-mlx", "dummy"},
            "translation": {
                "nllb-ct2",
                "translategemma-mlx",
                "hymt-mlx",
                "mlx-chat",
                "none",
                "passthrough",
                "dummy",
            },
            "tts": {"kokoro", "piper", "moss-onnx", "f5", "dummy", "none"},
        }
        cloud_engines = {
            "translation": {"google-free"},
            "tts": {"edge", "openai", "elevenlabs"},
        }
        if engine in local_engines.get(kind, set()):
            scope = "local"
        elif engine in cloud_engines.get(kind, set()):
            scope = "external"
        elif engine == "openai-compatible":
            scope = "local" if endpoint and self._is_local_endpoint(endpoint) else "external"
        else:
            # Плагины не обязаны сообщать, где работают: не обещаем приватность.
            scope = "unknown"
        item = {"kind": kind, "engine": engine, "scope": scope}
        if endpoint:
            item["endpoint"] = endpoint
        return item

    def _metadata(
        self, cfg: AppConfig | None = None, profile_name: str | None = None
    ) -> dict:
        cfg = cfg or self.cfg
        selected_profile = profile_name or self.profile_name
        engines = [
            self._engine_metadata("stt", cfg.stt),
            self._engine_metadata("translation", cfg.translation),
            self._engine_metadata("tts", cfg.tts),
        ]
        external = [item for item in engines if item["scope"] == "external"]
        unknown = [item for item in engines if item["scope"] == "unknown"]
        local = [item for item in engines if item["scope"] == "local"]
        if external and local:
            profile_kind = "mixed"
        elif external:
            profile_kind = "cloud"
        elif unknown:
            profile_kind = "unknown"
        else:
            profile_kind = "local"
        local_host = _is_loopback_host(self.listen_host)
        compatible_voices = self._compatible_voices(cfg)
        installed_voice_languages = sorted(
            {
                language for voice in self._voice_catalog(cfg)
                if voice["installed"] is True
                for language in local_voice_languages(voice)
            }
        )
        tts_engine = str(cfg.tts.engine)
        selectable_tts = bool(compatible_voices) if tts_engine in {"piper", "moss-onnx"} else tts_engine in {
            "openai",
            "elevenlabs",
        }
        selected_setup = self._profile_setup_readiness.get(selected_profile)
        if selected_setup is not None and selected_setup.get("installed") is False:
            selected_readiness: dict[str, object] = {
                "status": "missing",
                "detail": (
                    f"{selected_setup.get('detail')}; выполните "
                    f"uvt setup-mac-local --preset {selected_setup.get('preset')}"
                ),
            }
        elif self._stt_signature(cfg) == self._stt_signature(self.cfg):
            selected_readiness = dict(self._model_readiness)
        else:
            selected_readiness = {
                "status": "on-demand",
                "detail": f"{cfg.stt.engine} загрузится при запуске этого профиля",
            }
        return {
            "api_version": 2,
            "mode": "batch",
            "capabilities": {
                "workspace": True,
                "file_upload": True,
                "subtitle_export": True,
                "batch_dubbing": True,
                "live_translation": False,
                "streaming_audio": False,
                "profile_selection": len(self._profile_configs) > 1,
                "voice_selection": selectable_tts,
                "tts_preview": selectable_tts,
            },
            "profile": {
                "name": selected_profile,
                "kind": profile_kind,
                "source_lang": cfg.source_lang,
                "target_lang": cfg.target_lang,
                "engines": {item["kind"]: item["engine"] for item in engines},
            },
            "route": {
                "label": self.route_label,
                "profile": selected_profile,
                "port": self.listen_port,
            },
            "model_readiness": selected_readiness,
            "defaults": {
                "profile_id": self.profile_name,
                "voice_gender": str(self.cfg.tts.voice_gender or "auto"),
                "voice_id": getattr(self.cfg.tts, "voice_id", None),
            },
            "profiles": self._profile_catalog(),
            "voices": self._voice_catalog(cfg),
            "limits": {
                "preview_text_chars": _PREVIEW_MAX_CHARS,
                "local_tts_languages": installed_voice_languages,
            },
            "privacy": {
                "server_scope": "localhost" if local_host else "network",
                "browser_input": "Браузер передаёт URL страницы и медиапотока; исходный звук скачивает локальный UVT-сервер.",
                "storage": "Временные исходные файлы удаляются после задачи; готовая M4A-дорожка хранится в локальном кэше до 7 дней.",
                "data_leaves_device": bool(external),
                "external_components": external,
                "unknown_components": unknown,
                "local_components": local,
                "notice": (
                    "Режим batch: это подготовка готовой дорожки, а не live-перевод. "
                    "Локальный сервер сам по себе не делает внешние движки приватными."
                ),
            },
        }

    def _metadata_for_request(self, data: dict) -> dict:
        """Отражает выбранные языки конкретной кнопки, не только дефолт сервера."""
        cfg, profile_name = self._config_for_request(data)
        return self._metadata(cfg, profile_name)

    def _queue_stats(self, job: Job) -> tuple[int | None, int, float | None]:
        """(позиция, задач впереди, оценка до готовности в секундах)."""
        active = sorted(
            (
                item
                for item in self.jobs.values()
                if item.status in {"queued", "running", "awaiting_approval"}
            ),
            key=lambda item: item.created_at,
        )
        if job.status == "queued":
            try:
                ahead = active.index(job)
            except ValueError:
                ahead = 0
            position: int | None = ahead + 1
        else:
            ahead = 0
            position = None

        if not self._completed_job_seconds:
            return position, ahead, None
        average = sum(self._completed_job_seconds) / len(self._completed_job_seconds)
        now = time.time()
        if job.status == "queued":
            # У каждого элемента очереди пока только средняя длительность —
            # это сознательно помечается в ответе как estimate.
            eta = average * (ahead + 1)
        elif job.status in ("running", "awaiting_approval") and job.started_at is not None:
            eta = max(0.0, average - (now - job.started_at))
        else:
            eta = 0.0 if job.status == "done" else None
        return position, ahead, round(eta, 1) if eta is not None else None

    def _job_payload(self, job: Job) -> dict:
        now = time.time()
        queue_position, queue_ahead, eta_seconds = self._queue_stats(job)
        elapsed_from = job.started_at or job.created_at
        video = getattr(self, "_video_results", None)
        video_payload = video.payload(job) if video and job.status == "done" else {}
        payload = asdict(job)
        payload.update(video_payload)
        job_cfg = self._job_configs.get(
            job.id, self._profile_configs.get(job.profile_name, self.cfg)
        )
        job_meta = self._metadata(job_cfg, job.profile_name or self.profile_name)
        payload.update(
            {
                "mode": "batch",
                "is_live": False,
                "route": job_meta["route"],
                "engines": job.engines or job_meta["profile"]["engines"],
                "queue_position": queue_position,
                "queue_ahead": queue_ahead,
                "eta_seconds": eta_seconds,
                "eta_is_estimate": eta_seconds is not None
                and job.status in {"queued", "running", "awaiting_approval"},
                "timing": {
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                    "elapsed_seconds": round(max(0.0, (job.finished_at or now) - elapsed_from), 1),
                    "stage_elapsed_seconds": round(max(0.0, (job.finished_at or now) - job.stage_started_at), 1),
                },
            }
        )
        return payload

    # --- обработка задач ---

    async def _resolve_source(
        self,
        data: dict,
        workdir: Path,
        *,
        progress: _ProgressCallback | None = None,
    ) -> Path:
        file_path = data.get("file")
        if file_path:
            path = Path(file_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(path)
            if progress is not None:
                progress(1.0, "локальный исходный файл готов")
            return path

        raw_page_url = data.get("page_url")
        page_url = raw_page_url.strip() if isinstance(raw_page_url, str) else None
        # Быстрый путь: src текущего плеера и замеченные браузером manifest/
        # audio resources. Это не требует, чтобы yt-dlp уже знал конкретный
        # сайт, и обычно не скачивает целый видеоролик.
        candidates = _browser_media_candidates(data)

        # Длительность видео в плеере — фильтр от роликов-превью related-видео
        try:
            duration_hint = float(data.get("duration_hint") or 0) or None
        except (TypeError, ValueError):
            duration_hint = None

        last_error: RuntimeError | None = None
        for index, candidate in enumerate(candidates):
            progress_callback: _ProgressCallback | None = None
            if progress is not None:
                attempt = f"поток {index + 1} из {len(candidates)}"

                def progress_callback(fraction: float | None, detail: str, *, _attempt=attempt) -> None:
                    progress(fraction, f"{_attempt}: {detail}")

            try:
                kwargs = {
                    "duration_hint": duration_hint,
                    "progress": progress_callback,
                }
                # Keep direct unit/plugin callers compatible: only extend the
                # downloader call when a live job actually asked for progress.
                if progress_callback is None and duration_hint is None:
                    kwargs = {}
                path = await _download_media(
                    candidate,
                    workdir,
                    referer=page_url,
                    out_name=f"media_{index}.m4a",
                    **kwargs,
                )
            except RuntimeError as exc:
                last_error = exc
                log.info("поток не подошёл: %.100s", candidate)
                if progress is not None:
                    progress(0.0, f"поток {index + 1} недоступен, пробую следующий…")
                continue
            if duration_hint:
                actual = _probe_duration(path)
                if actual and abs(actual - duration_hint) > max(15.0, duration_hint * 0.1):
                    log.info(
                        "поток не совпал по длительности (%.0f с вместо ~%.0f с) — похоже, превью",
                        actual, duration_hint,
                    )
                    last_error = RuntimeError(
                        "найденные потоки не совпали с длительностью видео в плеере"
                    )
                    path.unlink(missing_ok=True)
                    continue
            return path

        # Медиа-подсказки браузера могут быть временными/защищёнными. Только
        # после их исчерпания используем более медленный page extractor.
        if page_url:
            if candidates:
                log.info(
                    "браузерные медиапотоки не подошли — пробую страницу через yt-dlp (%d шт.)",
                    len(candidates),
                )
            try:
                if progress is None:
                    return await _download_page(page_url, workdir)
                return await _download_page(page_url, workdir, progress=progress)
            except RuntimeError as yt_error:
                if last_error is not None:
                    log.info("yt-dlp тоже не справился после медиапотоков: %s", yt_error)
                raise yt_error

        if last_error is not None:
            raise last_error
        raise RuntimeError("не передан ни адрес страницы, ни ссылка на поток, ни файл")

    async def _resolve_cached_source(
        self,
        data: dict,
        workdir: Path,
        *,
        progress: _ProgressCallback | None = None,
    ) -> Path:
        """Reuse downloaded source audio across Free, GPT and ElevenLabs jobs."""
        cache_key = _source_cache_key(data)
        if cache_key is None:
            return await self._resolve_source(data, workdir, progress=progress)

        cached = self._cached_source(cache_key)
        if cached is not None:
            detail = "исходный звук взят из общего кэша — повторно не скачиваю"
            log.info("кэш исходного звука: использую %s", cached.name)
            if progress is not None:
                progress(1.0, detail)
            return cached

        lock = _SOURCE_CACHE_LOCKS.setdefault(cache_key, asyncio.Lock())
        async with lock:
            # Another route may have completed the download while this job was
            # waiting for the shared key.
            cached = self._cached_source(cache_key)
            if cached is not None:
                detail = "исходный звук взят из общего кэша — повторно не скачиваю"
                log.info("кэш исходного звука: использую %s", cached.name)
                if progress is not None:
                    progress(1.0, detail)
                return cached

            source = await self._resolve_source(data, workdir, progress=progress)
            cached = self._store_cached_source(cache_key, source)
            log.info(
                "исходный звук сохранён в общий кэш: %s (%.1f МБ)",
                cached.name,
                cached.stat().st_size / 1e6,
            )
            return cached

    async def _run_job(
        self,
        job: Job,
        data: dict,
        cfg_snapshot: AppConfig | None = None,
        profile_snapshot: str | None = None,
    ) -> None:
        handed_stt: STTEngine | None = None
        stt_stage_started = False
        try:
            async with self._lock:  # по одной задаче: Whisper не параллелим
                now = time.time()
                job.status = "running"
                job.started_at = now
                job.updated_at = now
                self._set_stage(job, "download")
                if cfg_snapshot is None:
                    cfg, selected_profile = self._config_for_request(data)
                else:
                    cfg = cfg_snapshot.model_copy(deep=True)
                    selected_profile = profile_snapshot or job.profile_name
                job.profile_name = selected_profile

                preprocess_share = 0.15 if cfg.separation.enabled else 0.0

                def on_progress(done: int, total: int) -> None:
                    self._set_render_progress(job, done, total, preprocess_share=preprocess_share)

                def on_stage(stage: str, fraction: float, detail: str) -> None:
                    self._set_preprocess_progress(job, stage, fraction, detail)

                def on_download_progress(fraction: float | None, detail: str) -> None:
                    self._set_download_progress(job, fraction, detail)

                def publish_clip(clip) -> None:
                    # Реплика готова окончательно — публикуем сразу, чтобы
                    # браузер начал озвучивать видео, не дожидаясь остальных.
                    self._store_clip(job, clip, self.audio_dir / "clips" / job.id)

                approval = ApprovalGate(
                    on_request=lambda kind, cause: self._request_approval(job, kind, cause)
                )
                self._approval_gates[job.id] = approval

                with tempfile.TemporaryDirectory(prefix="uvt-serve-") as td:
                    source = await self._resolve_cached_source(
                        data,
                        Path(td),
                        progress=on_download_progress,
                    )
                    # A downloader calls this on success too; repeat it here
                    # for plugin/local sources that only return a path.
                    self._set_download_progress(job, 1.0, "исходный звук получен")
                    self._set_stage(job, "decode")
                    stt_stage_started = True
                    if cfg.separation.enabled:
                        # On 16 GB keep Parakeet/Whisper out of memory while
                        # Demucs processes the source; render loads STT later.
                        await self.close_prepared_models()
                        self._model_readiness = {
                            "status": "on-demand",
                            "detail": "распознавание загрузится после отделения речи, чтобы сэкономить память",
                        }
                    else:
                        handed_stt = await self.take_prepared_stt(cfg)
                    # Для браузера — только голос перевода: оригинал играет сам
                    # плеер на странице (приглушённо), иначе звук двоится.
                    video_results = getattr(self, "_video_results", None)
                    if video_results:
                        await video_results.retain_source(job, source, data)
                    mixed, entries = await render_dub_track(
                        cfg,
                        source,
                        progress=on_progress,
                        mix_original=data.get("mix_original") is True,
                        approval=approval,
                        stt_engine=handed_stt,
                        on_clip=publish_clip,
                        on_stage=on_stage,
                    )

                    import soundfile as sf

                    self._set_stage(job, "mix", stage_progress=max(job.stage_progress, 0.78))
                    wav = Path(td) / "mix.wav"
                    await _run_blocking(sf.write, wav, mixed, 48000, subtype="PCM_16")
                    del mixed
                    audio_path = self.audio_dir / f"{job.id}.m4a"
                    await _run_process(
                        [
                            "ffmpeg", "-v", "error", "-y", "-i", str(wav),
                            "-c:a", "aac", "-b:a", "160k", str(audio_path),
                        ],
                        1800, "сохранение переведённой дорожки",
                    )
                    if data.get("export_video") is True:
                        self._set_stage(job, "mix", detail="сохраняю видео с переводом; видеоряд копируется без перекодирования")
                        await _run_process(
                            ["ffmpeg", "-v", "error", "-y", "-i", str(source),
                             "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0",
                             "-map", "0:a?", "-c", "copy", "-disposition:a", "0",
                             "-disposition:a:0", "default", "-metadata:s:a:0", "title=UVT translation",
                             str(self.audio_dir / f"{job.id}.mkv")],
                            1800, "сохранение видео с переводом",
                        )

                job.entries = [asdict(e) for e in entries]
                if self.api_token:
                    # Тот же токен, что уже ушёл в браузер вместе с клипами:
                    # новый сделал бы выданные ссылки недействительными.
                    audio_token = self._clip_token(job)
                    job.audio_url = f"/audio/{job.id}.m4a?access={audio_token}"
                else:
                    job.audio_url = f"/audio/{job.id}.m4a"
                access_query = f"?access={self._clip_token(job)}" if self.api_token else ""
                formats = ["m4a", "srt", "vtt", "txt", "json"]
                if data.get("export_video") is True:
                    formats.append("mkv")
                job.downloads = {
                    kind: f"/download/{job.id}/{kind}{access_query}" for kind in formats
                }
                job.progress = 1.0
                job.status = "done"
                job.finished_at = time.time()
                self._set_stage(job, "done", stage_progress=1.0)
                if video_results:
                    try:
                        video_results.persist(job)
                    except OSError as exc:
                        log.warning("не удалось сохранить историю готового перевода %s: %s", job.id, exc)
                    if data.get("workspace_video") and video_results.records.get(job.id, {}).get("has_video") is not False:
                        video_results.start(job)

                if job.started_at is not None:
                    self._completed_job_seconds = (
                        self._completed_job_seconds + [max(0.0, job.finished_at - job.started_at)]
                    )[-8:]
                log.info("задача %s готова: %d реплик", job.id, len(entries))
        except asyncio.CancelledError:
            # Ловим отмену и при ожидании lock: иначе задача навсегда оставалась
            # бы queued и вводила браузер в заблуждение.
            job.status = "cancelled"
            job.finished_at = time.time()
            self._set_stage(job, "cancelled", stage_progress=1.0)
            # Уже озвученные реплики отменённой задачи не понадобятся.
            job.clips = []
            self._drop_clips(job.id)
            log.info("задача %s отменена", job.id)
        except Exception as exc:  # noqa: BLE001 — статус уходит клиенту
            job.failed_stage = job.stage
            job.status = "error"
            job.finished_at = time.time()
            # Keep the selected route in the browser-visible error as well as
            # in the terminal log. This matters when several personal servers
            # share one userscript: a paid 402 must never look like a Free
            # route failure.
            route_context = f"{self.route_label}/{job.profile_name or self.profile_name}"
            error_detail = f"маршрут {route_context}: {exc}"
            self._set_stage(job, "error", detail=error_detail, stage_progress=1.0)
            if isinstance(exc, (RuntimeError, FileNotFoundError)):
                # ожидаемые сбои (не скачалось, нет речи) — без простыни traceback
                log.error("задача %s [%s] провалилась: %s", job.id, route_context, exc)
            else:
                log.exception("задача %s [%s] провалилась", job.id, route_context)
        finally:
            if job.status in {"error", "cancelled"}:
                for suffix in ("m4a", "mkv"):
                    (self.audio_dir / f"{job.id}.{suffix}").unlink(missing_ok=True)
                job.downloads.clear()
                video_results = getattr(self, "_video_results", None)
                if video_results:
                    video_results.discard(job)
            if handed_stt is not None:
                # Normally released by the STT stage; the second idempotent
                # close also covers decode errors before transcription starts.
                await asyncio.gather(handed_stt.close(), return_exceptions=True)
                if job.status == "cancelled":
                    # Cancellation must become truly idle. Do not immediately
                    # start a well-intentioned preload that looks exactly like
                    # the ghost Metal worker the user just stopped.
                    self._model_readiness = {
                        "status": "cancelled",
                        "detail": "задача отменена; фоновых MLX-вычислений нет",
                    }
                elif self._model_readiness.get("status") != "error":
                    self._model_readiness = {
                        "status": "idle",
                        "detail": "задача завершена; повторно прогреваю STT в фоне",
                    }
                    self.schedule_local_model_prepare()
            elif stt_stage_started and job.status == "cancelled" and str(self.cfg.stt.engine) in {
                "parakeet-mlx", "mlx-whisper", "nemotron-mlx"
            }:
                self._model_readiness = {
                    "status": "cancelled",
                    "detail": "задача отменена; фоновых MLX-вычислений нет",
                }
            elif job.status != "cancelled":
                self.schedule_local_model_prepare()
            if self._tasks.get(job.id) is asyncio.current_task():
                self._tasks.pop(job.id, None)
            self._download_log_buckets.pop(job.id, None)
            self._download_last_log_at.pop(job.id, None)
            self._approval_gates.pop(job.id, None)

    # --- HTTP ---

    def app(self):
        from aiohttp import web

        @web.middleware
        async def cors(request, handler):
            if request.method == "OPTIONS":
                response = web.Response()
            else:
                try:
                    response = await handler(request)
                except web.HTTPException as exc:
                    response = exc
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-UVT-Token"
            if isinstance(response, web.HTTPException):
                raise response
            return response

        @web.middleware
        async def auth(request, handler):
            # Дорожка проверяет короткоживущий URL-токен в _get_audio: HTMLAudio
            # не позволяет передать X-UVT-Token. OPTIONS остаётся доступным для
            # корректного preflight userscript.
            public_dashboard = request.method == "GET" and request.path in {
                "/", "/workspace", "/workspace/app.js", "/workspace/style.css",
                "/workspace/userscript.user.js",
            }
            # Дорожка и отдельные реплики прогрессивного дубляжа проверяют
            # собственный URL-токен в своих обработчиках.
            token_in_url = request.path.startswith(("/audio/", "/clip/", "/download/"))
            if (
                request.method != "OPTIONS"
                and not public_dashboard
                and not token_in_url
            ):
                if not self._request_has_api_token(request):
                    raise web.HTTPUnauthorized(text="нужен заголовок X-UVT-Token")
            return await handler(request)

        async def options(_request):
            return web.Response()

        app = web.Application(middlewares=[cors, auth])
        app.router.add_route("OPTIONS", "/{tail:.*}", options)
        app.router.add_get("/", self._index)
        app.router.add_get("/meta", self._get_meta)
        app.router.add_get("/settings", self._get_settings)
        app.router.add_put("/settings", self._put_settings)
        app.router.add_delete("/settings", self._delete_settings)
        app.router.add_get("/provider/voices", self._get_provider_voices)
        app.router.add_post("/dub", self._post_dub)
        app.router.add_post("/tts/preview", self._post_tts_preview)
        app.router.add_post("/voices/reference", self._post_voice_reference)
        app.router.add_get("/job/{jid}", self._get_job)
        app.router.add_post("/job/{jid}/cancel", self._cancel_job)
        app.router.add_post("/job/{jid}/approve", self._approve_job)
        app.router.add_get("/audio/{name}", self._get_audio)
        app.router.add_get("/clip/{jid}/{name}", self._get_clip)
        from uvt.server_workspace import WorkspaceRoutes
        WorkspaceRoutes(self).register(app)
        return app

    async def _index(self, request):
        from aiohttp import web

        forwarded_host = request.headers.get("X-Forwarded-Host", "").split(",", 1)[0].strip()
        visible_host = forwarded_host or request.host
        forwarded_scheme = request.headers.get("X-Forwarded-Proto", "").split(",", 1)[0].strip()
        visible_scheme = forwarded_scheme if forwarded_scheme in {"http", "https"} else request.scheme
        try:
            request_hostname = urlsplit(f"//{visible_host}").hostname or self.listen_host
        except ValueError:
            request_hostname = self.listen_host
        current_url = f"{visible_scheme}://{visible_host.rstrip('/')}/"
        routes = self.dashboard_routes or [
            {
                "label": self.route_label,
                "profile": self.profile_name,
                "url": current_url,
                "engines": {
                    "stt": str(self.cfg.stt.engine),
                    "translation": str(self.cfg.translation.engine),
                    "tts": str(self.cfg.tts.engine),
                },
            }
        ]
        readiness = dict(self._model_readiness)
        readiness_status = str(readiness.get("status") or "pending")
        protected_shell = bool(self.api_token)
        server_ready = readiness_status in {"ready", "not-applicable", "idle", "in-use"}
        server_error = readiness_status in {"error", "missing", "cancelled"}
        if server_error:
            server_label = "Есть проблемы"
            server_state_class = "error"
        elif server_ready:
            server_label = "Сервер готов"
            server_state_class = "ready"
        else:
            server_label = "Сервер запускается"
            server_state_class = "pending"
        if protected_shell:
            server_label = "Требуется вход"
            server_state_class = "pending"

        def readiness_presentation(value: str) -> tuple[str, str]:
            return {
                "ready": ("Модели готовы", "ok"),
                "not-applicable": ("Маршрут готов", "ok"),
                "idle": ("Маршрут готов", "ok"),
                "in-use": ("Модель занята задачей", "ok"),
                "on-demand": ("Загрузится по запросу", "pending"),
                "pending": ("Ожидает подготовки", "pending"),
                "checking": ("Проверяю модели", "pending"),
                "loading": ("Загружаю модель", "pending"),
                "error": ("Ошибка модели", "error"),
                "missing": ("Модели не установлены", "error"),
                "cancelled": ("Подготовка остановлена", "error"),
            }.get(value, ("Состояние неизвестно", "pending"))

        route_rows: list[str] = []
        settings_routes: list[dict[str, object]] = []
        for route_number, item in enumerate(routes, start=1):
            label = html.escape(str(item.get("label") or "UVT"))
            profile = html.escape(str(item.get("profile") or "configured"))
            configured_url = str(item.get("url") or current_url)
            try:
                route_port = urlsplit(configured_url).port or self.listen_port or 8765
            except ValueError:
                route_port = self.listen_port or 8765
            route_url = (
                configured_url
                if item.get("public_url") is True
                else _dashboard_url(request_hostname, int(route_port))
            )
            route_url_escaped = html.escape(route_url, quote=True)
            meta_url = html.escape(f"{route_url.rstrip('/')}/meta", quote=True)
            engines = dict(item.get("engines") or {})
            engine_chain = " → ".join(
                str(engines.get(kind) or "?") for kind in ("stt", "translation", "tts")
            )
            if protected_shell:
                profile = "—"
                engine_chain = "доступно после входа"
            is_current = (
                str(item.get("label") or "").casefold() == self.route_label.casefold()
            )
            settings_routes.append(
                {
                    "id": settings_route_key(str(item.get("label") or "uvt")),
                    "label": str(item.get("label") or "UVT"),
                    "port": int(route_port),
                    "url": route_url.rstrip("/"),
                }
            )
            if is_current:
                status_text, status_class = readiness_presentation(readiness_status)
                status_detail = str(readiness.get("detail") or "")
                row_state = readiness_status
            else:
                status_text = "Проверяю маршрут"
                status_class = "pending"
                status_detail = ""
                row_state = "checking"
            if protected_shell:
                status_text, status_class = "Нужен токен", "pending"
                status_detail = ""
                row_state = "pending"
            route_rows.append(
                f"""
                <tr data-route data-route-port="{int(route_port)}" data-state="{html.escape(row_state, quote=True)}" data-meta-url="{meta_url}">
                  <td data-label="Маршрут">
                    <a class="route-link" href="{route_url_escaped}">
                      <span>{route_number}. {label}</span>
                      <svg aria-hidden="true" viewBox="0 0 20 20"><path d="m7 4 6 6-6 6"/></svg>
                    </a>
                  </td>
                  <td data-label="Профиль"><code data-profile>{profile}</code></td>
                  <td data-label="Движок"><code data-engines>{html.escape(engine_chain)}</code></td>
                  <td data-label="Состояние">
                    <span class="route-state {status_class}">
                      <svg class="status-icon" aria-hidden="true" viewBox="0 0 20 20">
                        <circle cx="10" cy="10" r="8"/>
                        <path class="status-check" d="m6.2 10.1 2.4 2.5 5.2-5.5"/>
                        <path class="status-error-mark" d="m7.1 7.1 5.8 5.8m0-5.8-5.8 5.8"/>
                      </svg>
                      <span data-status>{status_text}</span>
                    </span>
                    <small data-detail>{html.escape(status_detail)}</small>
                  </td>
                  <td data-label="Действия">
                    <button class="configure-link" type="button" data-configure-route="{html.escape(settings_route_key(str(item.get('label') or 'uvt')), quote=True)}">
                      Настроить
                    </button>
                    <a class="meta-link secondary" href="{meta_url}">
                      <svg aria-hidden="true" viewBox="0 0 20 20">
                        <path d="M11 3h6v6M17 3l-8 8M8 5H4a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-4"/>
                      </svg>
                      Открыть /meta
                    </a>
                  </td>
                </tr>
                """
            )

        active_jobs = [
            job
            for job in self.jobs.values()
            if job.status in {"queued", "running", "awaiting_approval"}
        ]
        if protected_shell:
            active_jobs = []
        if active_jobs:
            job_items = []
            for job in active_jobs:
                progress = min(100, max(0, round(float(job.progress) * 100)))
                job_items.append(
                    f"""
                    <li class="job-row">
                      <div><strong>{html.escape(job.id)}</strong>
                        <span>{html.escape(job.status)} · {html.escape(job.stage)}</span></div>
                      <div class="job-progress" aria-label="Прогресс {progress}%">
                        <span style="width:{progress}%"></span></div>
                      <div>{progress}%</div>
                      <small>{html.escape(job.detail)}</small>
                    </li>
                    """
                )
            jobs_html = f'<ul class="job-list">{"".join(job_items)}</ul>'
        else:
            jobs_html = f"""
                <div class="empty-jobs">
                  <svg aria-hidden="true" viewBox="0 0 32 32">
                    <path d="M5 13.5 10 6h12l5 7.5V25H5Z"/>
                    <path d="M5 16h7l2 3h4l2-3h7"/>
                  </svg>
                  <span>{"Данные задач доступны после входа" if protected_shell else "Активных задач нет"}</span>
                </div>
            """

        settings_routes_json = json.dumps(
            settings_routes, ensure_ascii=False, separators=(",", ":")
        ).replace("</", "<\\/")

        page = (
            "<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>UVT · Локальная панель</title>"
            """
            <style>
              :root { color-scheme: dark; --bg:#080d13; --surface:#0d141d;
                --border:#2a3441; --text:#f4f7fb; --muted:#9aa6b5;
                --blue:#6593ff; --green:#79c85a; --amber:#d8a84e; }
              * { box-sizing:border-box; }
              body { margin:0; min-height:100vh; background:var(--bg); color:var(--text);
                font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
              a { color:inherit; }
              a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible,
              textarea:focus-visible { outline:2px solid var(--blue); outline-offset:3px; }
              .shell { width:min(1586px,100%); margin:0 auto; min-height:100vh;
                border-inline:1px solid var(--border); display:flex; flex-direction:column; }
              header { display:flex; justify-content:space-between; align-items:center;
                gap:24px; padding:34px 46px; border-bottom:1px solid var(--border); }
              .brand { display:flex; align-items:center; gap:22px; min-width:0; }
              h1 { margin:0; font-size:42px; line-height:1; letter-spacing:.02em; }
              .subtitle { color:var(--muted); font-size:20px; border-left:1px solid #46505d;
                padding-left:22px; white-space:nowrap; }
              .server-state { display:flex; align-items:center; gap:10px; color:var(--green);
                font-size:19px; white-space:nowrap; }
              .server-icon { width:24px; height:24px; fill:currentColor; stroke:#07100a;
                stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round; }
              .server-state.pending { color:var(--amber); }
              .server-state.error { color:#ef7777; }
              main { padding:28px 48px 34px; flex:1; }
              .routes { width:100%; border-collapse:collapse; table-layout:fixed; }
              .routes th { color:var(--muted); font-weight:600; text-align:left;
                padding:14px 10px; border-bottom:1px solid var(--border); }
              .routes td { padding:22px 10px; border-bottom:1px solid var(--border);
                vertical-align:middle; overflow-wrap:anywhere; }
              .routes th:nth-child(1){width:17%}.routes th:nth-child(2){width:16%}
              .routes th:nth-child(3){width:35%}.routes th:nth-child(4){width:20%}
              .routes th:nth-child(5){width:12%}
              code { color:#c6d1e0; font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
              .route-link { display:inline-flex; align-items:center; gap:10px; color:var(--blue);
                font-size:23px; font-weight:700; text-decoration-thickness:1px;
                text-underline-offset:5px; }
              .route-link svg { width:20px; height:20px; fill:none; stroke:currentColor;
                stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round; }
              .route-state { display:flex; align-items:center; gap:8px; color:var(--amber); font-weight:650; }
              .route-state.ok { color:var(--green); }
              .route-state.error { color:#ef7777; }
              .status-icon { width:20px; height:20px; flex:0 0 auto; fill:none;
                stroke:currentColor; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
              .status-error-mark { display:none; }
              .route-state.error .status-check { display:none; }
              .route-state.error .status-error-mark { display:block; }
              .route-state.pending .status-check,.route-state.pending .status-error-mark { display:none; }
              td small { display:block; color:var(--muted); margin-top:3px; line-height:1.35; }
              .meta-link { display:inline-flex; justify-content:center; min-height:36px;
                align-items:center; gap:8px; padding:8px 12px; border:1px solid #4874d5;
                border-radius:7px; color:var(--blue); text-decoration:none;
                font-weight:650; white-space:nowrap; }
              .meta-link svg { width:19px; height:19px; fill:none; stroke:currentColor;
                stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
              .meta-link:hover { background:#101d34; }
              .configure-link,.route-tab,.action-button { min-height:44px; border:1px solid #4874d5;
                border-radius:8px; background:#132241; color:#dce7ff; padding:9px 14px;
                font:inherit; font-weight:700; cursor:pointer; }
              .configure-link:hover,.route-tab:hover,.action-button:hover { background:#1a2e55; }
              .meta-link.secondary { margin-top:7px; border-color:transparent; color:var(--muted);
                font-size:13px; padding:4px 0; min-height:28px; }
              .settings-panel { margin-top:26px; padding:26px 30px; border:1px solid var(--border);
                border-radius:11px; background:var(--surface); }
              .settings-heading { display:flex; align-items:flex-start; justify-content:space-between;
                gap:20px; margin-bottom:18px; }
              .settings-heading h2 { margin:0 0 4px; font-size:25px; }
              .settings-heading p,.field-help,.settings-notice,.provider-status { margin:0; color:var(--muted); }
              .route-tabs { display:flex; flex-wrap:wrap; gap:9px; margin:18px 0 22px; }
              .route-tab[aria-selected="true"] { background:var(--blue); border-color:var(--blue);
                color:#07101f; }
              .settings-form[aria-busy="true"] { opacity:.62; pointer-events:none; }
              .settings-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:17px 22px; }
              .field { display:flex; flex-direction:column; gap:7px; min-width:0; }
              .field.full { grid-column:1/-1; }
              .field label { font-weight:700; }
              .field select,.field input,.field textarea { width:100%; min-height:44px; border:1px solid #3a4758;
                border-radius:8px; background:#080e16; color:var(--text); padding:10px 12px; font:inherit; }
              .field textarea { min-height:88px; resize:vertical; }
              .engine-chain { min-height:44px; display:flex; align-items:center; padding:10px 12px;
                border:1px solid var(--border); border-radius:8px; color:#c6d1e0; overflow-wrap:anywhere; }
              [data-cloud-fields][hidden],[data-local-fields][hidden],[hidden] { display:none!important; }
              .settings-actions { display:flex; flex-wrap:wrap; align-items:center; gap:10px;
                margin-top:22px; padding-top:20px; border-top:1px solid var(--border); }
              .action-button.primary { background:var(--blue); border-color:var(--blue); color:#07101f; }
              .action-button.ghost { background:transparent; border-color:#485464; color:#c2ccd8; }
              .action-button:disabled { opacity:.45; cursor:not-allowed; }
              .settings-status { min-height:24px; margin-left:auto; color:var(--muted); }
              .settings-status.ok { color:var(--green); }.settings-status.error { color:#ef7777; }
              .preview-row { display:grid; grid-template-columns:minmax(0,1fr) auto auto; gap:10px; align-items:end; }
              .preview-row .field { min-width:0; }.preview-note { color:var(--amber); font-size:14px; }
              .auth-box { margin:0 0 22px; padding:18px; border:1px solid var(--amber);
                border-radius:9px; background:#211a0e; }
              .auth-row { display:flex; gap:10px; margin-top:12px; }
              .auth-row input { flex:1; min-height:44px; border:1px solid #66522e; border-radius:8px;
                background:#0c0e12; color:var(--text); padding:10px 12px; font:inherit; }
              .instruction { min-height:126px; margin-top:26px; padding:24px 30px;
                border:1px solid var(--border); border-radius:11px; background:var(--surface);
                display:flex; align-items:center; gap:24px; }
              .instruction > svg { width:52px; height:52px; flex:0 0 auto; fill:none;
                stroke:var(--blue); stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round;
                border-right:1px solid var(--border); padding-right:20px; box-sizing:content-box; }
              .instruction h2,.jobs h2 { margin:0 0 7px; font-size:19px; }
              .instruction p { margin:0; color:#b5bfcc; }
              .jobs { min-height:214px; margin-top:18px; padding:26px 30px;
                border:1px solid var(--border); border-radius:11px; }
              .empty-jobs { margin-top:18px; min-height:116px; display:flex; align-items:center;
                justify-content:center; gap:16px;
                color:#7f8997; border:1px dashed #3a4552; border-radius:8px; }
              .empty-jobs svg { width:34px; height:34px; fill:none; stroke:currentColor;
                stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
              .job-list { list-style:none; margin:16px 0 0; padding:0; }
              .job-row { display:grid; grid-template-columns:minmax(220px,1fr) minmax(140px,2fr) 50px;
                gap:12px; align-items:center; padding:13px 0; border-top:1px solid var(--border); }
              .job-row div:first-child { display:flex; flex-direction:column; }
              .job-row span,.job-row small { color:var(--muted); }
              .job-row small { grid-column:1/-1; }
              .job-progress { height:6px; background:#202a36; border-radius:10px; overflow:hidden; }
              .job-progress span { display:block; height:100%; background:var(--blue); }
              footer { display:flex; justify-content:space-between; gap:20px; padding:24px 48px;
                border-top:1px solid var(--border); color:#7f8997; }
              footer a { color:var(--blue); text-decoration:none; font-family:ui-monospace,monospace; }
              @media (max-width:820px) {
              main > section:first-child { flex-wrap:wrap; }
              .meta-link { max-width:100%; white-space:normal; text-align:center; }
              .field { min-width:0; }
                .shell { border:0; } header,main,footer { padding-inline:18px; }
                header { align-items:flex-start; } .brand { gap:12px; flex-wrap:wrap; }
                h1 { font-size:28px; }.subtitle { font-size:15px; padding-left:12px; }
                .server-state { font-size:14px; }
                .routes thead { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); }
                .routes,.routes tbody,.routes tr,.routes td { display:block; width:100%; }
                .routes tr { padding:14px 0; border-bottom:1px solid var(--border); }
                .routes td { display:grid; grid-template-columns:92px minmax(0,1fr); gap:10px;
                  padding:5px 0; border:0; }
                .routes td::before { content:attr(data-label); color:var(--muted); font-size:12px; }
                .meta-link { justify-self:start; }.job-row { grid-template-columns:1fr 42px; }
                .job-row div:first-child,.job-row small { grid-column:1/-1; }
                .instruction { align-items:flex-start; min-height:0; padding:20px; }
                .instruction > svg { width:34px; height:34px; padding-right:12px; }
                footer { flex-direction:column; }
                .settings-panel { padding:20px 16px; }.settings-heading { flex-direction:column; }
                .settings-grid { grid-template-columns:1fr; }.field.full { grid-column:auto; }
                .preview-row { grid-template-columns:1fr; }.settings-actions { align-items:stretch; }
                .settings-actions .action-button { flex:1 1 100%; }.settings-status { margin-left:0; }
                .auth-row { flex-direction:column; }
              }
              @media (prefers-reduced-motion:reduce) { * { scroll-behavior:auto!important; } }
            </style></head><body><div class="shell">
            """
            f"""
            <header><div class="brand"><h1>UVT</h1><div class="subtitle">Локальная панель</div></div>
              <div class="server-state {server_state_class}" role="status">
                <svg class="server-icon" aria-hidden="true" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10"/><path d="m7.5 12 3 3 6-7"/>
                </svg><span data-server-label>{server_label}</span>
              </div>
            </header><main>
              <section class="instruction"><div><h2>Перевести скачанное видео</h2>
                <p>Файлы, ссылки, субтитры и результаты в одном окне.</p></div>
                <a class="meta-link" href="/workspace">Открыть рабочую панель</a></section>
              <table class="routes"><thead><tr><th>Маршрут</th><th>Профиль</th>
                <th>Движок (цепочка)</th><th>Состояние</th><th>Действия</th></tr></thead>
                <tbody>{''.join(route_rows)}</tbody></table>
              <section class="settings-panel" id="settings" aria-labelledby="settings-title">
                <div class="settings-heading"><div><h2 id="settings-title">Настройки обработки</h2>
                  <p>Выбор здесь станет значением по умолчанию для следующих видео.</p></div>
                  <span class="provider-status" data-saved-state></span></div>
                <div class="auth-box" data-auth-box hidden>
                  <strong>Сервер защищён UVT_API_TOKEN</strong>
                  <div class="field-help">Введите токен из окружения сервера. Он хранится только до закрытия вкладки.</div>
                  <div class="auth-row"><input type="password" autocomplete="off" data-token-input aria-label="UVT API token">
                    <button class="action-button" type="button" data-token-submit>Войти</button></div>
                </div>
                <div class="route-tabs" role="tablist" aria-label="Маршрут обработки" data-settings-tabs></div>
                <form class="settings-form" data-settings-form aria-busy="true">
                  <div class="settings-grid">
                    <div class="field"><label for="setting-source">Язык оригинала</label>
                      <select id="setting-source" name="source_lang"></select></div>
                    <div class="field"><label for="setting-target">Язык перевода</label>
                      <select id="setting-target" name="target_lang"></select></div>
                    <div class="field" data-local-field><label for="setting-profile">Локальный профиль</label>
                      <select id="setting-profile" name="profile_id"></select>
                      <span class="field-help">Для диалогов с контекстом выбирайте профиль с Hy-MT2.</span></div>
                    <div class="field" data-local-field><label>Цепочка движков</label>
                      <div class="engine-chain" data-engine-chain></div></div>
                    <div class="field" data-voice-pair><label for="setting-male-voice">Мужской голос</label>
                    <select id="setting-male-voice" name="male_voice_id"></select>
                    <button type="button" class="action-button" data-role-preview="male">▶ Послушать мужской</button></div>
                  <div class="field" data-voice-pair><label for="setting-female-voice">Женский голос</label>
                    <select id="setting-female-voice" name="female_voice_id"></select>
                    <button type="button" class="action-button" data-role-preview="female">▶ Послушать женский</button></div>
                  <div class="field full"><span class="field-help" data-voice-pair-help>Выберите голоса для диалогов. Пара запоминается отдельно для движка и языка.</span></div>
                  <details class="field full" data-reference-panel hidden><summary>Добавить свой образец голоса</summary>
                    <p class="field-help">От 3 до 30 секунд чистой речи одного человека. Образец хранится локально. Укажите точный текст, который слышен на записи.</p>
                    <label for="reference-label">Название голоса</label><input id="reference-label" maxlength="100" placeholder="Например, мягкий женский голос">
                    <label for="reference-role">Роль</label><select id="reference-role"><option value="female">Женский</option><option value="male">Мужской</option></select>
                    <label for="reference-audio">Аудиообразец</label><input id="reference-audio" type="file" accept="audio/*">
                    <label for="reference-text">Текст из записи</label><textarea id="reference-text" maxlength="2000"></textarea>
                    <button type="button" class="action-button" data-reference-upload>Сохранить образец</button>
                  </details>
                  <div class="field"><label for="setting-gender">Тембр / пол голоса</label>
                      <select id="setting-gender" name="voice_gender"></select></div>
                    <div class="field" data-local-field><label for="setting-local-voice">Один голос для всех (необязательно)</label>
                      <select id="setting-local-voice" name="voice_id"></select></div>
                    <div class="field" data-cloud-field><label for="setting-stt-model">Модель распознавания</label>
                      <select id="setting-stt-model" name="stt_model"></select></div>
                    <div class="field" data-cloud-field><label for="setting-translation-model">Модель перевода</label>
                      <select id="setting-translation-model" name="translation_model"></select></div>
                    <div class="field" data-cloud-field><label for="setting-tts-model">Модель озвучки</label>
                      <select id="setting-tts-model" name="tts_model"></select></div>
                    <div class="field" data-openai-voice><label for="setting-openai-voice">Один голос OpenAI для всех (необязательно)</label>
                      <select id="setting-openai-voice" name="openai_voice"></select></div>
                    <div class="field" data-eleven-voice><label for="setting-eleven-voice">Один голос ElevenLabs для всех (необязательно)</label>
                      <input id="setting-eleven-voice" name="eleven_voice" list="eleven-voice-list" autocomplete="off" placeholder="auto или Voice ID">
                      <datalist id="eleven-voice-list"></datalist>
                      <span class="field-help" data-eleven-help>Загружаю голоса аккаунта…</span></div>
                    <div class="field full"><label>Ключи провайдеров</label>
                      <div class="provider-status" data-provider-status>—</div></div>
                    <div class="field full"><label for="preview-text">Проба озвучки</label><label for="preview-role">Какой голос проверить</label><select id="preview-role"><option value="female">Женский</option><option value="male">Мужской</option></select>
                      <div class="preview-row"><div class="field"><textarea id="preview-text" maxlength="240">Привет! Это проба голоса для перевода видео.</textarea></div>
                        <button class="action-button" type="button" data-preview-play>▶ Прослушать</button>
                        <button class="action-button ghost" type="button" data-preview-stop disabled>■ Стоп</button></div>
                      <span class="preview-note" data-preview-note></span>
                      <audio data-preview-audio preload="none"></audio></div>
                  </div>
                  <div class="settings-actions">
                    <button class="action-button primary" type="submit" data-save>Сохранить</button>
                    <button class="action-button ghost" type="button" data-reset-form>Отменить правки</button>
                    <button class="action-button ghost" type="button" data-restore>Вернуть YAML по умолчанию</button>
                    <span class="settings-status" role="status" aria-live="polite" data-settings-status></span>
                  </div>
                </form>
              </section>
              <section class="instruction">
                <svg aria-hidden="true" viewBox="0 0 40 40"><circle cx="20" cy="20" r="16"/>
                  <path d="M20 18v10m0-16h.01"/></svg>
                <div><h2>Как запустить перевод</h2>
                <p>Оставьте этот терминал запущенным, откройте страницу с видео и нажмите <strong>UVT · перевести</strong>.</p></div>
              </section>
              <section class="jobs"><h2>Текущие задачи · {html.escape(self.route_label)}</h2>{jobs_html}</section>
            </main><script type="application/json" id="uvt-settings-routes">{settings_routes_json}</script>
              <footer><a href="{html.escape(current_url, quote=True)}">{html.escape(current_url)}</a>
              <span>UVT — персональный сервер перевода видео</span></footer>
            """
            """
            </div><script>
              const tokenStorageKey = "uvt-dashboard-token";
              let apiToken = "";
              try { apiToken = window.sessionStorage.getItem(tokenStorageKey) || ""; } catch (_) {}
              function showAuth(message) {
                const box = document.querySelector("[data-auth-box]");
                box.hidden = false;
                setSettingsStatus(message || "Нужен UVT_API_TOKEN", "error");
              }
              async function dashboardFetch(url, options = {}) {
                const headers = new Headers(options.headers || {});
                if (apiToken) headers.set("X-UVT-Token", apiToken);
                const response = await fetch(url, {...options, headers, cache:"no-store"});
                if (response.status === 401) {
                  showAuth("Неверный или отсутствующий UVT_API_TOKEN");
                }
                return response;
              }
              const readinessViews = {
                "ready": ["Модели готовы", "ok", false],
                "not-applicable": ["Маршрут готов", "ok", false],
                "idle": ["Маршрут готов", "ok", false],
                "in-use": ["Модель занята задачей", "ok", true],
                "on-demand": ["Загрузится по запросу", "pending", true],
                "pending": ["Ожидает подготовки", "pending", true],
                "checking": ["Проверяю модели", "pending", true],
                "loading": ["Загружаю модель", "pending", true],
                "error": ["Ошибка модели", "error", false],
                "missing": ["Модели не установлены", "error", false],
                "cancelled": ["Подготовка остановлена", "error", false]
              };
              function setRouteState(row, label, kind, state) {
                const statusLabel = row.querySelector("[data-status]");
                const status = statusLabel.closest(".route-state");
                statusLabel.textContent = label;
                status.classList.remove("ok", "pending", "error");
                status.classList.add(kind);
                row.dataset.state = state;
              }
              function refreshServerState() {
                const states = Array.from(document.querySelectorAll("[data-route]"),
                  row => row.dataset.state || "checking");
                const serverState = document.querySelector(".server-state");
                const serverLabel = serverState.querySelector("[data-server-label]");
                serverState.classList.remove("ready", "pending", "error");
                if (states.some(state => ["error", "missing", "cancelled", "unavailable"].includes(state))) {
                  serverLabel.textContent = "Есть проблемы";
                  serverState.classList.add("error");
                } else if (states.some(state => ["pending", "checking", "loading", "on-demand"].includes(state))) {
                  serverLabel.textContent = "Сервер запускается";
                  serverState.classList.add("pending");
                } else {
                  serverLabel.textContent = "Сервер готов";
                  serverState.classList.add("ready");
                }
              }
              async function refreshRoute(row) {
                let retryDelay = 0;
                try {
                  const response = await dashboardFetch(row.dataset.metaUrl);
                  if (!response.ok) throw new Error(String(response.status));
                  const meta = await response.json();
                  const profile = meta.profile || {};
                  const engines = profile.engines || {};
                  row.querySelector("[data-profile]").textContent = profile.name || "configured";
                  row.querySelector("[data-engines]").textContent =
                    [engines.stt, engines.translation, engines.tts].filter(Boolean).join(" → ");
                  const readiness = meta.model_readiness || {};
                  const readinessStatus = readiness.status || "unknown";
                  const view = readinessViews[readinessStatus] || ["Состояние неизвестно", "pending", true];
                  setRouteState(row, view[0], view[1], readinessStatus);
                  row.querySelector("[data-detail]").textContent = readiness.detail || "";
                  retryDelay = view[2] ? 1500 : 10000;
                } catch (_) {
                  setRouteState(row, "Маршрут недоступен", "error", "unavailable");
                  row.querySelector("[data-detail]").textContent = "Повторяю проверку…";
                  retryDelay = 3000;
                } finally {
                  refreshServerState();
                  if (retryDelay) window.setTimeout(() => refreshRoute(row), retryDelay);
                }
              }
              const routeDefinitions = JSON.parse(
                document.getElementById("uvt-settings-routes").textContent
              );
              const form = document.querySelector("[data-settings-form]");
              const statusNode = document.querySelector("[data-settings-status]");
              const audio = document.querySelector("[data-preview-audio]");
              let activeRoute = routeDefinitions[0] || null;
              let settingsDocument = null;
              let previewUrl = "";
              let previewController = null;

              function routeBase(route) {
                if (route.url) return String(route.url).replace(/\\/$/, "");
                const rawHost = window.location.hostname;
                const host = rawHost.includes(":") ? `[${rawHost}]` : rawHost;
                return `${window.location.protocol}//${host}:${route.port}`;
              }
              function setSettingsStatus(message, kind = "") {
                if (!statusNode) return;
                statusNode.textContent = message || "";
                statusNode.className = `settings-status ${kind}`.trim();
              }
              async function responseError(response) {
                const text = (await response.text()).trim();
                return text || `HTTP ${response.status}`;
              }
              function setOptions(select, items, value, emptyLabel = "") {
                select.replaceChildren();
                if (emptyLabel) {
                  const option = document.createElement("option");
                  option.value = "";
                  option.textContent = emptyLabel;
                  select.append(option);
                }
                for (const raw of items || []) {
                  const item = typeof raw === "string" ? {id:raw, label:raw} : raw;
                  const option = document.createElement("option");
                  option.value = String(item.id || "");
                  option.textContent = String(item.label || item.id || "");
                  option.disabled = item.installed === false;
                  select.append(option);
                }
                select.value = value == null ? "" : String(value);
                if (select.value !== String(value == null ? "" : value) && select.options.length) {
                  select.selectedIndex = 0;
                }
              }
              function setKindVisibility(kind) {
                document.querySelectorAll("[data-local-field]").forEach(
                  node => { node.hidden = kind !== "local"; }
                );
                document.querySelectorAll("[data-cloud-field]").forEach(
                  node => { node.hidden = kind === "local"; }
                );
                document.querySelector("[data-openai-voice]").hidden = kind !== "openai";
                document.querySelector("[data-eleven-voice]").hidden = kind !== "elevenlabs";
              }
              let draftVoicePairs = {};
              let currentPairScope = "";
              let providerPairVoices = [];
              function selectedVoiceEngine() {
                if (!settingsDocument) return "";
                if (settingsDocument.route.kind !== "local") return settingsDocument.engines.tts;
                const selected = (settingsDocument.catalog.profiles || []).find(item => item.id === form.elements.profile_id.value);
                return selected ? selected.engines.tts : settingsDocument.engines.tts;
              }
              function rememberVoicePair() {
                if (currentPairScope) draftVoicePairs[currentPairScope] = {
                  male_voice_id:form.elements.male_voice_id.value,
                  female_voice_id:form.elements.female_voice_id.value
                };
              }
              function pairVoices() {
                const kind = settingsDocument.route.kind;
                if (kind === "elevenlabs") return providerPairVoices;
                if (kind === "openai") return (settingsDocument.catalog.voices || []).filter(voice => !voice.models || voice.models.includes(form.elements.tts_model.value));
                const selected = (settingsDocument.catalog.profiles || []).find(item => item.id === form.elements.profile_id.value);
                return ((selected && selected.voices) || []).filter(voice => voice.installed !== false && (voice.languages || [voice.language]).includes(form.elements.target_lang.value));
              }
              function updateRoleVoiceOptions() {
                if (!settingsDocument) return;
                rememberVoicePair();
                const engine = selectedVoiceEngine();
                currentPairScope = `${engine}:${form.elements.target_lang.value}`;
                const pair = draftVoicePairs[currentPairScope] || {};
                for (const role of ["male", "female"]) {
                  const field = `${role}_voice_id`;
                  const choices = pairVoices().filter(voice => voice.id && voice.id !== "auto").slice().sort((a,b) => Number(b.gender === role) - Number(a.gender === role));
                  if (pair[field] && !choices.some(item => item.id === pair[field])) choices.push({id:pair[field], label:`${pair[field]} · сохранённый голос`});
                  setOptions(form.elements[field], choices, pair[field] || "", engine === "f5" || engine === "moss-onnx" ? "Из оригинала / автоматически" : "Автоматический голос движка");
                }
                document.querySelector("[data-reference-panel]").hidden = !["f5", "moss-onnx"].includes(engine);
                document.querySelector("[data-voice-pair-help]").textContent = engine === "f5"
                  ? "F5 использует образцы голоса. Добавьте свои образцы и выберите их для диалога; пустой выбор сохраняет голос из оригинала."
                  : "Пара запоминается отдельно для движка и языка. В режиме «Авто по спикеру» диалог озвучивается выбранными мужским и женским голосами.";
              }
              function clearSingleVoice() {
                form.elements.voice_id.value = "";
                form.elements.openai_voice.value = "";
                form.elements.eleven_voice.value = "auto";
              }
              function updateLocalVoiceOptions(preferred) {
                if (!settingsDocument || settingsDocument.route.kind !== "local") return;
                const target = form.elements.target_lang.value;
                const selected = (settingsDocument.catalog.profiles || []).find(
                  item => item.id === form.elements.profile_id.value
                );
                const voices = ((selected && selected.voices) || settingsDocument.catalog.voices || []).filter(
                  voice => (voice.languages || [voice.language]).includes(target)
                    && voice.installed !== false && voice.id
                );
                setOptions(
                  form.elements.voice_id,
                  voices,
                  preferred == null ? form.elements.voice_id.value : preferred,
                  "Авто по спикеру"
                );
              }
              function updateOpenAIVoiceOptions(preferred) {
                if (!settingsDocument || settingsDocument.route.kind !== "openai") return;
                const model = form.elements.tts_model.value;
                const voices = (settingsDocument.catalog.voices || []).filter(
                  voice => !voice.models || voice.models.includes(model)
                );
                setOptions(
                  form.elements.openai_voice,
                  voices,
                  preferred == null ? form.elements.openai_voice.value : preferred,
                  "Авто по спикеру"
                );
              }
              function updateEngineChain() {
                if (!settingsDocument || settingsDocument.route.kind !== "local") return;
                const selected = (settingsDocument.catalog.profiles || []).find(
                  item => item.id === form.elements.profile_id.value
                );
                const engines = selected ? selected.engines || {} : {};
                document.querySelector("[data-engine-chain]").textContent =
                  [engines.stt, engines.translation, engines.tts].filter(Boolean).join(" → ") || "—";
                const allSources = settingsDocument.catalog.all_source_languages || [];
                const supported = selected && selected.source_languages && selected.source_languages.length
                  ? allSources.filter(item => selected.source_languages.includes(item.id))
                  : allSources;
                const previousSource = form.elements.source_lang.value;
                setOptions(form.elements.source_lang, supported, previousSource);
                const allTargets = settingsDocument.catalog.all_target_languages || settingsDocument.catalog.target_languages || [];
                const targetLanguages = selected && selected.target_languages;
                const targets = targetLanguages && targetLanguages.length
                  ? allTargets.filter(item => targetLanguages.includes(item.id)) : allTargets;
                const previousTarget = form.elements.target_lang.value;
                setOptions(form.elements.target_lang, targets, previousTarget);
                updateLocalVoiceOptions();
              }
              function providerStatusText(items) {
                if (!items || !items.length) return "Все движки локальные — API-ключи не нужны";
                return items.map(item =>
                  `${item.env}: ${item.configured ? "настроен" : "не задан"}`
                ).join(" · ");
              }
              function renderSettings(documentData) {
                settingsDocument = documentData;
                const current = documentData.effective || {};
                draftVoicePairs = JSON.parse(JSON.stringify(current.voice_pairs || {}));
                currentPairScope = "";
                const initialScope = `${documentData.engines.tts}:${current.target_lang}`;
                draftVoicePairs[initialScope] = {male_voice_id:current.male_voice_id || "", female_voice_id:current.female_voice_id || ""};
                providerPairVoices = [
                  {id:"EXAVITQu4vr4xnSDxMaL", label:"Sarah · стандартный", gender:"female"},
                  {id:"ErXwobaYiN019PkySvjV", label:"Adam · стандартный", gender:"male"}
                ];
                const catalog = documentData.catalog || {};
                const kind = documentData.route.kind;
                setKindVisibility(kind);
                setOptions(form.elements.source_lang, catalog.source_languages, current.source_lang);
                setOptions(form.elements.target_lang, catalog.target_languages, current.target_lang);
                setOptions(form.elements.voice_gender, catalog.voice_genders, current.voice_gender);
                if (kind === "local") {
                  setOptions(form.elements.profile_id, catalog.profiles, current.profile_id);
                  updateEngineChain();
                  updateLocalVoiceOptions(current.voice_id || "");
                } else {
                  setOptions(form.elements.stt_model, catalog.stt_models, current.stt_model);
                  setOptions(form.elements.translation_model, catalog.translation_models, current.translation_model);
                  setOptions(form.elements.tts_model, catalog.tts_models, current.tts_model);
                  if (kind === "openai") {
                    updateOpenAIVoiceOptions(current.tts_voice === "auto" ? "" : current.tts_voice);
                  } else {
                    form.elements.eleven_voice.value = current.tts_voice || "auto";
                  }
                }
                document.querySelector("[data-provider-status]").textContent =
                  providerStatusText(documentData.provider_status);
                document.querySelector("[data-saved-state]").textContent = documentData.saved
                  ? "Сохранено в web-панели"
                  : "Значения из YAML-профиля";
                document.querySelector("[data-preview-note]").textContent = kind === "local"
                  ? "Проба создаётся локально."
                  : "Проба отправит текст в облачный TTS и использует API-квоту.";
                form.querySelector("[data-save]").disabled = !documentData.can_save;
                form.setAttribute("aria-busy", "false");
                setSettingsStatus(documentData.warning || documentData.notice || "", documentData.warning ? "error" : "");
                updateRoleVoiceOptions();
                if (kind === "elevenlabs") loadProviderVoices();
              }
              function collectSettings() {
                const kind = settingsDocument.route.kind;
                rememberVoicePair();
                const value = {
                  source_lang: form.elements.source_lang.value,
                  target_lang: form.elements.target_lang.value,
                  voice_gender: form.elements.voice_gender.value,
                  male_voice_id: form.elements.male_voice_id.value,
                  female_voice_id: form.elements.female_voice_id.value,
                  voice_pairs: draftVoicePairs
                };
                if (kind === "local") {
                  value.profile_id = form.elements.profile_id.value;
                  value.voice_id = form.elements.voice_id.value;
                } else {
                  value.stt_model = form.elements.stt_model.value;
                  value.translation_model = form.elements.translation_model.value;
                  value.tts_model = form.elements.tts_model.value;
                  value.tts_voice = kind === "openai"
                    ? (form.elements.openai_voice.value || "auto")
                    : (form.elements.eleven_voice.value.trim() || "auto");
                }
                return value;
              }
              async function loadSettings(route, focusPanel = false) {
                activeRoute = route;
                form.setAttribute("aria-busy", "true");
                setSettingsStatus("Загружаю настройки…");
                document.querySelectorAll(".route-tab").forEach(button =>
                  button.setAttribute("aria-selected", String(button.dataset.routeId === route.id))
                );
                try {
                  const response = await dashboardFetch(`${routeBase(route)}/settings`);
                  if (!response.ok) throw new Error(await responseError(response));
                  renderSettings(await response.json());
                  if (focusPanel) document.getElementById("settings").scrollIntoView({behavior:"smooth", block:"start"});
                } catch (error) {
                  form.setAttribute("aria-busy", "false");
                  setSettingsStatus(error.message || "Не удалось загрузить настройки", "error");
                }
              }
              async function loadProviderVoices() {
                const help = document.querySelector("[data-eleven-help]");
                const requestRoute = activeRoute;
                help.textContent = "Загружаю голоса вашего ElevenLabs…";
                try {
                  const response = await dashboardFetch(`${routeBase(activeRoute)}/provider/voices`);
                  if (!response.ok) throw new Error(await responseError(response));
                  const result = await response.json();
                  if (activeRoute !== requestRoute) return;
                  const merged = new Map(providerPairVoices.map(voice => [voice.id, voice]));
                  for (const voice of result.voices || []) merged.set(voice.id, {...voice, label:`${voice.label}${voice.category === "professional" ? " · библиотека, возможен платный API" : ""}`});
                  providerPairVoices = Array.from(merged.values());
                  updateRoleVoiceOptions();
                  const list = document.getElementById("eleven-voice-list");
                  list.replaceChildren();
                  for (const voice of result.voices || []) {
                    const option = document.createElement("option");
                    option.value = voice.id;
                    option.label = voice.label;
                    list.append(option);
                  }
                  help.textContent = result.voices.length
                    ? `Найдено голосов: ${result.voices.length}. Можно ввести Voice ID вручную.`
                    : "Голоса не найдены; введите Voice ID вручную.";
                } catch (error) {
                  help.textContent = `${error.message}. Voice ID можно ввести вручную.`;
                }
              }
              function buildTabs() {
                const tabs = document.querySelector("[data-settings-tabs]");
                for (const route of routeDefinitions) {
                  const button = document.createElement("button");
                  button.type = "button";
                  button.className = "route-tab";
                  button.role = "tab";
                  button.dataset.routeId = route.id;
                  button.textContent = route.label;
                  button.setAttribute("aria-selected", "false");
                  button.addEventListener("click", () => loadSettings(route));
                  tabs.append(button);
                }
              }
              form.addEventListener("submit", async event => {
                event.preventDefault();
                if (!settingsDocument) return;
                form.setAttribute("aria-busy", "true");
                setSettingsStatus("Сохраняю…");
                try {
                  const response = await dashboardFetch(`${routeBase(activeRoute)}/settings`, {
                    method:"PUT",
                    headers:{"Content-Type":"application/json"},
                    body:JSON.stringify({revision:settingsDocument.revision, settings:collectSettings()})
                  });
                  if (!response.ok) throw new Error(await responseError(response));
                  const result = await response.json();
                  renderSettings(result);
                  setSettingsStatus(result.message || "Сохранено", "ok");
                  document.querySelectorAll("[data-route]").forEach(refreshRoute);
                } catch (error) {
                  form.setAttribute("aria-busy", "false");
                  setSettingsStatus(error.message || "Не удалось сохранить", "error");
                }
              });
              form.querySelector("[data-reset-form]").addEventListener("click", () => {
                if (settingsDocument) renderSettings(settingsDocument);
                setSettingsStatus("Несохранённые правки отменены");
              });
              form.querySelector("[data-restore]").addEventListener("click", async () => {
                if (!settingsDocument || !window.confirm("Вернуть все настройки этого маршрута к YAML-профилю?")) return;
                form.setAttribute("aria-busy", "true");
                try {
                  const response = await dashboardFetch(`${routeBase(activeRoute)}/settings`, {
                    method:"DELETE", headers:{"Content-Type":"application/json"},
                    body:JSON.stringify({revision:settingsDocument.revision})
                  });
                  if (!response.ok) throw new Error(await responseError(response));
                  const result = await response.json();
                  renderSettings(result);
                  setSettingsStatus(result.message || "Восстановлено", "ok");
                } catch (error) {
                  form.setAttribute("aria-busy", "false");
                  setSettingsStatus(error.message || "Не удалось сбросить", "error");
                }
              });
              form.elements.target_lang.addEventListener("change", () => { updateLocalVoiceOptions(""); updateRoleVoiceOptions(); });
              form.elements.profile_id.addEventListener("change", () => {
                updateEngineChain();
                updateLocalVoiceOptions("");
                updateRoleVoiceOptions();
                setSettingsStatus("Есть несохранённые изменения");
              });
              form.elements.tts_model.addEventListener("change", () => { updateOpenAIVoiceOptions(""); updateRoleVoiceOptions(); });
              for (const role of ["male", "female"]) form.elements[`${role}_voice_id`].addEventListener("change", () => { clearSingleVoice(); rememberVoicePair(); });
              form.querySelectorAll("[data-role-preview]").forEach(button => button.addEventListener("click", () => {
                if (selectedVoiceEngine() === "f5" && !form.elements[`${button.dataset.rolePreview}_voice_id`].value) {
                  setSettingsStatus("Для пробы F5 добавьте и выберите образец голоса. Голос из оригинала доступен во время перевода видео.", "error");
                  return;
                }
                document.getElementById("preview-role").value = button.dataset.rolePreview;
                form.querySelector("[data-preview-play]").click();
              }));
              form.querySelector("[data-reference-upload]").addEventListener("click", async () => {
                const file = document.getElementById("reference-audio").files[0];
                if (!file) { setSettingsStatus("Выберите аудиообразец", "error"); return; }
                const body = new FormData();
                body.append("audio", file);
                body.append("text", document.getElementById("reference-text").value);
                body.append("label", document.getElementById("reference-label").value);
                body.append("gender", document.getElementById("reference-role").value);
                body.append("language", form.elements.target_lang.value);
                const button = form.querySelector("[data-reference-upload]");
                button.disabled = true;
                const route = activeRoute;
                try {
                  const response = await dashboardFetch(`${routeBase(route)}/voices/reference`, {method:"POST", body});
                  if (!response.ok) throw new Error(await responseError(response));
                  const result = await response.json();
                  if (activeRoute !== route) return;
                  for (const profile of settingsDocument.catalog.profiles || []) if (["f5", "moss-onnx"].includes(profile.engines.tts)) profile.voices.push({...result.voice,engine:profile.engines.tts});
                  updateLocalVoiceOptions();
                  updateRoleVoiceOptions();
                  form.elements[`${result.voice.gender}_voice_id`].value = result.voice.id;
                  clearSingleVoice(); rememberVoicePair();
                  setSettingsStatus("Образец добавлен и выбран. Сохраните настройки, чтобы применять его к переводам.", "ok");
                } catch(error) { setSettingsStatus(error.message || "Не удалось добавить образец", "error"); }
                finally { button.disabled = false; }
              });
              form.addEventListener("input", event => {
                if (event.target.closest("#preview-text")) return;
                setSettingsStatus("Есть несохранённые изменения");
              });
              form.querySelector("[data-preview-play]").addEventListener("click", async () => {
                if (!settingsDocument) return;
                if (previewController) previewController.abort();
                previewController = new AbortController();
                setSettingsStatus("Создаю пробу голоса…");
                const button = form.querySelector("[data-preview-play]");
                button.disabled = true;
                try {
                  const response = await dashboardFetch(`${routeBase(activeRoute)}/tts/preview`, {
                    method:"POST", headers:{"Content-Type":"application/json"},
                    body:JSON.stringify({...collectSettings(), voice_gender:document.getElementById("preview-role").value, voice_id:"", tts_voice:"auto", settings_mode:"override", text:document.getElementById("preview-text").value}),
                    signal:previewController.signal
                  });
                  if (!response.ok) throw new Error(await responseError(response));
                  if (previewUrl) URL.revokeObjectURL(previewUrl);
                  previewUrl = URL.createObjectURL(await response.blob());
                  audio.src = previewUrl;
                  form.querySelector("[data-preview-stop]").disabled = false;
                  await audio.play();
                  setSettingsStatus("Проба готова", "ok");
                } catch (error) {
                  if (error.name !== "AbortError") setSettingsStatus(error.message || "Ошибка пробы", "error");
                } finally {
                  button.disabled = false;
                }
              });
              form.querySelector("[data-preview-stop]").addEventListener("click", () => {
                if (previewController) previewController.abort();
                audio.pause(); audio.currentTime = 0;
                setSettingsStatus("Воспроизведение остановлено");
              });
              document.querySelector("[data-token-submit]").addEventListener("click", () => {
                apiToken = document.querySelector("[data-token-input]").value.trim();
                try { window.sessionStorage.setItem(tokenStorageKey, apiToken); } catch (_) {}
                document.querySelector("[data-auth-box]").hidden = true;
                document.querySelectorAll("[data-route]").forEach(refreshRoute);
                if (activeRoute) loadSettings(activeRoute);
              });
              document.querySelectorAll("[data-configure-route]").forEach(button => {
                button.addEventListener("click", () => {
                  const route = routeDefinitions.find(item => item.id === button.dataset.configureRoute);
                  if (route) loadSettings(route, true);
                });
              });
              window.addEventListener("beforeunload", () => {
                if (previewController) previewController.abort();
                if (previewUrl) URL.revokeObjectURL(previewUrl);
              });
              buildTabs();
              document.querySelectorAll("[data-route]").forEach(refreshRoute);
              if (activeRoute) loadSettings(activeRoute);
            </script></body></html>
            """
        )
        return web.Response(
            text=page,
            content_type="text/html",
            charset="utf-8",
            headers={"Cache-Control": "no-store"},
        )

    async def _get_meta(self, request):
        from aiohttp import web

        data = dict(request.query)
        if not data:
            return web.json_response(self._metadata())
        try:
            return web.json_response(self._metadata_for_request(data))
        except ValueError as exc:
            raise web.HTTPUnprocessableEntity(text=str(exc)) from None

    async def _get_settings(self, request):
        from aiohttp import web

        if not self._dashboard_request_allowed(request):
            raise web.HTTPForbidden(
                text="настройки доступны только локальной web-панели"
            )
        return web.json_response(
            self._settings_payload(), headers={"Cache-Control": "no-store"}
        )

    async def _put_settings(self, request):
        from aiohttp import web

        if not self._dashboard_request_allowed(request):
            raise web.HTTPForbidden(
                text="менять настройки можно только из локальной web-панели"
            )
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            raise web.HTTPBadRequest(text="ожидается JSON") from None
        if not isinstance(payload, dict) or set(payload) - {"revision", "settings"}:
            raise web.HTTPBadRequest(text="ожидаются revision и settings")
        raw_settings = payload.get("settings")
        if not isinstance(raw_settings, dict):
            raise web.HTTPBadRequest(text="settings должен быть JSON-object")
        try:
            revision = self._expected_revision(payload)
            normalized = normalize_settings(
                raw_settings,
                current=effective_settings(
                    self.cfg,
                    kind=self._settings_kind,
                    profile_name=self.profile_name,
                ),
                kind=self._settings_kind,
                profile_ids=set(self._profile_configs),
                local_voices=self._settings_voice_catalog(payload["settings"]),
                    voice_engine=self._settings_voice_engine(payload["settings"]),
                    voice_catalogs=self._settings_voice_catalogs(),
            )
            if self._settings_kind == "local":
                self._require_profile_ready(str(normalized["profile_id"]))
            candidate, _candidate_profile = apply_settings(
                self._base_cfg,
                normalized,
                kind=self._settings_kind,
                base_profile_name=self._base_profile_name,
                profiles=self._profile_configs,
            )
            self._validate_piper_voice_support(candidate)
            self._validate_stt_language_support(candidate)
        except ValueError as exc:
            raise web.HTTPUnprocessableEntity(text=str(exc)) from None

        async with self._settings_lock:
            if self._has_active_jobs():
                raise web.HTTPConflict(
                    text="дождитесь завершения текущего перевода"
                )
            try:
                entry = self.settings_store.set(
                    self.settings_key,
                    normalized,
                    expected_revision=revision,
                )
            except SettingsConflictError as exc:
                raise web.HTTPConflict(text=str(exc)) from None
            except OSError as exc:
                log.error("не удалось сохранить UVT settings: %s", exc)
                raise web.HTTPInternalServerError(
                    text="не удалось сохранить настройки на диск"
                ) from None
            self._settings_revision = int(entry["revision"])
            self._settings_saved = True
            self._settings_load_error = None
            await self._apply_effective_settings(normalized)
        response = self._settings_payload()
        response["message"] = "сохранено; новые настройки применятся к следующему переводу"
        return web.json_response(response, headers={"Cache-Control": "no-store"})

    async def _delete_settings(self, request):
        from aiohttp import web

        if not self._dashboard_request_allowed(request):
            raise web.HTTPForbidden(
                text="сбрасывать настройки можно только из локальной web-панели"
            )
        try:
            payload = await request.json() if request.can_read_body else {}
        except Exception:  # noqa: BLE001
            raise web.HTTPBadRequest(text="ожидается JSON") from None
        if not isinstance(payload, dict) or set(payload) - {"revision"}:
            raise web.HTTPBadRequest(text="ожидается revision")
        try:
            revision = self._expected_revision(payload)
        except ValueError as exc:
            raise web.HTTPUnprocessableEntity(text=str(exc)) from None
        async with self._settings_lock:
            if self._has_active_jobs():
                raise web.HTTPConflict(
                    text="дождитесь завершения текущего перевода"
                )
            try:
                entry = self.settings_store.reset(
                    self.settings_key, expected_revision=revision
                )
            except SettingsConflictError as exc:
                raise web.HTTPConflict(text=str(exc)) from None
            except OSError as exc:
                log.error("не удалось сбросить UVT settings: %s", exc)
                raise web.HTTPInternalServerError(
                    text="не удалось сбросить настройки на диске"
                ) from None
            self._settings_revision = int(entry["revision"])
            self._settings_saved = False
            self._settings_load_error = None
            await self._apply_effective_settings(None)
        response = self._settings_payload()
        response["message"] = "восстановлены значения из YAML-профиля"
        return web.json_response(response, headers={"Cache-Control": "no-store"})

    async def _post_voice_reference(self, request):
        from aiohttp import web
        from uvt.voice_references import library_dir, catalog
        from uvt.server_settings import LANGUAGES
        import tempfile
        if not self._dashboard_request_allowed(request):
            raise web.HTTPForbidden(text="Добавлять образцы можно только из настроек UVT")
        if len(catalog()) >= 100:
            raise web.HTTPConflict(text="В библиотеке уже 100 образцов")
        limit = 20 * 1024**2
        total = 0
        fields = {}
        root = library_dir()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(prefix="upload-", dir=root) as temporary:
            source = Path(temporary) / "source"
            try:
                reader = await request.multipart()
                async with asyncio.timeout(60):
                    while (part := await reader.next()) is not None:
                        if part.name == "audio" and not source.exists():
                            with source.open("wb") as output:
                                while chunk := await part.read_chunk():
                                    total += len(chunk)
                                    if total > limit:
                                        raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=total)
                                    output.write(chunk)
                        elif part.name in {"label", "text", "language", "gender"}:
                            value = bytearray()
                            while chunk := await part.read_chunk():
                                value.extend(chunk)
                                if len(value) > 8000:
                                    raise web.HTTPBadRequest(text="Текст образца слишком длинный")
                            fields[part.name] = value.decode("utf-8").strip()
                        else:
                            raise web.HTTPBadRequest(text="Неизвестное или повторное поле образца")
            except (ValueError, UnicodeDecodeError, TimeoutError):
                raise web.HTTPBadRequest(text="Не удалось прочитать образец голоса") from None
            if not source.is_file() or not source.stat().st_size:
                raise web.HTTPBadRequest(text="Выберите аудиофайл")
            if not fields.get("text") or len(fields["text"]) > 2000:
                raise web.HTTPBadRequest(text="Введите точный текст из образца, до 2000 символов")
            if fields.get("gender") not in {"male", "female"}:
                raise web.HTTPBadRequest(text="Выберите мужской или женский голос")
            if fields.get("language") not in {item[0] for item in LANGUAGES} - {"auto"}:
                raise web.HTTPBadRequest(text="Выберите язык озвучки")
            lines = []
            try:
                await _run_process(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(source)], 15, "проверка образца", on_line=lines.append)
                probe = json.loads("\n".join(lines))
                duration = float(probe["format"]["duration"])
                if not 3 <= duration <= 30 or not any(item.get("codec_type") == "audio" for item in probe.get("streams", [])):
                    raise ValueError("duration")
                output = Path(temporary) / "sample.wav"
                await _run_process(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:a:0", "-ac", "1", "-ar", "24000", "-t", "30", "-c:a", "pcm_s16le", str(output)], 20, "подготовка образца")
            except (RuntimeError, ValueError, KeyError):
                raise web.HTTPUnprocessableEntity(text="Нужен читаемый образец чистой речи длительностью от 3 до 30 секунд") from None
            voice_id = "ref_" + uuid.uuid4().hex
            metadata = {"id": voice_id, "label": (fields.get("label") or "Мой голос")[:100],
                        "text": fields["text"], "language": fields["language"], "gender": fields["gender"]}
            audio_path = root / (voice_id + ".wav")
            metadata_path = root / (voice_id + ".json")
            try:
                output.chmod(0o600)
                output.replace(audio_path)
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
                metadata_path.chmod(0o600)
            except OSError:
                audio_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                raise web.HTTPInsufficientStorage(text="Не удалось сохранить образец; проверьте свободное место") from None
        return web.json_response({"voice": next(item for item in catalog() if item["id"] == voice_id)}, status=201)

    async def _get_provider_voices(self, request):
        from aiohttp import web

        if not self._dashboard_request_allowed(request):
            raise web.HTTPForbidden(
                text="список голосов доступен из локальной web-панели"
            )
        if self._settings_kind != "elevenlabs":
            raise web.HTTPUnprocessableEntity(
                text="голоса аккаунта доступны только для ElevenLabs"
            )
        now = time.time()
        if self._provider_voice_cache and now - self._provider_voice_cache[0] < 600:
            return web.json_response({"voices": self._provider_voice_cache[1], "cached": True})
        env_name = str(getattr(self.cfg.tts, "api_key_env", "ELEVENLABS_API_KEY"))
        key = os.environ.get(env_name, "").strip()
        if not key:
            raise web.HTTPServiceUnavailable(
                text=f"задайте {env_name} и перезапустите сервер"
            )
        try:
            import httpx

            base = str(
                getattr(self.cfg.tts, "base_url", "https://api.elevenlabs.io/v1")
            ).rstrip("/")
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{base}/voices",
                    headers={"xi-api-key": key, "Accept": "application/json"},
                )
                response.raise_for_status()
                raw = response.json()
        except Exception as exc:  # noqa: BLE001 - provider errors are sanitized
            log.warning("ElevenLabs voice catalog unavailable: %s", type(exc).__name__)
            raise web.HTTPBadGateway(
                text="ElevenLabs не отдал список голосов; проверьте ключ и сеть"
            ) from None
        voices: list[dict[str, str]] = []
        for item in raw.get("voices", []) if isinstance(raw, dict) else []:
            if not isinstance(item, dict):
                continue
            voice_id = str(item.get("voice_id") or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{3,128}", voice_id):
                continue
            voices.append(
                {
                    "id": voice_id,
                    "label": str(item.get("name") or voice_id)[:100],
                    "category": str(item.get("category") or "")[:50],
                        "gender": str((item.get("labels") or {}).get("gender") or ""),
                }
            )
        voices.sort(key=lambda item: item["label"].casefold())
        self._provider_voice_cache = (now, voices)
        return web.json_response({"voices": voices, "cached": False})

    async def _post_dub(self, request):
        from aiohttp import web

        try:
            data = await request.json()
        except Exception:  # noqa: BLE001
            raise web.HTTPBadRequest(text="ожидается JSON") from None
        if not isinstance(data, dict):
            raise web.HTTPBadRequest(text="ожидается JSON-object")
        return await self._submit_dub_job(data)

    async def _submit_dub_job(self, data: dict):
        """Submit a validated request from JSON or the trusted upload handler."""
        from aiohttp import web

        if self._settings_lock.locked():
            raise web.HTTPConflict(
                text="настройки моделей обновляются; повторите через несколько секунд"
            )

        try:
            cfg_snapshot, selected_profile = self._config_for_request(data)
            self._require_profile_ready(selected_profile)
            self._validate_piper_voice_support(cfg_snapshot)
            request_meta = self._metadata(cfg_snapshot, selected_profile)
            key = self._cache_key(data, (cfg_snapshot, selected_profile))
        except ValueError as exc:
            raise web.HTTPUnprocessableEntity(text=str(exc)) from None
        cached_id = self._job_cache.get(key)
        if cached_id:
            cached = self.jobs.get(cached_id)
            if cached is not None and cached.status not in ("error", "cancelled"):
                log.info("задача из кэша: %s", cached_id)
                payload = self._job_payload(cached)
                payload.update({"job_url": f"/job/{cached_id}", "meta": request_meta})
                return web.json_response(payload)

        from uvt.server_workspace import upload_source_name

        source_label = data.get("source_name")
        if not isinstance(source_label, str) or not source_label.strip():
            source_label = str(data.get("file") or "")
        if not source_label:
            source_label = urlsplit(str(data.get("page_url") or data.get("media_url") or "")).hostname or "Перевод видео"
        job = Job(
            id=uuid.uuid4().hex[:12],
            source_name=upload_source_name(source_label),
            profile_name=str(request_meta["profile"]["name"]),
            engines=dict(request_meta["profile"]["engines"]),
        )
        self._set_stage(job, "queue")
        self.jobs[job.id] = job
        video_results = getattr(self, "_video_results", None)
        if video_results:
            video_results.record_request(job, data)
        self._job_configs[job.id] = cfg_snapshot.model_copy(deep=True)
        self._job_cache[key] = job.id
        self._tasks[job.id] = asyncio.get_running_loop().create_task(
            self._run_job(
                job,
                data,
                cfg_snapshot.model_copy(deep=True),
                selected_profile,
            )
        )
        self._tasks[job.id].add_done_callback(
            lambda task: self._job_task_finished(job, task)
        )
        engines = request_meta["profile"]["engines"]
        log.info(
            "новая задача %s [%s/%s, порт %s; STT=%s, перевод=%s, TTS=%s]: %s",
            job.id,
            self.route_label,
            job.profile_name,
            self.listen_port or "?",
            engines["stt"],
            engines["translation"],
            engines["tts"],
            data.get("page_url") or data.get("media_url") or data.get("file"),
        )
        payload = self._job_payload(job)
        payload.update({"job_url": f"/job/{job.id}", "meta": request_meta})
        return web.json_response(payload)

    def _job_task_finished(self, job: Job, task: asyncio.Task) -> None:
        # A task cancelled before its first event-loop turn never enters
        # _run_job, so its try/finally cannot finish the queued job.
        if task.cancelled() and job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
            self._set_stage(job, "cancelled", stage_progress=1.0)
        if self._tasks.get(job.id) is task:
            self._tasks.pop(job.id, None)

    async def _post_tts_preview(self, request):
        from aiohttp import web

        try:
            data = await request.json()
        except Exception:  # noqa: BLE001
            raise web.HTTPBadRequest(text="ожидается JSON") from None
        if not isinstance(data, dict):
            raise web.HTTPBadRequest(text="ожидается JSON-object")

        text = str(data.get("text") or "").strip()
        if not text:
            raise web.HTTPUnprocessableEntity(text="введите текст для пробы голоса")
        if len(text) > _PREVIEW_MAX_CHARS or len(text.encode("utf-8")) > 1024:
            raise web.HTTPRequestEntityTooLarge(
                max_size=_PREVIEW_MAX_CHARS,
                actual_size=len(text),
            )

        try:
            cfg, profile_name = self._config_for_request(data)
            self._require_profile_ready(profile_name)
            self._validate_piper_voice_support(cfg)
        except ValueError as exc:
            raise web.HTTPUnprocessableEntity(text=str(exc)) from None
        # Local Piper preview is also used by the in-player userscript. Cloud
        # preview can spend quota, so it is restricted to the dashboard/token.
        if (
            str(cfg.tts.engine) in {"openai", "elevenlabs"}
            and not self._dashboard_request_allowed(request)
        ):
            raise web.HTTPForbidden(
                text="пробу голоса можно запустить только из web-панели"
            )
        if (
            str(cfg.tts.engine) == "piper"
            and not cfg.tts.voice_id
            and cfg.tts.voice_gender == "auto"
        ):
            raise web.HTTPUnprocessableEntity(
                text="для пробы выберите мужской или женский голос"
            )
        if any(
            job.status in {"queued", "running", "awaiting_approval"}
            for job in self.jobs.values()
        ) or self._lock.locked():
            raise web.HTTPConflict(
                text="сначала дождитесь завершения текущего перевода"
            )

        cache_key = (
            self.settings_key,
            profile_name,
            cfg.target_lang,
            str(cfg.tts.engine),
            str(getattr(cfg.tts, "model", "") or ""),
            str(getattr(cfg.tts, "voice", "") or ""),
            str(cfg.tts.voice_gender),
            str(cfg.tts.voice_id or ""),
            str(getattr(cfg.tts, "male_voice_id", "") or ""),
            str(getattr(cfg.tts, "female_voice_id", "") or ""),
            text,
        )
        cached = self._preview_cache.get(cache_key)
        if cached is None:
            async with self._lock:
                cached = self._preview_cache.get(cache_key)
                if cached is None:
                    registry.load_plugin_dirs(cfg.plugin_dirs)
                    engine = registry.create("tts", cfg.tts.engine, cfg.tts)
                    try:
                        try:
                            async with asyncio.timeout(_PREVIEW_TIMEOUT_S):
                                await engine.warmup()
                                samples, sample_rate = await engine.synthesize(
                                    text, cfg.target_lang
                                )
                        except TimeoutError:
                            raise web.HTTPGatewayTimeout(
                                text=(
                                    "голосовой движок не ответил за "
                                    f"{_PREVIEW_TIMEOUT_S:.0f} с; попробуйте ещё раз"
                                )
                            ) from None
                        except web.HTTPException:
                            raise
                        except Exception as exc:  # noqa: BLE001 - provider details stay in logs
                            log.warning(
                                "TTS preview failed [%s/%s]: %s",
                                self.route_label,
                                cfg.tts.engine,
                                type(exc).__name__,
                            )
                            raise web.HTTPBadGateway(
                                text=(
                                    "не удалось создать пробу; проверьте API-ключ, "
                                    "модель и голос"
                                )
                            ) from None
                    finally:
                        await engine.close()
                    if len(samples) == 0:
                        raise web.HTTPUnprocessableEntity(text="голос не создал аудио")
                    if len(samples) / max(sample_rate, 1) > 20:
                        raise web.HTTPUnprocessableEntity(
                            text="пример получился длиннее 20 секунд; сократите текст"
                        )

                    import soundfile as sf

                    output = io.BytesIO()
                    sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
                    cached = output.getvalue()
                    self._preview_cache[cache_key] = cached
                    while len(self._preview_cache) > 12:
                        self._preview_cache.pop(next(iter(self._preview_cache)))

        return web.Response(
            body=cached,
            content_type="audio/wav",
            headers={"Cache-Control": "private, no-store"},
        )

    async def _get_job(self, request):
        from aiohttp import web

        job = self.jobs.get(request.match_info["jid"])
        if job is None:
            raise web.HTTPNotFound(text="нет такой задачи")
        return web.json_response(self._job_payload(job))

    async def _cancel_job(self, request):
        from aiohttp import web

        job_id = request.match_info["jid"]
        job = self.jobs.get(job_id)
        if job is None:
            raise web.HTTPNotFound(text="нет такой задачи")
        task = self._tasks.get(job_id)
        if job.status in ("queued", "running", "awaiting_approval") and task is not None:
            task.cancel()
        return web.json_response({"ok": True, "job": self._job_payload(job)})

    async def _approve_job(self, request):
        from aiohttp import web

        job_id = request.match_info["jid"]
        job = self.jobs.get(job_id)
        if job is None:
            raise web.HTTPNotFound(text="нет такой задачи")
        if job.status != "awaiting_approval":
            raise web.HTTPBadRequest(text="задача не ожидает решения")
        gate = self._approval_gates.get(job_id)
        if gate is None:
            raise web.HTTPNotFound(text="нет ожидающего решения для этой задачи")
        try:
            data = await request.json()
        except Exception:  # noqa: BLE001
            raise web.HTTPBadRequest(text="ожидается JSON") from None
        approved = bool(data.get("approved"))
        job.status = "running"
        job.approval_kind = None
        job.approval_cause = None
        job.updated_at = time.time()
        log.info(
            "задача %s: пользователь %s переход на локальный резерв",
            job.id, "разрешил" if approved else "отклонил",
        )
        gate.resolve(approved)
        return web.json_response({"ok": True, "job": self._job_payload(job)})

    async def _get_clip(self, request):
        """Отдаёт одну готовую реплику прогрессивного дубляжа."""
        from aiohttp import web

        job_id = Path(request.match_info["jid"]).name  # без обхода каталогов
        name = Path(request.match_info["name"]).name
        if self.api_token:
            expected = self._audio_access_tokens.get(job_id, "")
            supplied = request.query.get("access", "")
            if not expected or not supplied or not hmac.compare_digest(supplied, expected):
                raise web.HTTPUnauthorized(text="нужен корректный токен дорожки")
        path = self.audio_dir / "clips" / job_id / name
        if not path.is_file():
            raise web.HTTPNotFound(text="реплика не найдена")
        return web.FileResponse(path, headers={"Content-Type": "audio/wav"})

    async def _get_audio(self, request):
        from aiohttp import web

        name = Path(request.match_info["name"]).name  # без обхода каталогов
        if self.api_token:
            job_id = Path(name).stem
            expected = self._audio_access_tokens.get(job_id, "")
            supplied = request.query.get("access", "")
            if not expected or not supplied or not hmac.compare_digest(supplied, expected):
                raise web.HTTPUnauthorized(text="нужен корректный токен дорожки")
        path = self.audio_dir / name
        if not path.is_file():
            raise web.HTTPNotFound(text="дорожка не найдена")
        return web.FileResponse(path, headers={"Content-Type": "audio/mp4"})


class ServerBindError(OSError):
    """An occupied listener reported without a CLI traceback."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        super().__init__(
            f"Порт {port} уже занят ({_dashboard_url(host, port)}). "
            "Если UVT уже работает, откройте этот адрес. "
            "Для перезапуска остановите предыдущий UVT через Ctrl+C в его терминале."
        )


async def run_server(
    cfg: AppConfig,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    stop_event: asyncio.Event | None = None,
    route_label: str = "UVT",
    profile_name: str = "configured",
    selectable_profiles: dict[str, AppConfig] | None = None,
    dashboard_routes: list[dict[str, object]] | None = None,
    bound_event: asyncio.Event | None = None,
    settings_store: ServerSettingsStore | None = None,
    settings_key: str | None = None,
) -> None:
    from aiohttp import web

    server_url = _dashboard_url(host, port)
    if not _is_loopback_host(host) and not os.environ.get(
        "UVT_API_TOKEN", ""
    ).strip():
        raise RuntimeError(
            "сетевой UVT-сервер требует UVT_API_TOKEN; без токена используйте host 127.0.0.1"
        )

    server = DubServer(
        cfg,
        route_label=route_label,
        profile_name=profile_name,
        listen_port=port,
        selectable_profiles=selectable_profiles,
        dashboard_routes=dashboard_routes,
        settings_store=settings_store,
        settings_key=settings_key,
    )
    server.listen_host = host
    runner = web.AppRunner(server.app())
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    try:
        await site.start()
    except BaseException as exc:
        await server.close_prepared_models()
        await runner.cleanup()
        if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE:
            raise ServerBindError(host, port) from None
        raise
    if bound_event is not None:
        bound_event.set()
    server.schedule_local_model_prepare()
    if server._prepare_stt_task is not None:
        # The socket is already reachable and /meta exposes checking/loading;
        # the startup log below is emitted only after readiness is known.
        try:
            await server._prepare_stt_task
        except BaseException:
            await server.close_prepared_models()
            await runner.cleanup()
            raise
    log.info(
        "UVT-сервер запущен: %s — установите userscript browser/uvt.user.js "
        "и нажимайте кнопку UVT на видео; Ctrl+C — остановка",
        server_url,
    )

    own_stop_event = stop_event is None
    stop_event = stop_event or asyncio.Event()
    installed_signals: list[signal.Signals] = []
    if own_stop_event:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, stop_event.set)
                installed_signals.append(sig)
    try:
        await stop_event.wait()
    finally:
        await server.close_prepared_models()
        await runner.cleanup()
        if own_stop_event:
            loop = asyncio.get_running_loop()
            for sig in installed_signals:
                with contextlib.suppress(NotImplementedError, RuntimeError):
                    loop.remove_signal_handler(sig)
        log.info("сервер %s остановлен", server_url)


async def run_personal_servers(
    host: str = "127.0.0.1",
    free_port: int = 8765,
    gpt_port: int = 8766,
    eleven_port: int = 8767,
    free_profile: str = "free-vps",
    open_browser: bool = False,
) -> None:
    """Поднимает три batch-маршрута для одного userscript и останавливает вместе."""
    settings_store = ServerSettingsStore.default()
    if settings_store.load_error:
        log.warning("настройки web-панели не загружены: %s", settings_store.load_error)
    local_profiles: dict[str, AppConfig] | None = None
    if free_profile in _LOCAL_PROFILE_LABELS:
        local_profiles = {name: load_config(name) for name in _LOCAL_PROFILE_LABELS}
    routes = (
        ("Free", free_profile, free_port, load_config(free_profile), local_profiles),
        ("GPT", "cloud-fast", gpt_port, load_config("cloud-fast"), None),
        ("ElevenLabs", "cloud-eleven", eleven_port, load_config("cloud-eleven"), None),
    )
    public_url_env = {
        "Free": "UVT_FREE_PUBLIC_URL",
        "GPT": "UVT_GPT_PUBLIC_URL",
        "ElevenLabs": "UVT_ELEVEN_PUBLIC_URL",
    }
    dashboard_routes: list[dict[str, object]] = []
    public_route_flags: list[bool] = []
    for label, profile, port, cfg, _profiles in routes:
        route_url, is_public = _configured_dashboard_url(
            public_url_env[label], _dashboard_url(host, port)
        )
        public_route_flags.append(is_public)
        dashboard_routes.append(
            {
                "label": label,
                "profile": profile,
                "url": route_url,
                "public_url": is_public,
                "engines": {
                    "stt": str(cfg.stt.engine),
                    "translation": str(cfg.translation.engine),
                    "tts": str(cfg.tts.engine),
                },
            }
        )
    if any(public_route_flags) and not all(public_route_flags):
        raise ValueError(
            "для web-панели за reverse proxy задайте все три: "
            "UVT_FREE_PUBLIC_URL, UVT_GPT_PUBLIC_URL и UVT_ELEVEN_PUBLIC_URL"
        )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop_event.set)
            installed_signals.append(sig)

    for label, profile, port, _cfg, _profiles in routes:
        log.info(
            "маршрут %-11s %s (профиль %s)",
            label,
            _dashboard_url(host, port),
            profile,
        )

    bound_events = [asyncio.Event() for _route in routes]
    tasks = [
        asyncio.create_task(
            run_server(
                cfg,
                host,
                port,
                stop_event=stop_event,
                route_label=label,
                profile_name=profile,
                selectable_profiles=profiles,
                dashboard_routes=dashboard_routes,
                bound_event=bound_event,
                settings_store=settings_store,
                settings_key=settings_route_key(label),
            ),
            name=f"uvt-{profile}",
        )
        for (label, profile, port, cfg, profiles), bound_event in zip(
            routes, bound_events, strict=True
        )
    ]

    async def wait_until_all_routes_bound() -> None:
        await asyncio.gather(*(event.wait() for event in bound_events))

    bind_waiter = asyncio.create_task(
        wait_until_all_routes_bound(), name="uvt-personal-bind-barrier"
    )
    try:
        done, _pending = await asyncio.wait(
            {bind_waiter, *tasks}, return_when=asyncio.FIRST_COMPLETED
        )
        if bind_waiter not in done:
            # A route stopped before the three-port barrier. Await it to retain
            # the original bind/startup exception (for example Errno 48).
            stopped_task = next(task for task in tasks if task in done)
            await stopped_task
            raise RuntimeError(
                f"маршрут {stopped_task.get_name()} остановился до запуска всех портов"
            )
        await bind_waiter

        # A bound route may still fail immediately during startup cleanup. Do
        # not open a dashboard for a set which is already incomplete.
        stopped_tasks = [task for task in tasks if task.done()]
        if stopped_tasks:
            if stop_event.is_set():
                await asyncio.gather(*tasks)
                return
            stopped_task = stopped_tasks[0]
            await stopped_task
            raise RuntimeError(
                f"маршрут {stopped_task.get_name()} остановился сразу после bind"
            )

        dashboard_url = _dashboard_url(host, free_port)
        log.info("панель UVT: %s", dashboard_url)
        if open_browser:
            block_reason = _dashboard_open_block_reason(host)
            if block_reason is None:
                await _open_dashboard_in_browser(dashboard_url)
            else:
                log.warning(
                    "панель UVT не открыта автоматически: %s; адрес: %s",
                    block_reason,
                    dashboard_url,
                )

        await asyncio.gather(*tasks)
    finally:
        stop_event.set()
        if not bind_waiter.done():
            bind_waiter.cancel()
        await asyncio.gather(bind_waiter, return_exceptions=True)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for sig in installed_signals:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(sig)
