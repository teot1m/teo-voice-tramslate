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
import logging
import os
import re
import signal
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from uvt.config import AppConfig
from uvt.dub import render_dub_track
from uvt.fallback import ApprovalGate

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

# Download is a short, separate phase before render_dub_track's 0–100% work.
# Keeping it in a small prefix makes the externally visible progress monotonic.
_DOWNLOAD_PROGRESS_SHARE = 0.08
_ProgressCallback = Callable[[float | None, str], None]
_MAX_BROWSER_MEDIA_CANDIDATES = 6

# Это именно стадии подготовки готовой дорожки. Они не означают потоковый
# перевод: браузер получает результат только после завершения всей задачи.
_STAGE_DETAILS = {
    "queue": "ожидает свободный обработчик…",
    "download": "получаю исходный звук…",
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


async def _kill_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()


async def _read_process_lines(stream, on_line: Callable[[str], None]) -> None:
    """Drain a subprocess stream so progress parsing can never deadlock it."""
    while True:
        line = await stream.readline()
        if not line:
            return
        try:
            on_line(line.decode("utf-8", errors="replace").strip())
        except Exception:  # noqa: BLE001 — progress must never break download
            log.debug("не удалось разобрать прогресс подпроцесса", exc_info=True)


async def _run_process(
    cmd: list[str],
    timeout_s: float,
    what: str,
    *,
    on_line: Callable[[str], None] | None = None,
) -> None:
    """Запустить подпроцесс с опциональным безопасным парсером progress.

    При включённом progress читаем оба pipe параллельно: иначе заполненный
    stdout/stderr способен подвесить ffmpeg/yt-dlp. Отмена по-прежнему сначала
    завершает дочерний процесс, затем дожидается drain-задач.
    """
    pipe = asyncio.subprocess.PIPE if on_line is not None else None
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=pipe, stderr=pipe)
    readers: list[asyncio.Task] = []
    if on_line is not None:
        if proc.stdout is not None:
            readers.append(asyncio.create_task(_read_process_lines(proc.stdout, on_line)))
        if proc.stderr is not None:
            readers.append(asyncio.create_task(_read_process_lines(proc.stderr, on_line)))
    try:
        code = await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        if readers:
            await asyncio.gather(*readers, return_exceptions=True)
    except asyncio.TimeoutError:
        await _kill_process(proc)
        if readers:
            await asyncio.gather(*readers, return_exceptions=True)
        raise RuntimeError(f"{what} не уложился в {timeout_s / 60:.0f} мин — прерван") from None
    except asyncio.CancelledError:
        await _kill_process(proc)
        if readers:
            await asyncio.gather(*readers, return_exceptions=True)
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


def _yt_dlp_progress_parser(progress: _ProgressCallback) -> Callable[[str], None]:
    """Read the explicit marker emitted by yt-dlp's progress template."""
    pattern = re.compile(r"UVT_PROGRESS:\s*([0-9]+(?:[.,][0-9]+)?)%")

    def on_line(line: str) -> None:
        match = pattern.search(line)
        if match is None:
            return
        try:
            percent = float(match.group(1).replace(",", "."))
        except ValueError:
            return
        percent = min(99.0, max(0.0, percent))
        progress(percent / 100.0, f"скачиваю звук со страницы: {percent:.0f}%")

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
    from uvt.dub import _find_ytdlp

    ytdlp = _find_ytdlp()
    if ytdlp is None:
        raise RuntimeError("для ссылок нужен yt-dlp: pip install yt-dlp")
    log.info("скачиваю ролик через yt-dlp…")
    if progress is not None:
        progress(0.0, "подключаюсь к странице через yt-dlp…")
    cmd = [
        ytdlp,
        # Серверу нужен только звук: сначала отдельная original-дорожка, затем
        # любой audio-only формат. Если у сайта только muxed-видео, берём
        # наименьший аудио-содержащий вариант, а не максимальное качество.
        "--no-playlist",
        "--no-color",
        "-f", _YT_DLP_AUDIO_SELECTOR,
        "--progress-delta", "3",
    ]
    if progress is not None:
        cmd += ["--progress-template", "download:UVT_PROGRESS:%(progress._percent_str)s"]
    cmd += [
        "-o", str(dest_dir / "%(title).80s.%(ext)s"), page_url,
    ]
    try:
        if progress is None:
            await _run_process(cmd, 1800, "yt-dlp")
        else:
            await _run_process(cmd, 1800, "yt-dlp", on_line=_yt_dlp_progress_parser(progress))
    except RuntimeError as exc:
        raise RuntimeError(
            f"yt-dlp не поддержал или не смог скачать {page_url}. UVT не обходит "
            "авторизацию, DRM и ограничения сайта; используйте законно сохранённый "
            "локальный файл или публичную ссылку поддерживаемого сервиса."
        ) from exc
    files = sorted(dest_dir.iterdir(), key=lambda p: p.stat().st_size, reverse=True)
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

    Ранжируем и текущий ``media_url``, и сетевые ресурсы: явный audio/HLS/DASH
    идёт раньше muxed MP4/WebM. yt-dlp получает страницу только после всех
    этих попыток.
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
    ordered = sorted(candidates, key=_rank_candidate)
    if len(ordered) <= _MAX_BROWSER_MEDIA_CANDIDATES or primary not in ordered:
        return ordered[:_MAX_BROWSER_MEDIA_CANDIDATES]

    # userscript уже ограничивает extras шестью URL, но добавляет к ним
    # media_url. Если он стал седьмым после сортировки, не теряем известный
    # текущий src: он остаётся последней попыткой вместо худшего ресурса.
    if primary not in ordered[:_MAX_BROWSER_MEDIA_CANDIDATES]:
        return ordered[: _MAX_BROWSER_MEDIA_CANDIDATES - 1] + [primary]
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
    entries: list = field(default_factory=list)
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


class DubServer:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.jobs: dict[str, Job] = {}
        self.audio_dir = _cache_dir()
        self._lock = asyncio.Lock()
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
        self._cleanup_audio_cache()

    def _cleanup_audio_cache(self, max_age_days: float = 7.0) -> None:
        """Дорожки старше недели из ~/.cache/uvt/serve удаляются при старте."""
        cutoff = time.time() - max_age_days * 86400
        removed = 0
        for file in self.audio_dir.glob("*.m4a"):
            try:
                if file.stat().st_mtime < cutoff:
                    file.unlink()
                    removed += 1
            except OSError:
                continue
        if removed:
            log.info("кэш дорожек: удалено %d старых файлов", removed)

    def _cache_key(self, data: dict) -> tuple:
        source = data.get("page_url") or data.get("media_url") or data.get("file") or ""
        # Отсутствующий выбор должен значить текущий auto/manual маршрут
        # профиля, а не старый неявный male. Иначе запрос без voice_gender
        # мог получить из кэша дорожку, созданную с явным male voice.
        voice_gender = data.get("voice_gender") or self.cfg.tts.voice_gender or "auto"
        return (
            str(source),
            str(data.get("source_lang") or "auto"),
            str(data.get("target_lang") or self.cfg.target_lang),
            str(voice_gender),
        )

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

    def _set_render_progress(self, job: Job, done: int, total: int) -> None:
        """Переводит существующий progress render_dub_track в понятные этапы.

        Внутренний batch-конвейер уже сообщает 0–70 % для STT, 70–85 % для
        перевода, 85–97 % для TTS и финальный 100 % после сборки. Не меняем его
        API: только даём этой информации имена для браузера.
        """
        progress = self._clamp_progress(done / max(total, 1))
        overall = round(_DOWNLOAD_PROGRESS_SHARE + (1.0 - _DOWNLOAD_PROGRESS_SHARE) * progress, 3)
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
                _DOWNLOAD_PROGRESS_SHARE + (1.0 - _DOWNLOAD_PROGRESS_SHARE) * min(progress, 0.985),
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
            "stt": {"faster-whisper", "mlx-whisper", "dummy"},
            "translation": {"none", "passthrough", "dummy"},
            "tts": {"kokoro", "piper", "dummy", "none"},
        }
        cloud_engines = {
            "translation": {"google-free"},
            "tts": {"edge", "openai"},
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

    def _metadata(self, cfg: AppConfig | None = None) -> dict:
        cfg = cfg or self.cfg
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
        local_host = self.listen_host in {"localhost", "127.0.0.1", "::1"}
        return {
            "api_version": 1,
            "mode": "batch",
            "capabilities": {
                "batch_dubbing": True,
                "live_translation": False,
                "streaming_audio": False,
            },
            "profile": {
                # Имя YAML-профиля до AppConfig не доходит, поэтому не
                # угадываем free/cloud: показываем реальную конфигурацию.
                "name": "configured",
                "kind": profile_kind,
                "source_lang": cfg.source_lang,
                "target_lang": cfg.target_lang,
                "engines": {item["kind"]: item["engine"] for item in engines},
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
        cfg = self.cfg.model_copy(deep=True)
        if data.get("source_lang"):
            cfg.source_lang = str(data["source_lang"])
        if data.get("target_lang"):
            cfg.target_lang = str(data["target_lang"])
        if data.get("voice_gender"):
            cfg.tts.voice_gender = str(data["voice_gender"])
        return self._metadata(cfg)

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
        payload = asdict(job)
        payload.update(
            {
                "mode": "batch",
                "is_live": False,
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

    async def _run_job(self, job: Job, data: dict) -> None:
        try:
            async with self._lock:  # по одной задаче: Whisper не параллелим
                now = time.time()
                job.status = "running"
                job.started_at = now
                job.updated_at = now
                self._set_stage(job, "download")
                cfg = self.cfg.model_copy(deep=True)
                if data.get("target_lang"):
                    cfg.target_lang = str(data["target_lang"])
                if data.get("source_lang"):
                    cfg.source_lang = str(data["source_lang"])  # "auto" — автоопределение
                if data.get("voice_gender"):
                    cfg.tts.voice_gender = str(data["voice_gender"])  # male | female

                def on_progress(done: int, total: int) -> None:
                    self._set_render_progress(job, done, total)

                def on_download_progress(fraction: float | None, detail: str) -> None:
                    self._set_download_progress(job, fraction, detail)

                approval = ApprovalGate(
                    on_request=lambda kind, cause: self._request_approval(job, kind, cause)
                )
                self._approval_gates[job.id] = approval

                with tempfile.TemporaryDirectory(prefix="uvt-serve-") as td:
                    source = await self._resolve_source(
                        data,
                        Path(td),
                        progress=on_download_progress,
                    )
                    # A downloader calls this on success too; repeat it here
                    # for plugin/local sources that only return a path.
                    self._set_download_progress(job, 1.0, "исходный звук получен")
                    self._set_stage(job, "transcribe")
                    # Для браузера — только голос перевода: оригинал играет сам
                    # плеер на странице (приглушённо), иначе звук двоится.
                    mixed, entries = await render_dub_track(
                        cfg, source, progress=on_progress, mix_original=False, approval=approval
                    )

                    import soundfile as sf

                    self._set_stage(job, "mix", stage_progress=max(job.stage_progress, 0.78))
                    wav = Path(td) / "mix.wav"
                    sf.write(wav, mixed, 48000, subtype="PCM_16")
                    audio_path = self.audio_dir / f"{job.id}.m4a"
                    subprocess.run(
                        [
                            "ffmpeg", "-v", "error", "-y", "-i", str(wav),
                            "-c:a", "aac", "-b:a", "160k", str(audio_path),
                        ],
                        check=True,
                    )

                job.entries = [asdict(e) for e in entries]
                job.audio_url = f"/audio/{job.id}.m4a"
                job.progress = 1.0
                job.status = "done"
                job.finished_at = time.time()
                self._set_stage(job, "done", stage_progress=1.0)
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
            log.info("задача %s отменена", job.id)
        except Exception as exc:  # noqa: BLE001 — статус уходит клиенту
            job.status = "error"
            job.finished_at = time.time()
            self._set_stage(job, "error", detail=str(exc), stage_progress=1.0)
            if isinstance(exc, (RuntimeError, FileNotFoundError)):
                # ожидаемые сбои (не скачалось, нет речи) — без простыни traceback
                log.error("задача %s провалилась: %s", job.id, exc)
            else:
                log.exception("задача %s провалилась", job.id)
        finally:
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
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            if isinstance(response, web.HTTPException):
                raise response
            return response

        async def options(_request):
            return web.Response()

        app = web.Application(middlewares=[cors])
        app.router.add_route("OPTIONS", "/{tail:.*}", options)
        app.router.add_get("/", self._index)
        app.router.add_get("/meta", self._get_meta)
        app.router.add_post("/dub", self._post_dub)
        app.router.add_get("/job/{jid}", self._get_job)
        app.router.add_post("/job/{jid}/cancel", self._cancel_job)
        app.router.add_post("/job/{jid}/approve", self._approve_job)
        app.router.add_get("/audio/{name}", self._get_audio)
        return app

    async def _index(self, request):
        from aiohttp import web

        lines = [f"UVT server: перевод {self.cfg.target_lang}, задач: {len(self.jobs)}"]
        for job in self.jobs.values():
            lines.append(
                f"  {job.id}: {job.status}/{job.stage} {job.progress:.0%} {job.detail}"
            )
        return web.Response(text="\n".join(lines))

    async def _get_meta(self, request):
        from aiohttp import web

        return web.json_response(self._metadata())

    async def _post_dub(self, request):
        from aiohttp import web

        try:
            data = await request.json()
        except Exception:  # noqa: BLE001
            raise web.HTTPBadRequest(text="ожидается JSON") from None

        key = self._cache_key(data)
        cached_id = self._job_cache.get(key)
        if cached_id:
            cached = self.jobs.get(cached_id)
            if cached is not None and cached.status not in ("error", "cancelled"):
                log.info("задача из кэша: %s", cached_id)
                payload = self._job_payload(cached)
                payload.update({"job_url": f"/job/{cached_id}", "meta": self._metadata_for_request(data)})
                return web.json_response(payload)

        job = Job(id=uuid.uuid4().hex[:12])
        self._set_stage(job, "queue")
        self.jobs[job.id] = job
        self._job_cache[key] = job.id
        self._tasks[job.id] = asyncio.get_running_loop().create_task(self._run_job(job, data))
        log.info(
            "новая задача %s: %s",
            job.id, data.get("page_url") or data.get("media_url") or data.get("file"),
        )
        payload = self._job_payload(job)
        payload.update({"job_url": f"/job/{job.id}", "meta": self._metadata_for_request(data)})
        return web.json_response(payload)

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

    async def _get_audio(self, request):
        from aiohttp import web

        name = Path(request.match_info["name"]).name  # без обхода каталогов
        path = self.audio_dir / name
        if not path.is_file():
            raise web.HTTPNotFound(text="дорожка не найдена")
        return web.FileResponse(path, headers={"Content-Type": "audio/mp4"})


async def run_server(cfg: AppConfig, host: str = "127.0.0.1", port: int = 8765) -> None:
    from aiohttp import web

    server = DubServer(cfg)
    server.listen_host = host
    runner = web.AppRunner(server.app())
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info(
        "UVT-сервер запущен: http://%s:%d — установите userscript browser/uvt.user.js "
        "и нажимайте кнопку UVT на видео; Ctrl+C — остановка", host, port,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop_event.set)
    try:
        await stop_event.wait()
    finally:
        await runner.cleanup()
        log.info("сервер остановлен")
