"""Локальный сервер браузерной кнопки (uvt serve) — движок для userscript.

Userscript (browser/uvt.user.js) вешает кнопку на любой <video> на странице.
По нажатию он присылает сюда адрес страницы и прямую ссылку на медиапоток;
сервер скачивает звук, готовит дублированную дорожку (render_dub_track) и
отдаёт её как .m4a — скрипт проигрывает её синхронно с видео, приглушив
оригинал. Тот же UX, что у voice-over-translation, но перевод локальный/ваш.

API (JSON, CORS открыт):
  POST /dub {page_url?, media_url?, file?, target_lang?} → {id, job_url}
  GET  /job/{id} → {status: queued|running|done|error, progress, detail,
                    audio_url, entries[]}
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from uvt.config import AppConfig
from uvt.dub import render_dub_track

log = logging.getLogger("uvt.server")

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


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


async def _run_process(cmd: list[str], timeout_s: float, what: str) -> None:
    """Запускает подпроцесс так, чтобы отмена задачи убивала и его тоже."""
    proc = await asyncio.create_subprocess_exec(*cmd)
    try:
        code = await asyncio.wait_for(proc.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        await _kill_process(proc)
        raise RuntimeError(f"{what} не уложился в {timeout_s / 60:.0f} мин — прерван") from None
    except asyncio.CancelledError:
        await _kill_process(proc)
        raise
    if code != 0:
        raise RuntimeError(f"{what} завершился с ошибкой (код {code})")


async def _download_media(
    url: str, dest_dir: Path, referer: str | None = None, out_name: str = "media.m4a"
) -> Path:
    """Скачивает только звук прямого медиапотока (mp4/m3u8/webm) через ffmpeg."""
    out = dest_dir / out_name
    log.info("скачиваю поток через ffmpeg: %.120s…", url)
    cmd = ["ffmpeg", "-v", "error", "-y", "-user_agent", _UA]
    if referer:
        # многие CDN отдают поток только со ссылающейся страницы
        cmd += ["-headers", f"Referer: {referer}\r\n"]
    cmd += ["-i", url, "-vn", "-acodec", "aac", "-b:a", "192k", str(out)]
    await _run_process(cmd, 900, "ffmpeg")
    if not out.is_file() or out.stat().st_size < 10_000:
        raise RuntimeError("поток скачался пустым — вероятно, нужны куки сессии")
    log.info("поток скачан: %.1f МБ", out.stat().st_size / 1e6)
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


async def _download_page(page_url: str, dest_dir: Path) -> Path:
    """Скачивает ролик по адресу страницы через yt-dlp (асинхронно, убиваемо)."""
    from uvt.dub import _find_ytdlp

    ytdlp = _find_ytdlp()
    if ytdlp is None:
        raise RuntimeError("для ссылок нужен yt-dlp: pip install yt-dlp")
    log.info("скачиваю ролик через yt-dlp…")
    cmd = [
        ytdlp,
        # Серверу нужен только ЗВУК: отдельная аудиодорожка (оригинальная, не
        # авто-дубляж), а если сайт отдаёт лишь склеенные файлы — САМЫЙ лёгкий
        # из них (w): звук во всех качествах одинаковый.
        "-f", "ba[format_note*=original]/ba/w/b",
        "--progress-delta", "15",
        "--merge-output-format", "mkv",
        "-o", str(dest_dir / "%(title).80s.%(ext)s"), page_url,
    ]
    try:
        await _run_process(cmd, 1800, "yt-dlp")
    except RuntimeError as exc:
        raise RuntimeError(
            f"yt-dlp не смог скачать {page_url} — проверьте, что это настоящая "
            "ссылка на видео"
        ) from exc
    files = sorted(dest_dir.iterdir(), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("yt-dlp ничего не скачал")
    return files[0]


def _rank_candidate(url: str) -> int:
    """Master-манифесты вперёд, явные видеодорожки (1080p/av1/…) — в конец:
    нам нужен звук, а не гигабайты видео."""
    lowered = url.lower()
    score = 0
    if re.search(r"master|playlist|manifest", lowered):
        score -= 2
    if re.search(r"audio|/aud", lowered):
        score -= 1
    if re.search(r"(2160|1440|1080|720|480|360|240)p?|av1|h26[45]|hevc|vp9|video", lowered):
        score += 1
    return score


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    detail: str = ""
    progress: float = 0.0
    audio_url: str | None = None
    entries: list = field(default_factory=list)


class DubServer:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.jobs: dict[str, Job] = {}
        self.audio_dir = _cache_dir()
        self._lock = asyncio.Lock()
        # (источник, языки) → id задачи: повторное нажатие кнопки не пересчитывает
        self._job_cache: dict[tuple, str] = {}
        self._tasks: dict[str, asyncio.Task] = {}
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
        return (
            str(source),
            str(data.get("source_lang") or "auto"),
            str(data.get("target_lang") or self.cfg.target_lang),
            str(data.get("voice_gender") or "male"),
        )

    # --- обработка задач ---

    async def _resolve_source(self, data: dict, workdir: Path) -> Path:
        file_path = data.get("file")
        if file_path:
            path = Path(file_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path

        page_url = data.get("page_url")
        # Прямая ссылка плеера + потоки, замеченные скриптом в ресурсах страницы
        candidates: list[str] = []
        if data.get("media_url"):
            candidates.append(str(data["media_url"]))
        for extra in data.get("media_candidates") or []:
            if isinstance(extra, str) and extra not in candidates:
                candidates.append(extra)
        candidates.sort(key=_rank_candidate)

        yt_error: RuntimeError | None = None
        if page_url:
            try:
                return await _download_page(page_url, workdir)
            except RuntimeError as exc:
                yt_error = exc
                if candidates:
                    log.info(
                        "yt-dlp не справился — пробую медиапотоки со страницы (%d шт.)",
                        len(candidates),
                    )

        # Длительность видео в плеере — фильтр от роликов-превью related-видео
        try:
            duration_hint = float(data.get("duration_hint") or 0) or None
        except (TypeError, ValueError):
            duration_hint = None

        last_error: RuntimeError | None = None
        for index, candidate in enumerate(candidates[:6]):
            try:
                path = await _download_media(
                    candidate, workdir, referer=page_url, out_name=f"media_{index}.m4a"
                )
            except RuntimeError as exc:
                last_error = exc
                log.info("поток не подошёл: %.100s", candidate)
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
                    continue
            return path

        if last_error is not None or yt_error is not None:
            raise last_error or yt_error
        raise RuntimeError("не передан ни адрес страницы, ни ссылка на поток, ни файл")

    async def _run_job(self, job: Job, data: dict) -> None:
        async with self._lock:  # по одной задаче: Whisper не параллелим
            job.status = "running"
            try:
                cfg = self.cfg.model_copy(deep=True)
                if data.get("target_lang"):
                    cfg.target_lang = str(data["target_lang"])
                if data.get("source_lang"):
                    cfg.source_lang = str(data["source_lang"])  # "auto" — автоопределение
                if data.get("voice_gender"):
                    cfg.tts.voice_gender = str(data["voice_gender"])  # male | female

                def on_progress(done: int, total: int) -> None:
                    job.progress = round(done / max(total, 1), 3)

                with tempfile.TemporaryDirectory(prefix="uvt-serve-") as td:
                    job.detail = "скачиваю исходное видео…"
                    source = await self._resolve_source(data, Path(td))
                    job.detail = ""
                    # Для браузера — только голос перевода: оригинал играет сам
                    # плеер на странице (приглушённо), иначе звук двоится.
                    mixed, entries = await render_dub_track(
                        cfg, source, progress=on_progress, mix_original=False
                    )

                    import soundfile as sf

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
                log.info("задача %s готова: %d реплик", job.id, len(entries))
            except asyncio.CancelledError:
                job.status = "cancelled"
                job.detail = "отменено пользователем"
                log.info("задача %s отменена", job.id)
            except Exception as exc:  # noqa: BLE001 — статус уходит клиенту
                job.status = "error"
                job.detail = str(exc)
                if isinstance(exc, (RuntimeError, FileNotFoundError)):
                    # ожидаемые сбои (не скачалось, нет речи) — без простыни traceback
                    log.error("задача %s провалилась: %s", job.id, exc)
                else:
                    log.exception("задача %s провалилась", job.id)

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
        app.router.add_post("/dub", self._post_dub)
        app.router.add_get("/job/{jid}", self._get_job)
        app.router.add_post("/job/{jid}/cancel", self._cancel_job)
        app.router.add_get("/audio/{name}", self._get_audio)
        return app

    async def _index(self, request):
        from aiohttp import web

        lines = [f"UVT server: перевод {self.cfg.target_lang}, задач: {len(self.jobs)}"]
        for job in self.jobs.values():
            lines.append(f"  {job.id}: {job.status} {job.progress:.0%} {job.detail}")
        return web.Response(text="\n".join(lines))

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
                return web.json_response({"id": cached_id, "job_url": f"/job/{cached_id}"})

        job = Job(id=uuid.uuid4().hex[:12])
        self.jobs[job.id] = job
        self._job_cache[key] = job.id
        self._tasks[job.id] = asyncio.get_running_loop().create_task(self._run_job(job, data))
        log.info(
            "новая задача %s: %s",
            job.id, data.get("page_url") or data.get("media_url") or data.get("file"),
        )
        return web.json_response({"id": job.id, "job_url": f"/job/{job.id}"})

    async def _get_job(self, request):
        from aiohttp import web

        job = self.jobs.get(request.match_info["jid"])
        if job is None:
            raise web.HTTPNotFound(text="нет такой задачи")
        return web.json_response(asdict(job))

    async def _cancel_job(self, request):
        from aiohttp import web

        job_id = request.match_info["jid"]
        job = self.jobs.get(job_id)
        if job is None:
            raise web.HTTPNotFound(text="нет такой задачи")
        task = self._tasks.get(job_id)
        if job.status in ("queued", "running") and task is not None:
            task.cancel()
        return web.json_response({"ok": True})

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
