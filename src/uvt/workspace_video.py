"""Owned video originals, synchronized playback and restart-safe completed results."""
from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
import time
from pathlib import Path

from aiohttp import web

MAX_VIDEO_BYTES = 2 * 1024**3
MAX_VIDEO_SECONDS = 90 * 60
VIDEO_FORMAT = "bv*[height<=720][vcodec^=avc]+ba/b[height<=720][vcodec^=avc]/bv*[height<=720]+ba/b[height<=720]/b"


class VideoResults:
    def __init__(self, server):
        self.server = server
        self.root = server.audio_dir / "video-results"
        self.records = {}
        self.tasks = {}
        self.lock = asyncio.Lock()
        self.prune(time.time() - 7 * 86400)
        self.restore()

    def directory(self, jid):
        return self.root / jid

    def record_request(self, job, data):
        allowed = ("page_url", "media_url", "media_candidates", "duration_hint")
        self.records[job.id] = {
            "version": 1, "request": {key: data[key] for key in allowed if key in data},
            "source": None, "has_video": data.get("source_has_video"), "independent_audio": data.get("mix_original") is not True,
            "status": "idle", "detail": "", "progress": None,
        }

    async def retain_source(self, job, source, data):
        """Reuse a video already downloaded for audio; uploads remain task-owned."""
        from uvt.server_workspace import validate_media
        import os
        record = self.records.get(job.id)
        if record is None or data.get("source_has_video") is False:
            return
        source = Path(source)
        uploads = (self.server.audio_dir / "uploads").resolve()
        if data.get("file"):
            if source.resolve().parent == uploads:
                record.update(source=str(source), has_video=data.get("source_has_video"))
            return
        # The audio-first downloader sometimes receives a complete low-resolution
        # MP4. Keep that existing video without downloading the source a second time.
        if source.suffix.lower() not in {".mp4", ".mkv", ".mov", ".webm"}:
            return
        try:
            if not await validate_media(source):
                return
        except web.HTTPException:
            return
        directory = self.directory(job.id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        retained = directory / f"original{source.suffix.lower()}"
        try:
            os.link(source, retained)
        except OSError:
            shutil.copyfile(source, retained)
        retained.chmod(0o600)
        record.update(source=str(retained), has_video=True)

    def persist(self, job):
        if job.status != "done" or job.id not in self.records:
            return
        from dataclasses import asdict
        record = self.records[job.id]
        data = {key: value for key, value in record.items() if key not in {"status", "detail", "progress"}}
        payload = asdict(job)
        # Access secrets are always regenerated for the running server.
        payload["audio_url"] = None
        payload["downloads"] = {}
        payload["clips"] = []
        data["job"] = payload
        directory = self.directory(job.id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / "result.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)

    def restore(self):
        from uvt.server import Job
        from dataclasses import fields
        allowed = {field.name for field in fields(Job)}
        if not self.root.is_dir():
            return
        for directory in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
            if not re.fullmatch(r"[a-f0-9]{12}", directory.name):
                continue
            path = directory / "result.json"
            try:
                if path.stat().st_size > 16 * 1024**2:
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                saved = record.pop("job")
                if saved.get("id") != directory.name or saved.get("status") != "done":
                    continue
                if not (self.server.audio_dir / f"{directory.name}.m4a").is_file():
                    continue
                saved.update(downloads={}, audio_url=None, clips=[])
                job = Job(**{key: value for key, value in saved.items() if key in allowed})
                source = record.get("source")
                if source and not self.owned_path(source):
                    record["source"] = None
                    record["has_video"] = None
                record.update(status="idle", detail="", progress=None)
                self.records[job.id] = record
                self.server.jobs.setdefault(job.id, job)
                self.refresh_urls(self.server.jobs[job.id])
            except (OSError, ValueError, TypeError, KeyError):
                continue

    def owned_path(self, value):
        try:
            path = Path(value).resolve()
            return path.is_relative_to(self.server.audio_dir.resolve()) and path.is_file()
        except (OSError, ValueError, TypeError):
            return False

    def path(self, job, kind):
        record = self.records.get(job.id, {})
        if kind == "original":
            value = record.get("source")
            return Path(value) if value and self.owned_path(value) else None
        name = {"video": "preview.mp4", "translated": "translated.mp4"}.get(kind)
        path = self.directory(job.id) / name if name else None
        return path if path and path.is_file() else None

    def refresh_urls(self, job):
        query = f"?access={self.server._clip_token(job)}" if self.server.api_token else ""
        job.audio_url = f"/audio/{job.id}.m4a{query}"
        job.downloads.update({kind: f"/download/{job.id}/{kind}{query}" for kind in ("m4a", "srt", "vtt", "txt", "json")})
        for kind in ("original", "translated"):
            if self.path(job, kind):
                job.downloads[kind] = f"/download/{job.id}/{kind}{query}"

    def payload(self, job):
        record = self.records.get(job.id)
        if record is None:
            return {}
        self.refresh_urls(job)
        query = f"?access={self.server._clip_token(job)}" if self.server.api_token else ""
        preview = self.path(job, "video")
        return {"video": {
            "status": record.get("status", "idle"), "detail": record.get("detail", ""),
            "progress": record.get("progress"), "has_video": record.get("has_video"),
            "can_prepare": bool(record.get("source") or record.get("request", {}).get("page_url") or record.get("request", {}).get("media_url")),
            "independent_audio": record.get("independent_audio", True),
            "url": f"/download/{job.id}/video{query}" if preview else None,
            "original_volume": record.get("original_volume", 0.15),
            "translation_volume": record.get("translation_volume", 1.0),
            "resolution": record.get("resolution"),
        }}

    def start(self, job, *, original_volume=0.15, translation_volume=1.0):
        if job.id in self.tasks and not self.tasks[job.id].done():
            return
        if len(self.tasks) >= 8:
            raise web.HTTPTooManyRequests(text="В очереди уже 8 видео. Дождитесь завершения или отмените лишнее.")
        if job.id not in self.records:
            raise web.HTTPConflict(text="Не сохранился источник этого задания. Откройте исходное видео новым заданием.")
        record = self.records[job.id]
        if record.get("has_video") is False:
            raise web.HTTPUnprocessableEntity(text="В исходном файле только звук. Для просмотра загрузите видео.")
        for value in (original_volume, translation_volume):
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise web.HTTPBadRequest(text="Громкость должна быть от 0 до 100%.")
        record.update(status="queued", detail="Видео в очереди на подготовку…", progress=None)
        task = asyncio.create_task(self.prepare(job, original_volume, translation_volume))
        self.tasks[job.id] = task
        task.add_done_callback(lambda completed: self.tasks.pop(job.id, None) if self.tasks.get(job.id) is completed else None)

    async def close(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def discard(self, job):
        self.records.pop(job.id, None)
        shutil.rmtree(self.directory(job.id), ignore_errors=True)

    def prune(self, cutoff):
        if not self.root.is_dir():
            return
        for directory in self.root.iterdir():
            if not directory.is_dir() or not re.fullmatch(r"[a-f0-9]{12}", directory.name):
                continue
            if directory.name in self.tasks:
                continue
            try:
                if directory.stat().st_mtime < cutoff:
                    shutil.rmtree(directory)
                    self.records.pop(directory.name, None)
            except OSError:
                pass

    async def _bounded_process(self, command, directory, *, max_bytes=6 * 1024**3, **kwargs):
        from uvt.server import _run_process
        task = asyncio.create_task(_run_process(command, 1800, "подготовка видео", **kwargs))
        try:
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=0.5)
                if done:
                    break
                size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
                if size > max_bytes:
                    raise RuntimeError("Видео больше 2 ГБ. Сохраните фрагмент и загрузите его в студию.")
            result = await task
            size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
            if size > max_bytes:
                raise RuntimeError("Видео больше 2 ГБ. Сохраните фрагмент и загрузите его в студию.")
            return result
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _download(self, job, directory):
        from uvt.server import _browser_media_candidates, _UA, _yt_dlp_progress_parser, _ffmpeg_progress_parser
        from uvt.dub import _find_ytdlp, _ytdlp_js_args
        from uvt.media_discovery import discover_page_media
        from uvt.source_download import run_source_download
        record = self.records[job.id]
        request = record.get("request", {})
        page_url = request.get("page_url")
        errors = []

        def progress(fraction, detail):
            record.update(progress=fraction, detail=detail)

        async def runner(command, timeout, label, **kwargs):
            return await self._bounded_process(command, directory, max_bytes=MAX_VIDEO_BYTES, **kwargs)

        ytdlp = _find_ytdlp()
        if page_url and ytdlp:
            command = [ytdlp, "--ignore-config", "--no-playlist", "--no-color", "--newline", "--progress", "--socket-timeout", "15", "--retries", "2", "--fragment-retries", "2", "--extractor-retries", "1", "--retry-sleep", "2", "--max-filesize", str(MAX_VIDEO_BYTES), "--match-filter", f"duration <= {MAX_VIDEO_SECONDS}", "-f", VIDEO_FORMAT, "--merge-output-format", "mp4", "--progress-template", "download:UVT_PROGRESS:%(progress._percent_str)s;UVT_BYTES:%(progress.downloaded_bytes)s", *_ytdlp_js_args(), "-o", str(directory / "original.%(ext)s"), "--", page_url]
            try:
                await run_source_download(runner, command, _yt_dlp_progress_parser(progress, errors), progress)
                found = [path for path in directory.glob("original.*") if path.suffix not in {".part", ".ytdl"}]
                if len(found) == 1:
                    return found[0]
                raise RuntimeError("Сервис не вернул видео допустимого размера или длительности.")
            except RuntimeError as exc:
                errors.append(str(exc))
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink(missing_ok=True)
        candidates = _browser_media_candidates(request)
        if page_url:
            try:
                candidates += await discover_page_media(page_url)
            except Exception as exc:
                errors.append(str(exc))
        try:
            duration_hint = float(request.get("duration_hint") or 0)
            if not math.isfinite(duration_hint) or duration_hint <= 0:
                duration_hint = None
        except (ValueError, TypeError):
            duration_hint = None
        for url in list(dict.fromkeys(candidates))[:3]:
            output = directory / "original.mp4"
            command = ["ffmpeg", "-v", "error", "-nostats", "-progress", "pipe:1", "-y", "-rw_timeout", "15000000", "-user_agent", _UA]
            if page_url:
                command += ["-headers", f"Referer: {page_url}\r\n"]
            command += ["-i", url, "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-t", str(MAX_VIDEO_SECONDS + 1), "-movflags", "+faststart", str(output)]
            try:
                await self._bounded_process(command, directory, max_bytes=MAX_VIDEO_BYTES, on_line=_ffmpeg_progress_parser(duration_hint, progress))
                return output
            except RuntimeError as exc:
                errors.append(str(exc))
                output.unlink(missing_ok=True)
        raise RuntimeError("Не удалось получить исходное видео. Откройте ссылку в Chrome с UVT или загрузите локальный файл. " + (errors[-1][-500:] if errors else "Нет доступного видеопотока."))

    async def prepare(self, job, original_volume, translation_volume):
        from uvt.server_workspace import validate_media
        from uvt.server import _run_process
        import tempfile
        record = self.records[job.id]
        directory = self.directory(job.id)
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            async with self.lock:
                source = self.path(job, "original")
                needed = max(1024**3, source.stat().st_size * 2 + 512 * 1024**2) if source else 6 * 1024**3
                if shutil.disk_usage(directory).free < needed:
                    raise RuntimeError(f"Для подготовки видео нужно хотя бы {math.ceil(needed / 1024**3)} ГБ свободного места.")
                record.update(status="running", detail="Подготавливаю исходное видео…", progress=None)
                if source is None:
                    with tempfile.TemporaryDirectory(prefix="download-", dir=directory) as temporary:
                        found = await self._download(job, Path(temporary))
                        has_video = await validate_media(found)
                        if not has_video:
                            record["has_video"] = False
                            raise RuntimeError("Источник содержит только звук. Загрузите файл с видео.")
                        source = directory / f"original{found.suffix.lower()}"
                        found.replace(source)
                        source.chmod(0o600)
                        record.update(source=str(source), has_video=True)
                elif not await validate_media(source):
                    record["has_video"] = False
                    raise RuntimeError("В исходном файле только звук. Для просмотра загрузите видео.")
                else:
                    record["has_video"] = True
                metadata_lines = []
                await _run_process(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height", "-of", "json", str(source)], 20, "проверка видеодорожки", on_line=metadata_lines.append)
                metadata = json.loads("\n".join(metadata_lines))["streams"][0]
                record["resolution"] = f'{metadata.get("width", "?")}×{metadata.get("height", "?")}'
                self.refresh_urls(job)
                self.persist(job)
                preview = directory / "preview.mp4"
                if not preview.is_file():
                    record.update(detail="Готовлю видео для встроенного плеера…", progress=None)
                    temporary = directory / "preview.pending.mp4"
                    # Copy the video when the source is already MP4/MOV. Other codecs
                    # are transcoded once; the retained original is never altered.
                    video_args = ["-c:v", "copy"] if metadata.get("codec_name") == "h264" else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22"]
                    await self._bounded_process(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0", *video_args, "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(temporary)], directory)
                    temporary.replace(preview)
                record.update(detail="Сохраняю видео с переводом и выбранной громкостью…", progress=None)
                translation = self.server.audio_dir / f"{job.id}.m4a"
                temporary = directory / "translated.pending.mp4"
                if record.get("independent_audio", True):
                    filters = f"[0:a:0]volume={original_volume}[original];[1:a:0]volume={translation_volume},apad[dub];[original][dub]amix=inputs=2:duration=first:normalize=0:dropout_transition=0,alimiter=limit=0.95:level=0:latency=1[mix]"
                else:
                    filters = f"[1:a:0]volume={translation_volume},apad[mix]"
                await self._bounded_process(["ffmpeg", "-v", "error", "-y", "-i", str(preview), "-i", str(translation), "-filter_complex", filters, "-map", "0:v:0", "-map", "[mix]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(temporary)], directory)
                temporary.replace(directory / "translated.mp4")
                for path in directory.iterdir():
                    if path.is_file():
                        path.chmod(0o600)
                record.update(status="ready", detail="Видео готово. Громкость в плеере можно менять отдельно.", progress=1.0, original_volume=original_volume, translation_volume=translation_volume)
                self.refresh_urls(job)
                self.persist(job)
        except asyncio.CancelledError:
            record.update(status="cancelled", detail="Подготовка видео остановлена. Перевод сохранён.", progress=None)
            raise
        except Exception as exc:
            record.update(status="error", detail=str(exc)[:1600], progress=None)
        finally:
            for path in directory.glob("*.pending.mp4"):
                path.unlink(missing_ok=True)
