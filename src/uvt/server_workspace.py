"""File workspace: bounded streaming uploads and authenticated exports.

The browser never sends a local filesystem path. Only files uploaded through
the dashboard are owned here; their lifetime is tied to the processing task.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import math
import shutil
import uuid
import unicodedata
from pathlib import Path

from aiohttp import web

from uvt.history import EXPORTERS, HistoryEntry

MAX_UPLOAD_BYTES = 2 * 1024**3
MAX_DURATION_SECONDS = 90 * 60
MAX_WORKSPACE_JOBS = 8
MEDIA_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus"}
STATIC_DIR = Path(__file__).with_name("web")
# Explicit media demuxers prevent a playlist renamed .mp4 from opening URLs
# or unrelated local files. The same accepted formats are checked in metadata.
MEDIA_FORMATS = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm", "avi", "mp3", "wav", "flac", "ogg", "aac"}
PROBE_TIMEOUT_SECONDS = 20
MAX_PROBE_BYTES = 64 * 1024


def upload_source_name(filename: str) -> str:
    """Keep a short display name, never a caller-supplied filesystem path."""
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(char for char in basename if not unicodedata.category(char).startswith("C"))
    return cleaned.strip()[:180] or "Загруженный файл"


async def validate_media(path: Path) -> bool:
    """Read bounded metadata from supported local containers before model work."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
            "-format_whitelist", ",".join(sorted(MEDIA_FORMATS)),
            "-show_entries", "format=duration,format_name:stream=codec_type",
            "-of", "json", str(path), stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as exc:
        raise web.HTTPServiceUnavailable(text="Не удалось запустить ffprobe. Проверьте установку FFmpeg и перезапустите UVT.") from exc

    async def read_metadata():
        output = bytearray()
        while chunk := await proc.stdout.read(8192):
            if len(output) + len(chunk) > MAX_PROBE_BYTES:
                raise web.HTTPUnprocessableEntity(text="У файла слишком большой список дорожек. Выберите обычный видео- или аудиофайл.")
            output.extend(chunk)
        await proc.wait()
        return bytes(output)

    try:
        output = await asyncio.wait_for(read_metadata(), timeout=PROBE_TIMEOUT_SECONDS)
    except BaseException as exc:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        if isinstance(exc, asyncio.TimeoutError):
            raise web.HTTPUnprocessableEntity(text="Проверка файла заняла слишком много времени. Попробуйте другой файл или сохраните его в MP4/WAV.") from exc
        raise
    try:
        info = json.loads(output)
        if not isinstance(info, dict) or not isinstance(info.get("format"), dict):
            raise ValueError("invalid format metadata")
        duration = float(info["format"].get("duration"))
        formats = set(str(info["format"].get("format_name", "")).split(","))
        raw_streams = info.get("streams")
        if not isinstance(raw_streams, list) or not all(isinstance(item, dict) for item in raw_streams):
            raise ValueError("invalid stream metadata")
        streams = {item.get("codec_type") for item in raw_streams}
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        raise web.HTTPUnprocessableEntity(text="Не удалось прочитать медиафайл. Выберите MP4, MKV, WAV или другой поддерживаемый формат.") from exc
    if proc.returncode or not formats.intersection(MEDIA_FORMATS):
        raise web.HTTPUnprocessableEntity(text="Формат файла не поддерживается. Списки воспроизведения вместо видео не принимаются.")
    if "audio" not in streams or not math.isfinite(duration) or duration <= 0:
        raise web.HTTPUnprocessableEntity(text="В файле нет доступной звуковой дорожки.")
    if duration > MAX_DURATION_SECONDS:
        raise web.HTTPUnprocessableEntity(
            text="Для 16 ГБ памяти обрабатывайте файлы частями до 90 минут: полная дорожка пока собирается в памяти."
        )
    return "video" in streams


class WorkspaceRoutes:
    def __init__(self, server):
        self.server = server
        self.upload_lock = asyncio.Lock()

    def register(self, app):
        app.router.add_get("/workspace", self.index)
        app.router.add_get("/workspace/app.js", self.script)
        app.router.add_get("/workspace/style.css", self.style)
        app.router.add_get("/workspace/userscript.user.js", self.userscript)
        app.router.add_get("/workspace/jobs", self.jobs)
        app.router.add_post("/workspace/upload", self.upload)
        app.router.add_get("/download/{jid}/{kind}", self.download)

    def require_dashboard(self, request):
        if not self.server._dashboard_request_allowed(request):
            raise web.HTTPForbidden(text="Откройте рабочую панель UVT на этом компьютере.")

    async def index(self, request):
        return web.FileResponse(STATIC_DIR / "workspace.html", headers={"Cache-Control": "no-store"})

    async def script(self, request):
        return web.FileResponse(STATIC_DIR / "workspace.js", headers={"Cache-Control": "no-cache"})

    async def style(self, request):
        return web.FileResponse(STATIC_DIR / "workspace.css", headers={"Cache-Control": "no-cache"})

    async def userscript(self, request):
        script = Path(__file__).resolve().parents[2] / "browser" / "uvt.user.js"
        if not script.is_file():
            raise web.HTTPNotFound(text="Откройте browser/uvt.user.js в исходной папке проекта.")
        return web.FileResponse(script, headers={"Content-Type": "text/javascript; charset=utf-8", "Cache-Control": "no-store"})

    async def jobs(self, request):
        self.require_dashboard(request)
        items = sorted(self.server.jobs.values(), key=lambda job: job.created_at, reverse=True)[:30]
        jobs = []
        for job in items:
            payload = self.server._job_payload(job)
            payload.pop("entries", None)
            payload.pop("clips", None)
            jobs.append(payload)
        return web.json_response({"jobs": jobs}, headers={"Cache-Control": "no-store"})

    async def upload(self, request):
        self.require_dashboard(request)
        if self.upload_lock.locked():
            raise web.HTTPConflict(text="Уже загружается файл. Дождитесь завершения загрузки.")
        active = sum(job.status in {"queued", "running", "awaiting_approval"} for job in self.server.jobs.values())
        if active >= MAX_WORKSPACE_JOBS:
            raise web.HTTPTooManyRequests(text="В очереди уже 8 задач. Дождитесь завершения или отмените лишние.")
        if request.content_type != "multipart/form-data":
            raise web.HTTPBadRequest(text="Выберите видео или аудиофайл.")
        # A fixed budget bounds disk consumption before multipart parsing starts.
        if request.content_length and request.content_length > MAX_UPLOAD_BYTES + 65536:
            raise web.HTTPRequestEntityTooLarge(max_size=MAX_UPLOAD_BYTES, actual_size=request.content_length)
        async with self.upload_lock:
            upload_dir = self.server.audio_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            if shutil.disk_usage(upload_dir).free < MAX_UPLOAD_BYTES + 1024**3:
                raise web.HTTPInsufficientStorage(text="Для загрузки и результата освободите минимум 3 ГБ на диске.")
            path = None
            handed_off = False
            options = {}
            source_name = "Загруженный файл"
            try:
                reader = await request.multipart()
                count = 0
                async for part in reader:
                    count += 1
                    if count > 2:
                        raise web.HTTPBadRequest(text="Отправьте один файл и его настройки.")
                    if part.name == "options" and part.filename is None:
                        raw = bytearray()
                        while chunk := await part.read_chunk(8192):
                            raw.extend(chunk)
                            if len(raw) > 16384:
                                raise web.HTTPBadRequest(text="Слишком большой запрос настроек.")
                        try:
                            options = json.loads(raw)
                        except (ValueError, UnicodeDecodeError):
                            raise web.HTTPBadRequest(text="Не удалось прочитать настройки.") from None
                        if not isinstance(options, dict):
                            raise web.HTTPBadRequest(text="Настройки должны быть объектом.")
                    elif part.name == "file" and part.filename and path is None:
                        suffix = Path(part.filename).suffix.lower()
                        if suffix not in MEDIA_EXTENSIONS:
                            raise web.HTTPUnsupportedMediaType(text="Выберите MP4, MKV, MOV, WebM или аудиофайл.")
                        source_name = upload_source_name(part.filename)
                        path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
                        total = 0
                        with path.open("xb") as handle:
                            path.chmod(0o600)
                            while chunk := await part.read_chunk(256 * 1024):
                                total += len(chunk)
                                if total > MAX_UPLOAD_BYTES:
                                    raise web.HTTPRequestEntityTooLarge(max_size=MAX_UPLOAD_BYTES, actual_size=total)
                                handle.write(chunk)
                        if not total:
                            raise web.HTTPBadRequest(text="Файл пустой.")
                    else:
                        raise web.HTTPBadRequest(text="Отправьте один файл и его настройки.")
                if path is None:
                    raise web.HTTPBadRequest(text="Файл не выбран.")
                has_video = await validate_media(path)
                # Ignore caller paths/URLs and allow only ordinary job preferences.
                data = {key: options[key] for key in (
                    "settings_mode", "profile_id", "source_lang", "target_lang", "voice_gender", "voice_id"
                ) if key in options}
                data.update(file=str(path), source_name=source_name, mix_original=True, export_video=has_video and options.get("export_video") is True)
                response = await self.server._submit_dub_job(data)
                job_id = json.loads(response.body)["id"]
                task = self.server._tasks[job_id]
                task.add_done_callback(lambda _task, owned_path=path: owned_path.unlink(missing_ok=True))
                handed_off = True
                return response
            except (ValueError, AssertionError) as exc:
                # aiohttp's multipart parser rejects absent/broken boundaries
                # with these exceptions; malformed uploads are client errors.
                raise web.HTTPBadRequest(text="Не удалось прочитать загрузку. Выберите файл заново и повторите отправку.") from exc
            except OSError as exc:
                if exc.errno == 28:
                    raise web.HTTPInsufficientStorage(text="Недостаточно места для файла.") from None
                raise
            finally:
                if path is not None and not handed_off:
                    path.unlink(missing_ok=True)

    async def download(self, request):
        job_id, kind = request.match_info["jid"], request.match_info["kind"]
        job = self.server.jobs.get(job_id)
        if job is None or job.status != "done":
            raise web.HTTPNotFound(text="Результат ещё не готов или задача недоступна после перезапуска сервера.")
        if self.server.api_token:
            expected = self.server._audio_access_tokens.get(job_id, "")
            supplied = request.query.get("access", "")
            if not self.server._request_has_api_token(request) and not (expected and hmac.compare_digest(expected, supplied)):
                raise web.HTTPUnauthorized(text="Нужен токен доступа к результату.")
        headers = {"Cache-Control": "private, no-store", "Content-Disposition": f'attachment; filename="uvt-{job_id}.{kind}"'}
        if kind in {"m4a", "mkv"}:
            path = self.server.audio_dir / f"{job_id}.{kind}"
            if not path.is_file():
                raise web.HTTPNotFound(text="Этот формат результата недоступен.")
            return web.FileResponse(path, headers=headers)
        if kind not in EXPORTERS:
            raise web.HTTPNotFound(text="Неизвестный формат экспорта.")
        entries = [HistoryEntry(**entry) for entry in job.entries]
        return web.Response(text=EXPORTERS[kind](entries), content_type="application/json" if kind == "json" else "text/plain", charset="utf-8", headers=headers)
