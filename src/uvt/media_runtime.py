"""Repair a known Homebrew x265 loader mismatch for this process only.

Never rename libraries, create ABI aliases, install packages, or change shell
configuration. A fallback is committed only after both media tools start.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger("uvt.media_runtime")
_X265_ROOTS = (Path("/opt/homebrew/Cellar/x265"), Path("/usr/local/Cellar/x265"))
_MISSING_X265 = re.compile(r"Library not loaded:[^\r\n]*(libx265\.\d+(?:\.\d+)*\.dylib)")
_PROBE_TIMEOUT = 3
_guard = threading.Lock()
_checked = False
_repaired = False
_error_message: str | None = None


class MediaRuntimeError(RuntimeError):
    """FFmpeg cannot start and no verified process-local repair is available."""


def _probe(binary: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [binary, "-version"], env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            timeout=_PROBE_TIMEOUT, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaRuntimeError("FFmpeg/ffprobe не ответил за 3 секунды. Проверьте установку FFmpeg.") from exc
    except OSError as exc:
        raise MediaRuntimeError("Не удалось запустить FFmpeg/ffprobe. Проверьте установку и права запуска.") from exc


def _candidate_library_dirs(names: set[str]) -> list[str]:
    candidates = set()
    for root in _X265_ROOTS:
        for name in sorted(names):
            for library in root.glob(f"*/lib/{name}"):
                if library.is_file():
                    candidates.add(str(library.parent))
    return sorted(candidates, reverse=True)[:4]


def _check_and_repair() -> bool:
    binaries = [shutil.which(name) for name in ("ffmpeg", "ffprobe")]
    if not all(binaries):
        raise MediaRuntimeError("Для работы со звуком нужны ffmpeg и ffprobe. Установите FFmpeg и повторите запуск.")
    env = os.environ.copy()
    results = [_probe(binary, env) for binary in binaries]
    failed = [result for result in results if result.returncode != 0]
    if not failed:
        return False
    missing = []
    for result in failed:
        match = _MISSING_X265.search(result.stderr or "")
        if not match:
            raise MediaRuntimeError("FFmpeg/ffprobe не запускается. Это не известная ошибка библиотеки x265; проверьте установку FFmpeg.")
        missing.append(match[1])
    directories = _candidate_library_dirs(set(missing))
    # Most pairs need one ABI. The combined candidate also supports a pair
    # needing different existing ABIs, without fabricating compatibility.
    candidates = [[directory] for directory in directories]
    if len(directories) > 1:
        candidates.append(directories)
    previous = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    for paths in candidates:
        candidate = os.pathsep.join([*paths, *([previous] if previous else [])])
        candidate_env = {**env, "DYLD_FALLBACK_LIBRARY_PATH": candidate}
        if all(_probe(binary, candidate_env).returncode == 0 for binary in binaries):
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = candidate
            log.info("Совместимая библиотека FFmpeg подключена только для текущего запуска UVT.")
            return True
    raise MediaRuntimeError(
        "FFmpeg требует прежнюю библиотеку x265, но совместимая установленная версия не найдена. "
        "Обновите или переустановите FFmpeg; библиотеки и настройки системы UVT не менял."
    )


def ensure_media_runtime() -> bool:
    """Probe once on macOS; return whether a verified fallback was applied.

    Linux/Windows and non-media CLI commands need no Homebrew workaround.
    Failures are cached too, so repeated callers do not repeat slow probes.
    """
    if sys.platform != "darwin":
        return False
    global _checked, _repaired, _error_message
    with _guard:
        if not _checked:
            try:
                _repaired = _check_and_repair()
            except MediaRuntimeError as exc:
                _error_message = str(exc)
                _checked = True
                raise
            _checked = True
        if _error_message:
            raise MediaRuntimeError(_error_message)
        return _repaired


def prepare_media_runtime_for_cli(argv: list[str] | None = None) -> None:
    """Set native loader paths before Python starts, if a repair is needed.

    Updating os.environ repairs child FFmpeg processes but macOS dyld does
    not use that update for libraries loaded inside this Python process.
    Re-exec only at CLI startup, before servers, threads or models exist.
    The next process sees working media tools and does not re-exec again.
    """
    previous = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
    repaired = ensure_media_runtime()
    if not repaired or previous == os.environ.get("DYLD_FALLBACK_LIBRARY_PATH"):
        return
    arguments = list(sys.argv[1:] if argv is None else argv)
    log.info("Перезапускаю UVT с совместимой библиотекой FFmpeg для локальной озвучки.")
    try:
        os.execve(sys.executable, [sys.executable, "-m", "uvt", *arguments], dict(os.environ))
    except OSError as exc:
        raise MediaRuntimeError(
            "Не удалось перезапустить UVT с совместимой библиотекой FFmpeg. "
            "Остановите UVT и откройте «Запустить UVT.command»."
        ) from exc
