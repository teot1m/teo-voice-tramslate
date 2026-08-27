"""Реестр движков и загрузчик плагинов (ТЗ §15).

Встроенные движки регистрируются декоратором при импорте своих модулей;
пользовательские — из .py-файлов в каталогах plugin_dirs, без изменения ядра.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

log = logging.getLogger("uvt.registry")

_REGISTRY: dict[str, dict[str, type]] = {}
_builtins_loaded = False
_loaded_plugin_files: set[Path] = set()

_BUILTIN_MODULES = [
    "uvt.engines.capture_sounddevice",
    "uvt.engines.capture_wasapi",
    "uvt.engines.vad_silero",
    "uvt.engines.vad_energy",
    "uvt.engines.vad_webrtc",
    "uvt.engines.stt_faster_whisper",
    "uvt.engines.stt_mlx_whisper",
    "uvt.engines.stt_parakeet_mlx",
    "uvt.engines.stt_openai",
    "uvt.engines.translate_openai",
    "uvt.engines.translate_google",
    "uvt.engines.translate_nllb",
    "uvt.engines.translate_mlx_translategemma",
    "uvt.engines.tts_edge",
    "uvt.engines.tts_openai",
    "uvt.engines.tts_elevenlabs",
    "uvt.engines.tts_kokoro",
    "uvt.engines.tts_piper",
    "uvt.engines.testing",
]

KINDS = ("capture", "vad", "stt", "translation", "tts")


def register(kind: str, name: str):
    """Декоратор регистрации движка: @register("stt", "my-engine")."""

    def deco(cls: type) -> type:
        _REGISTRY.setdefault(kind, {})[name] = cls
        return cls

    return deco


def available(kind: str) -> list[str]:
    load_builtins()
    return sorted(_REGISTRY.get(kind, {}))


def create(kind: str, name: str, cfg):
    load_builtins()
    try:
        cls = _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(
            f"движок {kind}/'{name}' не найден; доступны: {', '.join(available(kind)) or '—'}"
        ) from None
    return cls(cfg)


def load_builtins() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    for module in _BUILTIN_MODULES:
        try:
            importlib.import_module(module)
        except Exception:  # noqa: BLE001 — один сломанный модуль не валит остальные
            log.exception("не удалось импортировать встроенный модуль %s", module)


def load_plugin_dirs(dirs: list[str]) -> None:
    """Загружает *.py из каталогов плагинов. Плагины исполняются как код —
    подключайте только доверенные."""
    for d in dirs:
        root = Path(d).expanduser()
        if not root.is_dir():
            continue
        for file in sorted(root.glob("*.py")):
            _load_plugin_file(file.resolve())


def _load_plugin_file(path: Path) -> None:
    if path in _loaded_plugin_files:
        return
    _loaded_plugin_files.add(path)
    module_name = f"uvt_plugins.{path.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        log.info("плагин загружен: %s", path)
    except Exception:  # noqa: BLE001
        log.exception("не удалось загрузить плагин %s", path)
