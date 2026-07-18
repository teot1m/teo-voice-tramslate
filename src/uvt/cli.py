"""CLI: uvt run | dub | serve | serve-personal | devices | profiles | gui."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from uvt import __version__
from uvt.config import load_config
from uvt.logs import setup_logging


def _strip_malloc_env() -> None:
    """Отладочные Malloc*-переменные macOS (их выставляют VS Code/Xcode)
    заставляют каждый дочерний процесс печатать предупреждение — убираем."""
    for key in list(os.environ):
        if key.startswith("Malloc"):
            os.environ.pop(key, None)

def _load_env_file(path: str = ".env") -> None:
    """Подхватывает переменные окружения (OPENAI_API_KEY и т.п.) из .env
    в текущей папке. Уже выставленные переменные не перезаписывает."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip().removeprefix("export ").strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_VIRTUAL_MARKERS = ("blackhole", "vb-audio", "cable", "monitor", "loopback", "virtual", "voicemeeter")
_LEGACY_LIVE_MODES = {"replace", "dual"}
_PROFILE_OVERVIEW = (
    ("local", "private local: Whisper + Ollama + Piper; модели и Piper нужно настроить"),
    ("free", "без платных API: локальные STT/перевод + Microsoft Edge TTS через сеть"),
    ("free-vps", "Linux VPS: CPU Whisper + Ollama Qwen 3B последовательно + Edge TTS"),
    ("free-quality", "Apple Silicon: MLX Whisper large + Qwen 3B последовательно + Edge TTS"),
    ("cloud-fast", "облачные STT/перевод/TTS с упором на минимальную задержку"),
    ("cloud-eleven", "OpenAI STT/GPT-перевод + естественная озвучка ElevenLabs"),
    ("cloud-quality", "облачный пакетный дубляж с упором на качество"),
    ("live", "системный звук через виртуальный вход + бесплатный Edge TTS через сеть"),
    ("cloud", "устаревшее совместимое имя cloud-fast"),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uvt",
        description="Universal Voice Translator — закадровый перевод системного звука в реальном времени",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="запустить конвейер в консоли")
    run.add_argument("--profile", "-p", help="имя профиля из profiles/ или путь к YAML")
    run.add_argument("--source-lang", help="язык оригинала (по умолчанию auto)")
    run.add_argument("--target-lang", help="язык перевода (по умолчанию ru)")
    run.add_argument(
        "--mode",
        choices=["subtitles", "voiceover", "replace", "dual"],
        metavar="{subtitles,voiceover}",
        help="Live-вывод: subtitles или voiceover; replace/dual принимаются только как legacy-алиасы voiceover",
    )
    run.add_argument("--debug", action="store_true", help="подробные логи и тайминги стадий")

    dub = sub.add_parser(
        "dub",
        help="перевести видео/аудиофайл или ссылку: замена голоса + субтитры",
    )
    dub.add_argument("input", help="путь к файлу (mp4/mkv/…) или http(s)-ссылка (нужен yt-dlp)")
    dub.add_argument("--output", "-o", help="куда сохранить результат (по умолчанию рядом с входом)")
    dub.add_argument("--profile", "-p", help="профиль движков (STT/перевод/TTS)")
    dub.add_argument("--source-lang")
    dub.add_argument("--target-lang")
    dub.add_argument(
        "--duck-db", type=float, default=-12.0,
        help="приглушение оригинала под репликами, дБ (по умолчанию -12; -60 — почти убрать)",
    )
    dub.add_argument(
        "--no-original-track", action="store_true",
        help="не сохранять оригинальную аудиодорожку в итоговом mkv",
    )
    dub.add_argument("--debug", action="store_true")

    serve = sub.add_parser(
        "serve",
        help="сервер для браузерной кнопки (userscript browser/uvt.user.js)",
    )
    serve.add_argument("--profile", "-p", help="профиль движков (STT/перевод/TTS)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--target-lang")
    serve.add_argument("--source-lang")
    serve.add_argument("--debug", action="store_true")

    personal = sub.add_parser(
        "serve-personal",
        help="одной командой поднять Free, GPT и ElevenLabs для userscript",
    )
    personal.add_argument("--host", default=os.environ.get("UVT_HOST", "127.0.0.1"))
    personal.add_argument(
        "--free-port", type=int, default=int(os.environ.get("UVT_FREE_PORT", "8765"))
    )
    personal.add_argument(
        "--gpt-port", type=int, default=int(os.environ.get("UVT_CLOUD_PORT", "8766"))
    )
    personal.add_argument(
        "--eleven-port", type=int, default=int(os.environ.get("UVT_ELEVEN_PORT", "8767"))
    )
    personal.add_argument("--debug", action="store_true")

    sub.add_parser("devices", help="список аудиоустройств (вход/выход, виртуальные помечены)")
    sub.add_parser("profiles", help="показать готовые профили и их маршрут данных")

    gui = sub.add_parser("gui", help="графический интерфейс: отдельные Live и Batch сценарии")
    gui.add_argument("--profile", "-p", help="профиль при старте; в GUI можно переключить маршрут")
    gui.add_argument("--debug", action="store_true")

    sub.add_parser("version", help="показать версию")
    return parser


def _overrides(args: argparse.Namespace) -> dict:
    mode = getattr(args, "mode", None)
    # Старые скрипты не ломаем, но больше не создаём иллюзию, что оригинальная
    # речь реально будет удалена/разнесена по каналам.
    if mode in _LEGACY_LIVE_MODES:
        mode = "voiceover"
    pairs = {
        "source_lang": getattr(args, "source_lang", None),
        "target_lang": getattr(args, "target_lang", None),
        "mode": mode,
    }
    return {k: v for k, v in pairs.items() if v}


def _print_devices() -> int:
    import sounddevice as sd

    print("Аудиоустройства (🔁 — виртуальные/loopback, годятся для захвата системного звука):\n")
    for index, dev in enumerate(sd.query_devices()):
        name = dev["name"]
        marker = "🔁" if any(m in name.lower() for m in _VIRTUAL_MARKERS) else "  "
        io = []
        if dev["max_input_channels"]:
            io.append(f"вход×{dev['max_input_channels']}")
        if dev["max_output_channels"]:
            io.append(f"выход×{dev['max_output_channels']}")
        print(f" {marker} [{index:2d}] {name}  ({', '.join(io)}, {dev['default_samplerate']:.0f} Гц)")
    print(
        "\nПодсказка: для системного звука на macOS установите BlackHole "
        "(brew install blackhole-2ch), на Windows используйте backend wasapi-loopback, "
        "на Linux выберите monitor-источник PipeWire/PulseAudio."
    )
    return 0


def _print_profiles() -> int:
    print("Готовые профили UVT:\n")
    for name, description in _PROFILE_OVERVIEW:
        print(f"  {name:14} {description}")
    print(
        "\n`local` не ходит в сеть во время работы после установки моделей. "
        "`free` бесплатен по цене, но Edge TTS отправляет текст в Microsoft.\n"
        "Выберите: uvt run -p <имя>, uvt dub <файл> -p <имя> или uvt gui -p <имя>."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    _strip_malloc_env()
    _load_env_file()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in (None,):
        parser.print_help()
        return 0
    if args.command == "version":
        print(__version__)
        return 0

    setup_logging(getattr(args, "debug", False))

    if args.command == "devices":
        return _print_devices()
    if args.command == "profiles":
        return _print_profiles()

    if getattr(args, "mode", None) in _LEGACY_LIVE_MODES:
        print(
            f"Предупреждение: --mode {args.mode} устарел и работает как voiceover; "
            "реальной замены/разделения оригинала в Live пока нет.",
            file=sys.stderr,
        )

    cfg = load_config(getattr(args, "profile", None), overrides=_overrides(args))

    if args.command == "run":
        from uvt.app import run_headless

        try:
            asyncio.run(run_headless(cfg))
        except KeyboardInterrupt:
            pass
        return 0

    if args.command == "dub":
        from uvt.dub import dub

        try:
            asyncio.run(
                dub(
                    cfg,
                    args.input,
                    output=args.output,
                    duck_db=args.duck_db,
                    keep_original=not args.no_original_track,
                )
            )
        except KeyboardInterrupt:
            print("Прервано.", file=sys.stderr)
            return 130
        except Exception as exc:  # noqa: BLE001 — пользователю нужна одна строка
            if args.debug:
                raise
            print(f"Ошибка: {exc}", file=sys.stderr)
            print("Подробности: повторите команду с флагом --debug", file=sys.stderr)
            return 1
        return 0

    if args.command == "serve":
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            print('серверу нужен aiohttp — установите: pip install "uvt[server]"', file=sys.stderr)
            return 1
        from uvt.server import run_server

        try:
            asyncio.run(run_server(cfg, host=args.host, port=args.port))
        except KeyboardInterrupt:
            pass
        return 0

    if args.command == "serve-personal":
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            print('серверу нужен aiohttp — установите: pip install "uvt[server]"', file=sys.stderr)
            return 1
        from uvt.server import run_personal_servers

        try:
            asyncio.run(
                run_personal_servers(
                    host=args.host,
                    free_port=args.free_port,
                    gpt_port=args.gpt_port,
                    eleven_port=args.eleven_port,
                )
            )
        except KeyboardInterrupt:
            pass
        return 0

    if args.command == "gui":
        try:
            from uvt.gui.main_window import run_gui
        except ImportError:
            print('GUI требует PySide6 — установите: pip install "uvt[gui]"', file=sys.stderr)
            return 1
        return run_gui(cfg, profile_name=getattr(args, "profile", None))

    parser.print_help()
    return 1
