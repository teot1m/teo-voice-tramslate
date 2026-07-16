"""Настройка логирования: rich-консоль, если доступна."""
from __future__ import annotations

import logging


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=False, show_path=False, markup=False
        )
        fmt, datefmt = "%(name)s: %(message)s", "[%X]"
    except ImportError:
        handler = logging.StreamHandler()
        fmt, datefmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=[handler], force=True)
    for noisy in ("httpx", "httpcore", "urllib3", "numba", "faster_whisper", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
