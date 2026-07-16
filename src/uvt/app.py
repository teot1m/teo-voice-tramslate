"""Сборка и запуск конвейера из конфигурации."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from typing import Callable

from uvt import registry
from uvt.bus import Bus
from uvt.config import AppConfig
from uvt.events import SubtitleEvent
from uvt.history import HistoryService
from uvt.metrics import Metrics
from uvt.services.base import Service
from uvt.services.capture import CaptureService
from uvt.services.output import OutputService
from uvt.services.overlay import OverlayService
from uvt.services.stt import STTService
from uvt.services.translate import TranslationService
from uvt.services.tts import TTSService
from uvt.services.vad import VADService

log = logging.getLogger("uvt.app")


class Pipeline:
    """Capture → VAD → STT → Translation → [TTS → Output] + Overlay + History."""

    def __init__(
        self,
        cfg: AppConfig,
        subtitle_sink: Callable[[SubtitleEvent], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.bus = Bus()
        self.metrics = Metrics()

        registry.load_builtins()
        registry.load_plugin_dirs(cfg.plugin_dirs)

        if cfg.mode in ("replace", "dual"):
            log.warning(
                "режим '%s': приглушение/панорамирование оригинала появится в v1.5 — "
                "пока работаю как voiceover", cfg.mode,
            )

        services: list[Service] = [
            CaptureService(self.bus, cfg, self.metrics),
            VADService(self.bus, cfg, self.metrics),
            STTService(self.bus, cfg, self.metrics),
            TranslationService(self.bus, cfg, self.metrics),
        ]
        if cfg.mode != "subtitles":
            services.append(TTSService(self.bus, cfg, self.metrics))
            services.append(OutputService(self.bus, cfg, self.metrics))
        services.append(OverlayService(self.bus, cfg, self.metrics, sink=subtitle_sink))
        services.append(HistoryService(self.bus, cfg, self.metrics))
        self.services = services

    def get_service(self, name: str) -> Service:
        for service in self.services:
            if service.name == name:
                return service
        raise KeyError(name)

    async def start(self) -> None:
        # Задачи создаются с конца конвейера: потребители подписываются на
        # топики раньше, чем источник начнёт публиковать.
        for service in reversed(self.services):
            await service.start()

    async def stop(self) -> None:
        # Останавливаем с начала: сперва замолкает источник.
        for service in self.services:
            await service.stop()


async def run_headless(cfg: AppConfig) -> None:
    """Консольный режим: работает до Ctrl+C / SIGTERM."""
    pipeline = Pipeline(cfg)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop_event.set)

    await pipeline.start()
    log.info(
        "конвейер запущен: %s → %s, режим '%s', пресет задержки '%s'; Ctrl+C — остановка",
        cfg.source_lang, cfg.target_lang, cfg.mode, cfg.latency.preset,
    )

    async def metrics_ticker() -> None:
        while True:
            await asyncio.sleep(10)
            line = pipeline.metrics.format_line()
            if line:
                log.info("задержки (медианы): %s", line)

    ticker = asyncio.create_task(metrics_ticker())
    stop_task = asyncio.create_task(stop_event.wait())
    waiting = {stop_task}
    capture_task = pipeline.get_service("capture").task
    if capture_task is not None:
        waiting.add(capture_task)
    try:
        # Работаем до Ctrl+C; если источник конечен (демо, файл) — до его конца.
        done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        if stop_task not in done:
            log.info("источник звука завершился — доигрываю хвост конвейера")
            await asyncio.sleep(2.0)
    finally:
        stop_task.cancel()
        ticker.cancel()
        await asyncio.gather(stop_task, ticker, return_exceptions=True)
        await pipeline.stop()
        log.info("конвейер остановлен")
