"""Базовый сервис конвейера.

Сервис-источник (consumes=None) реализует run_source(),
сервис-обработчик — handle(item); ошибки одного события не роняют конвейер.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC

from uvt.bus import Bus
from uvt.config import AppConfig
from uvt.events import TOPIC_STATUS, ServiceStatus
from uvt.metrics import Metrics


class Service(ABC):
    name: str = "service"
    consumes: str | None = None
    produces: str | None = None

    def __init__(self, bus: Bus, cfg: AppConfig, metrics: Metrics) -> None:
        self.bus = bus
        self.cfg = cfg
        self.metrics = metrics
        self.log = logging.getLogger(f"uvt.{self.name}")
        self._task: asyncio.Task | None = None

    # --- переопределяемые точки ---

    async def setup(self) -> None:
        """Создание движков, прогрев моделей."""

    async def teardown(self) -> None:
        """Освобождение ресурсов; вызывается всегда, в том числе при отмене."""

    async def handle(self, item):
        """Обработка события; вернуть событие, список событий или None."""
        raise NotImplementedError

    async def run_source(self) -> None:
        """Цикл сервиса-источника (когда consumes=None)."""
        raise NotImplementedError

    # --- инфраструктура ---

    def publish(self, item) -> None:
        if self.produces is not None:
            self.bus.topic(self.produces).publish(item)

    def set_status(self, state: str, detail: str = "") -> None:
        self.bus.topic(TOPIC_STATUS).publish(ServiceStatus(self.name, state, detail))

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"uvt:{self.name}")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _run(self) -> None:
        # Подписка до setup(): пока движок греется, события копятся в очереди,
        # а не теряются.
        inbox = self.bus.topic(self.consumes).subscribe() if self.consumes else None
        try:
            try:
                await self.setup()
            except Exception as exc:  # noqa: BLE001
                self.log.exception("сбой инициализации")
                self.set_status("error", str(exc))
                return
            self.set_status("running")

            if inbox is None:
                await self.run_source()
                self.set_status("done")
                return

            while True:
                item = await inbox.get()
                try:
                    out = await self.handle(item)
                except Exception as exc:  # noqa: BLE001
                    self.log.exception("ошибка обработки события")
                    self.set_status("error", str(exc))
                    continue
                if out is None:
                    continue
                for event in out if isinstance(out, list) else [out]:
                    self.publish(event)
        finally:
            with contextlib.suppress(Exception):
                await self.teardown()
            with contextlib.suppress(Exception):
                self.set_status("stopped")
