"""Базовый сервис конвейера.

Сервис-источник (consumes=None) реализует run_source(),
сервис-обработчик — handle(item); ошибки одного события не роняют конвейер.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC
from dataclasses import dataclass, field

from uvt.bus import Bus
from uvt.config import AppConfig
from uvt.events import TOPIC_STATUS, ServiceStatus
from uvt.metrics import Metrics


async def _finish_lifecycle(task: asyncio.Future):
    """Drain setup/teardown before propagating cancellation of its waiter.

    Cancelling a native model loader's to_thread await does not stop that
    loader. Teardown must not race it or allow it to resurrect a model.
    """
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except Exception:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
    if cancelled:
        raise asyncio.CancelledError
    return result


@dataclass
class _Readiness:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    initialized: bool = False
    error: BaseException | None = None


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
        self._readiness = _Readiness()

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

    async def handle_with_inbox(self, item, inbox: asyncio.Queue):
        """Вариант handle для сервисов, которым нужен доступ к свежей очереди.

        Обычно достаточно handle(). OutputService переопределяет этот хук,
        чтобы новую реплику можно было предпочесть во время ожидания target.
        """
        return await self.handle(item)

    # --- инфраструктура ---

    def publish(self, item) -> None:
        if self.produces is not None:
            self.bus.topic(self.produces).publish(item)

    def set_status(self, state: str, detail: str = "") -> None:
        self.bus.topic(TOPIC_STATUS).publish(ServiceStatus(self.name, state, detail))

    # --- realtime admission control ---

    @staticmethod
    def _trace_for(item):
        trace = getattr(item, "trace", None)
        if trace is not None:
            return trace
        # Сохраняем совместимость с внешними plugin events, которые держат
        # trace только в исходном сегменте.
        for attr in ("segment", "transcript", "translation"):
            parent = getattr(item, attr, None)
            trace = getattr(parent, "trace", None)
            if trace is not None:
                return trace
        return None

    def enforces_deadline(self) -> bool:
        """Только тяжёлые live-стадии должны терять просроченную работу.

        Overlay/History продолжают получать готовый текст: пропуск аудио не
        должен скрывать перевод или ломать историю сессии.
        """
        return self.name in {"stt", "translate", "tts", "output"}

    def latest_wins_enabled(self) -> bool:
        return bool(getattr(self.cfg.output, "latest_wins", True)) and self.enforces_deadline()

    def is_expired(self, item) -> bool:
        trace = self._trace_for(item)
        return bool(trace is not None and trace.is_expired())

    def drop_item(self, item, reason: str) -> None:
        """Отметить намеренный drop, не превращая его в service error."""
        trace = self._trace_for(item)
        if trace is not None:
            trace.mark_dropped(self.name, reason)
        self.metrics.record_drop(self.name, reason)
        self.log.debug("сегмент пропущен (%s)", reason)

    def _take_latest(self, inbox: asyncio.Queue, item):
        """Слить уже накопившуюся очередь, сохранив newest event."""
        if not self.latest_wins_enabled():
            return item
        latest = item
        while True:
            try:
                newer = inbox.get_nowait()
            except asyncio.QueueEmpty:
                return latest
            self.drop_item(latest, "superseded")
            latest = newer

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            raise RuntimeError(f"Сервис {self.name} уже запущен")
        state = self._readiness = _Readiness()
        self._task = asyncio.create_task(self._run(), name=f"uvt:{self.name}")

        def wake_waiters(task: asyncio.Task) -> None:
            # A task cancelled before its first scheduled turn never enters
            # _run's finally block. A per-start state also isolates restarts.
            if not state.event.is_set():
                state.error = (
                    asyncio.CancelledError() if task.cancelled() else task.exception()
                )
                state.event.set()

        self._task.add_done_callback(wake_waiters)

    async def wait_ready(self) -> None:
        """Wait for this start's setup; start() itself remains nonblocking."""
        state = self._readiness
        if self._task is None:
            raise RuntimeError(f"Сервис {self.name} не запущен")
        await state.event.wait()
        if isinstance(state.error, asyncio.CancelledError):
            raise asyncio.CancelledError(f"Запуск сервиса {self.name} остановлен")
        if state.error is not None:
            raise RuntimeError(
                f"Не удалось подготовить сервис {self.name}: {state.error}"
            ) from state.error
        if not state.initialized:
            raise RuntimeError(f"Сервис {self.name} завершился до готовности")

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await _finish_lifecycle(asyncio.gather(task, return_exceptions=True))
        finally:
            if task.done() and self._task is task:
                self._task = None

    async def _run(self) -> None:
        # Подписка до setup(): пока движок греется, события копятся в очереди,
        # а не теряются.
        inbox = self.bus.topic(self.consumes).subscribe() if self.consumes else None
        try:
            try:
                await _finish_lifecycle(asyncio.create_task(self.setup()))
            except asyncio.CancelledError as exc:
                self._readiness.error = exc
                raise
            except Exception as exc:  # noqa: BLE001
                self._readiness.error = exc
                self.log.exception("сбой инициализации")
                self.set_status("error", str(exc))
                return
            self._readiness.initialized = True
            self._readiness.event.set()
            self.set_status("running")

            if inbox is None:
                await self.run_source()
                self.set_status("done")
                return

            while True:
                item = await inbox.get()
                item = self._take_latest(inbox, item)
                if self.enforces_deadline() and self.is_expired(item):
                    self.drop_item(item, "deadline")
                    continue
                try:
                    out = await self.handle_with_inbox(item, inbox)
                except Exception as exc:  # noqa: BLE001
                    self.log.exception("ошибка обработки события")
                    self.set_status("error", str(exc))
                    continue
                if out is None:
                    continue
                for event in out if isinstance(out, list) else [out]:
                    # STT и TTS имеют одного downstream-потребителя, поэтому
                    # можно не публиковать работу, просроченную во время модели.
                    # Translation сохраняем для субтитров/истории даже поздней.
                    if self.name in {"stt", "tts"} and self.is_expired(event):
                        self.drop_item(event, "deadline_after_processing")
                        continue
                    self.publish(event)
        finally:
            self._readiness.event.set()
            if inbox is not None:
                # A stopped/restarted live service must not retain queued audio
                # or keep receiving events while its engine is shutting down.
                self.bus.topic(self.consumes).unsubscribe(inbox)
            with contextlib.suppress(Exception):
                await _finish_lifecycle(asyncio.create_task(self.teardown()))
            with contextlib.suppress(Exception):
                self.set_status("stopped")
