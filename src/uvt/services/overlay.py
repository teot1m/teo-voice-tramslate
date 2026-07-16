"""Сервис оверлея: превращает переводы в события субтитров.

Всегда пишет пары «оригинал ⇒ перевод» в лог (консольный режим), а если задан
sink (GUI-оверлей или плагин) — отдаёт события и туда.
"""
from __future__ import annotations

from typing import Callable

from uvt.bus import Bus
from uvt.config import AppConfig
from uvt.events import TOPIC_TRANSLATION, SubtitleEvent, Translation
from uvt.metrics import Metrics
from uvt.services.base import Service


class OverlayService(Service):
    name = "overlay"
    consumes = TOPIC_TRANSLATION
    produces = None

    def __init__(
        self,
        bus: Bus,
        cfg: AppConfig,
        metrics: Metrics,
        sink: Callable[[SubtitleEvent], None] | None = None,
    ) -> None:
        super().__init__(bus, cfg, metrics)
        self.sink = sink

    async def handle(self, translation: Translation):
        segment = translation.transcript.segment
        event = SubtitleEvent(
            original=translation.transcript.text,
            translated=translation.text,
            language=translation.transcript.language,
            target_lang=translation.target_lang,
            start_ts=segment.start_ts,
            end_ts=segment.end_ts,
        )
        prefix = "⚠ " if translation.failed else ""
        self.log.info(
            "%s[%s→%s] %s ⇒ %s",
            prefix, event.language, event.target_lang, event.original, event.translated,
        )
        # В режиме «только субтитры» конец конвейера здесь — фиксируем метрики
        if self.cfg.mode == "subtitles":
            self.metrics.observe(translation.trace)
        if self.sink is not None:
            try:
                self.sink(event)
            except Exception:  # noqa: BLE001
                self.log.exception("сбой отрисовки оверлея")
        return None
