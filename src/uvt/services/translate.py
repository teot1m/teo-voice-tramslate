"""Сервис перевода: держит контекст последних реплик, деградирует без падений.

Если перевод не удался или язык оригинала совпадает с целевым — дальше идёт
оригинальный текст (failed=True блокирует озвучку, но субтитры остаются).
"""
from __future__ import annotations

from collections import deque

from uvt import registry
from uvt.events import TOPIC_TRANSCRIPT, TOPIC_TRANSLATION, Transcript, Translation
from uvt.services.base import Service


def _same_lang(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return a.split("-")[0].lower() == b.split("-")[0].lower()


class TranslationService(Service):
    name = "translate"
    consumes = TOPIC_TRANSCRIPT
    produces = TOPIC_TRANSLATION

    async def setup(self) -> None:
        tcfg = self.cfg.translation
        if tcfg.engine in ("none", "passthrough"):
            self.engine = None
            self.log.info("перевод отключён (engine=none) — сквозной режим")
        else:
            self.engine = registry.create("translation", tcfg.engine, tcfg)
            await self.engine.warmup()
        pairs = max(0, tcfg.context_pairs)
        self._context_enabled = pairs > 0
        self._context: deque[tuple[str, str]] = deque(maxlen=pairs or 1)

    async def teardown(self) -> None:
        if getattr(self, "engine", None) is not None:
            await self.engine.close()

    async def handle(self, transcript: Transcript):
        target = self.cfg.target_lang
        failed = False
        if self.engine is None or _same_lang(transcript.language, target):
            text = transcript.text
        else:
            try:
                context = list(self._context) if self._context_enabled else []
                text = await self.engine.translate(
                    transcript.text, transcript.language, target, context
                )
                if self._context_enabled:
                    self._context.append((transcript.text, text))
            except Exception as exc:  # noqa: BLE001 — конвейер должен жить дальше
                self.log.error("перевод не удался: %s", exc)
                self.set_status("error", str(exc))
                text = transcript.text
                failed = True
        transcript.trace.mark("translate")
        return Translation(
            transcript=transcript,
            text=(text or transcript.text).strip(),
            target_lang=target,
            failed=failed,
            trace=transcript.trace,
        )
