"""Сервис распознавания речи + фильтр типовых галлюцинаций Whisper на тишине."""
from __future__ import annotations

from uvt.events import TOPIC_SPEECH, TOPIC_TRANSCRIPT, SpeechSegment, Transcript
from uvt.fallback import create_stt_engine
from uvt.services.base import Service

# Типовые фразы, которые Whisper выдаёт на тишине/музыке
_JUNK = {
    "", ".", "..", "...",
    "thank you.", "thanks for watching!", "thank you for watching!",
    "share this video with your friends on social media",
    "спасибо за просмотр!", "продолжение следует...",
    "субтитры делал dimatorzok", "субтитры сделал dimatorzok",
    "редактор субтитров а.семкин корректор а.егорова",
    "дякую за перегляд!", "つづく", "ご視聴ありがとうございました",
}


def _is_junk(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in _JUNK or len(normalized) < 2


class STTService(Service):
    name = "stt"
    consumes = TOPIC_SPEECH
    produces = TOPIC_TRANSCRIPT

    async def setup(self) -> None:
        self.engine = create_stt_engine(self.cfg)
        await self.engine.warmup()
        lang = self.cfg.source_lang
        self._language = None if lang in (None, "", "auto") else lang

    async def teardown(self) -> None:
        if hasattr(self, "engine"):
            await self.engine.close()

    async def handle(self, segment: SpeechSegment):
        result = await self.engine.transcribe(segment.samples, segment.sample_rate, self._language)
        if result is None:
            return None
        text = " ".join(result.text.split())
        if _is_junk(text):
            self.log.debug("отброшено как мусор STT: %r", text)
            return None
        segment.trace.mark("stt")
        return Transcript(
            segment=segment,
            text=text,
            language=(result.language or self._language or "und"),
            confidence=result.confidence,
            trace=segment.trace,
        )
