"""Сервис синтеза речи: переводит текст в звук единой частоты вывода."""
from __future__ import annotations

from uvt import registry
from uvt.audio import resample
from uvt.events import TOPIC_TRANSLATION, TOPIC_TTS, Translation, TtsAudio
from uvt.services.base import Service


class TTSService(Service):
    name = "tts"
    consumes = TOPIC_TRANSLATION
    produces = TOPIC_TTS

    async def setup(self) -> None:
        tcfg = self.cfg.tts
        if tcfg.engine == "none":
            self.engine = None
            self.log.info("озвучка отключена (engine=none)")
        else:
            self.engine = registry.create("tts", tcfg.engine, tcfg)
            await self.engine.warmup()
        self._out_rate = self.cfg.output.sample_rate

    async def teardown(self) -> None:
        if getattr(self, "engine", None) is not None:
            await self.engine.close()

    async def handle(self, translation: Translation):
        if self.engine is None or translation.failed or not translation.text:
            return None
        samples, rate = await self.engine.synthesize(translation.text, translation.target_lang)
        if len(samples) == 0:
            return None
        if rate != self._out_rate:
            samples = resample(samples, rate, self._out_rate)
        translation.trace.mark("tts")
        return TtsAudio(
            translation=translation,
            samples=samples,
            sample_rate=self._out_rate,
            trace=translation.trace,
        )
