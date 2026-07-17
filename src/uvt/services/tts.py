"""Сервис синтеза речи: переводит текст в звук единой частоты вывода."""
from __future__ import annotations

from contextlib import contextmanager, suppress

from uvt import registry
from uvt.audio import resample
from uvt.events import TOPIC_TRANSLATION, TOPIC_TTS, Translation, TtsAudio
from uvt.services.base import Service
from uvt.services.speaker import SpeakerAssignment, SpeakerService


class TTSService(Service):
    name = "tts"
    consumes = TOPIC_TRANSLATION
    produces = TOPIC_TTS

    async def setup(self) -> None:
        tcfg = self.cfg.tts
        self.speakers = SpeakerService(self.cfg.speaker)
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
        segment = translation.transcript.segment
        assignment = self.speakers.assign(segment.samples, segment.sample_rate)
        segment.speaker_id = assignment.speaker_id
        segment.speaker_timbre = assignment.timbre
        segment.speaker_confidence = assignment.confidence
        if assignment.speaker_id is not None:
            self.log.debug(
                "speaker %s: timbre=%s confidence=%.2f f0=%s",
                assignment.speaker_id,
                assignment.timbre,
                assignment.confidence,
                f"{assignment.f0_hz:.0f}Hz" if assignment.f0_hz is not None else "n/a",
            )
        # Existing engines read voice/voice_gender from their config rather than
        # from TTSEngine.synthesize(). Keep that API compatible, but scope the
        # per-replica override to this serial TTS call.
        with self._speaker_voice(assignment):
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

    @contextmanager
    def _speaker_voice(self, assignment: SpeakerAssignment):
        """Временно выбрать target voice для auto-режима одной реплики."""
        engine_cfg = self.engine.cfg
        configured_voice = str(getattr(engine_cfg, "voice", "auto") or "auto")
        configured_role = str(getattr(engine_cfg, "voice_gender", "auto") or "auto").lower()
        # Явный voice или manual voice_gender всегда важнее auto speaker map.
        if configured_voice != "auto" or configured_role in {"male", "female"}:
            yield
            return

        had_role = hasattr(engine_cfg, "voice_gender")
        original_voice = getattr(engine_cfg, "voice", None)
        original_role = getattr(engine_cfg, "voice_gender", None)
        try:
            if assignment.voice:
                engine_cfg.voice = assignment.voice
            elif assignment.voice_gender:
                engine_cfg.voice_gender = assignment.voice_gender
            yield
        finally:
            engine_cfg.voice = original_voice
            if had_role:
                engine_cfg.voice_gender = original_role
            else:
                # Extra-enabled plugin configs may not have declared this field.
                with suppress(AttributeError):
                    delattr(engine_cfg, "voice_gender")
