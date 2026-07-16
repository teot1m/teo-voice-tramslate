"""Сервис VAD: режет непрерывный звук на реплики.

Вся логика нарезки — в uvt.segmenter (общая с офлайн-дубляжом); сервис лишь
подключает её к шине. Отметки времени сегментов — секунды от начала потока.
"""
from __future__ import annotations

from uvt.config import PRESETS
from uvt.events import TOPIC_AUDIO, TOPIC_SPEECH, AudioChunk
from uvt.segmenter import FRAME_MS, Segmenter, SegmenterParams, create_vad_engine
from uvt.services.base import Service


class VADService(Service):
    name = "vad"
    consumes = TOPIC_AUDIO
    produces = TOPIC_SPEECH

    async def setup(self) -> None:
        vcfg = self.cfg.vad
        preset = PRESETS[self.cfg.latency.preset]
        params = SegmenterParams.from_config(vcfg, preset)
        self.engine = await create_vad_engine(vcfg, self.log)
        self.segmenter = Segmenter(self.engine, params)
        self.log.info(
            "VAD '%s': отсечка по паузе ~%d мс, сегмент до ~%.0f с (пресет '%s')",
            type(self.engine).__name__,
            params.min_silence_frames * FRAME_MS,
            params.max_frames * FRAME_MS / 1000,
            self.cfg.latency.preset,
        )

    async def teardown(self) -> None:
        if hasattr(self, "engine"):
            await self.engine.close()

    async def handle(self, chunk: AudioChunk):
        segments = self.segmenter.feed(chunk.samples)
        for segment in segments:
            self.log.debug("сегмент речи %.2f с (позиция %.1f с)", segment.duration_s, segment.start_ts)
        return segments or None
