"""Сервис VAD: режет непрерывный звук на реплики.

Вся логика нарезки — в uvt.segmenter (общая с офлайн-дубляжом); сервис лишь
подключает её к шине. Отметки времени сегментов — секунды от начала потока.
"""
from __future__ import annotations

from dataclasses import dataclass

from uvt.config import PRESETS
from uvt.events import TOPIC_AUDIO, TOPIC_SPEECH, AudioChunk
from uvt.segmenter import FRAME_MS, Segmenter, SegmenterParams, create_vad_engine
from uvt.services.base import Service


@dataclass(slots=True)
class _ClockSpan:
    """Соответствие sample-position Segmenter → source audio clock."""

    first_sample: int
    last_sample: int
    start_ts: float
    end_ts: float


class _AudioClock:
    """Сохраняет timestamp каждого входного диапазона без изменения Segmenter.

    Segmenter исторически возвращает секунды от старта потока. В live-режиме
    этого недостаточно: выход должен знать, когда именно прозвучала реплика.
    Этот тонкий адаптер переводит его sample offsets обратно в source clock.
    """

    def __init__(self) -> None:
        self._spans: list[_ClockSpan] = []
        self._next_sample = 0

    def append(self, chunk: AudioChunk) -> None:
        size = len(chunk.samples)
        if size <= 0:
            return
        first = self._next_sample
        last = first + size
        self._spans.append(
            _ClockSpan(first, last, chunk.source_start_ts, chunk.source_end_ts)
        )
        self._next_sample = last

    def timestamp(self, sample: int, fallback: float) -> float:
        if not self._spans:
            return fallback
        # Небольшие чанки, но много их за сессию: начинаем с конца, потому что
        # VAD всегда выдаёт сегменты в хронологическом порядке около хвоста.
        for span in reversed(self._spans):
            if sample >= span.first_sample:
                if span.last_sample <= span.first_sample:
                    return span.end_ts
                pos = min(max(sample, span.first_sample), span.last_sample)
                ratio = (pos - span.first_sample) / (span.last_sample - span.first_sample)
                return span.start_ts + (span.end_ts - span.start_ts) * ratio
        return self._spans[0].start_ts

    def discard_before(self, sample: int) -> None:
        """Освободить spans, которые Segmenter больше не сможет процитировать.

        ``sample`` — начало самого раннего VAD-контекста, который ещё может
        попасть в будущую реплику. Span хранит полуоткрытый диапазон
        ``[first_sample, last_sample)``; span, содержащий эту границу, остаётся
        для точного timestamp неполного входного буфера.
        """
        first_live = 0
        for span in self._spans:
            if span.last_sample > sample:
                break
            first_live += 1
        if first_live:
            del self._spans[:first_live]


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
        self._audio_clock = _AudioClock()
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
        self._audio_clock.append(chunk)
        segments = self.segmenter.feed(chunk.samples)
        for segment in segments:
            # Segmenter использует offsets от начала. Конвертируем их после
            # нарезки, не теряя точность clock source при ресемплинге/паузах.
            start_offset = segment.start_ts
            end_offset = segment.end_ts
            start_sample = round(start_offset * segment.sample_rate)
            end_sample = round(end_offset * segment.sample_rate)
            segment.start_ts = self._audio_clock.timestamp(start_sample, start_offset)
            segment.end_ts = self._audio_clock.timestamp(end_sample, end_offset)

            trace = segment.trace
            trace.source_start_ts = segment.start_ts
            trace.source_end_ts = segment.end_ts
            target_delay_s = max(0.0, float(getattr(self.cfg.output, "target_delay_s", 0.0)))
            max_backlog_s = max(0.0, float(self.cfg.output.max_backlog_s))
            trace.target_ts = segment.end_ts + target_delay_s
            trace.deadline_ts = trace.target_ts + max_backlog_s
            self.log.debug(
                "сегмент речи %.2f с (source %.3f–%.3f, target %.3f)",
                segment.duration_s,
                segment.start_ts,
                segment.end_ts,
                trace.target_ts,
            )
        # При долгой тишине Segmenter не выдаёт сегменты, поэтому прежняя
        # очистка только после финализации удерживала все AudioChunk с начала
        # сессии. Оставляем лишь pre-roll/активную реплику и текущий хвост VAD.
        self._audio_clock.discard_before(self.segmenter.clock_retention_start)
        return segments or None
