"""Сервис захвата: движок отдаёт звук с нативной частотой, сервис приводит
его к 16 кГц моно — общему формату VAD и STT."""
from __future__ import annotations

import numpy as np

from uvt import registry
from uvt.events import TOPIC_AUDIO, AudioChunk, now
from uvt.services.base import Service

TARGET_RATE = 16000


class CaptureService(Service):
    name = "capture"
    consumes = None
    produces = TOPIC_AUDIO

    async def setup(self) -> None:
        self.engine = registry.create("capture", self.cfg.capture.backend, self.cfg.capture)
        await self.engine.warmup()

    async def teardown(self) -> None:
        await self.engine.close()

    async def run_source(self) -> None:
        resampler = None
        resampler_rate: int | None = None
        # Движки захвата отдают только сэмплы, без hardware timestamp. Поэтому
        # строим непрерывные часы источника по числу захваченных сэмплов и
        # подхватываем реальный monotonic clock при паузе устройства. Эти часы
        # затем проходят через VAD до target/deadline OutputService.
        clock_end_ts: float | None = None
        pending_start_ts: float | None = None
        pending_end_ts: float | None = None
        async for samples, rate in self.engine.stream():
            raw = np.asarray(samples, dtype=np.float32)
            if len(raw) == 0 or rate <= 0:
                continue
            raw_duration_s = len(raw) / rate
            observed_end_ts = now()
            if clock_end_ts is None:
                source_end_ts = observed_end_ts
            else:
                # При нормальном realtime-захвате это previous + duration; при
                # паузе устройства сохраняем реальный пропуск на source clock.
                source_end_ts = max(clock_end_ts + raw_duration_s, observed_end_ts)
            source_start_ts = source_end_ts - raw_duration_s
            clock_end_ts = source_end_ts

            if pending_start_ts is None:
                pending_start_ts = source_start_ts
            pending_end_ts = source_end_ts

            if rate != TARGET_RATE:
                if resampler is None or resampler_rate != rate:
                    import soxr

                    resampler = soxr.ResampleStream(rate, TARGET_RATE, 1, dtype="float32")
                    resampler_rate = rate
                raw = resampler.resample_chunk(raw)
            if len(raw) == 0:
                continue
            self.publish(
                AudioChunk(
                    samples=np.asarray(raw, dtype=np.float32),
                    sample_rate=TARGET_RATE,
                    ts=pending_start_ts,
                    end_ts=pending_end_ts,
                )
            )
            pending_start_ts = None
            pending_end_ts = None
        self.log.info("источник звука завершил поток")
