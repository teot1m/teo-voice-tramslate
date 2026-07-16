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
        async for samples, rate in self.engine.stream():
            if rate != TARGET_RATE:
                if resampler is None or resampler_rate != rate:
                    import soxr

                    resampler = soxr.ResampleStream(rate, TARGET_RATE, 1, dtype="float32")
                    resampler_rate = rate
                samples = resampler.resample_chunk(samples)
            if len(samples) == 0:
                continue
            self.publish(
                AudioChunk(
                    samples=np.asarray(samples, dtype=np.float32),
                    sample_rate=TARGET_RATE,
                    ts=now(),
                )
            )
        self.log.info("источник звука завершил поток")
