"""Сервис вывода: играет озвучку в выбранное устройство.

Устройством может быть виртуальный кабель (VB-Cable, BlackHole) — так перевод
попадает в OBS/Discord/Zoom (ТЗ §10). Если конвейер отстаёт сильнее
max_backlog_s, сегмент пропускается: перевод в реальном времени важнее полноты.
"""
from __future__ import annotations

import asyncio

import numpy as np

from uvt.events import TOPIC_TTS, TtsAudio, now
from uvt.services.base import Service


class OutputService(Service):
    name = "output"
    consumes = TOPIC_TTS
    produces = None

    async def setup(self) -> None:
        ocfg = self.cfg.output
        self._stream = None
        self._closing = False
        if ocfg.backend == "null":
            self.log.info("аудиовыход: null (без воспроизведения)")
            return
        try:
            import sounddevice as sd

            self._stream = sd.OutputStream(
                device=ocfg.device,
                samplerate=ocfg.sample_rate,
                channels=1,
                dtype="float32",
            )
            self._stream.start()
            if ocfg.device is not None:
                device_name = sd.query_devices(ocfg.device, "output")["name"]
            else:
                device_name = sd.query_devices(kind="output")["name"]
            self.log.info("аудиовыход: %s @ %d Гц", device_name, ocfg.sample_rate)
        except Exception as exc:  # noqa: BLE001 — без звука, но с субтитрами
            self.log.error("аудиовыход недоступен (%s) — работаю без воспроизведения", exc)
            self._stream = None

    async def teardown(self) -> None:
        self._closing = True
        if self._stream is not None:
            stream, self._stream = self._stream, None
            await asyncio.to_thread(self._close_stream, stream)

    @staticmethod
    def _close_stream(stream) -> None:
        stream.stop()
        stream.close()

    async def handle(self, audio: TtsAudio):
        lag = now() - audio.trace.marks.get("speech_end", now())
        if lag > self.cfg.output.max_backlog_s:
            self.log.warning("отстаём на %.1f с — сегмент пропущен", lag)
            return None
        audio.trace.mark("play")
        self.metrics.observe(audio.trace)
        spans = " | ".join(f"{name} {ms:.0f} мс" for name, ms in audio.trace.spans_ms())
        self.log.debug("задержки сегмента: %s", spans)
        if self._stream is not None:
            await asyncio.to_thread(self._write, audio.samples)
        return None

    def _write(self, samples: np.ndarray) -> None:
        stream = self._stream
        if stream is None:
            return
        # Пишем кусками по 100 мс, чтобы остановка была отзывчивой
        step = max(1, self.cfg.output.sample_rate // 10)
        try:
            for i in range(0, len(samples), step):
                if self._closing:
                    break
                stream.write(np.ascontiguousarray(samples[i : i + step]))
        except Exception as exc:  # noqa: BLE001 — устройство могло исчезнуть
            if not self._closing:
                self.log.error("сбой воспроизведения: %s", exc)
