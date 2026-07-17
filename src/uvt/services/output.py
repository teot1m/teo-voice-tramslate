"""Сервис вывода: играет озвучку в выбранное устройство.

Устройством может быть виртуальный кабель (VB-Cable, BlackHole) — так перевод
попадает в OBS/Discord/Zoom (ТЗ §10). Реплика планируется на target timestamp
исходного аудио; просроченная или вытесненная свежей работа отбрасывается.
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

    async def handle_with_inbox(self, audio: TtsAudio, inbox: asyncio.Queue):
        """Ждать target, но уступить его более свежей готовой реплике.

        Без этого простое ``sleep(target-now)`` блокирует output на старом
        сегменте, хотя новый уже успел пройти STT/MT/TTS. Саму начавшуюся запись
        прерывать нельзя (иначе артефакты), но до write действует latest-wins.
        """
        while True:
            audio = self._take_latest(inbox, audio)
            if self.is_expired(audio):
                self.drop_item(audio, "deadline")
                return None

            target_ts = audio.trace.target_ts
            if (
                not self.latest_wins_enabled()
                or target_ts is None
                or getattr(self, "_stream", None) is None
            ):
                return await self.handle(audio)
            wait_s = target_ts - now()
            if wait_s <= 0:
                return await self.handle(audio)

            sleep_task = asyncio.create_task(asyncio.sleep(wait_s))
            next_task = asyncio.create_task(inbox.get())
            try:
                done, _ = await asyncio.wait(
                    {sleep_task, next_task}, return_when=asyncio.FIRST_COMPLETED
                )
            except BaseException:
                # stop() отменяет service во время target wait. Не оставляем
                # orphan get(), иначе он может съесть событие уже после stop.
                sleep_task.cancel()
                next_task.cancel()
                await asyncio.gather(sleep_task, next_task, return_exceptions=True)
                raise
            if next_task in done:
                # За время ожидания появился более свежий перевод. Убираем
                # старый до записи в устройство и заодно сливаем хвост очереди.
                self.drop_item(audio, "superseded")
                audio = next_task.result()
                sleep_task.cancel()
                await asyncio.gather(sleep_task, return_exceptions=True)
                continue

            # target наступил; не оставляем незавершённый get(), который мог бы
            # съесть следующую реплику уже после возврата в базовый service loop.
            next_task.cancel()
            await asyncio.gather(next_task, return_exceptions=True)
            return await self.handle(audio)

    async def handle(self, audio: TtsAudio):
        if self._is_late(audio):
            self.drop_item(audio, "deadline")
            return None

        # Direct users of handle() (тесты/плагины) также получают планирование;
        # штатный loop может прервать ожидание в handle_with_inbox выше.
        target_ts = audio.trace.target_ts
        if target_ts is not None and self._stream is not None:
            wait_s = target_ts - now()
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            if self._is_late(audio):
                self.drop_item(audio, "deadline")
                return None

        # Без устройства вывода нет фактического момента воспроизведения.
        # Не рисуем ``play`` / ``write_end`` для null или недоступного
        # sounddevice: иначе ускоренный dummy source даёт отрицательный
        # «source_end→play» и UI выдаёт его за реальную синхронность. При этом
        # processing latency до TTS всё равно полезна для smoke/debug metrics.
        if self._stream is None:
            self.metrics.observe(audio.trace)
            return None

        # ``play`` — фактическое начало вызова stream.write, а не момент TTS.
        audio.trace.mark("play", now())
        write_ok = await asyncio.to_thread(self._write, audio.samples)
        # sounddevice.write блокирует, пока данные приняты stream. Это не
        # притворяется временем, когда динамик физически доиграл буфер, зато
        # честно фиксирует реальный конец записи в API устройства.
        audio.trace.mark("write_end", now())
        if not write_ok:
            self.metrics.record_output_error()
        self.metrics.observe(audio.trace)
        spans = " | ".join(f"{name} {ms:.0f} мс" for name, ms in audio.trace.spans_ms())
        self.log.debug("задержки сегмента: %s", spans)
        return None

    def _is_late(self, audio: TtsAudio) -> bool:
        """Проверить deadline, сохранив старую max_backlog эвристику для API."""
        if audio.trace.deadline_ts is not None:
            if audio.trace.is_expired():
                self.log.warning("target просрочен на %.1f с — сегмент пропущен", now() - audio.trace.deadline_ts)
                return True
            return False
        speech_end = audio.trace.marks.get("speech_end")
        if speech_end is None:
            return False
        lag = now() - speech_end
        if lag > self.cfg.output.max_backlog_s:
            self.log.warning("отстаём на %.1f с — сегмент пропущен", lag)
            return True
        return False

    def _write(self, samples: np.ndarray) -> bool:
        stream = self._stream
        if stream is None:
            return True
        # Пишем кусками по 100 мс, чтобы остановка была отзывчивой
        step = max(1, self.cfg.output.sample_rate // 10)
        try:
            for i in range(0, len(samples), step):
                if self._closing:
                    return False
                stream.write(np.ascontiguousarray(samples[i : i + step]))
            return True
        except Exception as exc:  # noqa: BLE001 — устройство могло исчезнуть
            if not self._closing:
                self.log.error("сбой воспроизведения: %s", exc)
            return False
