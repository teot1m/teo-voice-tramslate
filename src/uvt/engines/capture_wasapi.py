"""Захват системного звука Windows: WASAPI loopback через PyAudioWPatch.

device в конфиге — подстрока имени устройства вывода (например, "Speakers");
без него берётся loopback устройства вывода по умолчанию.

ВНИМАНИЕ: движок написан для Windows и на других ОС сразу сообщает об этом.
"""
from __future__ import annotations

import asyncio
import logging
import sys

import numpy as np

from uvt.audio import to_mono
from uvt.interfaces import CaptureEngine
from uvt.registry import register

log = logging.getLogger("uvt.capture.wasapi")


@register("capture", "wasapi-loopback")
class WasapiLoopbackCapture(CaptureEngine):
    async def stream(self):
        if sys.platform != "win32":
            raise RuntimeError(
                "wasapi-loopback работает только на Windows; "
                "на macOS используйте backend 'sounddevice' + BlackHole, "
                "на Linux — 'sounddevice' + monitor-источник PipeWire"
            )
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        device = self._find_device(pa)
        rate = int(device["defaultSampleRate"])
        channels = max(1, int(device["maxInputChannels"]))

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)

        def _offer(data: np.ndarray) -> None:
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

        def callback(in_data, frame_count, time_info, status):
            data = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels).copy()
            try:
                loop.call_soon_threadsafe(_offer, data)
            except RuntimeError:
                pass
            return (None, pyaudio.paContinue)

        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(device["index"]),
            frames_per_buffer=int(rate * 0.02),
            stream_callback=callback,
        )
        log.info("WASAPI loopback: '%s' @ %d Гц, %d кан.", device["name"], rate, channels)
        try:
            while True:
                data = await queue.get()
                yield to_mono(data), rate
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def _find_device(self, pa) -> dict:
        if self.cfg.device is not None:
            needle = str(self.cfg.device).lower()
            for dev in pa.get_loopback_device_info_generator():
                if needle in dev["name"].lower():
                    return dev
            log.warning("loopback с именем '%s' не найден, беру по умолчанию", self.cfg.device)
        try:
            return pa.get_default_wasapi_loopback()
        except Exception:  # noqa: BLE001 — старые версии PyAudioWPatch
            dev = next(pa.get_loopback_device_info_generator(), None)
            if dev is None:
                raise RuntimeError("loopback-устройство WASAPI не найдено") from None
            return dev
