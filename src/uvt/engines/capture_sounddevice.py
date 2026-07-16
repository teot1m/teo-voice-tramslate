"""Захват с любого входного устройства через PortAudio (sounddevice).

Покрывает: BlackHole (macOS), VB-Cable/Virtual Audio Cable (Windows),
monitor-источники PulseAudio/PipeWire (Linux), микрофоны и линейные входы.
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from uvt.audio import to_mono
from uvt.interfaces import CaptureEngine
from uvt.registry import register

log = logging.getLogger("uvt.capture.sounddevice")


@register("capture", "sounddevice")
class SoundDeviceCapture(CaptureEngine):
    async def stream(self):
        import sounddevice as sd

        cfg = self.cfg
        device = cfg.device
        if device is not None:
            info = sd.query_devices(device, "input")
        else:
            info = sd.query_devices(kind="input")
        rate = int(cfg.sample_rate or info["default_samplerate"] or 48000)
        channels = int(cfg.channels or min(2, max(1, info["max_input_channels"])))

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)
        status_count = 0

        def _offer(data: np.ndarray) -> None:
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                pass  # конвейер отстаёт — новее важнее

        def callback(indata, frames, time_info, status) -> None:
            nonlocal status_count
            if status:
                status_count += 1
                if status_count % 100 == 1:
                    log.warning("переполнение буфера захвата: %s", status)
            data = np.array(indata, dtype=np.float32, copy=True)
            try:
                loop.call_soon_threadsafe(_offer, data)
            except RuntimeError:
                pass  # цикл событий уже остановлен

        stream = sd.InputStream(
            device=device,
            samplerate=rate,
            channels=channels,
            dtype="float32",
            blocksize=max(64, int(rate * 0.02)),
            callback=callback,
        )
        with stream:
            log.info("захват: '%s' @ %d Гц, %d кан.", info["name"], rate, channels)
            while True:
                data = await queue.get()
                yield to_mono(data), rate
