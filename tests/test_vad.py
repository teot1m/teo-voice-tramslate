"""Сегментация VAD: тишина → тон → тишина даёт ровно одну реплику."""
import numpy as np

from uvt.bus import Bus
from uvt.config import AppConfig, CaptureConfig, VADConfig
from uvt.events import AudioChunk
from uvt.metrics import Metrics
from uvt.services.vad import VADService

RATE = 16000


async def _feed(service: VADService, samples: np.ndarray):
    """Отдаёт сигнал чанками по 20 мс с согласованными метками времени."""
    segments = []
    chunk_len = int(RATE * 0.02)
    ts = 100.0
    for i in range(0, len(samples), chunk_len):
        chunk = samples[i : i + chunk_len]
        out = await service.handle(AudioChunk(samples=chunk, sample_rate=RATE, ts=ts))
        ts += len(chunk) / RATE
        if out:
            segments.extend(out)
    return segments


async def test_single_utterance_detected():
    cfg = AppConfig(
        capture=CaptureConfig(backend="dummy"),
        vad=VADConfig(engine="energy"),
    )
    cfg.latency.preset = "balanced"  # пауза-отсечка 500 мс
    service = VADService(Bus(), cfg, Metrics())
    await service.setup()

    t = np.arange(RATE) / RATE  # 1 с тона
    signal = np.concatenate([
        np.zeros(RATE // 2, dtype=np.float32),
        (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32),
        np.zeros(RATE, dtype=np.float32),
    ])
    segments = await _feed(service, signal)

    assert len(segments) == 1
    seg = segments[0]
    # тон + пред-буфер + оставленный хвост тишины
    assert 0.9 <= seg.duration_s <= 1.9
    assert seg.end_ts > seg.start_ts
    assert "speech_end" in seg.trace.marks


async def test_silence_produces_nothing():
    cfg = AppConfig(vad=VADConfig(engine="energy"))
    service = VADService(Bus(), cfg, Metrics())
    await service.setup()
    segments = await _feed(service, np.zeros(RATE * 3, dtype=np.float32))
    assert segments == []
