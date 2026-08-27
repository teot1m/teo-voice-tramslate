"""Latency guards for the batch dubbing pipeline."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from uvt.config import AppConfig
from uvt.dub import (
    _compact_stt_spans,
    _fit_audio_tempo,
    _synthesize_all,
    _transcribe_all,
)
from uvt.interfaces import STTResult, STTSpan


class _ParallelSTT:
    concurrency_hint = 3

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def transcribe_long(self, *_args, **_kwargs):
        return None

    async def transcribe(self, samples, _sample_rate, _language):
        index = int(samples[0])
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            # Finish in a different order to prove timestamps remain ordered.
            await asyncio.sleep(0.01 * (4 - index))
            return STTResult(f"segment-{index}", "en")
        finally:
            self.active -= 1


class _FakeVAD:
    async def close(self) -> None:
        return None


class _FakeSegmenter:
    def __init__(self, *_args, **_kwargs) -> None:
        self.sent = False

    def feed(self, _samples):
        if self.sent:
            return []
        self.sent = True
        return [
            SimpleNamespace(
                samples=np.full(32, index, dtype=np.float32),
                sample_rate=16_000,
                start_ts=float(index),
                end_ts=float(index + 1),
            )
            for index in range(4)
        ]

    def flush(self):
        return None


def test_whisper_fragments_are_compacted_without_breaking_sync_boundaries():
    spans = [
        STTSpan(0.0, 1.0, "Теперь мы примерно", "ru"),
        STTSpan(1.1, 1.8, "понимаем,", "ru"),
        STTSpan(1.9, 2.8, "что происходит.", "ru"),
        STTSpan(3.0, 3.8, "Новый вопрос", "ru"),
        STTSpan(5.2, 6.0, "после длинной паузы", "ru"),
    ]

    compacted = _compact_stt_spans(spans)

    assert [span.text for span in compacted] == [
        "Теперь мы примерно понимаем, что происходит.",
        "Новый вопрос",
        "после длинной паузы",
    ]
    assert compacted[0].start == 0.0
    assert compacted[0].end == 2.8


def test_ffmpeg_atempo_shortens_clip_without_pitch_resynthesis():
    source = np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000).astype(np.float32)
    fitted = _fit_audio_tempo(source, 16_000, 1.5)

    assert 0.55 < len(fitted) / 16_000 < 0.8


async def test_vad_stt_runs_in_parallel_and_preserves_source_order(monkeypatch):
    import uvt.dub as dub_module

    engine = _ParallelSTT()
    monkeypatch.setattr(dub_module, "create_stt_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(
        dub_module,
        "create_vad_engine",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=_FakeVAD()),
    )
    monkeypatch.setattr(dub_module, "Segmenter", _FakeSegmenter)

    progress: list[float] = []
    spans = await _transcribe_all(
        AppConfig(),
        np.zeros(16_000, dtype=np.float32),
        "en",
        progress.append,
    )

    assert engine.max_active == 3
    assert [span.text for span in spans] == [
        "segment-0",
        "segment-1",
        "segment-2",
        "segment-3",
    ]
    assert progress[-1] == 1.0


class _PaymentRequired(Exception):
    response = SimpleNamespace(status_code=402)


class _FatalTTSEngine:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    async def synthesize(self, *_args):
        self.calls += 1
        raise _PaymentRequired("payment required")


async def test_tts_payment_error_stops_queued_requests(monkeypatch):
    import uvt.dub as dub_module

    engine = _FatalTTSEngine()
    monkeypatch.setattr(dub_module.registry, "create", lambda *_args, **_kwargs: engine)
    cfg = AppConfig()
    cfg.tts.engine = "openai"
    cfg.tts.concurrency = 2
    spans = [STTSpan(float(i), float(i + 1), f"line {i}", "en") for i in range(12)]

    with pytest.raises(RuntimeError, match="OpenAI.*HTTP 402"):
        await _synthesize_all(
            cfg,
            spans,
            [f"строка {i}" for i in range(12)],
            ["female"] * 12,
            None,
        )

    assert engine.calls <= cfg.tts.concurrency
    assert engine.closed


class _RateLimitedTTSEngine:
    def __init__(self) -> None:
        self.calls = 0

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def synthesize(self, *_args):
        self.calls += 1
        if self.calls <= 3:
            response = SimpleNamespace(status_code=429, headers={"retry-after": "0"})
            error = RuntimeError("rate limited")
            error.response = response
            raise error
        return np.ones(16_000, dtype=np.float32), 16_000


async def test_elevenlabs_rate_limit_retries_with_provider_delay(monkeypatch):
    import uvt.dub as dub_module

    engine = _RateLimitedTTSEngine()
    monkeypatch.setattr(dub_module.registry, "create", lambda *_args, **_kwargs: engine)
    cfg = AppConfig()
    cfg.target_lang = "ru"
    cfg.tts.engine = "elevenlabs"
    cfg.tts.concurrency = 1
    cfg.tts.request_interval_s = 0
    spans = [STTSpan(0.0, 1.0, "line", "en")]

    clips = await _synthesize_all(cfg, spans, ["строка"], ["male"], None)

    assert len(clips) == 1 and clips[0] is not None
    assert engine.calls == 4


class _SpeedProbeEngine:
    def __init__(self) -> None:
        self.speeds: list[float] = []

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def synthesize(self, *_args):
        self.speeds.append(1.0)
        return np.ones(16_000, dtype=np.float32), 16_000

    async def synthesize_rated(self, _text, _language, speed):
        self.speeds.append(speed)
        return np.ones(int(16_000 / speed), dtype=np.float32), 16_000


async def test_tts_predicts_piper_speed_before_first_synthesis(monkeypatch):
    import uvt.dub as dub_module

    engine = _SpeedProbeEngine()
    monkeypatch.setattr(dub_module.registry, "create", lambda *_args, **_kwargs: engine)
    cfg = AppConfig()
    cfg.target_lang = "uk"
    cfg.tts.engine = "speed-probe"
    cfg.tts.duration_per_char = {"uk:male": 0.08}
    cfg.tts.concurrency = 2
    spans = [
        STTSpan(0.0, 1.0, "source", "ru"),
        STTSpan(1.1, 2.1, "next", "ru"),
    ]

    progress: list[float] = []
    await _synthesize_all(
        cfg,
        spans,
        ["довольно длинная переведенная строка", "конец"],
        ["male", "male"],
        progress.append,
    )

    assert engine.speeds[0] > 1.0
    assert engine.speeds[-1] == 1.0
    assert progress == sorted(progress)
    assert progress[-1] == 1.0
