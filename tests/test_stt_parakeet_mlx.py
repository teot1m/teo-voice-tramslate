"""Parakeet MLX adapter tests; the 2.5 GB model is never loaded here."""

from __future__ import annotations

import asyncio
import math
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import uvt.engines.stt_parakeet_mlx as parakeet_engine
from uvt.engines.stt_parakeet_mlx import (
    ParakeetMlxSTT,
    _normalise_language,
    _result_to_spans,
)

RATE = 16_000


@pytest.fixture(autouse=True)
def reset_process_caches():
    parakeet_engine._MODEL_CACHE.clear()
    parakeet_engine._LANGID_IDENTIFIER = None
    yield
    parakeet_engine._MODEL_CACHE.clear()
    parakeet_engine._LANGID_IDENTIFIER = None


def model_dir(tmp_path):
    root = tmp_path / "parakeet"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"mock")
    return root


def sentence(text: str, start, end, duration=None):
    return SimpleNamespace(text=text, start=start, end=end, duration=duration)


def result(text: str, sentences):
    return SimpleNamespace(text=text, sentences=sentences)


def install_fake_runtime(monkeypatch, model):
    loads: list[str] = []
    evaluated: list[object] = []
    cache_clears: list[None] = []

    def from_pretrained(path):
        loads.append(path)
        return model

    parakeet_module = SimpleNamespace(from_pretrained=from_pretrained)
    audio_module = SimpleNamespace(get_logmel=lambda audio, _config: audio)
    mx_module = SimpleNamespace(
        array=lambda audio: np.asarray(audio, dtype=np.float32),
        eval=lambda parameters: evaluated.append(parameters),
        clear_cache=lambda: cache_clears.append(None),
    )
    monkeypatch.setitem(sys.modules, "parakeet_mlx", parakeet_module)
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", audio_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mx_module)
    return loads, evaluated, cache_clears


class SequenceModel:
    preprocessor_config = SimpleNamespace(sample_rate=RATE)

    def __init__(self):
        self.calls: list[int] = []

    def parameters(self):
        return {"mock": np.array([1.0], dtype=np.float32)}

    def generate(self, audio):
        duration = len(audio) / RATE
        index = len(self.calls) + 1
        self.calls.append(len(audio))
        text = f"chunk {index}"
        return [result(text, [sentence(text, 0.0, duration, duration)])]


@pytest.mark.asyncio
async def test_warmup_loads_one_cached_model_and_bounds_chunk_config(
    monkeypatch, tmp_path
):
    model = SequenceModel()
    loads, evaluated, _ = install_fake_runtime(monkeypatch, model)
    root = model_dir(tmp_path)
    cfg = SimpleNamespace(
        model=str(root), chunk_seconds=20, overlap_seconds=90
    )
    first = ParakeetMlxSTT(cfg)
    second = ParakeetMlxSTT(cfg)

    await asyncio.gather(first.warmup(), second.warmup())

    assert loads == [str(root.resolve())]
    assert len(evaluated) == 1
    assert first._model is second._model is model
    assert first.chunk_seconds == 60.0
    assert first.overlap_seconds == 30.0
    assert first.concurrency_hint == 1


@pytest.mark.asyncio
async def test_long_audio_uses_overlapping_chunks_and_reports_seconds(
    monkeypatch, tmp_path
):
    model = SequenceModel()
    _, _, cache_clears = install_fake_runtime(monkeypatch, model)
    engine = ParakeetMlxSTT(
        SimpleNamespace(
            model=str(model_dir(tmp_path)),
            chunk_seconds=60,
            overlap_seconds=15,
        )
    )
    progress: list[tuple[float, float]] = []

    spans = await engine.transcribe_long(
        np.zeros(130 * RATE, dtype=np.float32),
        RATE,
        "uk-UA",
        progress=lambda done, total: progress.append((done, total)),
    )

    assert model.calls == [60 * RATE, 60 * RATE, 40 * RATE]
    assert [(span.start, span.end, span.language) for span in spans] == [
        (0.0, 60.0, "uk"),
        (45.0, 105.0, "uk"),
        (90.0, 130.0, "uk"),
    ]
    assert progress == [
        (0.0, 130.0),
        (60.0, 130.0),
        (105.0, 130.0),
        (130.0, 130.0),
    ]
    assert len(cache_clears) == 3


@pytest.mark.asyncio
async def test_model_calls_are_serial(monkeypatch, tmp_path):
    state_lock = threading.Lock()

    class SlowModel(SequenceModel):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.max_active = 0

        def generate(self, audio):
            with state_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.03)
                return super().generate(audio)
            finally:
                with state_lock:
                    self.active -= 1

    model = SlowModel()
    install_fake_runtime(monkeypatch, model)
    engine = ParakeetMlxSTT(SimpleNamespace(model=str(model_dir(tmp_path))))
    await engine.warmup()

    results = await asyncio.gather(
        engine.transcribe(np.zeros(RATE, dtype=np.float32), RATE, "en"),
        engine.transcribe(np.zeros(RATE, dtype=np.float32), RATE, "en"),
    )

    assert all(item is not None for item in results)
    assert model.max_active == 1


@pytest.mark.asyncio
async def test_close_evicts_model_before_translation_stage(monkeypatch, tmp_path):
    model = SequenceModel()
    _loads, _evaluated, cache_clears = install_fake_runtime(monkeypatch, model)
    engine = ParakeetMlxSTT(SimpleNamespace(model=str(model_dir(tmp_path))))

    await engine.warmup()
    model_path = engine._model_path
    assert model_path in parakeet_engine._MODEL_CACHE

    await engine.close()

    assert engine._model is None
    assert engine._runtime is None
    assert model_path not in parakeet_engine._MODEL_CACHE
    assert cache_clears == [None]


@pytest.mark.asyncio
async def test_cancellation_waits_for_active_chunk_and_starts_no_next_chunk(
    monkeypatch, tmp_path
):
    started = threading.Event()
    release = threading.Event()

    class BlockingModel(SequenceModel):
        def generate(self, audio):
            self.calls.append(len(audio))
            started.set()
            assert release.wait(timeout=2.0)
            duration = len(audio) / RATE
            return [result("done", [sentence("done", 0.0, duration, duration)])]

    model = BlockingModel()
    install_fake_runtime(monkeypatch, model)
    engine = ParakeetMlxSTT(
        SimpleNamespace(
            model=str(model_dir(tmp_path)),
            chunk_seconds=60,
            overlap_seconds=15,
        )
    )
    await engine.warmup()
    task = asyncio.create_task(
        engine.transcribe_long(np.zeros(130 * RATE, dtype=np.float32), RATE, "ru")
    )

    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.005)
    assert started.is_set()
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.calls == [60 * RATE]


@pytest.mark.asyncio
async def test_auto_language_is_classified_once_from_aggregate_text(
    monkeypatch, tmp_path
):
    class TwoSentenceModel(SequenceModel):
        def generate(self, _audio):
            return [
                result(
                    "Привіт, друже. Як справи?",
                    [
                        sentence("Привіт, друже.", 0.1, 0.8, 0.7),
                        sentence("Як справи?", 1.0, 1.8, 0.8),
                    ],
                )
            ]

    calls: list[str] = []

    def detect(text):
        calls.append(text)
        return "uk", 0.98

    install_fake_runtime(monkeypatch, TwoSentenceModel())
    monkeypatch.setattr(parakeet_engine, "_detect_language_blocking", detect)
    engine = ParakeetMlxSTT(SimpleNamespace(model=str(model_dir(tmp_path))))

    spans = await engine.transcribe_long(
        np.zeros(2 * RATE, dtype=np.float32), RATE, None
    )

    assert calls == ["Привіт, друже. Як справи?"]
    assert [span.language for span in spans] == ["uk", "uk"]


@pytest.mark.parametrize(
    ("given", "expected"),
    [("en-US", "en"), ("ru_RU", "ru"), ("uk-UA", "uk"), ("ua", "uk")],
)
def test_explicit_en_ru_uk_language_codes(given, expected):
    assert _normalise_language(given) == expected


def test_unsupported_explicit_language_is_rejected():
    with pytest.raises(ValueError, match="does not support language"):
        _normalise_language("ja")


def test_third_party_timestamps_are_finite_bounded_and_ordered():
    raw = result(
        "one two three",
        [
            sentence("one", math.nan, math.inf, None),
            sentence("two", 2.5, 1.5, None),
            sentence("three", 99.0, -4.0, None),
        ],
    )

    spans = _result_to_spans(
        raw, offset_seconds=10.0, chunk_seconds=3.0, language="en"
    )

    assert all(math.isfinite(span.start) and math.isfinite(span.end) for span in spans)
    assert all(10.0 <= span.start < span.end <= 13.0 for span in spans)
    assert [(span.start, span.end) for span in spans] == sorted(
        (span.start, span.end) for span in spans
    )
