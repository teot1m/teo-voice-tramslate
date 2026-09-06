"""Nemotron integration: cumulative streams, timestamps and native cleanup."""
from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from uvt.engines import stt_nemotron_mlx as nemotron
from uvt.engines.stt_nemotron_mlx import NemotronMLX, _native_call, _result_spans

RATE = 16000


def sentence(text, start, end):
    return SimpleNamespace(text=text, start=start, end=end)


def result(text, sentences=None):
    return SimpleNamespace(text=text, sentences=sentences)


class Model:
    prompt_dictionary = {"en": 0, "ru": 1, "uk": 2, "pt-BR": 3}

    def __init__(self, results):
        self.results = results
        self.calls = []
        self.closed = 0

    def stream_generate(self, audio, **kwargs):
        self.calls.append((audio.copy(), kwargs))
        try:
            yield from self.results
        finally:
            self.closed += 1


def ready(model, chunk=1.0):
    engine = NemotronMLX(SimpleNamespace(chunk_seconds=chunk))
    clears = []
    engine._model = model
    engine._mx = SimpleNamespace(array=lambda samples: samples, clear_cache=lambda: clears.append(True))
    return engine, clears


async def test_final_cumulative_result_is_returned_once_with_native_timestamps():
    model = Model([
        result("Hello", [sentence("Hello", 0.1, 0.5)]),
        result("Hello again.", [sentence("Hello again.", 0.1, 0.8)]),
        result("Hello again. How are you?", [sentence("Hello again.", 0.1, 0.8), sentence("How are you?", 1.0, 1.7)]),
    ])
    engine, _ = ready(model)
    progress = []
    event_thread = threading.get_ident()
    threads = []

    def on_progress(done, total):
        progress.append((done, total))
        threads.append(threading.get_ident())

    spans = await engine.transcribe_long(np.zeros(2*RATE), RATE, "en-US", on_progress)
    assert [(s.start, s.end, s.text, s.language) for s in spans] == [
        (0.1, 0.8, "Hello again.", "en"), (1.0, 1.7, "How are you?", "en"),
    ]
    assert len(model.calls) == 1
    assert model.calls[0][1] == {"language":"en", "chunk_duration":1.0}
    assert progress == [(1.0,2.0),(2.0,2.0),(2.0,2.0)]
    assert threads == [event_thread]*3
    assert model.closed == 1
    await engine.close()


async def test_transcribe_joins_final_spans_without_partial_duplicates():
    model = Model([result("Old partial."), result("Final one. Final two.", [sentence("Final one.",0,1),sentence("Final two.",1,2)])])
    engine, _ = ready(model)
    recognized = await engine.transcribe(np.zeros(2*RATE),RATE,"ru")
    assert recognized.text == "Final one. Final two."
    assert recognized.language == "ru"
    await engine.close()


@pytest.mark.parametrize("raw,expected", [(None,"auto"),("und","auto"),("auto","auto"),("ua","uk"),("UK_ua","uk"),("ru-RU","ru"),("EN_us","en"),("pt-br","pt-BR")])
def test_language_aliases_use_supported_prompt(raw, expected):
    engine, _ = ready(Model([]))
    assert engine._language(raw) == expected


async def test_unknown_language_fails_before_native_stream():
    model = Model([])
    engine, _ = ready(model)
    with pytest.raises(ValueError,match="не поддерживает"):
        await engine.transcribe_long(np.zeros(RATE),RATE,"zz")
    assert model.calls == []
    await engine.close()


async def test_auto_detects_language_from_final_text(monkeypatch):
    seen = []
    monkeypatch.setitem(sys.modules,"langid",SimpleNamespace(classify=lambda text: (seen.append(text) or "uk",1.0)))
    engine, _ = ready(Model([result("Проміжний текст."),result("Добрий день.",[sentence("Добрий день.",0,1)])]))
    spans = await engine.transcribe_long(np.zeros(RATE),RATE,"auto")
    assert [s.language for s in spans] == ["uk"]
    assert seen == ["Добрий день."]
    await engine.close()


@pytest.mark.parametrize("samples,rate,message", [
    (np.zeros(2),8000,"16000"),
    (np.array([np.nan]),RATE,"некорректные"),
    (np.array([np.inf]),RATE,"некорректные"),
    (np.zeros((4,2)),RATE,"одномерный"),
    (np.array(0),RATE,"одномерный"),
])
async def test_invalid_samples_do_not_enter_native_stream(samples,rate,message):
    model = Model([])
    engine, _ = ready(model)
    with pytest.raises(ValueError,match=message):
        await engine.transcribe_long(samples,rate,"en")
    assert model.calls == []
    await engine.close()


async def test_empty_audio_needs_no_model_and_empty_stream_returns_none():
    cold = NemotronMLX(SimpleNamespace(chunk_seconds=1.0))
    assert await cold.transcribe_long(np.array([]),RATE,"en") == []
    assert await cold.transcribe(np.array([]),RATE,"en") is None
    engine, _ = ready(Model([]))
    assert await engine.transcribe(np.zeros(RATE),RATE,"en") is None
    await engine.close()


async def test_unprepared_model_has_actionable_error():
    engine = NemotronMLX(SimpleNamespace(chunk_seconds=1.0))
    with pytest.raises(RuntimeError,match="подготовьте модель"):
        await engine.transcribe_long(np.zeros(RATE),RATE,"en")


def test_timestamps_are_clamped_and_invalid_intervals_not_fabricated():
    value = result("unused",[
        sentence("begin",-.2,.5), sentence("end",1.5,9),
        sentence("backwards",1,.5), sentence("past-end",3,4),
        sentence("nan",float("nan"),1), sentence("missing",None,None),
        sentence(None,0,1), sentence("   ",0,1),
    ])
    spans = _result_spans(value,2,"en")
    assert [(s.start,s.end,s.text) for s in spans] == [(0,.5,"begin"),(1.5,2,"end")]


def test_no_sentence_metadata_falls_back_to_complete_text():
    spans = _result_spans(result("Complete transcription.",None),2,"uk")
    assert [(s.start,s.end,s.text,s.language) for s in spans] == [(0,2,"Complete transcription.","uk")]
    assert _result_spans(result(None,None),2,"uk") == []
    assert _result_spans(None,2,"en") == []


async def test_cancellation_drains_current_chunk_closes_generator_and_starts_no_next():
    started = threading.Event()
    release = threading.Event()

    class Blocking(Model):
        def stream_generate(self,audio,**kwargs):
            try:
                self.calls.append("first")
                started.set()
                assert release.wait(timeout=3)
                yield result("Partial.",[sentence("Partial.",0,1)])
                self.calls.append("second")
                yield result("Must not run.")
            finally:
                self.closed += 1

    model = Blocking([])
    engine, clears = ready(model)
    task = asyncio.create_task(engine.transcribe_long(np.zeros(2*RATE),RATE,"en"))
    assert await asyncio.to_thread(started.wait,2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    closing = asyncio.create_task(engine.close())
    await asyncio.sleep(0)
    assert not task.done() and not closing.done()
    assert engine._model is model
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await closing
    assert model.calls == ["first"] and model.closed == 1
    assert engine._model is None and clears == [True]


async def test_native_error_closes_iterator_and_releases_lock():
    class Broken(Model):
        def stream_generate(self,audio,**kwargs):
            try:
                yield result("Partial.")
                raise RuntimeError("native decode failed")
            finally:
                self.closed += 1

    model = Broken([])
    engine, clears = ready(model)
    with pytest.raises(RuntimeError,match="native decode failed"):
        await engine.transcribe_long(np.zeros(RATE),RATE,"en")
    assert model.closed == 1
    await engine.close()
    assert clears == [True]


async def test_cancelled_close_queued_behind_native_guard_still_releases_model():
    model = Model([])
    engine, clears = ready(model)
    nemotron._NATIVE_GUARD.acquire()
    try:
        task = asyncio.create_task(engine.close())
        for _ in range(10):
            await asyncio.sleep(0)
            if engine._lock.locked():
                break
        assert engine._lock.locked()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        nemotron._NATIVE_GUARD.release()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine._model is None
    assert clears == [True]


async def test_cancelled_native_error_preserves_cancellation_instead_of_failed_job():
    started = threading.Event()
    release = threading.Event()

    def operation(cancel):
        started.set()
        assert release.wait(timeout=3)
        raise RuntimeError("native failure during cancellation")

    task = asyncio.create_task(_native_call(operation))
    assert await asyncio.to_thread(started.wait,2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("chunk",[-1,.01,31,float("nan")])
def test_invalid_chunk_duration_rejected(chunk):
    with pytest.raises(ValueError,match="chunk_seconds"):
        NemotronMLX(SimpleNamespace(chunk_seconds=chunk))
