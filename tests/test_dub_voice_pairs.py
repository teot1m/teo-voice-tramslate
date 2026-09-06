"""Whole dubbing calls preserve selected role pairs on one shared TTS model."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

import uvt.dub as dub
from uvt.config import AppConfig, configured_role_voice
from uvt.interfaces import STTSpan, TTSEngine, VoiceReference


class RoleEngine(TTSEngine):
    supports_reference = True

    def __init__(self, cfg, tracker, *, duration=False, shared=True):
        super().__init__(cfg)
        self.tracker = tracker
        self.supports_duration = duration
        self.supports_reference = shared
        self.calls = []
        self.attempts = {}
        self.closed = False
        self.active = 0

    async def warmup(self):
        self.initial_role = self.cfg.voice_gender

    async def close(self):
        assert self.active == 0, "models must outlive native inference"
        assert self.cfg.voice_gender == self.initial_role, "role must be restored before closing"
        self.closed = True

    async def _generate(self, text, *, target_duration=None, reference=None):
        self.active += 1
        self.tracker.active += 1
        self.tracker.max_active = max(self.tracker.max_active, self.tracker.active)
        before = self.cfg.voice_gender
        try:
            # A realistic lazy/native model reads its config after async setup.
            await asyncio.sleep(.002)
            await asyncio.sleep(0)
            role = self.cfg.voice_gender
            voice = self.cfg.voice_id or configured_role_voice(self.cfg)
            self.calls.append({"text": text, "before": before, "role": role,
                               "voice": voice, "target": target_duration,
                               "reference": reference})
            self.attempts[text] = self.attempts.get(text, 0) + 1
            if self.tracker.retry == text and self.attempts[text] == 1:
                raise httpx.HTTPStatusError("temporary test failure", request=httpx.Request("POST", "https://tts.invalid"), response=httpx.Response(503))
            if self.tracker.fail == text:
                raise ValueError("test synthesis failure")
            seconds = target_duration if target_duration is not None else (3 if self.supports_duration else .1)
            return np.full(round(seconds * 1000), .1 if role == "male" else .2, dtype=np.float32), 1000
        finally:
            self.active -= 1
            self.tracker.active -= 1

    async def synthesize(self, text, language):
        return await self._generate(text)

    async def synthesize_slot(self, text, language, *, target_duration=None, reference=None, **kwargs):
        return await self._generate(text, target_duration=target_duration, reference=reference)


def prepare(monkeypatch, *, duration=False, shared=True, retry=None, fail=None, engine_class=RoleEngine):
    tracker = SimpleNamespace(active=0, max_active=0, retry=retry, fail=fail, instances=[])
    def create(kind, name, cfg):
        assert kind == "tts"
        engine = engine_class(cfg, tracker, duration=duration, shared=shared)
        tracker.instances.append(engine)
        return engine
    monkeypatch.setattr(dub.registry, "create", create)
    monkeypatch.setattr(dub, "_tts_retry_delay", lambda *_: 0)
    cfg = AppConfig()
    cfg.tts.engine = "fake-shared-tts"
    cfg.tts.concurrency = 4
    cfg.tts.male_voice_id = "chosen-low"
    cfg.tts.female_voice_id = "chosen-high"
    cfg.tts.duration_per_char = {}
    genders = ["male", "female", "male", "female"]
    texts = [f"line{i}" for i in range(4)]
    spans = [STTSpan(i * 2., i * 2. + 1, text, "en") for i, text in enumerate(texts)]
    return cfg, spans, texts, genders, tracker


@pytest.mark.parametrize("duration", [False, True])
@pytest.mark.asyncio
async def test_shared_model_alternates_pair_through_await_and_duration_rerender(monkeypatch, duration):
    cfg, spans, texts, genders, tracker = prepare(monkeypatch, duration=duration)
    references = {role: VoiceReference(np.zeros(16000, dtype=np.float32), 16000, role) for role in set(genders)}
    clips = await dub._synthesize_all(cfg, spans, texts, genders, None, references=references)
    [engine] = tracker.instances
    assert len(clips) == 4
    assert tracker.max_active == 1
    assert engine.closed
    assert cfg.tts.voice_gender == "auto"
    for call in engine.calls:
        expected = genders[texts.index(call["text"])]
        assert call["before"] == call["role"] == expected
        assert call["voice"] == ("chosen-low" if expected == "male" else "chosen-high")
        assert call["reference"] is references[expected]
    if duration:
        assert any(call["target"] is not None for call in engine.calls)
        # A different speaker cannot interleave the first pass and its rerender.
        assert [call["text"] for call in engine.calls] == ["line0", "line0", "line1", "line1", "line2", "line2", "line3"]
    np.testing.assert_allclose([clip.samples[0] for clip in clips], [.1, .2, .1, .2])


@pytest.mark.asyncio
async def test_retry_keeps_replica_role_and_holds_shared_lock(monkeypatch):
    cfg, spans, texts, genders, tracker = prepare(monkeypatch, retry="line1")
    clips = await dub._synthesize_all(cfg, spans, texts, genders, None)
    [engine] = tracker.instances
    assert len(clips) == 4
    assert [call["text"] for call in engine.calls] == ["line0", "line1", "line1", "line2", "line3"]
    assert [call["voice"] for call in engine.calls if call["text"] == "line1"] == ["chosen-high"] * 2
    assert tracker.max_active == 1
    assert engine.closed


@pytest.mark.asyncio
async def test_failed_replica_restores_role_before_next_speaker(monkeypatch):
    cfg, spans, texts, genders, tracker = prepare(monkeypatch, fail="line1")
    clips = await dub._synthesize_all(cfg, spans, texts, genders, None)
    [engine] = tracker.instances
    assert [clip.index for clip in clips] == [0, 2, 3]
    assert [call["role"] for call in engine.calls] == genders
    assert engine.closed


@pytest.mark.asyncio
async def test_explicit_single_voice_stays_authoritative_on_shared_model(monkeypatch):
    cfg, spans, texts, genders, tracker = prepare(monkeypatch)
    cfg.tts.voice_id = "one-chosen-voice"
    await dub._synthesize_all(cfg, spans, texts, genders, None)
    [engine] = tracker.instances
    assert {call["voice"] for call in engine.calls} == {"one-chosen-voice"}


@pytest.mark.asyncio
async def test_separate_gender_models_keep_parallelism(monkeypatch):
    cfg, spans, texts, genders, tracker = prepare(monkeypatch, shared=False)
    clips = await dub._synthesize_all(cfg, spans, texts, genders, None)
    assert len(tracker.instances) == 2
    assert tracker.max_active > 1
    assert len(clips) == 4
    for engine in tracker.instances:
        assert engine.closed
        assert all(call["role"] == engine.initial_role for call in engine.calls)
        expected = "chosen-low" if engine.initial_role == "male" else "chosen-high"
        assert {call["voice"] for call in engine.calls} == {expected}


class NativeDrainEngine(RoleEngine):
    async def _generate(self, text, **kwargs):
        if text != "line1":
            return await super()._generate(text, **kwargs)
        self.active += 1
        self.tracker.active += 1
        self.tracker.started.set()
        async def native_worker():
            await self.tracker.release.wait()
            self.tracker.native_role = self.cfg.voice_gender
            self.tracker.native_voice = configured_role_voice(self.cfg)
            return np.zeros(100, dtype=np.float32), 1000
        worker = asyncio.create_task(native_worker())
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            self.tracker.draining.set()
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
            raise
        finally:
            self.active -= 1
            self.tracker.active -= 1


@pytest.mark.asyncio
async def test_repeated_cancel_drains_native_worker_before_role_restore_and_close(monkeypatch):
    cfg, spans, texts, genders, tracker = prepare(monkeypatch, engine_class=NativeDrainEngine)
    tracker.started = asyncio.Event()
    tracker.draining = asyncio.Event()
    tracker.release = asyncio.Event()
    task = asyncio.create_task(dub._synthesize_all(cfg, spans, texts, genders, None))
    try:
        await asyncio.wait_for(tracker.started.wait(), 2)
        [engine] = tracker.instances
        assert engine.cfg.voice_gender == "female"
        task.cancel()
        await asyncio.wait_for(tracker.draining.wait(), 2)
        task.cancel()
        await asyncio.sleep(.005)
        assert not task.done()
        assert not engine.closed
        assert engine.cfg.voice_gender == "female"
        tracker.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert tracker.native_role == "female"
        assert tracker.native_voice == "chosen-high"
        assert engine.cfg.voice_gender == "male"
        assert engine.closed
        assert tracker.active == 0
    finally:
        tracker.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
