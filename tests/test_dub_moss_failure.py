"""Exhausted local TTS recovery must not produce a successful, incomplete dub."""
from __future__ import annotations

import asyncio

import pytest

from uvt.config import AppConfig
from uvt.dub import _is_fatal_tts_error, _synthesize_all, _tts_failure_reason
from uvt.interfaces import STTSpan


class _ExhaustedMossRecovery(RuntimeError):
    fatal_tts = True
    user_message = "MOSS-TTS не смогла завершить реплику после повторных попыток"


class _FailingLocalEngine:
    def __init__(self):
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.closed = False

    async def warmup(self):
        pass

    async def synthesize(self, *_args):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            raise _ExhaustedMossRecovery("internal recovery details")
        finally:
            self.active -= 1

    async def close(self):
        assert self.active == 0, "Model closed before its native workers drained"
        self.closed = True


async def test_exhausted_moss_recovery_stops_queued_lines_and_drains_workers(monkeypatch):
    import uvt.dub as dub_module

    engine = _FailingLocalEngine()
    monkeypatch.setattr(dub_module.registry, "create", lambda *_args, **_kwargs: engine)
    cfg = AppConfig()
    cfg.target_lang = "ru"
    cfg.tts.engine = "moss-onnx"
    cfg.tts.concurrency = 2
    cfg.tts.duration_per_char = {}
    spans = [STTSpan(float(i * 4), float(i * 4 + 3), f"line {i}", "en") for i in range(12)]
    clips = []

    with pytest.raises(RuntimeError, match="озвучка остановлена: MOSS-TTS") as failure:
        await _synthesize_all(
            cfg, spans, [f"строка {i}" for i in range(12)], ["male"] * 12,
            None, on_clip=clips.append,
        )

    assert isinstance(failure.value.__cause__, _ExhaustedMossRecovery)
    assert _ExhaustedMossRecovery.user_message in str(failure.value)
    assert "HTTP" not in str(failure.value)
    assert "internal recovery details" not in str(failure.value)
    assert engine.calls == engine.max_active == cfg.tts.concurrency
    assert engine.active == 0
    assert engine.closed
    assert clips == []


def test_only_explicit_fatal_tts_marker_changes_local_error_handling():
    ordinary = RuntimeError("ordinary local error")
    assert not _is_fatal_tts_error(ordinary)
    ordinary.fatal_tts = False
    assert not _is_fatal_tts_error(ordinary)
    exhausted = _ExhaustedMossRecovery()
    assert _is_fatal_tts_error(exhausted)
    assert _tts_failure_reason("moss-onnx", exhausted) == exhausted.user_message
