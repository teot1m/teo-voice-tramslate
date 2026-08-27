"""MLX route resolves the pinned local snapshot without a network check."""
from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from uvt.engines.stt_mlx_whisper import MlxWhisperSTT


async def test_mlx_warmup_resolves_local_only_snapshot(monkeypatch, tmp_path):
    calls = []
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    loaded = []

    class FakeHolder:
        model = None
        model_path = None

        @classmethod
        def get_model(cls, model_path, dtype):
            cls.model_path = model_path
            cls.model = object()
            loaded.append((model_path, dtype))
            return cls.model

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper.transcribe",
        SimpleNamespace(ModelHolder=FakeHolder),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    cfg = SimpleNamespace(
        model="large-v3-turbo", revision="fixed-revision", allow_download=False
    )
    engine = MlxWhisperSTT(cfg)

    await engine.warmup()

    assert engine._repo == str(snapshot)
    assert calls == [
        {
            "repo_id": "mlx-community/whisper-large-v3-turbo",
            "revision": "fixed-revision",
            "local_files_only": True,
        }
    ]
    assert loaded and loaded[0][0] == str(snapshot)


@pytest.mark.asyncio
async def test_long_transcription_reports_each_completed_chunk(monkeypatch):
    engine = MlxWhisperSTT(
        SimpleNamespace(chunk_seconds=15, overlap_seconds=2, beam_size=1)
    )
    engine._repo = "/cached/whisper"
    calls = 0

    def fake_transcribe(samples, _language, _words):
        nonlocal calls
        calls += 1
        return {
            "text": f"word-{calls}.",
            "language": "en",
            "segments": [
                {
                    "text": f" word-{calls}.",
                    "start": 1.0,
                    "end": 2.0,
                    "words": [
                        {
                            "word": f" word-{calls}.",
                            "start": 1.0,
                            "end": 2.0,
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(engine, "_transcribe", fake_transcribe)
    progress = []
    spans = await engine.transcribe_long(
        np.zeros(30 * 16_000, dtype=np.float32),
        16_000,
        None,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert calls == 3
    assert progress == [(14.0, 30.0), (27.0, 30.0), (30.0, 30.0)]
    assert [span.text for span in spans] == ["word-1.", "word-2.", "word-3."]
    assert all(span.language == "en" for span in spans)


def test_mlx_transcribe_uses_supported_greedy_options(monkeypatch):
    calls = []

    def transcribe(samples, **kwargs):
        calls.append((samples, kwargs))
        return {"text": "ok", "language": "en", "segments": []}

    monkeypatch.setitem(
        sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe)
    )
    engine = MlxWhisperSTT(SimpleNamespace(beam_size=1))
    engine._repo = "/cached/whisper"
    engine._transcribe(np.zeros(16_000, dtype=np.float32), None, True)

    assert len(calls) == 1
    kwargs = calls[0][1]
    assert kwargs["temperature"] == 0.0
    assert kwargs["condition_on_previous_text"] is False
    assert "beam_size" not in kwargs


@pytest.mark.asyncio
async def test_cancellation_waits_for_active_whisper_chunk(monkeypatch):
    engine = MlxWhisperSTT(
        SimpleNamespace(chunk_seconds=15, overlap_seconds=0, beam_size=1)
    )
    engine._repo = "/cached/whisper"
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def blocking_transcribe(_samples, _language, _words):
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=3)
        return {"text": "done", "language": "en", "segments": []}

    monkeypatch.setattr(engine, "_transcribe", blocking_transcribe)
    task = asyncio.create_task(
        engine.transcribe_long(
            np.zeros(30 * 16_000, dtype=np.float32), 16_000, None
        )
    )
    assert await asyncio.to_thread(started.wait, 1.0)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == 1
