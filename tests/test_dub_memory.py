"""A fixed timeline memory budget protects 16 GB Macs from full-audio copies."""

import tracemalloc

import numpy as np
import pytest

from uvt.config import AppConfig
from uvt.interfaces import STTSpan


@pytest.mark.parametrize("mix_original", [False, True])
async def test_render_timeline_stays_within_memory_budget(tmp_path, monkeypatch, mix_original):
    import uvt.dub as dub

    duration = 30
    source = tmp_path / "input.wav"
    source.touch()
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.target_lang = "ru"
    cfg.translation.engine = "none"
    cfg.tts.engine = "dummy"
    cfg.tts.gender = "male"
    cfg.speaker.enabled = False

    def decode(_path, rate, channels):
        shape = (duration * rate, channels) if channels > 1 else duration * rate
        return np.full(shape, 0.2, dtype=np.float32)

    async def transcribe(*_args, **_kwargs):
        return [STTSpan(1.0, 2.0, "This is a complete sentence.", "en")]

    async def synthesize(*_args, **_kwargs):
        return [dub._SynthClip(0, 1.0, np.full(4800, 0.1, dtype=np.float32), 48000)]

    monkeypatch.setattr(dub, "_decode_file", decode)
    monkeypatch.setattr(dub, "_transcribe_all", transcribe)
    monkeypatch.setattr(dub, "_synthesize_all", synthesize)
    # Import registries before measuring only this job's audio allocations.
    dub.registry.load_builtins()
    tracemalloc.start()
    try:
        track, entries = await dub.render_dub_track(cfg, source, mix_original=mix_original)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    timeline_bytes = duration * dub.MIX_RATE * np.dtype(np.float32).itemsize
    # Voice only: STT mono + one output. Full mix: additionally stereo source
    # and a ducking envelope. Allow scratch space, not another full-track copy.
    assert peak < timeline_bytes * (5.2 if mix_original else 2.0)
    assert track.shape == ((duration * dub.MIX_RATE, 2) if mix_original else (duration * dub.MIX_RATE,))
    assert len(entries) == 1
    assert entries[0].tts_end > entries[0].tts_start
    np.testing.assert_allclose(track[:100], 0.2 if mix_original else 0.0)
    assert np.max(np.abs(track)) <= 1.0
