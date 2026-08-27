"""Отделение речи от фона: распознавание по чистой речи, фон целиком в микс."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
import soundfile as sf

from uvt.config import AppConfig
from uvt.dub import render_dub_track
from uvt.separate import SeparatedAudio, _as_stereo

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.target_lang = "ru"
    cfg.vad.engine = "energy"
    cfg.latency.preset = "ultra"
    cfg.stt.engine = "dummy"
    cfg.translation.engine = "dummy"
    cfg.tts.engine = "dummy"
    cfg.separation.enabled = True
    return cfg


def _speech_wav(path, rate: int = 44100) -> None:
    """Две «реплики» тоном с тишиной между ними."""
    t = np.arange(rate) / rate
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    silence = np.zeros(int(rate * 1.0), dtype=np.float32)
    sf.write(path, np.concatenate([silence[: rate // 2], tone, silence, tone, silence[: rate // 2]]), rate)


class TestStereoNormalisation:
    def test_mono_is_duplicated(self):
        out = _as_stereo(np.ones(10, dtype=np.float32))
        assert out.shape == (10, 2)

    def test_single_channel_column_is_duplicated(self):
        out = _as_stereo(np.ones((10, 1), dtype=np.float32))
        assert out.shape == (10, 2)

    def test_multichannel_is_trimmed_to_stereo(self):
        out = _as_stereo(np.ones((10, 6), dtype=np.float32))
        assert out.shape == (10, 2)


class TestPipelineIntegration:
    async def test_background_replaces_original_and_disables_ducking(
        self, tmp_path, monkeypatch
    ):
        import uvt.dub as dub_module

        src = tmp_path / "in.wav"
        _speech_wav(src)
        calls: list[int] = []
        # Фон — заметная постоянная величина, речь — тишина: если фон попал в
        # микс, его видно в результате; ducking его тронуть не должен.
        background_level = 0.4

        def fake_separate(samples, rate, **_kwargs):
            calls.append(len(samples))
            stereo = _as_stereo(samples)
            return SeparatedAudio(
                speech=np.asarray(samples, dtype=np.float32),
                background=np.full_like(stereo, background_level),
                sample_rate=rate,
            )

        monkeypatch.setattr(dub_module, "separate_speech", fake_separate)

        track, entries = await render_dub_track(_cfg(), src, mix_original=True)

        assert calls, "разделение должно быть вызвано"
        assert entries, "реплики должны распознаться по отделённой речи"
        # Фон сохранён полностью: под репликой уровень не просел
        first = entries[0]
        window = track[int(first.start * 48_000) + 2_400 : int(first.start * 48_000) + 4_800]
        assert np.mean(np.abs(window)) > background_level * 0.8

    async def test_failed_separation_falls_back_to_original(self, tmp_path, monkeypatch):
        import uvt.dub as dub_module

        src = tmp_path / "in.wav"
        _speech_wav(src)

        def broken(*_args, **_kwargs):
            raise RuntimeError("demucs недоступен")

        monkeypatch.setattr(dub_module, "separate_speech", broken)

        track, entries = await render_dub_track(_cfg(), src, mix_original=True)

        # Задача не падает: работаем по исходному звуку
        assert entries
        assert len(track) > 0

    async def test_disabled_separation_does_not_call_demucs(self, tmp_path, monkeypatch):
        import uvt.dub as dub_module

        src = tmp_path / "in.wav"
        _speech_wav(src)
        cfg = _cfg()
        cfg.separation.enabled = False

        def unexpected(*_args, **_kwargs):
            raise AssertionError("разделение выключено — вызова быть не должно")

        monkeypatch.setattr(dub_module, "separate_speech", unexpected)

        _track, entries = await render_dub_track(cfg, src, mix_original=True)
        assert entries
