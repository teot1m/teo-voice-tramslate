"""Piper uses reusable in-process voices and selects RU/UK gender models."""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from uvt.engines.tts_piper import PiperTTS


def write_model(root, name: str, sample_rate: int = 24000):
    model = root / name
    model.write_bytes(b"model")
    model.with_name(model.name + ".json").write_text(
        f'{{"audio": {{"sample_rate": {sample_rate}}}}}', encoding="utf-8"
    )
    return model


def install_fake_piper(monkeypatch):
    loaded = []

    class SynthesisConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Voice:
        def __init__(self, path):
            self.path = path
            self.config = SimpleNamespace(sample_rate=24000)
            self.configs = []
            self.active = 0
            self.max_active = 0

        def synthesize(self, text, syn_config=None):
            self.configs.append(syn_config)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                if text.startswith("parallel"):
                    time.sleep(0.03)
                value = 0.25 if "tetiana" in str(self.path) else 0.5
                yield SimpleNamespace(
                    audio_float_array=np.array([value, -value], dtype=np.float32),
                    sample_rate=24000,
                )
            finally:
                self.active -= 1

    class PiperVoice:
        @staticmethod
        def load(path):
            voice = Voice(path)
            loaded.append(voice)
            return voice

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=PiperVoice, SynthesisConfig=SynthesisConfig),
    )
    return loaded


def voice_cfg(tmp_path, **overrides):
    values = {
        "voice_dir": str(tmp_path),
        "voice_models": {
            "uk:male": "uk_UA-mykyta-high.onnx",
            "uk:female": "uk_UA-tetiana-high.onnx",
        },
        "voice_gender": "male",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


async def test_piper_loads_voice_once_and_selects_gender(monkeypatch, tmp_path):
    write_model(tmp_path, "uk_UA-mykyta-high.onnx")
    write_model(tmp_path, "uk_UA-tetiana-high.onnx")
    loaded = install_fake_piper(monkeypatch)
    cfg = voice_cfg(tmp_path)
    engine = PiperTTS(cfg)
    await engine.warmup()

    male, rate = await engine.synthesize("Вітаю", "uk-UA")
    again, _ = await engine.synthesize("Ще раз", "uk")
    cfg.voice_gender = "female"
    female, _ = await engine.synthesize("Вітаю", "uk")

    assert rate == 24000
    assert np.allclose(male, [0.5, -0.5])
    assert np.array_equal(male, again)
    assert np.allclose(female, [0.25, -0.25])
    assert [voice.path.name for voice in loaded] == [
        "uk_UA-mykyta-high.onnx",
        "uk_UA-tetiana-high.onnx",
    ]


async def test_piper_exact_voice_id_overrides_gender(monkeypatch, tmp_path):
    write_model(tmp_path, "uk_UA-mykyta-high.onnx")
    write_model(tmp_path, "uk_UA-tetiana-high.onnx")
    loaded = install_fake_piper(monkeypatch)
    engine = PiperTTS(
        voice_cfg(
            tmp_path,
            voice_gender="male",
            voice_id="uk_UA-tetiana-high",
        )
    )
    await engine.warmup()

    samples, _ = await engine.synthesize("Вітаю", "uk")

    assert np.allclose(samples, [0.25, -0.25])
    assert loaded[0].path.name == "uk_UA-tetiana-high.onnx"


async def test_piper_rejects_unknown_or_wrong_language_voice_id(monkeypatch, tmp_path):
    write_model(tmp_path, "uk_UA-mykyta-high.onnx")
    write_model(tmp_path, "uk_UA-tetiana-high.onnx")
    install_fake_piper(monkeypatch)

    unknown = PiperTTS(voice_cfg(tmp_path, voice_id="arbitrary-path"))
    await unknown.warmup()
    with pytest.raises(RuntimeError, match="отсутствует"):
        await unknown.synthesize("Вітаю", "uk")

    wrong_language = PiperTTS(
        voice_cfg(tmp_path, voice_id="uk_UA-mykyta-high")
    )
    await wrong_language.warmup()
    with pytest.raises(RuntimeError, match="не подходит"):
        await wrong_language.synthesize("Здравствуйте", "ru")


async def test_piper_rated_synthesis_uses_length_scale(monkeypatch, tmp_path):
    write_model(tmp_path, "uk_UA-mykyta-high.onnx")
    write_model(tmp_path, "uk_UA-tetiana-high.onnx")
    loaded = install_fake_piper(monkeypatch)
    engine = PiperTTS(voice_cfg(tmp_path))
    await engine.warmup()

    await engine.synthesize_rated("Швидше", "uk", 1.25)

    assert loaded[0].configs[0].length_scale == pytest.approx(0.8)
    assert loaded[0].configs[0].normalize_audio is True


async def test_piper_reuses_one_voice_for_parallel_inference(monkeypatch, tmp_path):
    write_model(tmp_path, "uk_UA-mykyta-high.onnx")
    write_model(tmp_path, "uk_UA-tetiana-high.onnx")
    loaded = install_fake_piper(monkeypatch)
    engine = PiperTTS(voice_cfg(tmp_path))
    await engine.warmup()

    await asyncio.gather(
        engine.synthesize("parallel one", "uk"),
        engine.synthesize("parallel two", "uk"),
    )

    assert len(loaded) == 1
    assert loaded[0].max_active == 2


async def test_piper_cancellation_drains_started_native_worker(monkeypatch, tmp_path):
    write_model(tmp_path, "uk_UA-mykyta-high.onnx")
    write_model(tmp_path, "uk_UA-tetiana-high.onnx")
    loaded = install_fake_piper(monkeypatch)
    engine = PiperTTS(voice_cfg(tmp_path))
    await engine.warmup()

    task = asyncio.create_task(engine.synthesize("parallel cancel", "uk"))
    await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert loaded[0].active == 0


async def test_piper_cancellation_drains_native_voice_load(monkeypatch, tmp_path):
    write_model(tmp_path, "uk_UA-mykyta-high.onnx")
    write_model(tmp_path, "uk_UA-tetiana-high.onnx")
    loaded = install_fake_piper(monkeypatch)
    piper_module = sys.modules["piper"]
    original_load = piper_module.PiperVoice.load
    started = threading.Event()
    release = threading.Event()

    def slow_load(path):
        started.set()
        release.wait(timeout=2)
        return original_load(path)

    monkeypatch.setattr(piper_module.PiperVoice, "load", staticmethod(slow_load))
    engine = PiperTTS(voice_cfg(tmp_path))
    await engine.warmup()

    task = asyncio.create_task(engine.synthesize("cancel load", "uk"))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0.01)
    assert task.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(loaded) == 1
    assert len(engine._voices) == 1
    await engine.close()


async def test_piper_requires_model(monkeypatch, tmp_path):
    install_fake_piper(monkeypatch)
    engine = PiperTTS(voice_cfg(tmp_path))
    with pytest.raises(RuntimeError, match="не установлен"):
        await engine.warmup()


async def test_piper_requires_local_metadata_sidecar(monkeypatch, tmp_path):
    install_fake_piper(monkeypatch)
    for name in ("uk_UA-mykyta-high.onnx", "uk_UA-tetiana-high.onnx"):
        (tmp_path / name).write_bytes(b"model")
    engine = PiperTTS(voice_cfg(tmp_path))

    with pytest.raises(RuntimeError, match=r"\.onnx\.json"):
        await engine.warmup()


@pytest.mark.parametrize(
    "sidecar",
    [
        "not json",
        '{"audio": {}}',
        '{"audio": []}',
        '{"audio": {"sample_rate": "22050"}}',
        '{"audio": {"sample_rate": 0}}',
    ],
)
async def test_piper_rejects_invalid_metadata_sample_rate(
    monkeypatch, tmp_path, sidecar
):
    install_fake_piper(monkeypatch)
    for name in ("uk_UA-mykyta-high.onnx", "uk_UA-tetiana-high.onnx"):
        model = tmp_path / name
        model.write_bytes(b"model")
        model.with_name(model.name + ".json").write_text(sidecar, encoding="utf-8")
    engine = PiperTTS(voice_cfg(tmp_path))

    with pytest.raises(RuntimeError, match="audio.sample_rate|прочитать"):
        await engine.warmup()


async def test_piper_rejects_configured_rate_mismatch(monkeypatch, tmp_path):
    install_fake_piper(monkeypatch)
    write_model(tmp_path, "uk_UA-mykyta-high.onnx", sample_rate=22050)
    write_model(tmp_path, "uk_UA-tetiana-high.onnx", sample_rate=22050)
    engine = PiperTTS(voice_cfg(tmp_path, sample_rate=24000))

    with pytest.raises(RuntimeError, match="не совпадает"):
        await engine.warmup()
