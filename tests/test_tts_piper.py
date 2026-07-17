"""Контракт локального Piper TTS без установленного Piper в test env."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from uvt.engines.tts_piper import PiperTTS


def write_piper_sidecar(model, sample_rate=24000):
    model.with_name(model.name + ".json").write_text(
        f'{{"audio": {{"sample_rate": {sample_rate}}}}}', encoding="utf-8"
    )


async def test_piper_decodes_raw_pcm(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    write_piper_sidecar(model)
    cfg = SimpleNamespace(model_path=str(model), binary="piper")
    engine = PiperTTS(cfg)
    raw = np.array([-32768, 0, 32767], dtype="<i2").tobytes()

    with patch("uvt.engines.tts_piper.shutil.which", return_value="/usr/bin/piper"), patch(
        "uvt.engines.tts_piper.subprocess.run",
        return_value=SimpleNamespace(stdout=raw),
    ):
        await engine.warmup()
        samples, rate = await engine.synthesize("hello", "ru")

    assert rate == 24000
    assert np.allclose(samples, [-1.0, 0.0, 32767 / 32768])


async def test_piper_requires_model(tmp_path):
    cfg = SimpleNamespace(model_path=str(tmp_path / "missing.onnx"), binary="piper")
    engine = PiperTTS(cfg)
    with patch("uvt.engines.tts_piper.shutil.which", return_value="/usr/bin/piper"):
        with pytest.raises(RuntimeError, match="model_path"):
            await engine.warmup()


async def test_piper_requires_local_metadata_sidecar(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    engine = PiperTTS(SimpleNamespace(model_path=str(model), binary="piper"))

    with patch("uvt.engines.tts_piper.shutil.which", return_value="/usr/bin/piper"):
        with pytest.raises(RuntimeError, match=r"voice\.onnx\.json"):
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
async def test_piper_rejects_invalid_metadata_sample_rate(tmp_path, sidecar):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    model.with_name(model.name + ".json").write_text(sidecar, encoding="utf-8")
    engine = PiperTTS(SimpleNamespace(model_path=str(model), binary="piper"))

    with patch("uvt.engines.tts_piper.shutil.which", return_value="/usr/bin/piper"):
        with pytest.raises(RuntimeError, match="audio.sample_rate|прочитать"):
            await engine.warmup()


async def test_piper_rejects_configured_rate_that_disagrees_with_model(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    write_piper_sidecar(model, sample_rate=22050)
    engine = PiperTTS(
        SimpleNamespace(model_path=str(model), binary="piper", sample_rate=24000)
    )

    with patch("uvt.engines.tts_piper.shutil.which", return_value="/usr/bin/piper"):
        with pytest.raises(RuntimeError, match="не совпадает"):
            await engine.warmup()
