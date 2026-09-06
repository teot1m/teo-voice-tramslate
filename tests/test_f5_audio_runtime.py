"""F5 must reject a broken native audio runtime before expensive model work."""
from __future__ import annotations

import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from uvt.config import AppConfig, TTSConfig
from uvt.engines import tts_f5


@pytest.fixture(autouse=True)
def clear_audio_check_cache():
    check = tts_f5.check_f5_audio_runtime
    check.cache_clear()
    yield
    check.cache_clear()


def test_audio_check_decodes_valid_temporary_wav_and_caches_success(monkeypatch):
    loaded_paths = []

    def load(path):
        path = Path(path)
        loaded_paths.append(path)
        with wave.open(str(path), "rb") as source:
            assert source.getnchannels() == 1
            assert source.getsampwidth() == 2
            assert source.getframerate() > 0
            assert source.getnframes() > 0
            assert source.getcomptype() == "NONE"
            assert len(source.readframes(source.getnframes())) > 0
        return object(), 24000

    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(load=load))
    tts_f5.check_f5_audio_runtime()
    tts_f5.check_f5_audio_runtime()

    assert len(loaded_paths) == 1
    assert not loaded_paths[0].exists()
    assert not loaded_paths[0].parent.exists()


def test_audio_check_failure_is_actionable_and_retried(monkeypatch):
    native_error = OSError("libtorchcodec_core8.dylib: missing dependency\n" * 100)
    loaded_paths = []

    def load(path):
        loaded_paths.append(Path(path))
        raise native_error

    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(load=load))
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            tts_f5.check_f5_audio_runtime()
        message = str(caught.value)
        assert "F5-TTS" in message
        assert "TorchCodec" in message
        assert "FFmpeg" in message
        assert "Запустить UVT.command" in message
        assert len(message) < 1000
        assert caught.value.__cause__ is native_error

    assert len(loaded_paths) == 2
    assert all(not path.parent.exists() for path in loaded_paths)
    assert tts_f5.check_f5_audio_runtime.cache_info().currsize == 0

    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(load=lambda path: (object(), 24000)))
    tts_f5.check_f5_audio_runtime()
    assert tts_f5.check_f5_audio_runtime.cache_info().currsize == 1


async def test_failed_audio_check_prevents_f5_model_construction(monkeypatch):
    failure = RuntimeError("audio runtime unavailable")
    built_models = []

    def check():
        raise failure

    def build(**kwargs):
        built_models.append(kwargs)
        raise AssertionError("model must not be constructed with a broken decoder")

    monkeypatch.setattr(tts_f5, "check_f5_audio_runtime", check)
    monkeypatch.setitem(sys.modules, "f5_tts", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "f5_tts.api", SimpleNamespace(F5TTS=build))
    engine = tts_f5.F5TTSEngine(TTSConfig(engine="f5"))

    with pytest.raises(RuntimeError) as caught:
        await engine.warmup()

    assert caught.value is failure
    assert built_models == []


async def test_failed_f5_audio_check_precedes_registry_and_decode(tmp_path, monkeypatch):
    import uvt.dub as dub

    source = tmp_path / "input.wav"
    source.write_bytes(b"preflight must run before decoding")
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.tts.engine = "f5"
    cfg.separation.enabled = True
    failure = RuntimeError("audio runtime unavailable")
    expensive_calls = []

    def check():
        raise failure

    def unexpected(*args, **kwargs):
        expensive_calls.append((args, kwargs))
        raise AssertionError("expensive work ran before F5 audio preflight")

    monkeypatch.setattr(tts_f5, "check_f5_audio_runtime", check)
    monkeypatch.setattr(dub.registry, "load_builtins", unexpected)
    monkeypatch.setattr(dub, "_decode_file", unexpected)
    monkeypatch.setattr(dub, "separate_speech", unexpected)

    with pytest.raises(RuntimeError) as caught:
        await dub.render_dub_track(cfg, source)

    assert caught.value is failure
    assert expensive_calls == []


async def test_non_f5_route_does_not_require_f5_audio_runtime(tmp_path, monkeypatch):
    import uvt.dub as dub

    source = tmp_path / "input.wav"
    source.write_bytes(b"probe ends before decoding")
    cfg = AppConfig()
    cfg.tts.engine = "dummy"
    called_checks = []

    class ReachedRegistry(Exception):
        pass

    def check():
        called_checks.append(True)
        raise AssertionError("non-F5 route must not depend on TorchCodec")

    def stop_at_registry():
        raise ReachedRegistry

    monkeypatch.setattr(tts_f5, "check_f5_audio_runtime", check)
    monkeypatch.setattr(dub.registry, "load_builtins", stop_at_registry)

    with pytest.raises(ReachedRegistry):
        await dub.render_dub_track(cfg, source)
    assert called_checks == []


@pytest.mark.parametrize(
    ("configured_seed", "random_seed"),
    [(None, 0), (None, 2**32 - 1), (0, None), (2**32 - 1, None)],
)
async def test_inference_seed_is_valid_for_child_python(
    tmp_path, monkeypatch, configured_seed, random_seed
):
    import asyncio
    import os
    import secrets
    import subprocess

    import numpy as np

    requested_bits = []
    inferred_seeds = []

    def randbits(bits):
        requested_bits.append(bits)
        assert random_seed is not None, "an explicit seed must be preserved"
        return random_seed

    def infer(**kwargs):
        seed = kwargs["seed"]
        inferred_seeds.append(seed)
        # F5 sets PYTHONHASHSEED before starting its FFmpeg/Python children.
        # Exercise Python's actual seed validation without changing this process.
        child = subprocess.run(
            [sys.executable, "-c", "print('child-ready')"],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert child.returncode == 0, child.stderr
        assert child.stdout.strip() == "child-ready"
        return np.ones(128, dtype=np.float32), 24000, None

    monkeypatch.setattr(secrets, "randbits", randbits)
    engine = tts_f5.F5TTSEngine(TTSConfig(engine="f5"))
    engine._seed = configured_seed
    engine._model = SimpleNamespace(infer=infer)
    engine._min_slot_s = 0.35
    engine._speed = 1.0
    engine._nfe_step = 8
    engine._cfg_strength = 2.0
    engine._lock = asyncio.Lock()
    monkeypatch.setattr(
        engine, "_resolve_reference", lambda _ref: (tmp_path / "reference.wav", "reference", 0.5)
    )

    audio, sample_rate = await engine.synthesize_slot("A neutral sentence.", "en")

    assert inferred_seeds == [configured_seed if configured_seed is not None else random_seed]
    assert requested_bits == ([32] if configured_seed is None else [])
    assert audio.shape == (128,)
    assert sample_rate == 24000


@pytest.mark.parametrize("seed", [-1, 2**32, 0, 2**32 - 1])
async def test_warmup_validates_configured_seed_before_building_model(monkeypatch, seed):
    built_models = []

    def build(**kwargs):
        built_models.append(kwargs)
        return object()

    monkeypatch.setattr(tts_f5, "check_f5_audio_runtime", lambda: None)
    monkeypatch.setitem(sys.modules, "f5_tts", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "f5_tts.api", SimpleNamespace(F5TTS=build))
    engine = tts_f5.F5TTSEngine(TTSConfig(engine="f5", seed=seed))
    monkeypatch.setattr(engine, "_hub_files", lambda: (None, None))
    monkeypatch.setattr(engine, "_optional_path", lambda _key: None)
    monkeypatch.setattr(engine, "_resolve_device", lambda: "cpu")

    if not 0 <= seed <= 2**32 - 1:
        with pytest.raises(RuntimeError, match="F5-TTS.*seed"):
            await engine.warmup()
        assert built_models == []
    else:
        try:
            await engine.warmup()
            assert engine._seed == seed
            assert len(built_models) == 1
        finally:
            if hasattr(engine, "_tmp"):
                engine._tmp.cleanup()
