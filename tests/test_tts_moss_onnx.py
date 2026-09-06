"""Contract, local-only runtime, and cancellation tests for MOSS ONNX."""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from uvt.engines import tts_moss_onnx as moss
from uvt.interfaces import VoiceReference


class Tokenizer:
    def encode(self, text, out_type=int):
        return list(range(len(text)))


class Codec:
    def __init__(self):
        self.inputs = []

    def run(self, _, inputs):
        self.inputs.append(inputs)
        return [np.ones((1, 4, 16), dtype=np.int32), np.array([4], dtype=np.int32)]

    def get_outputs(self):
        return [SimpleNamespace(name=n) for n in ("audio_codes", "audio_code_lengths")]


class Runtime:
    def __init__(self):
        self.codec_meta = {"codec_config": {"sample_rate": 48000, "channels": 2}}
        self.sessions = {"codec_encode": Codec()}
        self.requests = []
        self.frames = [[1] * 16] * 3
        self.started = threading.Event()
        self.finished = threading.Event()
        self.slow = False

    def list_builtin_voices(self):
        return [{"voice": "Adam", "prompt_audio_codes": [[1] * 16]},
                {"voice": "Bella", "prompt_audio_codes": [[2] * 16]}]

    def build_voice_clone_request_rows(self, codes, tokens):
        request = (codes, tokens)
        self.requests.append(request)
        return request

    def generate_audio_frames(self, request, on_frame=None):
        self.started.set()
        try:
            for i in range(100 if self.slow else 3):
                if self.slow:
                    time.sleep(0.002)
                if on_frame:
                    on_frame([], i, [1] * 16)
            return self.frames
        finally:
            self.finished.set()

    def decode_full_audio(self, frames):
        return [np.full(480, .1), np.full(480, .3)], 480


@pytest.fixture
def setup_engine(tmp_path, monkeypatch):
    model, codec = tmp_path / "tts", tmp_path / "codec"
    model.mkdir()
    codec.mkdir()
    for file in (model / "browser_poc_manifest.json", model / "tokenizer.model",
                 codec / "codec_browser_onnx_meta.json"):
        file.write_text("test")
    runtime = Runtime()
    calls = []

    def build(model_dir, codec_dir, **options):
        calls.append((model_dir, codec_dir, options))
        return runtime, Tokenizer()

    monkeypatch.setattr(moss, "_build_runtime", build)

    def make(**options):
        cfg = dict(model_path=str(model), codec_path=str(codec), voice_gender="auto")
        cfg.update(options)
        return moss.MossOnnxTTS(SimpleNamespace(**cfg))

    return make, runtime, calls


async def test_builtin_voice_local_runtime_and_mono_audio(setup_engine):
    make, runtime, calls = setup_engine
    engine = make(voice_gender="female")
    await engine.warmup()
    audio, rate = await engine.synthesize("Привет!", "ru-RU")
    assert rate == 48000
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    np.testing.assert_allclose(audio, .2)
    assert runtime.requests[0][0] == [[2] * 16]
    assert calls[0][2] == {"thread_count": 4, "max_new_frames": 375, "sample_mode": "fixed"}
    await engine.close()
    assert engine._runtime is None


async def test_missing_models_fail_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(moss, "_build_runtime", lambda *a, **k: pytest.fail("loaded missing model"))
    engine = moss.MossOnnxTTS(SimpleNamespace(model_path=str(tmp_path), codec_path=str(tmp_path)))
    with pytest.raises(RuntimeError, match="setup-mac-local --preset moss"):
        await engine.warmup()


async def test_unsupported_uk_fails_before_inference(setup_engine):
    make, runtime, _ = setup_engine
    engine = make()
    await engine.warmup()
    with pytest.raises(RuntimeError, match="украинского.*Piper"):
        await engine.synthesize("Привіт", "uk")
    assert not runtime.requests


async def test_invalid_voice_fails_before_job(setup_engine):
    make, _, _ = setup_engine
    engine = make(voice_id="missing")
    with pytest.raises(RuntimeError, match="неизвестный голос"):
        await engine.warmup()


async def test_reference_resampling_caching_and_correct_field(setup_engine):
    make, runtime, _ = setup_engine
    engine = make()
    await engine.warmup()
    reference = VoiceReference(np.ones(16000, dtype=np.float32) * .1, 16000, "speaker")
    await engine.synthesize_slot("Привет!", "ru", reference=reference)
    await engine.synthesize_slot("Пока!", "ru", reference=reference)
    codec = runtime.sessions["codec_encode"]
    assert len(codec.inputs) == 1
    assert codec.inputs[0]["waveform"].shape == (1, 2, 48000)
    assert codec.inputs[0]["input_lengths"].tolist() == [48000]
    assert len(runtime.requests[0][0]) == 4


async def test_explicit_voice_wins_over_clone(setup_engine):
    make, runtime, _ = setup_engine
    engine = make(voice_id="Bella")
    await engine.warmup()
    reference = VoiceReference(np.ones(16000, dtype=np.float32), 16000)
    await engine.synthesize_slot("Привет!", "ru", reference=reference)
    assert runtime.requests[0][0] == [[2] * 16]
    assert not runtime.sessions["codec_encode"].inputs


async def test_soundfile_reference_avoids_torchcodec(setup_engine, tmp_path):
    import soundfile as sf
    make, runtime, _ = setup_engine
    path = tmp_path / "reference.wav"
    sf.write(path, np.ones((16000, 2)) * .1, 16000)
    engine = make(reference_wav=str(path))
    await engine.warmup()
    await engine.synthesize("Привет!", "ru")
    assert runtime.sessions["codec_encode"].inputs[0]["waveform"].shape == (1, 2, 48000)


@pytest.mark.parametrize("samples,rate", [(np.zeros(10), 16000), (np.zeros(16000), 0),
                         (np.full(16000, np.nan), 16000), (np.zeros((2, 16000)), 16000)])
async def test_invalid_reference_rejected(setup_engine, samples, rate):
    make, runtime, _ = setup_engine
    engine = make()
    await engine.warmup()
    with pytest.raises(ValueError):
        await engine.synthesize_slot("Привет", "ru", reference=VoiceReference(samples, rate))
    assert not runtime.requests


def test_text_chunks_keep_all_content_and_token_budget():
    text = "Одна длинная фраза. И ещё одна фраза, которую нужно сохранить!"
    chunks = moss._text_chunks(text, Tokenizer(), 20)
    assert " ".join(chunks) == text
    assert all(len(Tokenizer().encode(c)) <= 20 for c in chunks)
    assert moss._text_chunks("  ", Tokenizer(), 20) == []
    chinese = "这是一段没有空格的长句子用于检查切分文本"
    assert "".join(moss._text_chunks(chinese, Tokenizer(), 8)) == chinese


async def test_long_output_never_silently_truncated(setup_engine):
    make, runtime, _ = setup_engine
    engine = make(max_new_frames=25)
    await engine.warmup()
    runtime.frames = [[1] * 16] * 25
    with pytest.raises(RuntimeError, match="предела реплики"):
        await engine.synthesize("Привет", "ru")


async def test_cancellation_drains_native_worker_before_close(setup_engine):
    make, runtime, _ = setup_engine
    engine = make()
    await engine.warmup()
    runtime.slow = True
    task = asyncio.create_task(engine.synthesize("Привет", "ru"))
    await asyncio.to_thread(runtime.started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.finished.is_set()
    await engine.close()
    assert engine._runtime is None


async def test_saved_role_reference_overrides_original_speaker(setup_engine, monkeypatch, tmp_path):
    import soundfile as sf
    import uvt.voice_references as library
    path = tmp_path / "saved.wav"
    sf.write(path, np.full(48000, .2, dtype=np.float32), 24000)
    selected = "ref_" + "a" * 32
    monkeypatch.setattr(library, "resolve_reference", lambda voice_id: (path, "Neutral sample") if voice_id == selected else (_ for _ in ()).throw(ValueError()))
    make, runtime, _ = setup_engine
    engine = make(voice_gender="female", female_voice_id=selected)
    await engine.warmup()
    try:
        assert engine._resolve_voice() == selected
        codes = engine._reference_codes(VoiceReference(np.full(48000, .8, dtype=np.float32), 48000, label="original"))
        assert codes and runtime.sessions["codec_encode"].inputs
        actual = runtime.sessions["codec_encode"].inputs[-1]
        audio = next(value for value in actual.values() if isinstance(value, np.ndarray) and value.dtype.kind == "f")
        assert float(audio.mean()) == pytest.approx(.2, abs=.005)
    finally:
        await engine.close()
