"""A selected role pair reaches every TTS backend without loading its model."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from uvt.config import TTSConfig, configured_role_voice
from uvt.engines.tts_edge import EdgeTTS
from uvt.engines.tts_elevenlabs import ElevenLabsTTS
from uvt.engines.tts_f5 import F5TTSEngine
from uvt.engines.tts_kokoro import KokoroTTS
from uvt.engines.tts_moss_onnx import MossOnnxTTS
from uvt.engines.tts_openai import OpenAITTS
from uvt.engines.tts_piper import PiperTTS
from uvt.interfaces import VoiceReference
from uvt.services.speaker import SpeakerAssignment
from uvt.services.tts import TTSService


def test_pair_config_round_trip_and_scope():
    data = {
        "voice_pairs": {
            "piper:ru": {"male_voice_id": "ru-male", "female_voice_id": "ru-female"},
            "piper:uk": {"female_voice_id": "uk-female"},
            "elevenlabs:ru": {"male_voice_id": "remote-male"},
        }
    }
    cfg = TTSConfig.model_validate(data)
    restored = TTSConfig.model_validate(cfg.model_dump())
    assert restored.voice_pairs["piper:ru"].male_voice_id == "ru-male"
    assert restored.voice_pairs["piper:uk"].female_voice_id == "uk-female"
    # Merely saving a pair for a different route/language does not apply it.
    assert configured_role_voice(cfg) is None


@pytest.mark.parametrize("data", [
    {"voice_pairs": {"piper": {"male_voice_id": "x"}}},
    {"voice_pairs": {"piper:ru": {"male_voice_id": 12}}},
    {"voice_pairs": {"piper:ru": {"other": "x"}}},
    {"male_voice_id": 12},
    {"female_voice_id": "x" * 201},
])
def test_pair_config_rejects_invalid_shape(data):
    with pytest.raises(ValidationError):
        TTSConfig.model_validate(data)


@pytest.mark.parametrize("engine_type,resolve", [
    (OpenAITTS, lambda engine: engine._voice()),
    (ElevenLabsTTS, lambda engine: engine._voice_id()),
    (EdgeTTS, lambda engine: engine._voice("ru")),
    (KokoroTTS, lambda engine: engine._voice("en")),
])
def test_cloud_and_catalog_engines_use_role_pair_and_preserve_single_voice(engine_type, resolve):
    cfg = TTSConfig(male_voice_id="chosen-low", female_voice_id="chosen-high")
    engine = engine_type(cfg)
    cfg.voice_gender = "male"
    assert resolve(engine) == "chosen-low"
    cfg.voice_gender = "female"
    assert resolve(engine) == "chosen-high"
    cfg.voice = "cloud"
    assert resolve(engine) == "chosen-high"
    cfg.voice = "explicit-one-voice"
    assert resolve(engine) == "explicit-one-voice"


def test_eleven_pair_overrides_environment_and_missing_role_retains_environment(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "shared-env")
    monkeypatch.setenv("ELEVENLABS_MALE_VOICE_ID", "male-env")
    monkeypatch.setenv("ELEVENLABS_FEMALE_VOICE_ID", "female-env")
    cfg = TTSConfig(male_voice_id="chosen-male", voice_gender="male")
    engine = ElevenLabsTTS(cfg)
    assert engine._voice_id() == "chosen-male"
    cfg.voice_gender = "female"
    assert engine._voice_id() == "female-env"
    cfg.voice = "single"
    assert engine._voice_id() == "single"


def test_missing_role_uses_existing_default():
    cfg = TTSConfig(female_voice_id="chosen-female", voice_gender="male")
    assert OpenAITTS(cfg)._voice() == "onyx"
    assert EdgeTTS(cfg)._voice("ru") == "ru-RU-DmitryNeural"


def test_piper_pair_resolves_allowlisted_model_and_enforces_language():
    cfg = TTSConfig(male_voice_id="ru-other-low", female_voice_id="ru-other-high")
    engine = PiperTTS(cfg)
    engine._legacy_model = None
    engine._voice_models = {
        "ru:male": Path("ru-default-low.onnx"),
        "ru:female": Path("ru-default-high.onnx"),
        "ru:custom1": Path("ru-other-low.onnx"),
        "ru:custom2": Path("ru-other-high.onnx"),
        "uk:female": Path("uk-female.onnx"),
    }
    cfg.voice_gender = "male"
    assert engine._resolve_model("ru-RU").stem == "ru-other-low"
    cfg.voice_gender = "female"
    assert engine._resolve_model("ru").stem == "ru-other-high"
    cfg.voice_id = "ru-default-low"
    assert engine._resolve_model("ru").stem == "ru-default-low"
    cfg.voice_id = None
    cfg.female_voice_id = "uk-female"
    with pytest.raises(RuntimeError, match="не подходит"):
        engine._resolve_model("ru")
    cfg.female_voice_id = "../not-allowlisted"
    with pytest.raises(RuntimeError, match="отсутствует"):
        engine._resolve_model("ru")


@pytest.mark.parametrize("role,expected", [("male", "Bella"), ("female", "Adam")])
def test_moss_pair_overrides_source_cloning_and_single_voice_wins(role, expected):
    # Gender labels are roles: a user may choose any available timbre for either.
    cfg = TTSConfig(male_voice_id="Bella", female_voice_id="Adam", voice_gender=role)
    engine = MossOnnxTTS(cfg)
    engine._voices = {"Adam": [[1]], "Bella": [[2]]}
    engine._reference_path = None
    reference = VoiceReference(np.zeros(24000, dtype=np.float32), 24000)
    assert engine._resolve_voice() == expected
    assert engine._reference_codes(reference) == engine._voices[expected]
    cfg.voice_id = "Bella"
    assert engine._reference_codes(reference) == [[2]]


def test_moss_invalid_pair_fails_instead_of_silently_cloning():
    engine = MossOnnxTTS(TTSConfig(male_voice_id="unknown"))
    engine._voices = {"Adam": [[1]], "Bella": [[2]]}
    with pytest.raises(RuntimeError, match="неизвестный голос"):
        engine._resolve_voice()


@pytest.mark.asyncio
async def test_kokoro_normal_and_rated_synthesis_use_selected_pair():
    calls = []
    def create(text, **kwargs):
        calls.append(kwargs)
        return np.zeros(8, dtype=np.float32), 24000
    cfg = TTSConfig(male_voice_id="am_michael", female_voice_id="af_bella", voice_gender="male")
    engine = KokoroTTS(cfg)
    engine._kokoro = SimpleNamespace(create=create)
    await engine.synthesize("Hello", "en")
    cfg.voice_gender = "female"
    await engine.synthesize_rated("Hello", "en", 1.2)
    assert [call["voice"] for call in calls] == ["am_michael", "af_bella"]
    assert calls[1]["speed"] == 1.2


def test_f5_pair_resolves_owned_reference_before_source_cloning(tmp_path, monkeypatch):
    import soundfile as sf
    path = tmp_path / "reference.wav"
    sf.write(path, np.zeros(24000, dtype=np.float32), 24000)
    requested = []
    def resolve(voice_id):
        requested.append(voice_id)
        return path, "Sample text."
    monkeypatch.setitem(sys.modules, "uvt.voice_references", SimpleNamespace(resolve_reference=resolve))
    cfg = TTSConfig(male_voice_id="ref_low", female_voice_id="ref_high", voice_gender="female")
    engine = F5TTSEngine(cfg)
    engine._store_reference = lambda _: pytest.fail("selected pair must override source cloning")
    reference = VoiceReference(np.zeros(24000, dtype=np.float32), 24000)
    assert engine._resolve_reference(reference) == (path, "Sample text.", 1.0)
    cfg.voice_gender = "male"
    engine._resolve_reference(reference)
    cfg.voice_id = "ref_explicit"
    engine._resolve_reference(reference)
    assert requested == ["ref_high", "ref_low", "ref_explicit"]


def test_f5_without_role_pair_keeps_source_cloning():
    engine = F5TTSEngine(TTSConfig())
    reference = VoiceReference(np.zeros(24000, dtype=np.float32), 24000)
    expected = (Path("source.wav"), "Source text", 1.0)
    engine._store_reference = lambda supplied: expected if supplied is reference else None
    assert engine._resolve_reference(reference) == expected


def speaker_assignment(role="female", voice="legacy-speaker-voice"):
    return SpeakerAssignment("speaker-1", "high", 0.8, 220.0, role, voice)


def service_with_cfg(cfg):
    service = TTSService.__new__(TTSService)
    service.engine = SimpleNamespace(cfg=cfg)
    return service


def test_live_role_pair_overrides_speaker_voice_and_restores_context_after_error():
    cfg = TTSConfig(male_voice_id="chosen-low", female_voice_id="chosen-high")
    service = service_with_cfg(cfg)
    with pytest.raises(RuntimeError, match="synthesis failure"):
        with service._speaker_voice(speaker_assignment()):
            assert cfg.voice_gender == "female"
            assert cfg.voice == "auto"
            assert configured_role_voice(cfg) == "chosen-high"
            raise RuntimeError("synthesis failure")
    assert cfg.voice == "auto"
    assert cfg.voice_gender == "auto"


@pytest.mark.parametrize("overrides", [
    {"voice": "single-voice"}, {"voice_id": "single-local"}, {"voice_gender": "male"},
])
def test_live_explicit_single_voice_and_manual_role_remain_authoritative(overrides):
    cfg = TTSConfig(female_voice_id="pair-high", **overrides)
    original = cfg.model_dump()
    with service_with_cfg(cfg)._speaker_voice(speaker_assignment()):
        assert cfg.model_dump() == original


def test_live_missing_role_pair_keeps_speaker_voice_override():
    cfg = TTSConfig(male_voice_id="pair-low")
    with service_with_cfg(cfg)._speaker_voice(speaker_assignment()):
        assert cfg.voice == "legacy-speaker-voice"
        assert cfg.voice_gender == "female"
    assert cfg.voice == "auto"
    assert cfg.voice_gender == "auto"
