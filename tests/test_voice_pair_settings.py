"""Remembered voice pairs survive profile, language and persistence boundaries."""
from __future__ import annotations

import pytest

from uvt.config import AppConfig
from uvt.server_settings import (
    ServerSettingsStore, apply_settings, effective_settings, normalize_settings,
)


PIPER = [
    {"id": f"{lang}-{gender}", "engine": "piper", "language": lang,
     "gender": gender, "installed": True}
    for lang in ("ru", "uk") for gender in ("male", "female")
]
MOSS = [
    {"id": name, "engine": "moss-onnx", "languages": ["ru", "en"],
     "gender": gender, "installed": True}
    for name, gender in (("Adam", "male"), ("Bella", "female"))
]
PROFILES = {"local-balanced", "local-moss"}


def current_local():
    cfg = AppConfig()
    cfg.tts.engine = "piper"
    return effective_settings(cfg, kind="local", profile_name="local-balanced")


def normalize(payload, current=None, *, engine="piper", voices=None, **options):
    return normalize_settings(
        payload, current=current if current is not None else current_local(),
        kind="local", profile_ids=PROFILES, local_voices=voices or PIPER,
        voice_engine=engine, **options,
    )


def test_pair_switch_piper_moss_piper_and_persistence(tmp_path):
    selected = normalize({"male_voice_id": "ru-male", "female_voice_id": "ru-female"})
    moss = normalize({"profile_id": "local-moss"}, selected, engine="moss-onnx", voices=MOSS)
    assert moss["male_voice_id"] == moss["female_voice_id"] == ""
    moss = normalize({"male_voice_id": "Adam", "female_voice_id": "Bella"},
                     moss, engine="moss-onnx", voices=MOSS)
    store = ServerSettingsStore(tmp_path / "settings.json", persistent=True)
    store.set("free", moss, expected_revision=0)
    reloaded = ServerSettingsStore(tmp_path / "settings.json", persistent=True).get("free")
    selected = normalize({"profile_id": "local-balanced"}, reloaded)
    assert selected["male_voice_id"] == "ru-male"
    assert selected["female_voice_id"] == "ru-female"
    assert selected["voice_pairs"]["moss-onnx:ru"] == {
        "male_voice_id": "Adam", "female_voice_id": "Bella",
    }


def test_language_switch_remembers_separate_pair():
    ru = normalize({"male_voice_id": "ru-male", "female_voice_id": "ru-female"})
    uk = normalize({"target_lang": "uk"}, ru)
    assert uk["male_voice_id"] == uk["female_voice_id"] == ""
    uk = normalize({"male_voice_id": "uk-male"}, uk)
    restored = normalize({"target_lang": "ru"}, uk)
    assert restored["female_voice_id"] == "ru-female"
    assert restored["voice_pairs"]["piper:uk"]["male_voice_id"] == "uk-male"


def test_pair_changes_clear_implicit_single_voice_and_preserve_gender_mode():
    old = {**current_local(), "voice_id": "ru-female", "voice_gender": "auto"}
    updated = normalize({"male_voice_id": "ru-male"}, old)
    assert updated["voice_id"] == ""
    assert updated["voice_gender"] == "auto"
    explicit = normalize({"male_voice_id": "ru-male", "voice_id": "ru-female"}, old)
    assert explicit["voice_id"] == "ru-female"
    assert explicit["voice_gender"] == "female"


def test_roles_can_use_any_installed_voice_without_forcing_gender():
    selected = normalize({"male_voice_id": "ru-female", "female_voice_id": "ru-male"})
    assert selected["voice_gender"] == "auto"
    assert selected["male_voice_id"] == "ru-female"


def test_explicit_blank_restores_engine_default_and_preserves_other_role():
    selected = normalize({"male_voice_id": "ru-male", "female_voice_id": "ru-female"})
    selected = normalize({"male_voice_id": "auto"}, selected)
    assert selected["male_voice_id"] == ""
    assert selected["female_voice_id"] == "ru-female"
    assert selected["voice_pairs"]["piper:ru"]["male_voice_id"] == ""


def test_map_payload_merges_other_engines_and_supports_unsaved_scope_edits():
    selected = normalize({"male_voice_id": "ru-male"})
    updated = normalize({"profile_id": "local-moss", "voice_pairs": {
        "piper:ru": {"male_voice_id": "ru-female"},
        "moss-onnx:ru": {"female_voice_id": "Bella"},
    }}, selected, engine="moss-onnx", voices=MOSS,
        voice_catalogs={"piper": PIPER})
    assert updated["female_voice_id"] == "Bella"
    restored = normalize({"profile_id": "local-balanced"}, updated)
    assert restored["male_voice_id"] == "ru-female"


def test_apply_settings_copies_selected_profile_and_effective_round_trip():
    cfg = AppConfig()
    cfg.tts.engine = "piper"
    moss = cfg.model_copy(deep=True)
    moss.tts.engine = "moss-onnx"
    selected = normalize({"profile_id": "local-moss", "male_voice_id": "Adam", "female_voice_id": "Bella"},
                         engine="moss-onnx", voices=MOSS)
    applied, name = apply_settings(cfg, selected, kind="local", base_profile_name="local-balanced",
                                  profiles={"local-balanced": cfg, "local-moss": moss})
    assert name == "local-moss"
    assert applied.tts.male_voice_id == "Adam"
    assert applied.tts.female_voice_id == "Bella"
    assert not getattr(cfg.tts, "male_voice_id", None)
    effective = effective_settings(applied, kind="local", profile_name=name)
    assert effective == selected
    # Persisted documents include readonly engine metadata; it is checked against cfg.
    assert normalize(effective, engine="moss-onnx", voices=MOSS) == effective


@pytest.mark.parametrize("payload", [
    {"voice_pairs": []}, {"voice_pairs": None}, {"voice_pairs": {"piper": {}}},
    {"voice_pairs": {"../f5:ru": {}}}, {"voice_pairs": {"piper:auto": {}}},
    {"voice_pairs": {"piper:ru": []}},
    {"voice_pairs": {"piper:ru": {"api_key": "secret"}}},
    {"voice_pairs": {"piper:ru": {"male_voice_id": ["ru-male"]}}},
    {"voice_pairs": {"piper:ru": {"male_voice_id": "../secret"}}},
    {"male_voice_id": 12}, {"female_voice_id": None},
    {"voice_engine": "moss-onnx"},
])
def test_malformed_pairs_rejected(payload):
    with pytest.raises(ValueError):
        normalize(payload)


@pytest.mark.parametrize("voice", ["unknown", "uk-male"])
def test_selected_local_pairs_validate_installed_voice_and_language(voice):
    with pytest.raises(ValueError):
        normalize({"male_voice_id": voice})


def test_inactive_pair_validates_against_supplied_catalog():
    with pytest.raises(ValueError, match="не установлен"):
        normalize({"voice_pairs": {"moss-onnx:ru": {"female_voice_id": "missing"}}},
                  voice_catalogs={"moss-onnx": MOSS})


def cloud_current(engine="openai"):
    cfg = AppConfig()
    cfg.tts.engine = engine
    cfg.stt.model = "gpt-4o-mini-transcribe"
    cfg.translation.model = "gpt-4o-mini"
    cfg.tts.model = "gpt-4o-mini-tts" if engine == "openai" else "eleven_turbo_v2_5"
    cfg.tts.voice = "onyx" if engine == "openai" else "SomeExplicitID"
    return cfg, effective_settings(cfg, kind=engine, profile_name="cloud")


def cloud_normalize(payload, *, engine="openai", current=None):
    return normalize_settings(payload, current=current or cloud_current(engine)[1],
                              kind=engine, profile_ids=set(), local_voices=[], voice_engine=engine)


def test_openai_pair_uses_model_allowlist_and_clears_legacy_override():
    selected = cloud_normalize({"male_voice_id": "onyx", "female_voice_id": "coral"})
    assert selected["tts_voice"] == "auto"
    assert selected["voice_gender"] == "auto"
    cfg, _ = cloud_current()
    cfg, _ = apply_settings(cfg, selected, kind="openai", base_profile_name="cloud", profiles={})
    assert cfg.tts.male_voice_id == "onyx"
    assert cfg.tts.voice == "auto"
    with pytest.raises(ValueError, match="OpenAI"):
        cloud_normalize({"male_voice_id": "made-up"})
    with pytest.raises(ValueError, match="OpenAI"):
        cloud_normalize({"tts_model": "tts-1", "male_voice_id": "ballad"})
    explicit = cloud_normalize({"male_voice_id": "onyx", "tts_voice": "coral"})
    assert explicit["tts_voice"] == "coral"


def test_eleven_pair_accepts_public_ids_and_auto_without_api_request():
    selected = cloud_normalize({"male_voice_id": "ErXwobaYiN019PkySvjV", "female_voice_id": "EXAVITQu4vr4xnSDxMaL"}, engine="elevenlabs")
    assert selected["tts_voice"] == "auto"
    selected = cloud_normalize({"female_voice_id": "auto"}, engine="elevenlabs", current=selected)
    assert selected["female_voice_id"] == ""
    with pytest.raises(ValueError):
        cloud_normalize({"female_voice_id": "https://evil.example/a"}, engine="elevenlabs")


def test_f5_blank_pair_keeps_clone_and_fixed_reference_uses_catalog():
    ref_id = "ref_" + "a" * 32
    voices = [
        {"id": "", "engine": "f5", "language": "ru", "installed": True},
        {"id": ref_id, "engine": "f5", "language": "ru", "installed": True},
    ]
    result = normalize({"male_voice_id": "", "female_voice_id": ref_id}, engine="f5", voices=voices)
    assert result["male_voice_id"] == ""
    assert result["female_voice_id"] == ref_id
    with pytest.raises(ValueError):
        normalize({"voice_pairs": {"f5:ru": {"male_voice_id": "bad-id"}}})


def test_cloud_eleven_catalog_is_not_local_installation_requirement():
    selected = normalize_settings(
        {"female_voice_id": "EXAVITQu4vr4xnSDxMaL"},
        current=cloud_current("elevenlabs")[1], kind="elevenlabs", profile_ids=set(),
        local_voices=[], voice_engine="elevenlabs", voice_catalogs={"elevenlabs": []},
    )
    assert selected["female_voice_id"] == "EXAVITQu4vr4xnSDxMaL"


def test_metadata_can_report_unsupported_language_but_fixed_voice_stays_validated():
    selected = normalize({"target_lang": "de"}, validate_target_availability=False)
    assert selected["target_lang"] == "de"
    with pytest.raises(ValueError, match="целевого языка"):
        normalize({"target_lang": "de", "male_voice_id": "ru-male"},
                  validate_target_availability=False)


@pytest.mark.parametrize("engine", ["piper", "f5"])
def test_empty_catalog_keeps_legacy_model_and_source_cloning(engine):
    selected = normalize_settings({}, current=current_local(), kind="local",
                                  profile_ids=PROFILES, local_voices=[], voice_engine=engine)
    assert selected["male_voice_id"] == selected["female_voice_id"] == ""


def test_f5_original_clone_language_does_not_depend_on_saved_sample_language():
    ref_id = "ref_" + "b" * 32
    catalog = [{"id": ref_id, "engine": "f5", "language": "ru", "installed": True}]
    selected = normalize({"target_lang": "uk"}, engine="f5", voices=catalog)
    assert selected["female_voice_id"] == ""
    with pytest.raises(ValueError, match="целевого языка"):
        normalize({"target_lang": "uk", "female_voice_id": ref_id}, engine="f5", voices=catalog)
