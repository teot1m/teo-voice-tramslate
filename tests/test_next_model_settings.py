"""New local stacks must select the requested models and their own voices."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from uvt.config import AppConfig, load_config
from uvt.server import DubServer
from uvt.server_settings import ServerSettingsStore, normalize_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("UVT_CACHE", str(tmp_path / "cache"))
    piper = AppConfig()
    piper.plugin_dirs = []
    piper.stt.engine = "dummy"
    piper.translation.engine = "dummy"
    piper.tts.engine = "piper"
    piper.tts.voice_dir = str(tmp_path / "voices")
    piper.tts.voice_models = {"ru:male": "ru-male.onnx", "ru:female": "ru-female.onnx",
                               "uk:male": "uk-male.onnx", "uk:female": "uk-female.onnx"}
    Path(piper.tts.voice_dir).mkdir()
    for name in piper.tts.voice_models.values():
        path = Path(piper.tts.voice_dir) / name
        path.write_bytes(b"test")
        Path(str(path) + ".json").write_text('{"audio":{"sample_rate":22050}}')
    moss = piper.model_copy(deep=True)
    moss.tts.engine = "moss-onnx"
    moss.tts.model_path = str(tmp_path / "moss")
    moss.tts.codec_path = str(tmp_path / "codec")
    for directory, files in ((moss.tts.model_path, ["browser_poc_manifest.json", "tokenizer.model",
                 "moss_tts_global_shared.data", "moss_tts_local_shared.data"]),
               (moss.tts.codec_path, ["codec_browser_onnx_meta.json", "moss_audio_tokenizer_encode.onnx",
                 "moss_audio_tokenizer_decode_full.onnx"])):
        Path(directory).mkdir()
        for name in files:
            (Path(directory) / name).write_bytes(b"test")
    moss.translation.engine = "hymt-mlx"
    return {"local-balanced": piper, "local-moss": moss}


def make_server(profiles, **options):
    return DubServer(profiles["local-balanced"], profile_name="local-balanced",
                     selectable_profiles=profiles, settings_key="free", **options)


def test_profile_catalog_owns_voices_and_supported_languages(profiles):
    server = make_server(profiles)
    catalog = {item["id"]: item for item in server._profile_catalog()}
    moss = catalog["local-moss"]
    assert moss["engines"]["tts"] == "moss-onnx"
    assert {voice["id"] for voice in moss["voices"]} == {"Adam", "Nathan", "Ava", "Bella"}
    assert "ru" in moss["target_languages"]
    assert "uk" not in moss["target_languages"]
    assert {voice["id"] for voice in catalog["local-balanced"]["voices"]} == {
        "ru-male", "ru-female", "uk-male", "uk-female"}
    # A model path never leaves local config via the public catalog.
    assert str(Path(profiles["local-moss"].tts.model_path)) not in json.dumps(moss)


def test_selected_profile_reaches_job_config_and_exact_voice(profiles):
    server = make_server(profiles)
    cfg, name = server._config_for_request({"settings_mode": "override",
        "profile_id": "local-moss", "target_lang": "en", "voice_id": "Bella"})
    assert name == "local-moss"
    assert cfg.translation.engine == "hymt-mlx"
    assert cfg.tts.engine == "moss-onnx"
    assert cfg.tts.voice_id == "Bella"
    assert cfg.tts.voice_gender == "female"
    assert cfg.target_lang == "en"
    assert server.cfg.tts.engine == "piper"


def test_new_profile_rejects_previous_piper_voice(profiles):
    server = make_server(profiles)
    with pytest.raises(ValueError, match="недоступен"):
        server._config_for_request({"settings_mode": "override", "profile_id": "local-moss",
                                     "voice_id": "ru-female"})


def test_normalization_resets_implicit_old_voice_but_keeps_explicit_choice(profiles):
    server = make_server(profiles)
    current = {"profile_id": "local-balanced", "source_lang": "auto", "target_lang": "ru",
               "voice_id": "ru-female", "voice_gender": "female"}
    payload = {"profile_id": "local-moss"}
    normalized = normalize_settings(payload, current=current, kind="local",
        profile_ids=set(profiles), local_voices=server._settings_voice_catalog(payload))
    assert normalized["voice_id"] == ""
    normalized = normalize_settings({**payload, "voice_id": "Bella", "target_lang": "en"},
        current=current, kind="local", profile_ids=set(profiles),
        local_voices=server._settings_voice_catalog(payload))
    assert normalized["voice_id"] == "Bella"
    assert normalized["target_lang"] == "en"


def test_moss_settings_reject_uk(profiles):
    server = make_server(profiles)
    values = {"profile_id": "local-moss", "source_lang": "auto", "target_lang": "uk",
              "voice_id": "", "voice_gender": "auto"}
    with pytest.raises(ValueError, match="целевого языка"):
        normalize_settings(values, current=values, kind="local", profile_ids=set(profiles),
                           local_voices=server._settings_voice_catalog(values))
    cfg = profiles["local-moss"].model_copy(deep=True)
    cfg.target_lang = "uk"
    with pytest.raises(ValueError, match="украинского"):
        DubServer._validate_piper_voice_support(cfg)


def test_saved_moss_selection_restores_against_selected_profile(profiles):
    store = ServerSettingsStore.memory()
    store.set("free", {"profile_id": "local-moss", "source_lang": "auto", "target_lang": "en",
                       "voice_id": "Bella", "voice_gender": "female"}, expected_revision=0)
    server = make_server(profiles, settings_store=store)
    assert server.profile_name == "local-moss"
    assert server.cfg.tts.engine == "moss-onnx"
    assert server.cfg.tts.voice_id == "Bella"
    assert not server._settings_load_error
    document = server._settings_payload()
    assert {v["id"] for v in document["catalog"]["voices"]} == {"Adam", "Nathan", "Ava", "Bella"}
    assert "en" in {v["id"] for v in document["catalog"]["target_languages"]}
    assert "uk" not in {v["id"] for v in document["catalog"]["target_languages"]}


async def test_put_settings_changes_model_and_next_job(profiles, monkeypatch):
    server = make_server(profiles)
    monkeypatch.setattr(server, "_dashboard_request_allowed", lambda request: True)
    monkeypatch.setattr(server, "_require_profile_ready", lambda name: None)

    class Request:
        async def json(self):
            return {"revision": 0, "settings": {"profile_id": "local-moss",
                "source_lang": "auto", "target_lang": "ru", "voice_id": "Adam"}}

    response = await server._put_settings(Request())
    assert response.status == 200
    document = json.loads(response.text)
    assert document["effective"]["profile_id"] == "local-moss"
    assert document["engines"]["tts"] == "moss-onnx"
    cfg, name = server._config_for_request({"settings_mode": "server"})
    assert name == "local-moss"
    assert cfg.tts.voice_id == "Adam"
    assert cfg.translation.engine == "hymt-mlx"


def test_moss_preview_metadata_reports_own_multilingual_voices(profiles):
    server = make_server(profiles)
    meta = server._metadata_for_request({"profile_id": "local-moss", "target_lang": "en"})
    assert meta["capabilities"]["tts_preview"] is True
    assert meta["profile"]["kind"] == "local"
    assert meta["capabilities"]["voice_selection"] is True
    assert {v["id"] for v in meta["voices"]} == {"Adam", "Nathan", "Ava", "Bella"}


def test_researched_profiles_are_loadable_and_allowlisted():
    from uvt.server import _LOCAL_PROFILE_LABELS
    from uvt.cli import _PROFILE_OVERVIEW
    for name, stt, tts in (("local-hymt", "parakeet-mlx", "piper"),
                          ("local-moss", "parakeet-mlx", "moss-onnx"),
                          ("local-nemotron", "nemotron-mlx", "piper")):
        cfg = load_config(str(ROOT / "profiles" / (name + ".yaml")))
        assert cfg.stt.engine == stt
        assert cfg.translation.engine == "hymt-mlx"
        assert cfg.tts.engine == tts
        assert name in _LOCAL_PROFILE_LABELS
        assert name in dict(_PROFILE_OVERVIEW)


def test_new_engines_are_registered_without_loading_weights():
    from uvt import registry
    registry.load_builtins()
    assert "hymt-mlx" in registry.available("translation")
    assert "moss-onnx" in registry.available("tts")
    assert "nemotron-mlx" in registry.available("stt")


@pytest.mark.parametrize("codec_ready", [True, False])
def test_preflight_does_not_require_piper_for_moss_but_keeps_model_checks(profiles, codec_ready):
    profiles["local-balanced"].stt.engine = "parakeet-mlx"
    profiles["local-balanced"].translation.engine = "translategemma-mlx"
    profiles["local-moss"].stt.engine = "parakeet-mlx"
    server = make_server(profiles)
    server._record_local_preflight({
        "ready": False,
        "piper": {"ready": False},
        "models": {key: {"ready": key != "moss-codec" or codec_ready}
                   for key in ("parakeet", "translategemma", "hymt2", "moss-tts", "moss-codec")},
    })
    moss = server._profile_setup_readiness["local-moss"]
    assert moss["installed"] is codec_ready
    assert moss["missing"] == ([] if codec_ready else ["moss-codec"])
    piper = server._profile_setup_readiness["local-balanced"]
    assert piper["installed"] is False
    assert piper["missing"] == ["piper"]
    if codec_ready:
        server._require_profile_ready("local-moss")
    else:
        with pytest.raises(ValueError, match="moss-codec"):
            server._require_profile_ready("local-moss")


def test_additional_moss_voices_can_be_chosen_for_dialogue(profiles):
    server = make_server(profiles)
    cfg, _ = server._config_for_request({"settings_mode": "override", "profile_id": "local-moss", "target_lang": "ru", "male_voice_id": "Nathan", "female_voice_id": "Ava"})
    assert cfg.tts.male_voice_id == "Nathan"
    assert cfg.tts.female_voice_id == "Ava"
    assert cfg.tts.voice_gender == "auto"
