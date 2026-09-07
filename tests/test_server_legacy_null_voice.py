"""Legacy userscript auto-voice requests must use the selected local engine."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from aiohttp import web

from uvt.config import AppConfig, VoicePairConfig, configured_role_voice
from uvt.server import DubServer
from uvt.server_settings import ServerSettingsStore, effective_settings


VOICE_PAIRS = {
    "moss-onnx:ru": {"male_voice_id": "Adam", "female_voice_id": "Bella"},
    "piper:ru": {"male_voice_id": "ru-male", "female_voice_id": "ru-female"},
    "piper:uk": {"male_voice_id": "uk-male", "female_voice_id": "uk-female"},
}
VOICES = {
    "moss-onnx": [
        {"id": voice, "engine": "moss-onnx", "languages": ["ru", "en"],
         "gender": role, "installed": True}
        for voice, role in (("Adam", "male"), ("Bella", "female"))
    ],
    "piper": [
        {"id": f"{lang}-{role}", "engine": "piper", "language": lang,
         "gender": role, "installed": True}
        for lang in ("ru", "uk") for role in ("male", "female")
    ],
}


@pytest.fixture
def moss_server(tmp_path, monkeypatch):
    monkeypatch.setenv("UVT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(DubServer, "_voice_catalog", staticmethod(
        lambda cfg: copy.deepcopy(VOICES.get(cfg.tts.engine, []))))
    monkeypatch.setattr(DubServer, "_require_profile_ready", lambda *args: None)
    monkeypatch.setattr(DubServer, "_dashboard_request_allowed", lambda *args: True)
    moss = AppConfig()
    moss.plugin_dirs = []
    moss.source_lang = "en"
    moss.target_lang = "ru"
    moss.stt.engine = "dummy"
    moss.translation.engine = "dummy"
    moss.tts.engine = "moss-onnx"
    moss.tts.voice_id = "Adam"
    moss.tts.male_voice_id = "Adam"
    moss.tts.female_voice_id = "Bella"
    moss.tts.voice_pairs = {key: VoicePairConfig(**pair) for key, pair in VOICE_PAIRS.items()}
    nemotron = moss.model_copy(deep=True)
    nemotron.stt.engine = "nemotron-mlx"
    nemotron.tts.engine = "piper"
    nemotron.tts.voice_id = None
    nemotron.tts.male_voice_id = ""
    nemotron.tts.female_voice_id = ""
    nemotron.tts.voice_pairs = {}
    store = ServerSettingsStore.memory()
    store.set("free", effective_settings(moss, kind="local", profile_name="local-moss"))
    return DubServer(moss, profile_name="local-moss", settings_key="free",
        selectable_profiles={"local-moss": moss, "local-nemotron": nemotron},
        settings_store=store)


class JsonRequest:
    def __init__(self, data):
        self.data = data

    async def json(self):
        return self.data


def job_payload(**overrides):
    return {"settings_mode": "override", "profile_id": "local-nemotron",
            "source_lang": "en", "target_lang": "ru", "voice_gender": "auto",
            "voice_id": None, "page_url": "https://example.test/watch/demo", **overrides}


def assert_piper_pair(cfg):
    assert cfg.stt.engine == "nemotron-mlx"
    assert cfg.tts.engine == "piper"
    assert not cfg.tts.voice_id
    assert cfg.tts.voice_gender == "auto"
    assert configured_role_voice(cfg.tts, "male") == "ru-male"
    assert configured_role_voice(cfg.tts, "female") == "ru-female"
    assert cfg.tts.model_dump()["voice_pairs"] == VOICE_PAIRS


async def test_legacy_null_voice_job_uses_selected_engine_and_scoped_pair(moss_server, monkeypatch):
    seen = []

    async def run_job(job, data, cfg, profile):
        seen.append(cfg)

    monkeypatch.setattr(moss_server, "_run_job", run_job)
    response = await moss_server._post_dub(JsonRequest(job_payload()))
    assert response.status == 200
    payload = json.loads(response.text)
    job_id = payload["id"]
    assert payload["meta"]["profile"]["name"] == "local-nemotron"
    assert_piper_pair(moss_server._job_configs[job_id])
    await moss_server._tasks[job_id]
    assert_piper_pair(seen[0])
    assert moss_server.profile_name == "local-moss"
    assert moss_server.cfg.tts.voice_id == "Adam"
    assert moss_server.cfg.tts.model_dump()["voice_pairs"] == VOICE_PAIRS


def test_null_clears_single_voice_without_erasing_same_engine_pair(moss_server):
    cfg, name = moss_server._config_for_request(job_payload(profile_id="local-moss"))
    assert name == "local-moss"
    assert cfg.tts.voice_id is None
    assert configured_role_voice(cfg.tts, "female") == "Bella"
    assert moss_server.cfg.tts.voice_id == "Adam"


async def test_meta_query_previews_selected_profile_after_global_settings_saved(moss_server):
    assert moss_server._settings_saved
    response = await moss_server._get_meta(SimpleNamespace(query={
        "profile_id": "local-nemotron", "target_lang": "ru"}))
    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["profile"]["name"] == "local-nemotron"
    assert payload["profile"]["engines"]["stt"] == "nemotron-mlx"
    assert payload["profile"]["engines"]["tts"] == "piper"
    assert {voice["engine"] for voice in payload["voices"]} == {"piper"}
    assert moss_server.profile_name == "local-moss"
    explicit_server = await moss_server._get_meta(SimpleNamespace(query={
        "profile_id": "local-nemotron", "settings_mode": "server"}))
    assert json.loads(explicit_server.text)["profile"]["name"] == "local-moss"


@pytest.mark.parametrize("value", [{}, 1, [], False])
@pytest.mark.parametrize("endpoint", ["_post_dub", "_post_tts_preview"])
async def test_non_string_voice_values_still_get_422(moss_server, endpoint, value):
    with pytest.raises(web.HTTPUnprocessableEntity) as error:
        await getattr(moss_server, endpoint)(JsonRequest(job_payload(voice_id=value, text="Hello")))
    assert error.value.text == "voice_id должно быть строкой"


@pytest.mark.parametrize("value", [None, {}, 1, [], False])
async def test_global_voice_validation_remains_strict(moss_server, value):
    before = moss_server.settings_store.get_entry("free")
    with pytest.raises(web.HTTPUnprocessableEntity) as error:
        await moss_server._put_settings(JsonRequest({"revision": before["revision"],
                                                     "settings": {"voice_id": value}}))
    assert error.value.text == "voice_id должно быть строкой"
    assert moss_server.settings_store.get_entry("free") == before
    assert moss_server.cfg.tts.voice_id == "Adam"


async def test_null_voice_preview_uses_selected_piper_female_pair(moss_server, monkeypatch):
    import numpy as np
    from uvt import server as server_module

    seen = []

    class FakeTTS:
        async def warmup(self):
            pass

        async def synthesize(self, text, language):
            return np.full(3200, 0.1, dtype=np.float32), 16000

        async def close(self):
            pass

    def create(kind, engine, cfg):
        assert kind == "tts"
        assert engine == "piper"
        seen.append(cfg)
        return FakeTTS()

    monkeypatch.setattr(server_module.registry, "load_plugin_dirs", lambda *args: None)
    monkeypatch.setattr(server_module.registry, "create", create)
    response = await moss_server._post_tts_preview(JsonRequest(job_payload(
        voice_gender="female", text="A neutral voice preview.")))
    assert response.status == 200
    assert bytes(response.body).startswith(b"RIFF")
    assert len(seen) == 1
    assert not seen[0].voice_id
    assert configured_role_voice(seen[0]) == "ru-female"
    assert moss_server.cfg.tts.voice_id == "Adam"
