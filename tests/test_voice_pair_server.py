"""Global voice choices reach real server request/config/cache boundaries."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from uvt.config import AppConfig, configured_role_voice
from uvt.server import DubServer
from uvt.server_settings import ServerSettingsStore


VOICES = {
    "piper": [
        {"id": f"{lang}-{gender}", "engine": "piper", "language": lang,
         "gender": gender, "installed": True}
        for lang in ("ru", "uk") for gender in ("male", "female")
    ],
    "moss-onnx": [
        {"id": name, "engine": "moss-onnx", "languages": ["ru", "en"],
         "gender": gender, "installed": True}
        for name, gender in (("Adam", "male"), ("Bella", "female"))
    ],
}


@pytest.fixture
def server_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("UVT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(DubServer, "_voice_catalog", staticmethod(
        lambda cfg: copy.deepcopy(VOICES.get(cfg.tts.engine, []))))
    monkeypatch.setattr(DubServer, "_dashboard_request_allowed", lambda *args: True)
    monkeypatch.setattr(DubServer, "_require_profile_ready", lambda *args: None)

    def build(*, store=None, engine="piper"):
        cfg = AppConfig()
        cfg.plugin_dirs = []
        cfg.stt.engine = "dummy"
        cfg.translation.engine = "dummy"
        cfg.tts.engine = engine
        if engine in {"openai", "elevenlabs"}:
            cfg.stt.model = "gpt-4o-mini-transcribe"
            cfg.translation.model = "gpt-4o-mini"
            cfg.tts.model = "gpt-4o-mini-tts" if engine == "openai" else "eleven_turbo_v2_5"
            return DubServer(cfg, profile_name="cloud", settings_store=store or ServerSettingsStore.memory(),
                             settings_key=engine)
        cfg.tts.voice_models = {f"{lang}:{gender}": f"{lang}-{gender}.onnx"
                                for lang in ("ru", "uk") for gender in ("male", "female")}
        moss = cfg.model_copy(deep=True)
        moss.tts.engine = "moss-onnx"
        return DubServer(cfg, profile_name="local-balanced",
                         selectable_profiles={"local-balanced": cfg, "local-moss": moss},
                         settings_store=store or ServerSettingsStore.memory(), settings_key="free")

    return build


async def save(client, revision, settings):
    response = await client.put("/settings", json={"revision": revision, "settings": settings})
    assert response.status == 200, await response.text()
    return await response.json()


async def test_global_put_restart_engine_and_language_restore_pairs(server_factory, tmp_path):
    path = tmp_path / "defaults.json"
    store = ServerSettingsStore(path, persistent=True)
    server = server_factory(store=store)
    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        doc = await save(client, 0, {"male_voice_id": "ru-male", "female_voice_id": "ru-female"})
        assert doc["effective"]["voice_gender"] == "auto"
        cfg, _ = server._config_for_request({"settings_mode": "server"})
        assert (cfg.tts.male_voice_id, cfg.tts.female_voice_id) == ("ru-male", "ru-female")
        doc = await save(client, 1, {"profile_id": "local-moss", "male_voice_id": "Adam", "female_voice_id": "Bella"})
        assert doc["engines"]["tts"] == "moss-onnx"
    finally:
        await client.close()

    server = server_factory(store=ServerSettingsStore(path, persistent=True))
    assert server._settings_load_error is None
    assert server.profile_name == "local-moss"
    assert (server.cfg.tts.male_voice_id, server.cfg.tts.female_voice_id) == ("Adam", "Bella")
    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        doc = await save(client, 2, {"profile_id": "local-balanced"})
        assert doc["effective"]["female_voice_id"] == "ru-female"
        doc = await save(client, 3, {"target_lang": "uk", "male_voice_id": "uk-male", "female_voice_id": "uk-female"})
        assert doc["effective"]["target_lang"] == "uk"
        doc = await save(client, 4, {"target_lang": "ru"})
        assert doc["effective"]["male_voice_id"] == "ru-male"
        assert doc["effective"]["voice_pairs"]["piper:uk"]["female_voice_id"] == "uk-female"
        assert doc["effective"]["voice_pairs"]["moss-onnx:ru"]["female_voice_id"] == "Bella"
    finally:
        await client.close()


async def test_per_video_override_keeps_global_pair_and_legacy_one_voice(server_factory):
    server = server_factory()
    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        await save(client, 0, {"male_voice_id": "ru-male", "female_voice_id": "ru-female"})
        override, _ = server._config_for_request({"settings_mode": "override", "female_voice_id": "ru-male"})
        assert override.tts.female_voice_id == "ru-male"
        assert override.tts.male_voice_id == "ru-male"
        assert server.cfg.tts.female_voice_id == "ru-female"
        explicit, _ = server._config_for_request({"settings_mode": "override", "voice_id": "ru-female"})
        assert explicit.tts.voice_id == "ru-female"
        assert explicit.tts.voice_gender == "female"
        assert explicit.tts.male_voice_id == "ru-male"
        switched, _ = server._config_for_request({"settings_mode": "override", "profile_id": "local-moss", "female_voice_id": "Bella"})
        assert switched.tts.engine == "moss-onnx"
        assert switched.tts.female_voice_id == "Bella"
        assert server.cfg.tts.engine == "piper"
    finally:
        await client.close()


@pytest.mark.parametrize("engine,male,female", [
    ("piper", "ru-male", "ru-female"), ("moss-onnx", "Adam", "Bella"),
    ("openai", "onyx", "coral"),
    ("elevenlabs", "ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL"),
])
async def test_gender_preview_uses_pair_and_cache_distinguishes_changed_pair(
    server_factory, monkeypatch, engine, male, female,
):
    from uvt import server as server_module

    calls = []

    class FakeTTS:
        def __init__(self, config):
            self.config = config
        async def warmup(self):
            pass
        async def synthesize(self, text, language):
            calls.append((self.config.engine, self.config.voice_gender,
                          configured_role_voice(self.config), text))
            return np.array([0.0, 0.1, -0.1], dtype=np.float32), 24000
        async def close(self):
            pass

    monkeypatch.setattr(server_module.registry, "create", lambda kind, name, cfg: FakeTTS(cfg))
    server = server_factory(engine=engine if engine in {"openai", "elevenlabs"} else "piper")
    client = TestClient(TestServer(server.app()))
    await client.start_server()
    settings = {"male_voice_id": male, "female_voice_id": female}
    if engine == "moss-onnx":
        settings["profile_id"] = "local-moss"
    try:
        await save(client, 0, settings)
        data = {"settings_mode": "override", "text": "Проба голоса", "voice_gender": "female"}
        for _ in range(2):
            response = await client.post("/tts/preview", json=data)
            assert response.status == 200, await response.text()
            assert (await response.read()).startswith(b"RIFF")
        assert calls == [(engine, "female", female, "Проба голоса")]
        response = await client.post("/tts/preview", json={**data, "voice_gender": "male"})
        assert response.status == 200, await response.text()
        assert calls[-1][2] == male
        response = await client.post("/tts/preview", json={**data, "female_voice_id": male})
        assert response.status == 200, await response.text()
        assert len(calls) == 3
        assert calls[-1][2] == male
        assert server.cfg.tts.female_voice_id == female
    finally:
        await client.close()


def test_output_cache_key_changes_for_each_effective_role(server_factory):
    server = server_factory()
    request = {"settings_mode": "override", "file": "/tmp/neutral.wav"}
    base = server._cache_key(request)
    male = server._cache_key({**request, "male_voice_id": "ru-male"})
    female = server._cache_key({**request, "female_voice_id": "ru-female"})
    both = server._cache_key({**request, "male_voice_id": "ru-male", "female_voice_id": "ru-female"})
    assert len({base, male, female, both}) == 4
    # A remembered inactive model's pair does not alter the actual output.
    unrelated = server._cache_key({**request, "voice_pairs": {
        "moss-onnx:ru": {"male_voice_id": "Adam", "female_voice_id": "Bella"},
    }})
    assert unrelated == base


async def test_invalid_pair_put_does_not_change_saved_or_effective_settings(server_factory):
    server = server_factory()
    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        await save(client, 0, {"male_voice_id": "ru-male"})
        response = await client.put("/settings", json={"revision": 1, "settings": {"female_voice_id": "uk-female"}})
        assert response.status == 422
        assert server._settings_revision == 1
        assert server.cfg.tts.male_voice_id == "ru-male"
        assert not server.cfg.tts.female_voice_id
        assert server.settings_store.get("free")["male_voice_id"] == "ru-male"
    finally:
        await client.close()
