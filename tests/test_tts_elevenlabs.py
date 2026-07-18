"""ElevenLabs TTS: ключ, voice ID и запрос к официальному endpoint."""
from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from uvt.config import TTSConfig
from uvt.engines import tts_elevenlabs
from uvt.engines.tts_elevenlabs import ElevenLabsTTS


async def test_elevenlabs_sends_key_voice_model_and_bounded_speed(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-secret")
    monkeypatch.setenv("ELEVENLABS_MALE_VOICE_ID", "male-test-voice")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"fake-mp3")

    real_async_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(
        tts_elevenlabs,
        "decode_bytes",
        lambda _data: (np.ones(16, dtype=np.float32), 44100),
    )
    cfg = TTSConfig(
        engine="elevenlabs",
        voice="auto",
        voice_gender="male",
        base_url="https://api.elevenlabs.io/v1",
        api_key_env="ELEVENLABS_API_KEY",
        model="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    engine = ElevenLabsTTS(cfg)
    await engine.warmup()
    try:
        samples, rate = await engine.synthesize_rated("Привет", "ru", 1.6)
    finally:
        await engine.close()

    assert len(samples) == 16 and rate == 44100
    assert len(requests) == 1
    request = requests[0]
    assert request.headers["xi-api-key"] == "eleven-secret"
    assert request.url.path == "/v1/text-to-speech/male-test-voice"
    assert request.url.params["output_format"] == "mp3_44100_128"
    payload = json.loads(request.content)
    assert payload["model_id"] == "eleven_multilingual_v2"
    assert payload["voice_settings"]["speed"] == 1.2


async def test_elevenlabs_requires_server_side_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    engine = ElevenLabsTTS(TTSConfig(engine="elevenlabs"))
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        await engine.warmup()
