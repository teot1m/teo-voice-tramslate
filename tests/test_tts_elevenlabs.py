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


def test_auto_uses_standard_voices_and_keeps_explicit_choices(monkeypatch):
    for key in ["ELEVENLABS_VOICE_ID", "ELEVENLABS_FEMALE_VOICE_ID", "ELEVENLABS_MALE_VOICE_ID"]:
        monkeypatch.delenv(key, raising=False)
    female = ElevenLabsTTS(TTSConfig(engine="elevenlabs", voice_gender="female"))
    assert female._voice_id() == "EXAVITQu4vr4xnSDxMaL"
    male = ElevenLabsTTS(TTSConfig(engine="elevenlabs", voice_gender="male"))
    assert male._voice_id() == "ErXwobaYiN019PkySvjV"
    explicit = ElevenLabsTTS(TTSConfig(engine="elevenlabs", voice="custom-library-voice"))
    assert explicit._voice_id() == "custom-library-voice"
    monkeypatch.setenv("ELEVENLABS_FEMALE_VOICE_ID", "custom-env-voice")
    assert female._voice_id() == "custom-env-voice"


def _error(status, code, message, headers=None):
    request = httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test", headers={"xi-api-key": "eleven-private-key"})
    response = httpx.Response(status, json={"detail": {"status": code, "message": message}}, headers=headers, request=request)
    return tts_elevenlabs.ElevenLabsAPIError(response)


def test_library_voice_error_is_not_misreported_as_empty_credit_balance():
    from uvt.dub import _is_fatal_tts_error, _tts_failure_reason
    error = _error(402, "payment_required", "Free users cannot use library voices via the API. Please upgrade your subscription to use this voice.")
    assert isinstance(error, httpx.HTTPStatusError)
    assert error.response.status_code == 402
    assert error.provider_status == "payment_required"
    assert "library voices" in error.provider_message
    assert "Voice Library" in str(error) and "стандартный голос" in str(error)
    assert "недостаточно" not in str(error)
    assert _is_fatal_tts_error(error)
    assert _tts_failure_reason("ElevenLabs", error) == error.user_message


@pytest.mark.parametrize("http_status,code,message,expected", [
    (402, "quota_exceeded", "Insufficient credits.", "лимит API-ключа"),
    (401, "missing_permissions", "Missing permission text_to_speech.", "разрешения"),
    (401, "invalid_api_key", "Invalid key.", "ключ не принят"),
    (403, "detected_unusual_activity", "Unusual activity.", "ограничил бесплатный API"),
    (401, "detected_unusual_activity", "Unusual activity.", "ограничил бесплатный API"),
    (402, "payment_required", "Plan required for the model.", "доступность выбранного голоса и модели"),
])
def test_provider_reason_distinguishes_permissions_plan_and_credits(http_status, code, message, expected):
    error = _error(http_status, code, message)
    assert expected in error.user_message
    assert f"HTTP {http_status}" in error.user_message


def test_detail_redacts_reflected_key_and_keeps_retry_headers():
    from uvt.dub import _is_retryable_tts_error, _tts_retry_delay
    error = _error(429, "too_many_concurrent_requests", "Rejected eleven-private-key and sk_sensitive_key_0123456789", {"Retry-After":"2"})
    assert "eleven-private-key" not in error.provider_message
    assert "sk_sensitive" not in error.provider_message
    assert "eleven-private-key" not in str(error)
    assert _is_retryable_tts_error(error)
    assert _tts_retry_delay(error, 1) == 2


def test_non_json_error_does_not_expose_server_html():
    request = httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test")
    response = httpx.Response(502, text="<html>private details</html>", request=request)
    error = tts_elevenlabs.ElevenLabsAPIError(response)
    assert error.provider_status == error.provider_message == ""
    assert "private details" not in str(error)


async def test_synthesis_raises_structured_error_before_audio_decode(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-secret")
    real_client = httpx.AsyncClient

    def handler(request):
        return httpx.Response(402, json={"detail": {"status":"payment_required", "message":"Free users cannot use library voices via the API."}})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(tts_elevenlabs, "decode_bytes", lambda _: pytest.fail("an error is not audio"))
    engine = ElevenLabsTTS(TTSConfig(engine="elevenlabs", voice="selected-library-id"))
    await engine.warmup()
    try:
        with pytest.raises(tts_elevenlabs.ElevenLabsAPIError, match="Voice Library") as result:
            await engine.synthesize("Проверка.", "ru")
        assert result.value.provider_status == "payment_required"
        assert result.value.response.status_code == 402
    finally:
        await engine.close()


async def test_fatal_library_voice_error_stops_remaining_lines(monkeypatch):
    import uvt.dub as dub
    from uvt.config import AppConfig
    from uvt.interfaces import STTSpan

    class RejectedVoice:
        calls = 0
        closed = False

        async def warmup(self):
            pass

        async def close(self):
            self.closed = True

        async def synthesize(self, text, language):
            self.calls += 1
            raise _error(402, "payment_required", "Free users cannot use library voices via the API.")

    voice = RejectedVoice()
    monkeypatch.setattr(dub.registry, "create", lambda *_args, **_kwargs: voice)
    cfg = AppConfig()
    cfg.tts.engine = "elevenlabs"
    cfg.tts.concurrency = 1
    spans = [STTSpan(float(i), float(i + 1), "test", "en") for i in range(10)]
    with pytest.raises(RuntimeError, match="Voice Library"):
        await dub._synthesize_all(cfg, spans, ["Проверка."] * 10, ["female"] * 10, None)
    assert voice.calls == 1
    assert voice.closed
