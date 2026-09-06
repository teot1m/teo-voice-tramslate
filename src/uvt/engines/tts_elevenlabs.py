"""ElevenLabs TTS через официальный Text-to-Speech API.

Профиль cloud-eleven оставляет распознавание и перевод в OpenAI, а этот
движок меняет только финальную озвучку. Ключ читается исключительно из
окружения; userscript его никогда не получает.
"""
from __future__ import annotations

import os
import re

import httpx

import numpy as np

from uvt.audio import decode_bytes
from uvt.config import configured_role_voice
from uvt.interfaces import TTSEngine
from uvt.registry import register

# Standard voices confirmed by the current API on 2026-09-06. The legacy
# Rachel ID now resolves to Janet/Voice Library and fails on Free (HTTP 402).
# Explicit/env choices keep priority; never replace a chosen voice silently.
_DEFAULT_MALE_VOICE = "ErXwobaYiN019PkySvjV"  # Adam
_DEFAULT_FEMALE_VOICE = "EXAVITQu4vr4xnSDxMaL"  # Sarah


def _provider_detail(response: httpx.Response) -> tuple[str, str]:
    """Keep useful JSON error details without leaking a reflected API key."""
    try:
        payload = response.json()
    except ValueError:
        return "", ""
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        code = str(detail.get("status") or "")
        message = str(detail.get("message") or "")
    else:
        code, message = "", detail if isinstance(detail, str) else ""
    code = code if re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", code) else ""
    # Do this before truncation: a long credential must not leak its prefix.
    for header in ("xi-api-key", "authorization"):
        secret = response.request.headers.get(header, "")
        if secret:
            message = message.replace(secret, "[redacted]")
    message = re.sub(r"\bsk[_-][A-Za-z0-9_-]{8,}\b", "[redacted]", message)
    return code, " ".join(message.split())[:500]


class ElevenLabsAPIError(httpx.HTTPStatusError):
    """Structured provider failure; response keeps status/retry headers."""

    def __init__(self, response: httpx.Response) -> None:
        self.provider_status, self.provider_message = _provider_detail(response)
        self.user_message = self._explain(response.status_code)
        super().__init__(self.user_message, request=response.request, response=response)

    def _explain(self, status: int) -> str:
        code = self.provider_status
        message = self.provider_message.lower()
        prefix = f"ElevenLabs (HTTP {status})"
        if code == "payment_required" and "free users cannot use library voices" in message:
            return (
                f"{prefix}: выбранный голос из Voice Library недоступен через API на бесплатном тарифе. "
                "Выберите «Авто» (Sarah/Adam) или доступный стандартный голос; "
                "наличие кредитов не снимает ограничение этого голоса"
            )
        if code in {"quota_exceeded", "insufficient_credits", "credit_limit_exceeded"} or "insufficient credits" in message:
            return (
                f"{prefix}: недостаточно доступных кредитов или достигнут лимит API-ключа. "
                "Проверьте остаток кредитов и ограничение ключа в ElevenLabs"
            )
        if code == "missing_permissions":
            return (
                f"{prefix}: API-ключу не хватает разрешения для этой операции. "
                "Проверьте разрешение Text to Speech у ключа; наличие кредитов этого не меняет"
            )
        if code in {"detected_unusual_activity", "unusual_activity"} or "unusual activity" in message:
            return (
                f"{prefix}: ElevenLabs ограничил бесплатный API для этого аккаунта или подключения. "
                "Проверьте уведомление провайдера в аккаунте или обратитесь в его поддержку"
            )
        if code == "invalid_api_key" or status == 401:
            return f"{prefix}: ключ не принят; проверьте серверный ELEVENLABS_API_KEY"
        if status == 402:
            return (
                f"{prefix}: провайдер ограничил доступ по тарифу или кредитам "
                f"({code or 'payment_required'}). Проверьте доступность выбранного голоса и модели "
                "для API, затем лимит аккаунта и ключа"
            )
        if status == 403:
            return f"{prefix}: нет доступа к выбранному голосу или модели; проверьте разрешения ключа"
        if status == 429:
            return f"{prefix}: временно превышена частота запросов или параллельность; повторите позже"
        return f"{prefix}: запрос отклонён ({code or 'provider_error'})"



@register("tts", "elevenlabs")
class ElevenLabsTTS(TTSEngine):
    async def warmup(self) -> None:
        import httpx

        self._base = str(
            getattr(self.cfg, "base_url", "https://api.elevenlabs.io/v1")
        ).rstrip("/")
        key_env = str(getattr(self.cfg, "api_key_env", "ELEVENLABS_API_KEY"))
        key = os.environ.get(key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"для ElevenLabs TTS задайте переменную окружения {key_env}"
            )
        timeout = float(getattr(self.cfg, "timeout_s", 90.0))
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"xi-api-key": key, "Accept": "audio/mpeg"},
        )
        self._model = str(
            getattr(self.cfg, "model", "eleven_multilingual_v2")
        )
        self._output_format = str(
            getattr(self.cfg, "output_format", "mp3_44100_128")
        )

    async def close(self) -> None:
        if hasattr(self, "_client"):
            await self._client.aclose()

    def _voice_id(self) -> str:
        explicit = str(getattr(self.cfg, "voice", "auto") or "auto").strip()
        if explicit not in {"auto", "cloud"}:
            return explicit

        gender = str(
            getattr(self.cfg, "voice_gender", "female") or "female"
        ).lower()
        female = gender.startswith(("f", "ж"))
        env_name = (
            "ELEVENLABS_FEMALE_VOICE_ID" if female else "ELEVENLABS_MALE_VOICE_ID"
        )
        configured = configured_role_voice(self.cfg, gender) or ""
        shared = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
        return (
            configured
            or os.environ.get(env_name, "").strip()
            or shared
            or (_DEFAULT_FEMALE_VOICE if female else _DEFAULT_MALE_VOICE)
        )

    def _voice_settings(self, speed: float) -> dict:
        return {
            "stability": float(getattr(self.cfg, "stability", 0.5)),
            "similarity_boost": float(
                getattr(self.cfg, "similarity_boost", 0.75)
            ),
            "style": float(getattr(self.cfg, "style", 0.0)),
            "use_speaker_boost": bool(
                getattr(self.cfg, "use_speaker_boost", True)
            ),
            # ElevenLabs принимает 0.7–1.2. Оставшуюся укладку в тайминг
            # делает общий batch-конвейер UVT.
            "speed": max(0.7, min(1.2, float(speed))),
        }

    async def _speech(self, text: str, speed: float) -> tuple[np.ndarray, int]:
        voice_id = self._voice_id()
        response = await self._client.post(
            f"{self._base}/text-to-speech/{voice_id}",
            params={"output_format": self._output_format},
            json={
                "text": text,
                "model_id": self._model,
                "voice_settings": self._voice_settings(speed),
            },
        )
        if response.is_error:
            raise ElevenLabsAPIError(response)
        return decode_bytes(response.content)

    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        return await self._speech(text, 1.0)

    async def synthesize_rated(
        self, text: str, language: str, speed: float
    ) -> tuple[np.ndarray, int]:
        return await self._speech(text, speed)
