"""OpenAI TTS — естественные голоса через /audio/speech (ключ OPENAI_API_KEY).

Заметно натуральнее Edge TTS; мультиязычный (включая русский), поддерживает
темп — дубляж использует его для укладки реплик в тайминги. Модель по
умолчанию tts-1; можно указать tts-1-hd или gpt-4o-mini-tts в tts.model.
Совместимые endpoint (base_url) тоже работают.
"""
from __future__ import annotations

import os

import numpy as np

from uvt.audio import decode_bytes
from uvt.interfaces import TTSEngine
from uvt.registry import register

# Голоса OpenAI не привязаны к языку; делим по тембру
_MALE_VOICE = "onyx"
_FEMALE_VOICE = "nova"


@register("tts", "openai")
class OpenAITTS(TTSEngine):
    async def warmup(self) -> None:
        import httpx

        from uvt.engines.translate_openai import _require_key_for_remote

        self._base = str(getattr(self.cfg, "base_url", "https://api.openai.com/v1")).rstrip("/")
        key_env = str(getattr(self.cfg, "api_key_env", "OPENAI_API_KEY"))
        key = os.environ.get(key_env, "")
        _require_key_for_remote(self._base, key, key_env)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.AsyncClient(timeout=60.0, headers=headers)
        self._model = str(getattr(self.cfg, "model", "tts-1"))

    async def close(self) -> None:
        if hasattr(self, "_client"):
            await self._client.aclose()

    def _voice(self) -> str:
        if self.cfg.voice and self.cfg.voice != "auto":
            return self.cfg.voice
        gender = str(getattr(self.cfg, "voice_gender", "male") or "male").lower()
        return _FEMALE_VOICE if gender.startswith(("f", "ж")) else _MALE_VOICE

    async def _speech(self, text: str, speed: float) -> tuple[np.ndarray, int]:
        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice(),
            "response_format": "mp3",
        }
        if abs(speed - 1.0) > 0.02:
            payload["speed"] = max(0.5, min(2.0, speed))
        response = await self._client.post(f"{self._base}/audio/speech", json=payload)
        response.raise_for_status()
        return decode_bytes(response.content)

    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        return await self._speech(text, 1.0)

    async def synthesize_rated(
        self, text: str, language: str, speed: float
    ) -> tuple[np.ndarray, int]:
        return await self._speech(text, speed)
