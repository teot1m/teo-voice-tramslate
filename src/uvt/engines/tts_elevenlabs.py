"""ElevenLabs TTS через официальный Text-to-Speech API.

Профиль cloud-eleven оставляет распознавание и перевод в OpenAI, а этот
движок меняет только финальную озвучку. Ключ читается исключительно из
окружения; userscript его никогда не получает.
"""
from __future__ import annotations

import os

import numpy as np

from uvt.audio import decode_bytes
from uvt.interfaces import TTSEngine
from uvt.registry import register

# Стабильные premade voice IDs ElevenLabs. Их можно заменить переменными
# ELEVENLABS_MALE_VOICE_ID / ELEVENLABS_FEMALE_VOICE_ID без правки профиля.
_DEFAULT_MALE_VOICE = "ErXwobaYiN019PkySvjV"  # Antoni
_DEFAULT_FEMALE_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel


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
        if explicit != "auto":
            return explicit

        gender = str(
            getattr(self.cfg, "voice_gender", "female") or "female"
        ).lower()
        female = gender.startswith(("f", "ж"))
        field = "female_voice_id" if female else "male_voice_id"
        env_name = (
            "ELEVENLABS_FEMALE_VOICE_ID" if female else "ELEVENLABS_MALE_VOICE_ID"
        )
        configured = str(getattr(self.cfg, field, "") or "").strip()
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
        response.raise_for_status()
        return decode_bytes(response.content)

    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        return await self._speech(text, 1.0)

    async def synthesize_rated(
        self, text: str, language: str, speed: float
    ) -> tuple[np.ndarray, int]:
        return await self._speech(text, speed)
