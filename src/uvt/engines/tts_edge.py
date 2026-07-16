"""Edge TTS — бесплатный облачный синтез Microsoft, без API-ключа (ТЗ §21, MVP).

voice="auto" подбирает нейронный голос по целевому языку. Поддерживает
ускорение темпа (rate=+N%) — дубляж использует его, чтобы длинные переводы
укладывались в тайминги оригинала.
"""
from __future__ import annotations

import re

import numpy as np

from uvt.audio import decode_bytes
from uvt.interfaces import TTSEngine
from uvt.registry import register

DEFAULT_VOICES = {
    "ru": "ru-RU-DmitryNeural",
    "en": "en-US-GuyNeural",
    "uk": "uk-UA-OstapNeural",
    "de": "de-DE-ConradNeural",
    "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural",
    "it": "it-IT-DiegoNeural",
    "pt": "pt-BR-AntonioNeural",
    "ja": "ja-JP-KeitaNeural",
    "zh": "zh-CN-YunxiNeural",
    "ko": "ko-KR-InJoonNeural",
    "tr": "tr-TR-AhmetNeural",
    "pl": "pl-PL-MarekNeural",
}

FEMALE_VOICES = {
    "ru": "ru-RU-SvetlanaNeural",
    "en": "en-US-JennyNeural",
    "uk": "uk-UA-PolinaNeural",
    "de": "de-DE-KatjaNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ja": "ja-JP-NanamiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ko": "ko-KR-SunHiNeural",
    "tr": "tr-TR-EmelNeural",
    "pl": "pl-PL-ZofiaNeural",
}


@register("tts", "edge")
class EdgeTTS(TTSEngine):
    async def warmup(self) -> None:
        import edge_tts  # noqa: F401 — ранняя проверка зависимости

    def _voice(self, language: str) -> str:
        if self.cfg.voice and self.cfg.voice != "auto":
            return self.cfg.voice
        lang = language.split("-")[0].lower()
        gender = str(getattr(self.cfg, "voice_gender", "male") or "male").lower()
        table = FEMALE_VOICES if gender.startswith(("f", "ж")) else DEFAULT_VOICES
        return table.get(lang) or DEFAULT_VOICES.get(lang, "en-US-GuyNeural")

    def _base_rate_pct(self) -> int:
        match = re.match(r"^([+-]?)(\d+)%$", str(self.cfg.rate).strip())
        if not match:
            return 0
        return int(match.group(2)) * (-1 if match.group(1) == "-" else 1)

    def _rate_string(self, speed: float) -> str:
        total = self._base_rate_pct() + max(0, round((speed - 1.0) * 100))
        return f"{'+' if total >= 0 else ''}{total}%"

    async def _synth(self, text: str, language: str, rate: str) -> tuple[np.ndarray, int]:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice=self._voice(language), rate=rate)
        raw = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                raw.extend(chunk["data"])
        if not raw:
            return np.zeros(0, dtype=np.float32), 24000
        return decode_bytes(bytes(raw))

    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        return await self._synth(text, language, str(self.cfg.rate))

    async def synthesize_rated(
        self, text: str, language: str, speed: float
    ) -> tuple[np.ndarray, int]:
        if speed <= 1.02:
            return await self.synthesize(text, language)
        return await self._synth(text, language, self._rate_string(speed))