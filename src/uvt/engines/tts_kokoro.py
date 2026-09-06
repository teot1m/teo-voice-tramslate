"""Kokoro TTS (ONNX) — быстрый локальный синтез (ТЗ §21, MVP).

Файлы модели крупные (~340 МБ суммарно), поэтому автоматически не качаются:
пути задаются в конфиге, иначе движок объясняет, откуда их взять.
Русского голоса в Kokoro v1.0 нет — для RU используйте edge.
"""
from __future__ import annotations

import asyncio

import numpy as np

from uvt.config import configured_role_voice
from uvt.interfaces import TTSEngine
from uvt.registry import register

_KOKORO_LANGS = {
    "en": "en-us", "fr": "fr-fr", "it": "it", "ja": "ja",
    "zh": "cmn", "es": "es", "hi": "hi", "pt": "pt-br",
}
_DOWNLOAD_HINT = (
    "укажите tts.model_path и tts.voices_path; файлы: "
    "https://github.com/thewh1teagle/kokoro-onnx/releases (kokoro-v1.0.onnx, voices-v1.0.bin)"
)


@register("tts", "kokoro")
class KokoroTTS(TTSEngine):
    async def warmup(self) -> None:
        if not (self.cfg.model_path and self.cfg.voices_path):
            raise RuntimeError(f"kokoro: не заданы пути к модели — {_DOWNLOAD_HINT}")
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(self.cfg.model_path, self.cfg.voices_path)

    def _voice(self, language: str) -> str:
        explicit = str(getattr(self.cfg, "voice", "auto") or "auto").strip()
        if explicit not in {"auto", "cloud"}:
            return explicit
        selected = configured_role_voice(self.cfg)
        if selected:
            return selected
        return "af_heart"

    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        lang_code = _KOKORO_LANGS.get(language.split("-")[0].lower())
        if lang_code is None:
            raise RuntimeError(
                f"kokoro не поддерживает язык '{language}' "
                f"(доступны: {', '.join(sorted(_KOKORO_LANGS))}); для него используйте tts.engine=edge"
            )
        voice = self._voice(language)
        samples, rate = await asyncio.to_thread(
            self._kokoro.create, text, voice=voice, speed=self.cfg.speed, lang=lang_code
        )
        return np.asarray(samples, dtype=np.float32), int(rate)

    async def synthesize_rated(
        self, text: str, language: str, speed: float
    ) -> tuple[np.ndarray, int]:
        lang_code = _KOKORO_LANGS.get(language.split("-")[0].lower())
        if lang_code is None or speed <= 1.02:
            return await self.synthesize(text, language)
        voice = self._voice(language)
        samples, rate = await asyncio.to_thread(
            self._kokoro.create, text, voice=voice,
            speed=float(self.cfg.speed) * speed, lang=lang_code,
        )
        return np.asarray(samples, dtype=np.float32), int(rate)
