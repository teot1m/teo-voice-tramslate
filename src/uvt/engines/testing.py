"""Тестовые движки: проверка конвейера без моделей, сети и звуковых карт.

Используются профилем demo и автотестами: uvt run --profile demo.
"""
from __future__ import annotations

import asyncio

import numpy as np

from uvt.interfaces import CaptureEngine, STTEngine, STTResult, TranslationEngine, TTSEngine
from uvt.registry import register

_RATE = 16000


@register("capture", "dummy")
class DummyCapture(CaptureEngine):
    """Генерирует «реплики»: тишина 0.5 с → тон 440 Гц → тишина 1 с, N раз."""

    async def stream(self):
        bursts = int(getattr(self.cfg, "bursts", 3))
        tone_s = float(getattr(self.cfg, "tone_s", 1.0))
        chunk = int(_RATE * 0.02)

        t = np.arange(int(_RATE * tone_s)) / _RATE
        tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        lead = np.zeros(int(_RATE * 0.5), dtype=np.float32)
        tail = np.zeros(int(_RATE * 1.0), dtype=np.float32)

        for _ in range(bursts):
            for part in (lead, tone, tail):
                for i in range(0, len(part), chunk):
                    yield part[i : i + chunk], _RATE
            await asyncio.sleep(0)


@register("stt", "dummy")
class DummySTT(STTEngine):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._count = 0

    async def transcribe(self, samples, sample_rate, language):
        self._count += 1
        return STTResult(text=f"hello {self._count}", language="en", confidence=0.99)


@register("translation", "dummy")
class DummyTranslator(TranslationEngine):
    async def translate(self, text, source_lang, target_lang, context):
        return f"{text.upper()} [{target_lang}]"


@register("tts", "dummy")
class DummyTTS(TTSEngine):
    async def synthesize(self, text, language):
        t = np.arange(int(24000 * 0.2)) / 24000
        return (0.2 * np.sin(2 * np.pi * 660.0 * t)).astype(np.float32), 24000
