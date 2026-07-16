"""Интерфейсы движков — контракт Plugin API (ТЗ §15).

Пользовательский плагин наследует один из классов ниже, регистрируется через
@uvt.registry.register(kind, name) и кладётся .py-файлом в каталог plugins/.
Ядро приложения при этом не меняется.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class STTResult:
    text: str
    language: str | None = None
    confidence: float | None = None


@dataclass(slots=True)
class STTSpan:
    """Реплика с таймкодами при пакетном распознавании целого файла."""

    start: float
    end: float
    text: str
    language: str | None = None


class Engine(ABC):
    """База: движок получает свою секцию конфига; тяжёлые импорты — в warmup()."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    async def warmup(self) -> None:
        """Загрузка моделей/клиентов. Вызывается до старта потока данных."""

    async def close(self) -> None:
        """Освобождение ресурсов при остановке конвейера."""


class CaptureEngine(Engine):
    """Источник звука."""

    @abstractmethod
    def stream(self) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Асинхронный генератор пар (сэмплы float32 моно, частота дискретизации)."""


class VADEngine(Engine):
    """Детектор речи. Получает кадры по frame_samples сэмплов при 16 кГц."""

    frame_samples: int = 512  # 32 мс при 16 кГц (требование Silero v5)

    @abstractmethod
    def prob(self, frame: np.ndarray) -> float:
        """Вероятность речи в кадре, 0.0–1.0."""


class STTEngine(Engine):
    """Распознавание речи."""

    @abstractmethod
    async def transcribe(
        self, samples: np.ndarray, sample_rate: int, language: str | None
    ) -> STTResult | None:
        """language=None → автоопределение. None в ответе — сегмент без речи."""

    async def transcribe_long(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str | None,
        progress=None,
    ) -> list[STTSpan] | None:
        """Пакетное распознавание целого файла с таймкодами (быстрый путь дубляжа).

        None — движок так не умеет, дубляж возьмёт путь VAD + transcribe().
        progress(секунд_готово, секунд_всего) — необязательный колбэк.
        """
        return None


class TranslationEngine(Engine):
    """Перевод текста."""

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Sequence[tuple[str, str]],
    ) -> str:
        """context — прошлые пары (оригинал, перевод) для связности диалога."""

    async def translate_batch(
        self, texts: Sequence[str], source_lang: str | None, target_lang: str
    ) -> list[str]:
        """Перевод набора реплик одним вызовом (дубляж). По умолчанию — по одной."""
        out: list[str] = []
        for text in texts:
            out.append(await self.translate(text, source_lang or "und", target_lang, []))
        return out

    async def translate_batch_tagged(
        self,
        texts: Sequence[str],
        source_lang: str | None,
        target_lang: str,
        genders: Sequence[str] | None,
    ) -> list[str]:
        """То же, но с полом говорящего для каждой реплики («я готова» vs
        «я готов»). По умолчанию пол игнорируется."""
        return await self.translate_batch(texts, source_lang, target_lang)


class TTSEngine(Engine):
    """Синтез речи."""

    @abstractmethod
    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        """Возвращает (сэмплы float32 моно, частота дискретизации)."""

    async def synthesize_rated(
        self, text: str, language: str, speed: float
    ) -> tuple[np.ndarray, int]:
        """Синтез с ускорением темпа (speed 1.2 = на 20 % быстрее) — чтобы
        озвучка укладывалась в тайминги оригинала. По умолчанию темп
        игнорируется; движки, умеющие темп, переопределяют."""
        return await self.synthesize(text, language)
