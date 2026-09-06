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
class VoiceReference:
    """Образец голоса из самого исходного звука — для клонирующих TTS.

    Референс берётся из оригинальной дорожки, поэтому переведённая реплика
    сохраняет тембр и манеру настоящего говорящего: это то, чего фиксированный
    голос вроде Piper дать не может.
    """

    samples: np.ndarray          # float32 моно
    sample_rate: int
    label: str = "speaker"       # ключ спикера/пола, к которому относится образец
    text: str | None = None      # расшифровка образца, если движку она нужна


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
    """Перевод текста.

    ``supports_shorten`` объявляет, что движок умеет переписать слишком
    длинный перевод короче (``shorten``). Для дубляжа это предпочтительный
    способ уложиться в тайминг: сжатие текстом не слышно, ускорение речи —
    слышно сразу.
    """

    supports_shorten: bool = False

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

    async def translate_batch_contextual(
        self, texts: Sequence[str], source_lang: str | None, target_lang: str,
        genders: Sequence[str] | None, *,
        before: Sequence[tuple[str, str]] = (),
        after: Sequence[tuple[str, str]] = (),
    ) -> list[str]:
        """File-only neighbouring source lines and roles; return only ``texts``.

        Engines without prompt context retain their existing batch implementation.
        Live translation uses ``translate`` and never waits for future speech.
        """
        return await self.translate_batch_tagged(texts, source_lang, target_lang, genders)

    async def shorten(self, text: str, target_lang: str, max_chars: int) -> str:
        """Переписать перевод короче указанного объёма, сохранив смысл.

        По умолчанию текст возвращается без изменений: движки, которые так не
        умеют, ничего не теряют, а дубляж проверяет ``supports_shorten``.
        """
        return text


class TTSEngine(Engine):
    """Синтез речи.

    Три уровня возможностей, которые дубляж выбирает сам:

    - ``synthesize`` — базовый синтез (обязателен);
    - ``synthesize_rated`` — синтез с ускорением темпа (Piper, OpenAI);
    - ``synthesize_slot`` — синтез сразу в заданную длительность и/или с
      клонированием голоса по образцу (F5-TTS, IndexTTS-2). Именно он
      позволяет не ускорять готовый звук через atempo, из-за которого речь
      звучит как перемотка.

    Движок объявляет, что умеет, флагами ``supports_duration`` и
    ``supports_reference`` — дубляж проверяет их до синтеза.
    """

    supports_duration: bool = False
    supports_reference: bool = False
    supports_emotion: bool = False

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

    async def synthesize_slot(
        self,
        text: str,
        language: str,
        *,
        target_duration: float | None = None,
        reference: VoiceReference | None = None,
        emotion: VoiceReference | None = None,
    ) -> tuple[np.ndarray, int]:
        """Синтез в тайминг оригинала и/или голосом из образца.

        ``target_duration`` — сколько секунд доступно под реплику; движок сам
        укладывается в них, вместо того чтобы отдавать длинный клип на
        последующее ускорение. None — говорить в естественном темпе.
        ``reference`` — образец голоса из исходной дорожки.
        ``emotion`` — звук самой переводимой реплики: движки с переносом
        просодии (IndexTTS-2) берут из него интонацию, сохраняя тембр из
        ``reference``.

        База игнорирует все параметры, поэтому старые движки продолжают
        работать без изменений.
        """
        return await self.synthesize(text, language)
