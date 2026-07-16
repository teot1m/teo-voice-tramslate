"""События конвейера — сообщения, которыми сервисы обмениваются через шину (ТЗ §14)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Имена топиков шины
TOPIC_AUDIO = "audio"              # AudioChunk: непрерывный звук 16 кГц моно
TOPIC_SPEECH = "speech"            # SpeechSegment: реплики, вырезанные VAD
TOPIC_TRANSCRIPT = "transcript"    # Transcript: распознанный текст
TOPIC_TRANSLATION = "translation"  # Translation: переведённый текст
TOPIC_TTS = "tts"                  # TtsAudio: синтезированная озвучка
TOPIC_STATUS = "status"            # ServiceStatus: состояние сервисов

# Порядок стадий для расчёта задержек (Debug Console, ТЗ §20)
STAGES = ("speech_end", "stt", "translate", "tts", "play")


def now() -> float:
    """Монотонное время — единые часы всего конвейера."""
    return time.monotonic()


@dataclass(slots=True)
class Trace:
    """Отметки времени по стадиям обработки одного сегмента речи."""

    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str) -> None:
        self.marks[stage] = now()

    def spans_ms(self) -> list[tuple[str, float]]:
        """Длительности между соседними пройденными стадиями + total, в мс."""
        present = [(s, self.marks[s]) for s in STAGES if s in self.marks]
        out = [
            (f"{a}→{b}", (tb - ta) * 1000.0)
            for (a, ta), (b, tb) in zip(present, present[1:])
        ]
        if len(present) >= 2:
            out.append(("total", (present[-1][1] - present[0][1]) * 1000.0))
        return out


@dataclass(slots=True)
class AudioChunk:
    """Кусок непрерывного аудио: float32 моно, обычно 16 кГц после ресемпла."""

    samples: np.ndarray
    sample_rate: int
    ts: float  # монотонное время поступления


@dataclass(slots=True)
class SpeechSegment:
    """Одна реплика, вырезанная VAD."""

    samples: np.ndarray
    sample_rate: int
    start_ts: float
    end_ts: float
    trace: Trace

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(slots=True)
class Transcript:
    """Распознанный текст реплики."""

    segment: SpeechSegment
    text: str
    language: str
    confidence: float | None
    trace: Trace


@dataclass(slots=True)
class Translation:
    """Переведённый текст. failed=True — перевод не удался, text = оригинал."""

    transcript: Transcript
    text: str
    target_lang: str
    failed: bool
    trace: Trace


@dataclass(slots=True)
class TtsAudio:
    """Синтезированная озвучка перевода."""

    translation: Translation
    samples: np.ndarray
    sample_rate: int
    trace: Trace


@dataclass(slots=True)
class SubtitleEvent:
    """Готовая пара строк для оверлея/интерфейса."""

    original: str
    translated: str
    language: str
    target_lang: str
    start_ts: float
    end_ts: float


@dataclass(slots=True)
class ServiceStatus:
    """Состояние сервиса: running | error | done | stopped."""

    service: str
    state: str
    detail: str = ""
