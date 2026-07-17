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
STAGES = ("speech_end", "stt", "translate", "tts", "play", "write_end")


def now() -> float:
    """Монотонное время — единые часы всего конвейера."""
    return time.monotonic()


@dataclass(slots=True)
class Trace:
    """Отметки времени по стадиям обработки одного сегмента речи."""

    marks: dict[str, float] = field(default_factory=dict)
    # marks — время выполнения кода. Эти поля — таймлайн исходного аудио в
    # монотонном clock-domain CaptureService.
    source_start_ts: float | None = None
    source_end_ts: float | None = None
    target_ts: float | None = None
    deadline_ts: float | None = None
    dropped_stage: str | None = None
    dropped_reason: str | None = None

    def mark(self, stage: str, ts: float | None = None) -> None:
        """Отметить стадию; ts нужен для честного времени конца write."""
        self.marks[stage] = now() if ts is None else ts

    def is_expired(self, at: float | None = None) -> bool:
        """Истёк ли дедлайн живого воспроизведения."""
        return self.deadline_ts is not None and (now() if at is None else at) > self.deadline_ts

    def mark_dropped(self, stage: str, reason: str) -> None:
        self.dropped_stage = stage
        self.dropped_reason = reason

    def spans_ms(self) -> list[tuple[str, float]]:
        """Длительности между соседними пройденными стадиями + total, в мс."""
        present = [(s, self.marks[s]) for s in STAGES if s in self.marks]
        out = [
            (f"{a}→{b}", (tb - ta) * 1000.0)
            for (a, ta), (b, tb) in zip(present, present[1:])
        ]
        if len(present) >= 2:
            out.append(("total", (present[-1][1] - present[0][1]) * 1000.0))
        # Не смешиваем processing latency и sync latency: последнее отсчитывается
        # от аудио-часов CaptureService, а не от окончания VAD-кода.
        if self.source_end_ts is not None and "play" in self.marks:
            out.append(("source_end→play", (self.marks["play"] - self.source_end_ts) * 1000.0))
        if self.source_end_ts is not None and "write_end" in self.marks:
            out.append(
                ("source_end→write_end", (self.marks["write_end"] - self.source_end_ts) * 1000.0)
            )
        if self.target_ts is not None and "play" in self.marks:
            out.append(("sync_drift", (self.marks["play"] - self.target_ts) * 1000.0))
        if self.target_ts is not None and "write_end" in self.marks:
            out.append(("sync_drift_end", (self.marks["write_end"] - self.target_ts) * 1000.0))
        return out


@dataclass(slots=True)
class AudioChunk:
    """Кусок непрерывного аудио: float32 моно, обычно 16 кГц после ресемпла.

    ``ts`` остаётся совместимым именем и означает начало аудио в монотонных
    часах источника. Для старых движков ``end_ts`` необязателен.
    """

    samples: np.ndarray
    sample_rate: int
    ts: float
    end_ts: float | None = None

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    @property
    def source_start_ts(self) -> float:
        return self.ts

    @property
    def source_end_ts(self) -> float:
        return self.end_ts if self.end_ts is not None else self.ts + self.duration_s


@dataclass(slots=True)
class SpeechSegment:
    """Одна реплика, вырезанная VAD."""

    samples: np.ndarray
    sample_rate: int
    start_ts: float
    end_ts: float
    trace: Trace
    # Это метка спикера/тембра, а не утверждение пола или личности человека.
    speaker_id: str | None = None
    speaker_timbre: str | None = None
    speaker_confidence: float | None = None

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def target_ts(self) -> float | None:
        return self.trace.target_ts

    @property
    def deadline_ts(self) -> float | None:
        return self.trace.deadline_ts


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
