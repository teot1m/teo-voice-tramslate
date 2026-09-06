"""Лёгкая session-local привязка реплик к спикерам для live TTS.

Это не diarization-модель и не определение личности. Для быстрого live пути
используются F0 и несколько акустических признаков, чтобы удержать одинаковый
target voice у одного спикера и не сваливаться в прежний ``male`` default.
Полные embedding/diarization движки можно подключить позднее без изменения
контракта ``SpeakerAssignment``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uvt.gender import estimate_gender_f0


@dataclass(slots=True, frozen=True)
class SpeakerAssignment:
    """Результат session-local speaker/timbre анализа одной реплики."""

    speaker_id: str | None
    timbre: str
    confidence: float
    f0_hz: float | None
    # Роль для существующих TTS engines (male/female), а не заявление о поле.
    voice_gender: str | None
    # Явный id голоса из speaker.voice_map, если пользователь его задал.
    voice: str | None = None


@dataclass(slots=True)
class _KnownSpeaker:
    speaker_id: str
    embedding: np.ndarray
    voice_gender: str
    assignments: int = 1


class SpeakerService:
    """Небольшой in-memory кластеризатор тембра с устойчивым voice mapping."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._speakers: list[_KnownSpeaker] = []

    @property
    def speakers(self) -> tuple[str, ...]:
        """Текущие session IDs; удобно для будущего UI ручного назначения."""
        return tuple(s.speaker_id for s in self._speakers)

    def assign(self, samples: np.ndarray, sample_rate: int) -> SpeakerAssignment:
        """Вернуть стабильную метку спикера и target voice role для сегмента."""
        if not bool(getattr(self.cfg, "enabled", True)):
            return SpeakerAssignment(None, "unknown", 0.0, None, None)

        f0_gender, f0_hz = estimate_gender_f0(samples, sample_rate)
        embedding = _embedding(samples, sample_rate, f0_hz)
        timbre = _timbre(f0_hz, embedding)
        known, distance = self._nearest(embedding)
        threshold = max(0.05, float(getattr(self.cfg, "match_threshold", 0.42)))

        is_distant = distance is not None and distance > threshold
        if known is None or (is_distant and self._can_add()):
            index = len(self._speakers) + 1
            speaker_id = f"speaker-{index}"
            voice_gender = self._role_for_new_speaker(index, f0_gender)
            known = _KnownSpeaker(speaker_id, embedding, voice_gender)
            self._speakers.append(known)
            distance = 0.0
        elif not is_distant:
            # EMA делает кластер терпимым к разной громкости/микрофону, но не
            # подменяет его резко новым голосом.
            known.embedding = (0.82 * known.embedding + 0.18 * embedding).astype(np.float32)
            known.assignments += 1
            # Role задаётся при создании кластера (или ручным voice_map ниже)
            # и не переписывается шумным F0 одной следующей реплики. F0 здесь
            # нужен для timbre/confidence, а не для определения личности.
        else:
            # Все роли уже заняты, а этот тембр заметно дальше порога. Возвращаем
            # ближайшую существующую роль как bounded fallback, но не "учим" её
            # чужой репликой: иначе один новый голос испортит и embedding, и
            # стабильно выбранный голос уже известного speaker-N.
            pass

        confidence = _confidence(f0_hz, distance, threshold)
        override = dict(getattr(self.cfg, "voice_map", {}) or {}).get(known.speaker_id)
        voice, role = _split_voice_override(override, known.voice_gender)
        return SpeakerAssignment(
            speaker_id=known.speaker_id,
            timbre=timbre,
            confidence=confidence,
            f0_hz=f0_hz,
            voice_gender=role,
            voice=voice,
        )

    def _nearest(self, embedding: np.ndarray) -> tuple[_KnownSpeaker | None, float | None]:
        if not self._speakers:
            return None, None
        distances = [float(np.linalg.norm(embedding - speaker.embedding)) for speaker in self._speakers]
        index = int(np.argmin(distances))
        return self._speakers[index], distances[index]

    def _can_add(self) -> bool:
        return len(self._speakers) < max(1, int(getattr(self.cfg, "max_speakers", 8)))

    def _role_for_new_speaker(self, index: int, inferred: str | None) -> str:
        if inferred in {"male", "female"}:
            return inferred
        roles = [
            str(role).lower()
            for role in (getattr(self.cfg, "fallback_voice_roles", None) or ["female", "male"])
            if str(role).lower() in {"male", "female"}
        ]
        # Неподтверждённому тембру назначается стабильная target role по
        # спикеру. Это не наследует последний male voice и не выдаёт роль за
        # определение пола.
        return (roles or ["female", "male"])[(index - 1) % len(roles or ["female", "male"])]


def _split_voice_override(value: object, default_role: str) -> tuple[str | None, str]:
    if value is None:
        return None, default_role
    text = str(value).strip()
    if text.lower() in {"male", "female"}:
        return None, text.lower()
    return text or None, default_role


def _embedding(samples: np.ndarray, sample_rate: int, f0_hz: float | None) -> np.ndarray:
    """Дешёвый, устойчивый к громкости fingerprint (не биометрический)."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(x) == 0 or sample_rate <= 0:
        return np.zeros(4, dtype=np.float32)
    # Двух секунд достаточно для тембра и держит work bounded для long VAD chunks.
    x = x[: min(len(x), sample_rate * 2)]
    x = x - float(np.mean(x))
    energy = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    if energy < 1e-5:
        return np.zeros(4, dtype=np.float32)

    window = np.hanning(len(x)).astype(np.float32)
    power = np.square(np.abs(np.fft.rfft(x * window)))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    mask = (freqs >= 60.0) & (freqs <= min(4000.0, sample_rate / 2))
    if not np.any(mask) or float(power[mask].sum()) <= 1e-12:
        centroid = 0.0
        rolloff = 0.0
    else:
        band_power = power[mask]
        band_freqs = freqs[mask]
        total = float(band_power.sum())
        centroid = float((band_freqs * band_power).sum() / total)
        cumulative = np.cumsum(band_power)
        rolloff = float(band_freqs[min(len(band_freqs) - 1, int(np.searchsorted(cumulative, total * 0.85)))])
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x))))) if len(x) > 1 else 0.0
    # F0 получают повышенный вес: это делает два явно разных голоса разными
    # кластерами даже при похожем микрофоне/громкости.
    # Accurate peak interpolation no longer inflates F0. Keep a clear octave
    # change distinguishable at the default 0.42 clustering threshold.
    f0_feature = 2.0 * (float(f0_hz) / 400.0) if f0_hz else 0.0
    return np.asarray(
        [f0_feature, centroid / 4000.0, rolloff / 4000.0, min(1.0, zcr * 4.0)],
        dtype=np.float32,
    )


def _timbre(f0_hz: float | None, embedding: np.ndarray) -> str:
    if f0_hz is not None:
        if f0_hz < 155.0:
            return "low"
        if f0_hz > 190.0:
            return "high"
        return "mid"
    centroid = float(embedding[1]) * 4000.0
    if centroid <= 1e-4:
        return "unknown"
    if centroid < 850.0:
        return "low"
    if centroid > 1800.0:
        return "high"
    return "mid"


def _confidence(f0_hz: float | None, distance: float | None, threshold: float) -> float:
    if f0_hz is not None:
        # 173 Hz — нейтральная зона между low/high F0; ближе к краю выше уверенность.
        pitch = min(0.98, 0.5 + min(0.45, abs(f0_hz - 173.0) / 180.0))
    else:
        pitch = 0.25
    if distance is None:
        return round(pitch, 3)
    clustering = max(0.0, min(1.0, 1.0 - distance / max(threshold, 1e-6)))
    return round(0.6 * pitch + 0.4 * clustering, 3)
