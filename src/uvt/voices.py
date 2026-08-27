"""Отбор образцов голоса из исходной дорожки для клонирующих TTS.

Клонирующему движку (F5-TTS, IndexTTS-2, Chatterbox) нужен короткий чистый
фрагмент речи говорящего. Брать первый попавшийся нельзя: в фильме это часто
шёпот, обрывок на музыке или полсекунды «Okay». Здесь выбирается лучший
фрагмент на каждого говорящего — по громкости, длительности и доле паузы, —
и вместе с ним отдаётся уже готовая расшифровка из STT, чтобы движку не
приходилось распознавать образец заново.
"""
from __future__ import annotations

import logging

import numpy as np

from uvt.interfaces import STTSpan, VoiceReference

log = logging.getLogger("uvt.voices")

MIN_REFERENCE_S = 2.5    # короче образца движку не хватает тембра
IDEAL_REFERENCE_S = 6.0  # рабочий диапазон zero-shot клонирования
MAX_REFERENCE_S = 10.0   # длиннее только замедляет синтез


def _rms(samples: np.ndarray) -> float:
    if not len(samples):
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _speech_ratio(samples: np.ndarray, rate: int) -> float:
    """Доля кадров с энергией выше относительного порога — грубая мера «речь/пауза»."""
    frame = max(1, int(0.02 * rate))
    frames = len(samples) // frame
    if frames == 0:
        return 0.0
    trimmed = samples[: frames * frame].reshape(frames, frame)
    energy = np.sqrt(np.mean(np.square(trimmed, dtype=np.float64), axis=1))
    floor = max(1e-4, 0.25 * float(np.max(energy)))
    return float(np.mean(energy >= floor))


def _score(duration: float, rms: float, speech_ratio: float) -> float:
    """Чем ближе к идеальной длительности, громче и плотнее — тем лучше образец."""
    if duration < MIN_REFERENCE_S:
        return 0.0
    # Штраф за отклонение от идеала мягкий: 3 с и 9 с оба пригодны.
    window = 1.0 / (1.0 + abs(duration - IDEAL_REFERENCE_S) / IDEAL_REFERENCE_S)
    return window * min(rms, 0.4) * max(speech_ratio, 0.1)


def pick_references(
    samples: np.ndarray,
    rate: int,
    spans: list[STTSpan],
    labels: list[str],
    *,
    max_seconds: float = MAX_REFERENCE_S,
) -> dict[str, VoiceReference]:
    """Лучший образец голоса на каждую метку говорящего.

    ``labels`` — метка (пол или id спикера) для каждой реплики из ``spans``.
    Возвращает метку → образец; метки, для которых чистого фрагмента не
    нашлось, в результат не попадают, и дубляж озвучит их обычным голосом.
    """
    if len(spans) != len(labels):
        raise ValueError("spans и labels должны быть одной длины")

    best: dict[str, tuple[float, VoiceReference]] = {}
    for span, label in zip(spans, labels):
        start = max(0, int(span.start * rate))
        end = min(len(samples), int(span.end * rate))
        if end <= start:
            continue
        chunk = samples[start:end]
        duration = len(chunk) / rate
        if duration > max_seconds:
            chunk = chunk[: int(max_seconds * rate)]
            duration = max_seconds
        score = _score(duration, _rms(chunk), _speech_ratio(chunk, rate))
        if score <= 0.0:
            continue
        current = best.get(label)
        if current is None or score > current[0]:
            best[label] = (
                score,
                VoiceReference(
                    samples=np.ascontiguousarray(chunk, dtype=np.float32),
                    sample_rate=rate,
                    label=label,
                    text=span.text or None,
                ),
            )

    references = {label: item[1] for label, item in best.items()}
    for label, reference in references.items():
        log.info(
            "образец голоса '%s': %.1f с из оригинала",
            label,
            len(reference.samples) / reference.sample_rate,
        )
    missing = sorted(set(labels) - set(references))
    if missing:
        log.info(
            "без чистого образца остались: %s — озвучу голосом движка",
            ", ".join(missing),
        )
    return references
