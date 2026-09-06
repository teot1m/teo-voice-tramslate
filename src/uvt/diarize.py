"""Пакетное разделение реплик по говорящим для дубляжа.

Live-путь назначает спикера на ходу (``uvt.services.speaker``): он видит только
прошлые реплики и вынужден решать сразу. Пакетный дубляж видит весь файл, и это
качественно другая задача — поэтому здесь офлайн-кластеризация:

- признаки F0 и спектра берутся из озвученных кадров; согласные и паузы
  не должны создавать нового говорящего;
- кластеризация агломеративная по всем репликам сразу, а не жадная по одной;
- роль голоса выбирается по уверенным фрагментам кластера, а не по каждой
  реплике отдельно. Именно из-за пореплечного F0 диалог двух человек
  раскладывался в «30 мужских, 100 женских», и голос прыгал посреди сцены;
- короткие реплики («Окей», «Да») не образуют своих кластеров: их F0 ненадёжен,
  поэтому они наследуют говорящего ближайшей по времени надёжной реплики.

Это не биометрическая идентификация: метки живут только внутри одной задачи и
нужны, чтобы закрепить за говорящим голос и образец для клонирования.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from uvt.gender import PitchEvidence, estimate_pitch, voiced_audio
from uvt.interfaces import STTSpan
from uvt.services.speaker import _embedding

log = logging.getLogger("uvt.diarize")

MIN_RELIABLE_S = 0.6      # короче — F0 и спектр слишком шумные для кластера
MAX_SPEAKERS = 8
# Абсолютного порога здесь нет намеренно. Live-путь сравнивает реплику с
# центроидом кластера и может позволить себе фиксированный 0.42; офлайн мы
# видим всю запись, и масштаб расстояний зависит от неё целиком. На реальном
# диалоге расстояния внутри голоса выходили 0.04–0.12, между голосами
# 0.21–0.34 — фиксированный 0.42 сливал оба голоса в один кластер.
# Поэтому число говорящих определяется по самому большому скачку расстояния
# слияния: он и есть граница между «тот же голос» и «другой голос».
MIN_MERGE_GAP = 0.05      # скачок меньше — запись однородна, один голос
MIN_SEPARATION = 0.12     # ближе этого кластеры не считаем разными голосами


@dataclass(slots=True)
class SpeakerLayout:
    """Раскладка реплик по говорящим для одной задачи дубляжа."""

    labels: list[str]        # метка говорящего на каждую реплику
    genders: list[str]       # роль голоса (male/female) на каждую реплику
    speakers: dict[str, str] # метка → роль голоса


def _average_linkage(
    features: np.ndarray,
    max_speakers: int,
    *,
    min_gap: float = MIN_MERGE_GAP,
    min_separation: float = MIN_SEPARATION,
) -> list[int]:
    """Кластеризация реплик с автоматическим выбором числа говорящих.

    Сначала строится полная последовательность слияний по среднему расстоянию,
    затем выбирается шаг с самым большим скачком расстояния: до него сливались
    реплики одного голоса, после — начинают сливаться разные. Абсолютный порог
    для этого не годится — масштаб расстояний зависит от записи.
    """
    count = len(features)
    if count <= 1:
        return [0] * count

    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)

    def cluster_distance(a: list[int], b: list[int]) -> float:
        return float(np.mean(distances[np.ix_(a, b)]))

    clusters: list[list[int]] = [[i] for i in range(count)]
    # История: расстояние слияния и состояние кластеров сразу после него.
    history: list[tuple[float, list[list[int]]]] = []
    while len(clusters) > 1:
        best = (float("inf"), 0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                value = cluster_distance(clusters[i], clusters[j])
                if value < best[0]:
                    best = (value, i, j)
        value, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)
        history.append((value, [list(item) for item in clusters]))

    merge_distances = [item[0] for item in history]
    # Ищем самый большой разрыв: слияние после него объединяет разные голоса.
    split_at: int | None = None
    best_gap = 0.0
    for index in range(len(merge_distances) - 1):
        gap = merge_distances[index + 1] - merge_distances[index]
        if gap <= best_gap or gap < min_gap:
            continue
        if merge_distances[index + 1] < min_separation:
            continue
        best_gap = gap
        split_at = index

    if split_at is None:
        # Разрыва нет: запись однородна — либо один голос, либо мы не можем
        # честно разделить её этими признаками.
        selected = history[-1][1]
    else:
        selected = history[split_at][1]
        # Лимит спикеров сильнее найденного разрыва: шумная дорожка иначе
        # рассыпается на десятки «говорящих».
        position = split_at
        while len(selected) > max_speakers and position + 1 < len(history):
            position += 1
            selected = history[position][1]

    assignment = [0] * count
    # Нумерация по первому появлению — метки стабильны и читаются в логах.
    order = sorted(range(len(selected)), key=lambda index: min(selected[index]))
    for number, index in enumerate(order):
        for item in selected[index]:
            assignment[item] = number
    return assignment


def assign_speakers(
    samples: np.ndarray,
    rate: int,
    spans: list[STTSpan],
    *,
    max_speakers: int = MAX_SPEAKERS,
    min_separation: float = MIN_SEPARATION,
    fallback_gender: str = "male",
) -> SpeakerLayout:
    """Раскладывает реплики по говорящим и назначает каждому роль голоса."""
    if not spans:
        return SpeakerLayout([], [], {})

    features: list[np.ndarray] = []
    evidence: list[PitchEvidence] = []
    reliable: list[int] = []
    for index, span in enumerate(spans):
        start = max(0, int(span.start * rate))
        end = min(len(samples), int(span.end * rate))
        chunk = samples[start:end] if end > start else np.zeros(0, dtype=np.float32)
        pitch = estimate_pitch(chunk, rate)
        evidence.append(pitch)
        feature = _embedding(voiced_audio(chunk, rate, pitch), rate, pitch.f0_hz)
        # ZCR mostly reflects the phonemes in a sentence, not speaker identity.
        # Do not let one consonant-heavy line create a new speaker by itself.
        feature[3] *= 0.15
        features.append(feature)
        if ((end - start) >= int(MIN_RELIABLE_S * rate)
                and pitch.voiced_seconds >= .3 and pitch.confidence >= .5
                and pitch.f0_hz is not None):
            reliable.append(index)

    if not reliable:
        log.info(
            "надёжных озвученных фрагментов нет — оставляю один голос на всех"
        )
        labels = ["speaker-1"] * len(spans)
        return SpeakerLayout(labels, [fallback_gender] * len(spans), {"speaker-1": fallback_gender})

    matrix = np.vstack([features[index] for index in reliable])
    assignment = _average_linkage(matrix, max_speakers, min_separation=min_separation)

    labels: list[str | None] = [None] * len(spans)
    for position, index in enumerate(reliable):
        labels[index] = f"speaker-{assignment[position] + 1}"

    # Короткие реплики наследуют ближайшую по времени надёжную: в диалоге
    # «Окей» почти всегда принадлежит одному из уже найденных голосов.
    for index, label in enumerate(labels):
        if label is not None:
            continue
        current = spans[index]
        def temporal_distance(candidate: int) -> tuple[float, float]:
            other = spans[candidate]
            gap = max(0.0, other.start - current.end, current.start - other.end)
            midpoint = abs((other.start + other.end) - (current.start + current.end))
            return gap, midpoint
        nearest = min(reliable, key=temporal_distance)
        labels[index] = labels[nearest]

    # Short/unvoiced phrases inherit identity but cannot overturn the role of
    # a speaker with stronger evidence. Roles need a voiced-duration majority;
    # two low (or two high) voices are never forced into opposite roles.
    votes_by_label: dict[str, dict[str, float]] = {}
    weights_by_label: dict[str, float] = {}
    for label in dict.fromkeys(labels):
        votes = {"male": 0.0, "female": 0.0}
        weight = 0.0
        for index in reliable:
            pitch = evidence[index]
            if labels[index] != label:
                continue
            amount = min(3.0, pitch.voiced_seconds) * pitch.confidence
            weight += amount
            if pitch.role is not None:
                votes[pitch.role] += amount
        votes_by_label[str(label)] = votes
        weights_by_label[str(label)] = weight
    totals = {role: sum(votes[role] for votes in votes_by_label.values()) for role in ("male", "female")}
    total = sum(weights_by_label.values())
    dominant = max(totals, key=totals.get)
    contextual_fallback = dominant if total >= .3 and totals[dominant] / total >= .75 else fallback_gender
    speakers = {}
    for label, votes in votes_by_label.items():
        total = weights_by_label[label]
        role = max(votes, key=votes.get)
        speakers[label] = role if total >= .3 and votes[role] / total >= .7 else contextual_fallback

    final_labels = [str(label) for label in labels]
    genders = [speakers[label] for label in final_labels]
    summary = ", ".join(
        f"{label}: {speakers[label]}, {final_labels.count(label)} реплик"
        for label in dict.fromkeys(final_labels)
    )
    log.info("говорящих найдено %d (%s)", len(speakers), summary)
    return SpeakerLayout(final_labels, genders, speakers)
