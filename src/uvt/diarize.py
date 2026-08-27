"""Пакетное разделение реплик по говорящим для дубляжа.

Live-путь назначает спикера на ходу (``uvt.services.speaker``): он видит только
прошлые реплики и вынужден решать сразу. Пакетный дубляж видит весь файл, и это
качественно другая задача — поэтому здесь офлайн-кластеризация:

- признаки те же, что в live (F0, спектральный центроид, rolloff, ZCR), чтобы
  два пути не расходились в оценках;
- кластеризация агломеративная по всем репликам сразу, а не жадная по одной;
- пол голоса определяется голосованием внутри кластера, а не по каждой
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

from uvt.gender import estimate_gender_f0
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
# Роль голоса: абсолютная граница мужской/женской медианы F0…
FEMALE_F0_HZ = 173.0
# …и относительное правило, когда все кластеры оказались по одну её сторону.
RELATIVE_F0_SPREAD = 0.25


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
    f0_values: list[float | None] = []
    reliable: list[int] = []
    for index, span in enumerate(spans):
        start = max(0, int(span.start * rate))
        end = min(len(samples), int(span.end * rate))
        chunk = samples[start:end] if end > start else np.zeros(0, dtype=np.float32)
        gender, f0 = estimate_gender_f0(chunk, rate) if len(chunk) else (None, None)
        features.append(_embedding(chunk, rate, f0))
        f0_values.append(f0)
        if (end - start) >= int(MIN_RELIABLE_S * rate) and float(np.any(features[-1])):
            reliable.append(index)

    if not reliable:
        log.info(
            "надёжных по длительности реплик нет — оставляю один голос на всех"
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
        nearest = min(reliable, key=lambda candidate: abs(candidate - index))
        labels[index] = labels[nearest]

    # Роль голоса — голосованием F0 внутри кластера, а не по каждой реплике.
    medians: dict[str, float | None] = {}
    for label in dict.fromkeys(labels):
        members = [i for i, value in enumerate(labels) if value == label]
        votes = [f0_values[i] for i in members if f0_values[i] is not None]
        medians[str(label)] = float(np.median(votes)) if votes else None

    speakers = {
        label: (
            fallback_gender
            if median is None
            else ("female" if median >= FEMALE_F0_HZ else "male")
        )
        for label, median in medians.items()
    }

    # Бывает, что все кластеры оказались по одну сторону границы: например у
    # обоих участников диалога высокая медиана F0. Абсолютное правило тогда
    # выдаёт всем одну роль, и диалог звучит одним голосом. Если кластеры
    # различаются заметно, назначаем роли относительно друг друга.
    known = {label: value for label, value in medians.items() if value is not None}
    if len(known) >= 2 and len(set(speakers[label] for label in known)) == 1:
        low = min(known, key=lambda label: known[label])
        high = max(known, key=lambda label: known[label])
        spread = (known[high] - known[low]) / max(known[low], 1e-6)
        if spread >= RELATIVE_F0_SPREAD:
            speakers[low] = "male"
            speakers[high] = "female"
            log.info(
                "медианы F0 по одну сторону границы (%.0f и %.0f Гц) — "
                "роли назначены относительно друг друга",
                known[low],
                known[high],
            )

    final_labels = [str(label) for label in labels]
    genders = [speakers[label] for label in final_labels]
    summary = ", ".join(
        f"{label}: {speakers[label]}, {final_labels.count(label)} реплик"
        for label in dict.fromkeys(final_labels)
    )
    log.info("говорящих найдено %d (%s)", len(speakers), summary)
    return SpeakerLayout(final_labels, genders, speakers)
