"""Определение пола говорящего по высоте тона (F0) — для авто-выбора голоса.

Базовая частота речи: мужчины ~85–155 Гц, женщины ~165–255 Гц. Оцениваем F0
автокорреляцией по озвученным кадрам и берём медиану — этого достаточно,
чтобы разнести реплики диалога по мужскому/женскому голосу озвучки без
тяжёлых моделей диаризации (полная диаризация — в дорожной карте v2.0).

Замечания по устойчивости на реальной речи:
- автокорреляция нормируется на число перекрывающихся сэмплов (иначе большие
  лаги занижены и оценка смещается вверх);
- берётся наименьший лаг с пиком ≥85 % от максимума — защита от «октавной»
  ошибки, когда у женского голоса выбирается субгармоника вдвое ниже;
- порог энергии кадра относительный (30 % от RMS реплики), чтобы тихая речь
  не выпадала из оценки.
"""
from __future__ import annotations

import numpy as np

_FRAME_S = 0.04
_HOP_S = 0.02
_F0_MIN = 60.0    # Гц — ниже не бывает речи
_F0_MAX = 400.0
_MALE_BELOW = 168.0    # высокие мужские голоса доходят до ~165 Гц
_FEMALE_ABOVE = 178.0  # женские ниже ~175 Гц — редкость
_MIN_VOICED_FRAMES = 5
_VOICING_THRESHOLD = 0.35  # доля энергии в пике автокорреляции
_PEAK_TOLERANCE = 0.85     # «почти максимум» для защиты от субгармоник


def estimate_gender_f0(samples: np.ndarray, rate: int) -> tuple[str | None, float | None]:
    """Возвращает (пол | None, медианная F0 | None)."""
    frame = int(_FRAME_S * rate)
    hop = int(_HOP_S * rate)
    if len(samples) < frame * 2:
        return None, None
    lag_min = max(2, int(rate / _F0_MAX))
    lag_max = min(frame - 1, int(rate / _F0_MIN))

    signal = samples.astype(np.float64)
    overall_rms = float(np.sqrt(np.mean(signal * signal)))
    energy_floor = max(0.008, 0.3 * overall_rms)
    overlap = frame - np.arange(frame)  # для несмещённой автокорреляции

    f0_values: list[float] = []
    for i in range(0, len(signal) - frame, hop):
        x = signal[i : i + frame]
        x = x - x.mean()
        if float(np.sqrt(np.mean(x * x))) < energy_floor:
            continue
        ac = np.correlate(x, x, mode="full")[frame - 1 :] / overlap
        if ac[0] <= 0:
            continue
        window = ac[lag_min:lag_max]
        if len(window) == 0:
            continue
        peak = float(window.max())
        if peak / ac[0] < _VOICING_THRESHOLD:  # кадр без явного тона
            continue
        near_peak = np.nonzero(window >= _PEAK_TOLERANCE * peak)[0]
        k = int(near_peak[0]) + lag_min  # наименьший лаг ≈ истинный период
        f0_values.append(rate / k)

    if len(f0_values) < _MIN_VOICED_FRAMES:
        return None, None
    f0 = float(np.median(f0_values))
    if f0 < _MALE_BELOW:
        return "male", f0
    if f0 > _FEMALE_ABOVE:
        return "female", f0
    return None, f0  # пограничный тембр — пусть решает контекст


def estimate_gender(samples: np.ndarray, rate: int) -> str | None:
    """Возвращает "male" / "female" или None, если голос не определился."""
    return estimate_gender_f0(samples, rate)[0]
