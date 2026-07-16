"""Энергетический VAD — резервный вариант без зависимостей.

Порог адаптивный: 10-й перцентиль RMS за последние ~3 секунды считается
уровнем фона; кадр громче фона в energy_ratio раз считается речью.
Хуже Silero на шумном фоне, но всегда доступен (и детерминирован в тестах).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from uvt.interfaces import VADEngine
from uvt.registry import register


@register("vad", "energy")
class EnergyVAD(VADEngine):
    frame_samples = 512

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._recent: deque[float] = deque(maxlen=94)  # ~3 с кадров по 32 мс
        self._ratio = float(getattr(cfg, "energy_ratio", 4.0))
        self._min_rms = float(getattr(cfg, "energy_min_rms", 0.005))

    def prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        self._recent.append(rms)
        if len(self._recent) >= 10:
            floor = max(float(np.percentile(self._recent, 10)), 1e-5)
        else:
            floor = max(rms, 1e-4)
        return 1.0 if rms > max(floor * self._ratio, self._min_rms) else 0.0
