"""Метрики задержек по стадиям конвейера (Debug Console, ТЗ §20)."""
from __future__ import annotations

from collections import defaultdict, deque
from statistics import median

from uvt.events import Trace


class Metrics:
    """Скользящие окна задержек: медиана по последним 50 сегментам."""

    def __init__(self, window: int = 50) -> None:
        self._window = window
        self.stages: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )

    def observe(self, trace: Trace) -> None:
        for name, ms in trace.spans_ms():
            self.stages[name].append(ms)

    def snapshot(self) -> dict[str, float]:
        return {name: median(vals) for name, vals in self.stages.items() if vals}

    def format_line(self) -> str:
        snap = self.snapshot()
        if not snap:
            return ""
        total = snap.pop("total", None)
        parts = [f"{name} {ms:.0f} мс" for name, ms in snap.items()]
        if total is not None:
            parts.append(f"итого {total:.0f} мс")
        return " | ".join(parts)
