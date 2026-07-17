"""Метрики задержек по стадиям конвейера (Debug Console, ТЗ §20)."""
from __future__ import annotations

from collections import defaultdict, deque
from statistics import median

from uvt.events import Trace


class Metrics:
    """Скользящие окна задержек и причины потерь в live-конвейере.

    Latency остаётся медианой в миллисекундах, а drops — отдельными счётчиками:
    их нельзя смешивать в одном числе и выдавать за задержку.
    """

    def __init__(self, window: int = 50) -> None:
        self._window = window
        self.stages: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        self._counters: dict[str, int] = defaultdict(int)

    def observe(self, trace: Trace) -> None:
        for name, ms in trace.spans_ms():
            self.stages[name].append(ms)

    def record_drop(self, stage: str, reason: str) -> None:
        """Зафиксировать намеренно отброшенный live-сегмент."""
        self._counters["dropped"] += 1
        self._counters[f"dropped.{stage}"] += 1
        self._counters[f"dropped.{stage}.{reason}"] += 1

    def record_output_error(self) -> None:
        self._counters["output_errors"] += 1

    def snapshot(self) -> dict[str, float]:
        return {name: median(vals) for name, vals in self.stages.items() if vals}

    def counters_snapshot(self) -> dict[str, int]:
        """Копия счётчиков для GUI/HTTP без доступа к внутреннему состоянию."""
        return dict(self._counters)

    def format_line(self) -> str:
        snap = self.snapshot()
        if not snap:
            return ""
        total = snap.pop("total", None)
        parts = [f"{name} {ms:.0f} мс" for name, ms in snap.items()]
        if total is not None:
            parts.append(f"итого {total:.0f} мс")
        dropped = self._counters.get("dropped", 0)
        if dropped:
            parts.append(f"пропущено {dropped}")
        return " | ".join(parts)
