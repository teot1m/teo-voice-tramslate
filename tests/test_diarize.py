"""Пакетное разделение реплик по говорящим.

Главное, что проверяется: диалог двух голосов не должен рассыпаться на
десятки «спикеров» и не должен менять голос посреди сцены — именно это делала
пореплечная оценка F0.
"""
from __future__ import annotations

import numpy as np

from uvt.diarize import assign_speakers
from uvt.interfaces import STTSpan


def _voice(rate: int, seconds: float, f0: float) -> np.ndarray:
    """Синтетический «голос»: основной тон плюс две гармоники."""
    t = np.arange(int(rate * seconds)) / rate
    signal = (
        0.35 * np.sin(2 * np.pi * f0 * t)
        + 0.18 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.09 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    return signal.astype(np.float32)


def _dialogue(rate: int = 16_000) -> tuple[np.ndarray, list[STTSpan], list[float]]:
    """Диалог: низкий голос (110 Гц) и высокий (220 Гц) по очереди."""
    pattern = [110.0, 220.0, 110.0, 220.0, 110.0, 220.0]
    chunks: list[np.ndarray] = []
    spans: list[STTSpan] = []
    cursor = 0.0
    for index, f0 in enumerate(pattern):
        seconds = 1.5
        chunks.append(_voice(rate, seconds, f0))
        spans.append(STTSpan(cursor, cursor + seconds, f"реплика {index}", "en"))
        cursor += seconds
        chunks.append(np.zeros(int(rate * 0.3), dtype=np.float32))
        cursor += 0.3
    return np.concatenate(chunks), spans, pattern


class TestAssignSpeakers:
    def test_two_voices_give_two_speakers(self):
        rate = 16_000
        audio, spans, pattern = _dialogue(rate)

        layout = assign_speakers(audio, rate, spans)

        assert len(layout.speakers) == 2, layout.speakers
        # Одинаковый тон — одна и та же метка на всех своих репликах
        low = {label for label, f0 in zip(layout.labels, pattern) if f0 == 110.0}
        high = {label for label, f0 in zip(layout.labels, pattern) if f0 == 220.0}
        assert len(low) == 1 and len(high) == 1
        assert low != high

    def test_voice_role_is_stable_within_speaker(self):
        rate = 16_000
        audio, spans, pattern = _dialogue(rate)

        layout = assign_speakers(audio, rate, spans)

        by_label: dict[str, set[str]] = {}
        for label, gender in zip(layout.labels, layout.genders):
            by_label.setdefault(label, set()).add(gender)
        assert all(len(roles) == 1 for roles in by_label.values()), by_label
        # Низкий голос — мужская роль, высокий — женская
        low_label = layout.labels[0]
        high_label = layout.labels[1]
        assert layout.speakers[low_label] == "male"
        assert layout.speakers[high_label] == "female"

    def test_short_line_inherits_nearest_speaker(self):
        rate = 16_000
        audio, spans, _ = _dialogue(rate)
        # Короткое «Да» внутри реплики первого голоса
        spans.insert(1, STTSpan(1.5, 1.7, "Да", "en"))

        layout = assign_speakers(audio, rate, spans)

        assert layout.labels[1] in {layout.labels[0], layout.labels[2]}
        assert len(layout.labels) == len(spans)

    def test_speaker_cap_is_respected(self):
        rate = 16_000
        chunks, spans = [], []
        cursor = 0.0
        for index in range(6):
            seconds = 1.2
            chunks.append(_voice(rate, seconds, 90.0 + index * 35.0))
            spans.append(STTSpan(cursor, cursor + seconds, f"line {index}", "en"))
            cursor += seconds

        layout = assign_speakers(np.concatenate(chunks), rate, spans, max_speakers=2)

        assert len(layout.speakers) <= 2

    def test_silence_falls_back_to_single_voice(self):
        rate = 16_000
        audio = np.zeros(rate * 6, dtype=np.float32)
        spans = [STTSpan(0.0, 2.0, "тишина", "en"), STTSpan(2.5, 4.0, "тоже", "en")]

        layout = assign_speakers(audio, rate, spans)

        assert set(layout.labels) == {"speaker-1"}
        assert set(layout.genders) == {"male"}

    def test_empty_input(self):
        layout = assign_speakers(np.zeros(0, dtype=np.float32), 16_000, [])
        assert layout.labels == [] and layout.speakers == {}
