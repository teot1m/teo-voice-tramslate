"""Lightweight pitch evidence for selecting a synthetic voice role.

Pitch is not a reliable statement about a person's gender or identity. Weak,
unvoiced and borderline evidence stays unknown so speaker context or a manual
voice choice can decide. No model download is required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FRAME_S = 0.04
_HOP_S = 0.02
_F0_MIN = 60.0
_F0_MAX = 400.0
_MALE_BELOW = 168.0
_FEMALE_ABOVE = 178.0
_MIN_VOICED_FRAMES = 15
_VOICING_THRESHOLD = 0.5
_PEAK_TOLERANCE = 0.85


@dataclass(frozen=True, slots=True)
class PitchEvidence:
    role: str | None = None
    f0_hz: float | None = None
    voiced_seconds: float = 0.0
    voiced_fraction: float = 0.0
    confidence: float = 0.0
    frame_starts: tuple[int, ...] = ()


def estimate_pitch(samples: np.ndarray, rate: int) -> PitchEvidence:
    """Use actual autocorrelation peaks, with enough stable voiced frames."""
    signal = np.asarray(samples, dtype=np.float64)
    if rate <= 0 or signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
        return PitchEvidence()
    frame = int(_FRAME_S * rate)
    hop = max(1, int(_HOP_S * rate))
    if frame < 4 or len(signal) < frame * 2:
        return PitchEvidence()
    lag_min = max(2, int(rate / _F0_MAX))
    lag_max = min(frame - 1, int(rate / _F0_MIN))
    if lag_max - lag_min < 3:
        return PitchEvidence()
    overall_rms = float(np.sqrt(np.mean(signal * signal)))
    energy_floor = max(0.008, 0.3 * overall_rms)
    overlap = frame - np.arange(frame)
    values: list[float] = []
    periodicities: list[float] = []
    starts: list[int] = []
    frame_count = 0
    for start in range(0, len(signal) - frame + 1, hop):
        frame_count += 1
        x = signal[start:start + frame]
        x = x - x.mean()
        if float(np.sqrt(np.mean(x * x))) < energy_floor:
            continue
        ac = np.correlate(x, x, mode="full")[frame - 1:] / overlap
        if ac[0] <= 0:
            continue
        window = ac[lag_min:lag_max]
        # A rising edge above 85% of a peak has a shorter lag and systematically
        # overstates pitch. Select the actual local maximum before interpolation.
        peaks = np.flatnonzero((window[1:-1] >= window[:-2]) & (window[1:-1] > window[2:])) + 1
        if not peaks.size:
            continue
        peak = float(window[peaks].max())
        if peak / ac[0] < _VOICING_THRESHOLD:
            continue
        candidates = peaks[window[peaks] >= _PEAK_TOLERANCE * peak]
        lag = int(candidates[0]) + lag_min
        denominator = ac[lag - 1] - 2 * ac[lag] + ac[lag + 1]
        offset = (0.5 * (ac[lag - 1] - ac[lag + 1]) / denominator) if denominator < -1e-12 else 0.0
        value = rate / (lag + float(np.clip(offset, -.5, .5)))
        if not _F0_MIN <= value <= _F0_MAX:
            continue
        values.append(value)
        periodicities.append(float(np.clip(ac[lag] / ac[0], 0, 1)))
        starts.append(start)

    seconds = min(len(signal) / rate, len(starts) * hop / rate)
    fraction = len(starts) / max(1, frame_count)
    if len(values) < _MIN_VOICED_FRAMES or fraction < 0.15:
        return PitchEvidence(voiced_seconds=seconds, voiced_fraction=fraction)
    f0 = float(np.median(values))
    # Large octave/pitch scatter weakens role evidence, even in a long segment.
    spread = float(np.median(np.abs(np.log2(np.asarray(values) / f0))))
    confidence = float(np.median(periodicities)) * max(0.0, 1.0 - spread / .4)
    role = None
    if confidence >= .6:
        role = "male" if f0 < _MALE_BELOW else "female" if f0 > _FEMALE_ABOVE else None
    return PitchEvidence(role, f0, seconds, fraction, confidence, tuple(starts))


def voiced_audio(samples: np.ndarray, rate: int, evidence: PitchEvidence) -> np.ndarray:
    """Spectral speaker cues from voiced frames, without consonant/silence bias."""
    if not evidence.frame_starts:
        return np.empty(0, dtype=np.float32)
    width = max(1, int(_FRAME_S * rate))
    return np.concatenate([samples[start:start + width] for start in evidence.frame_starts])


def estimate_gender_f0(samples: np.ndarray, rate: int) -> tuple[str | None, float | None]:
    evidence = estimate_pitch(samples, rate)
    return evidence.role, evidence.f0_hz


def estimate_gender(samples: np.ndarray, rate: int) -> str | None:
    return estimate_gender_f0(samples, rate)[0]
