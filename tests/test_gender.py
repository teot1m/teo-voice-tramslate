"""Определение пола голоса по базовой частоте."""
import numpy as np

from uvt.gender import estimate_gender

RATE = 16000


def _tone(freq: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    # немного гармоник, чтобы походить на голосовой сигнал
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
    return signal.astype(np.float32)


def test_low_pitch_is_male():
    assert estimate_gender(_tone(110.0), RATE) == "male"


def test_high_pitch_is_female():
    assert estimate_gender(_tone(220.0), RATE) == "female"


def test_silence_is_unknown():
    assert estimate_gender(np.zeros(RATE, dtype=np.float32), RATE) is None
