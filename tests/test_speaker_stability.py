"""Conservative timbre roles: identity should survive phonemes and intonation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from uvt.audio import resample
from uvt.diarize import assign_speakers
from uvt.gender import estimate_pitch
from uvt.interfaces import STTSpan

RATE = 16000


def tone(frequency, seconds=1.2):
    t = np.arange(round(RATE * seconds)) / RATE
    return (.3 * np.sin(2 * np.pi * frequency * t) + .12 * np.sin(4 * np.pi * frequency * t)).astype(np.float32)


def recording(frequencies):
    chunks = [tone(f) for f in frequencies]
    spans = [STTSpan(i * 1.2, (i + 1) * 1.2, str(i), "en") for i in range(len(chunks))]
    return np.concatenate(chunks), spans


@pytest.mark.parametrize("frequency", [110., 160., 173., 220.])
def test_pitch_uses_peak_not_its_rising_edge(frequency):
    pitch = estimate_pitch(tone(frequency), RATE)
    assert pitch.f0_hz == pytest.approx(frequency, rel=.01)
    assert pitch.voiced_seconds > 1
    assert pitch.confidence > .8


def test_brief_tonal_burst_cannot_establish_pitch_or_a_new_speaker():
    audio = np.concatenate([tone(220, .12), np.zeros(RATE * 2, dtype=np.float32)])
    pitch = estimate_pitch(audio, RATE)
    assert pitch.f0_hz is None
    assert pitch.role is None
    assert pitch.voiced_seconds < .3


def test_noise_has_no_confident_voice_role():
    audio = np.random.default_rng(42).normal(0, .1, RATE * 3).astype(np.float32)
    pitch = estimate_pitch(audio, RATE)
    assert pitch.role is None
    assert pitch.f0_hz is None


@pytest.mark.parametrize("pattern,role", [([100., 150.] * 3, "male"), ([200., 300.] * 3, "female")])
def test_distinct_same_range_voices_are_not_forced_into_opposite_roles(pattern, role):
    audio, spans = recording(pattern)
    layout = assign_speakers(audio, RATE, spans)
    assert len(layout.speakers) == 2
    assert set(layout.genders) == {role}


def test_one_raised_phrase_does_not_overrule_ambiguous_majority():
    audio, spans = recording([173., 173., 185., 173., 173.])
    layout = assign_speakers(audio, RATE, spans)
    assert len(layout.speakers) == 1
    assert set(layout.genders) == {"male"}  # fallback, not a gender claim


@pytest.mark.parametrize("fallback", ["male", "female"])
def test_ambiguous_pitch_uses_requested_fallback_role(fallback):
    audio, spans = recording([173.] * 3)
    layout = assign_speakers(audio, RATE, spans, fallback_gender=fallback)
    assert set(layout.genders) == {fallback}


def test_unvoiced_line_uses_nearest_time_not_nearest_list_index():
    audio = np.zeros(RATE * 104, dtype=np.float32)
    timings = [(0., 1.2, 110.), (2., 3.2, 110.), (98.8, 98.95, None), (99., 100.2, 220.), (101., 102.2, 220.)]
    spans = []
    for i, (start, end, frequency) in enumerate(timings):
        if frequency:
            chunk = tone(frequency, end - start)
            offset = round(start * RATE)
            audio[offset:offset + len(chunk)] = chunk
        spans.append(STTSpan(start, end, str(i), "en"))
    layout = assign_speakers(audio, RATE, spans)
    assert layout.labels[2] == layout.labels[3]
    assert layout.labels[2] != layout.labels[1]


@pytest.mark.parametrize("spans_file", ["nemotron-smoke.json", "pipeline-moss-ru.json"])
def test_neutral_single_voice_fixture_stays_one_speaker(spans_file):
    root = Path(__file__).resolve().parents[1]
    source = root / "artifacts/benchmark/source-en.wav"
    metadata = root / "artifacts/verification/next-local-models" / spans_file
    if not source.is_file() or not metadata.is_file():
        pytest.skip("optional local neutral speech verification artifacts")
    audio, rate = sf.read(source, dtype="float32")
    audio = resample(audio, rate, RATE)
    rows = json.loads(metadata.read_text())
    rows = rows["spans"] if isinstance(rows, dict) else rows
    spans = [STTSpan(r["start"], r["end"], r.get("text", r.get("original", "")), "en") for r in rows]
    layout = assign_speakers(audio, RATE, spans)
    assert len(layout.speakers) == 1
    assert len(set(layout.genders)) == 1
