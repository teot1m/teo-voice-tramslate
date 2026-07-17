"""Focused checks for live source-clock, admission control and auto voice mapping."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from uvt.bus import Bus
from uvt.config import AppConfig, SpeakerConfig, VADConfig
from uvt.events import AudioChunk, SpeechSegment, Trace, Transcript, Translation, TtsAudio, now
from uvt.metrics import Metrics
from uvt.segmenter import FRAME
from uvt.services.base import Service
from uvt.services.output import OutputService
from uvt.services.speaker import SpeakerService
from uvt.services.tts import TTSService
from uvt.services.vad import VADService


RATE = 16000


def _tone(freq: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


async def test_vad_preserves_source_audio_clock_and_assigns_target_deadline():
    cfg = AppConfig(vad=VADConfig(engine="energy"))
    cfg.latency.preset = "ultra"
    cfg.output.target_delay_s = 0.25
    cfg.output.max_backlog_s = 1.5
    service = VADService(Bus(), cfg, Metrics())
    await service.setup()

    signal = np.concatenate(
        [np.zeros(RATE // 2, dtype=np.float32), _tone(180.0), np.zeros(RATE, dtype=np.float32)]
    )
    source_origin = 1234.0
    segments = []
    step = RATE // 50
    for offset in range(0, len(signal), step):
        samples = signal[offset : offset + step]
        start = source_origin + offset / RATE
        out = await service.handle(
            AudioChunk(samples=samples, sample_rate=RATE, ts=start, end_ts=start + len(samples) / RATE)
        )
        if out:
            segments.extend(out)

    assert len(segments) == 1
    segment = segments[0]
    # Не offsets от нуля Segmenter, а тот же clock, который был у AudioChunk.
    assert source_origin <= segment.start_ts < segment.end_ts
    assert segment.trace.source_start_ts == segment.start_ts
    assert segment.trace.source_end_ts == segment.end_ts
    assert segment.target_ts == segment.end_ts + 0.25
    assert segment.deadline_ts == segment.target_ts + 1.5


async def test_vad_prunes_audio_clock_during_long_silence_but_keeps_preroll():
    """A quiet call must not retain every input timestamp forever."""
    cfg = AppConfig(vad=VADConfig(engine="energy"))
    cfg.latency.preset = "ultra"
    service = VADService(Bus(), cfg, Metrics())
    await service.setup()

    frame_samples = FRAME
    frame_s = frame_samples / RATE
    source_origin = 10_000.0
    quiet_frames = 240
    silence = np.zeros(frame_samples, dtype=np.float32)

    for index in range(quiet_frames):
        start = source_origin + index * frame_s
        await service.handle(AudioChunk(silence, RATE, start, start + frame_s))

    # The segmenter needs only its short pre-roll while inactive, not 7.7 s of
    # individual input spans. ``+1`` covers a span overlapping a partial tail.
    assert len(service._audio_clock._spans) <= service.segmenter.p.pre_roll_frames + 1

    segments = []
    signal = np.concatenate([_tone(180.0), np.zeros(RATE, dtype=np.float32)])
    for offset in range(0, len(signal), frame_samples):
        samples = signal[offset : offset + frame_samples]
        index = quiet_frames + offset // frame_samples
        start = source_origin + index * frame_s
        out = await service.handle(AudioChunk(samples, RATE, start, start + len(samples) / RATE))
        if out:
            segments.extend(out)

    assert len(segments) == 1
    # Pre-roll still maps to the source clock after a long silent period; it
    # must not fall back to Segmenter-relative seconds from the session start.
    assert segments[0].start_ts > source_origin + 7.0


class _FakeOutputStream:
    def __init__(self) -> None:
        self.writes: list[np.ndarray] = []

    def write(self, samples: np.ndarray) -> None:
        self.writes.append(samples.copy())

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


async def test_output_waits_for_target_and_records_actual_write_end():
    cfg = AppConfig()
    cfg.output.sample_rate = 24000
    cfg.output.max_backlog_s = 1.0
    metrics = Metrics()
    service = OutputService(Bus(), cfg, metrics)
    service._stream = _FakeOutputStream()
    service._closing = False

    trace = Trace(source_start_ts=now() - 0.1, source_end_ts=now() - 0.05)
    trace.mark("speech_end")
    trace.target_ts = now() + 0.03
    trace.deadline_ts = trace.target_ts + 0.5
    segment = SpeechSegment(np.zeros(160, dtype=np.float32), RATE, 0.0, 0.01, trace)
    transcript = Transcript(segment, "hello", "en", 1.0, trace)
    translation = Translation(transcript, "привет", "ru", False, trace)
    audio = TtsAudio(translation, np.ones(240, dtype=np.float32), 24000, trace)

    await service.handle(audio)

    assert trace.marks["play"] >= trace.target_ts
    assert trace.marks["write_end"] >= trace.marks["play"]
    assert service._stream.writes
    snapshot = metrics.snapshot()
    assert "play→write_end" in snapshot
    assert "source_end→write_end" in snapshot
    assert "sync_drift" in snapshot


def _audio_for_output(trace: Trace, samples: np.ndarray) -> TtsAudio:
    segment = SpeechSegment(np.zeros(160, dtype=np.float32), RATE, 0.0, 0.01, trace)
    transcript = Transcript(segment, "hello", "en", 1.0, trace)
    translation = Translation(transcript, "привет", "ru", False, trace)
    return TtsAudio(translation, samples, 24000, trace)


async def test_null_output_does_not_report_fake_playback_or_negative_sync():
    """Null sink keeps processing metrics, but has no physical playback clock."""
    cfg = AppConfig()
    metrics = Metrics()
    service = OutputService(Bus(), cfg, metrics)
    service._stream = None

    trace = Trace(source_end_ts=now() + 2.0, target_ts=now() + 2.75)
    trace.mark("speech_end")
    trace.mark("stt")
    trace.mark("translate")
    trace.mark("tts")
    await service.handle(_audio_for_output(trace, np.ones(240, dtype=np.float32)))

    assert "play" not in trace.marks and "write_end" not in trace.marks
    snapshot = metrics.snapshot()
    assert "total" in snapshot
    assert "source_end→play" not in snapshot
    assert "sync_drift" not in snapshot


async def test_output_prefers_newest_ready_audio_while_waiting_for_target():
    cfg = AppConfig()
    cfg.output.latest_wins = True
    bus = Bus()
    metrics = Metrics()
    service = OutputService(bus, cfg, metrics)
    stream = _FakeOutputStream()

    async def _fake_setup():
        service._stream = stream
        service._closing = False

    service.setup = _fake_setup  # type: ignore[method-assign]
    await service.start()
    await asyncio.sleep(0)
    try:
        old_trace = Trace(target_ts=now() + 0.2, deadline_ts=now() + 1.0)
        new_trace = Trace(target_ts=now() + 0.05, deadline_ts=now() + 1.0)
        bus.topic("tts").publish(_audio_for_output(old_trace, np.ones(240, dtype=np.float32)))
        await asyncio.sleep(0.01)
        bus.topic("tts").publish(_audio_for_output(new_trace, np.full(240, 2.0, dtype=np.float32)))
        await _wait_for(lambda: bool(stream.writes))

        assert float(stream.writes[0][0]) == 2.0
        assert old_trace.dropped_reason == "superseded"
        assert metrics.counters_snapshot()["dropped.output.superseded"] >= 1
    finally:
        await service.stop()


@dataclass
class _RealtimeEvent:
    trace: Trace
    label: str


class _RecordingRealtimeService(Service):
    name = "stt"
    consumes = "test-input"
    produces = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.handled: list[str] = []

    async def handle(self, item: _RealtimeEvent):
        self.handled.append(item.label)
        return None


async def _wait_for(predicate, timeout: float = 1.0) -> None:
    async def _check():
        while not predicate():
            await asyncio.sleep(0.002)

    await asyncio.wait_for(_check(), timeout)


async def test_realtime_service_drops_expired_and_uses_latest_wins_before_handle():
    cfg = AppConfig()
    cfg.output.latest_wins = True
    bus = Bus()
    metrics = Metrics()
    service = _RecordingRealtimeService(bus, cfg, metrics)
    await service.start()
    await asyncio.sleep(0)  # subscription is created inside the service task
    try:
        expired = _RealtimeEvent(Trace(deadline_ts=now() - 0.01), "expired")
        bus.topic("test-input").publish(expired)
        await _wait_for(lambda: metrics.counters_snapshot().get("dropped.stt.deadline") == 1)
        assert service.handled == []

        old = _RealtimeEvent(Trace(deadline_ts=now() + 1.0), "old")
        latest = _RealtimeEvent(Trace(deadline_ts=now() + 1.0), "latest")
        bus.topic("test-input").publish(old)
        bus.topic("test-input").publish(latest)
        await _wait_for(lambda: service.handled)
        assert service.handled == ["latest"]
        assert metrics.counters_snapshot()["dropped.stt.superseded"] == 1
    finally:
        await service.stop()


def test_speaker_service_maps_timbre_stably_without_male_default():
    speakers = SpeakerService(SpeakerConfig())
    low = speakers.assign(_tone(110.0), RATE)
    high = speakers.assign(_tone(220.0), RATE)
    low_again = speakers.assign(_tone(112.0), RATE)
    unknown = SpeakerService(SpeakerConfig()).assign(np.zeros(RATE, dtype=np.float32), RATE)

    assert low.voice_gender == "male"
    assert high.voice_gender == "female"
    assert low.speaker_id == low_again.speaker_id
    assert low.timbre == "low" and high.timbre == "high"
    # При неуверенном F0 это стабильная target voice role, не наследование
    # последнего / default male voice движка.
    assert unknown.voice_gender == "female"


def test_speaker_role_does_not_flip_when_f0_fluctuates_inside_cluster():
    # A deliberately wide threshold keeps these two F0 estimates in one
    # session-local cluster. The target role is assigned once, not per phrase.
    speakers = SpeakerService(SpeakerConfig(match_threshold=1.0))
    first = speakers.assign(_tone(110.0), RATE)
    fluctuating = speakers.assign(_tone(220.0), RATE)

    assert first.speaker_id == fluctuating.speaker_id
    assert first.voice_gender == "male"
    assert fluctuating.voice_gender == "male"
    assert speakers._speakers[0].voice_gender == "male"


def test_speaker_cap_does_not_mutate_nearest_cluster_for_distant_timbre():
    speakers = SpeakerService(SpeakerConfig(max_speakers=1, match_threshold=0.05))
    first = speakers.assign(_tone(110.0), RATE)
    known = speakers._speakers[0]
    embedding_before = known.embedding.copy()

    distant = speakers.assign(_tone(220.0), RATE)

    # At the configured cap UVT gives a bounded fallback role, but preserves
    # the established speaker's model instead of blending in a new voice.
    assert distant.speaker_id == first.speaker_id
    assert distant.voice_gender == first.voice_gender
    np.testing.assert_array_equal(known.embedding, embedding_before)
    assert known.assignments == 1


class _VoiceRecordingEngine:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.roles: list[str] = []

    async def synthesize(self, text: str, language: str):
        self.roles.append(self.cfg.voice_gender)
        return np.zeros(240, dtype=np.float32), 24000


async def test_tts_auto_uses_speaker_mapping_and_restores_config():
    cfg = AppConfig()
    cfg.tts.voice = "auto"
    cfg.tts.voice_gender = "auto"
    service = TTSService(Bus(), cfg, Metrics())
    engine = _VoiceRecordingEngine(cfg.tts)
    service.engine = engine
    service.speakers = SpeakerService(cfg.speaker)
    service._out_rate = 24000

    trace = Trace()
    segment = SpeechSegment(_tone(220.0), RATE, 1.0, 2.0, trace)
    transcript = Transcript(segment, "hello", "en", 0.9, trace)
    translation = Translation(transcript, "привет", "ru", False, trace)
    output = await service.handle(translation)

    assert output is not None
    assert engine.roles == ["female"]
    assert segment.speaker_id is not None
    assert segment.speaker_timbre == "high"
    assert cfg.tts.voice_gender == "auto"
