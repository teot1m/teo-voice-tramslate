"""Озвучка движками с контролем длительности и клонированием голоса.

Проверяется именно то, из-за чего дубляж звучал механически: реплика должна
синтезироваться сразу в свой тайминг (без atempo), голос — браться образцом из
оригинала, а на пол говорящего не должно уходить по копии весов в памяти.
"""
from __future__ import annotations

import numpy as np
import pytest

from uvt.config import AppConfig
from uvt.dub import _slot_seconds, _synthesize_all
from uvt.interfaces import STTSpan, TTSEngine, VoiceReference
from uvt.registry import register
from uvt.voices import pick_references


class _SlotEngine(TTSEngine):
    """Клонирующий движок: пишет, о чём его просили, и отдаёт ровно слот."""

    supports_duration = True
    supports_reference = True
    supports_emotion = True

    instances: list["_SlotEngine"] = []
    natural_seconds = 3.0

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.calls: list[dict] = []
        _SlotEngine.instances.append(self)

    async def synthesize(self, text, language):
        raise AssertionError("должен вызываться synthesize_slot")

    async def synthesize_slot(
        self, text, language, *, target_duration=None, reference=None, emotion=None
    ):
        self.calls.append(
            {
                "text": text,
                "target": target_duration,
                "reference": reference.label if reference else None,
                "emotion": emotion is not None,
            }
        )
        seconds = target_duration if target_duration else self.natural_seconds
        return np.ones(int(16_000 * seconds), dtype=np.float32), 16_000


register("tts", "test-slot")(_SlotEngine)


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.target_lang = "ru"
    cfg.tts.engine = "test-slot"
    cfg.tts.concurrency = 2
    # 0.06 с на символ — длинная реплика заведомо не влезает в 2-секундный слот
    cfg.tts.duration_per_char = {"ru:male": 0.06, "ru:female": 0.06}
    return cfg


@pytest.fixture(autouse=True)
def _reset_instances():
    _SlotEngine.instances.clear()
    _SlotEngine.natural_seconds = 3.0
    yield
    _SlotEngine.instances.clear()


class TestSlots:
    def test_slot_is_distance_to_next_line(self):
        spans = [STTSpan(0.0, 1.0, "a"), STTSpan(3.0, 4.0, "b")]
        assert _slot_seconds(spans, 0) == pytest.approx(2.85)

    def test_last_line_is_unbounded(self):
        spans = [STTSpan(0.0, 1.0, "a")]
        assert _slot_seconds(spans, 0) is None

    def test_too_short_slot_is_ignored(self):
        spans = [STTSpan(0.0, 0.2, "a"), STTSpan(0.4, 1.0, "b")]
        assert _slot_seconds(spans, 0) is None


class TestDurationAwareSynthesis:
    async def test_overlong_line_gets_target_duration_not_atempo(self):
        cfg = _cfg()
        spans = [
            STTSpan(0.0, 2.0, "source one"),
            STTSpan(3.0, 4.0, "source two"),
        ]
        long_text = (
            "очень длинная переведённая реплика, которая заведомо не влезает "
            "в отведённый ей слот и потребует укладки"
        )
        texts = [long_text, "коротко"]

        clips = await _synthesize_all(cfg, spans, texts, ["male", "male"], None)

        engine = _SlotEngine.instances[0]
        long_calls = [call for call in engine.calls if call["text"] == long_text]
        short_calls = [call for call in engine.calls if call["text"] == "коротко"]

        # Первый проход: жмём не сильнее чем в MAX_COMPRESSION раз от
        # естественной длительности (len × 0.06), а не до слота любой ценой.
        expected = max(2.85, len(long_text) * 0.06 / 1.25)
        assert long_calls[0]["target"] == pytest.approx(expected, abs=0.01)
        # Клип всё ещё длиннее слота → повторный синтез в меньшую длительность,
        # без ускорения готового звука.
        assert len(long_calls) == 2
        assert long_calls[1]["target"] < long_calls[0]["target"]
        # Последняя реплика ничем не ограничена
        assert short_calls == [{"text": "коротко", "target": None, "reference": None, "emotion": False}]
        assert len(clips) == 2

    async def test_short_line_keeps_natural_pace(self):
        cfg = _cfg()
        spans = [STTSpan(0.0, 1.0, "a"), STTSpan(9.0, 10.0, "b")]

        await _synthesize_all(cfg, spans, ["да", "нет"], ["male", "male"], None)

        assert all(call["target"] is None for call in _SlotEngine.instances[0].calls)

    async def test_overflowing_clip_is_resynthesized_not_sped_up(self, monkeypatch):
        import uvt.dub as dub_module

        def fail(*_args, **_kwargs):
            raise AssertionError("atempo не должен применяться к duration-aware движку")

        monkeypatch.setattr(dub_module, "_fit_audio_tempo", fail)

        cfg = _cfg()
        cfg.tts.duration_per_char = {}  # оценки нет → первый проход без тайминга
        _SlotEngine.natural_seconds = 6.0  # движок вернёт клип длиннее слота
        spans = [STTSpan(0.0, 2.0, "a"), STTSpan(3.0, 4.0, "b")]

        clips = await _synthesize_all(cfg, spans, ["длинно", "конец"], ["male", "male"], None)

        engine = _SlotEngine.instances[0]
        targets = [call["target"] for call in engine.calls]
        assert targets[0] is None, "первый проход — естественный темп"
        assert any(t is not None for t in targets[1:]), "нужен повторный синтез в тайминг"
        assert len(clips) == 2

    async def test_single_engine_instance_for_both_genders(self):
        cfg = _cfg()
        spans = [STTSpan(0.0, 1.0, "a"), STTSpan(9.0, 10.0, "b")]

        await _synthesize_all(cfg, spans, ["раз", "два"], ["male", "female"], None)

        assert len(_SlotEngine.instances) == 1, (
            "голос задаётся образцом — вторая копия весов не нужна"
        )

    async def test_reference_and_emotion_are_passed(self):
        cfg = _cfg()
        spans = [STTSpan(0.0, 1.0, "a"), STTSpan(9.0, 10.0, "b")]
        references = {
            "male": VoiceReference(np.ones(16_000, dtype=np.float32), 16_000, "male"),
            "female": VoiceReference(np.ones(16_000, dtype=np.float32), 16_000, "female"),
        }
        source = np.ones(16_000 * 12, dtype=np.float32)

        await _synthesize_all(
            cfg,
            spans,
            ["раз", "два"],
            ["male", "female"],
            None,
            references=references,
            source_audio=source,
            source_rate=16_000,
        )

        calls = {call["text"]: call for call in _SlotEngine.instances[0].calls}
        assert calls["раз"]["reference"] == "male"
        assert calls["два"]["reference"] == "female"
        assert all(call["emotion"] for call in calls.values())


class TestProgressivePublishing:
    """Реплика публикуется сразу после укладки — это основа прогрессивного дубляжа."""

    async def test_each_clip_is_published_once_and_final(self):
        cfg = _cfg()
        cfg.tts.duration_per_char = {}      # оценки нет → первый проход без тайминга
        _SlotEngine.natural_seconds = 6.0   # клип заведомо длиннее слота
        spans = [STTSpan(0.0, 2.0, "source", "en"), STTSpan(3.0, 4.0, "next", "en")]
        published: list = []

        clips = await _synthesize_all(
            cfg,
            spans,
            ["длинная реплика", "конец"],
            ["male", "male"],
            None,
            labels=["speaker-1", "speaker-2"],
            on_clip=published.append,
        )

        assert len(published) == 2, "каждая реплика публикуется ровно один раз"
        first = next(item for item in published if item.index == 0)
        # Опубликован уже уложенный вариант, а не первый длинный
        assert len(first.samples) / first.sample_rate < 6.0
        assert first.translated == "длинная реплика"
        assert first.original == "source"
        assert first.source_start == 0.0 and first.source_end == 2.0
        assert first.speaker == "speaker-1"
        assert first.voice_style == "male"
        # То, что опубликовано, попадает и в итоговую дорожку
        assert len(clips) == 2
        durations = sorted(len(c.samples) / c.sample_rate for c in clips)
        published_durations = sorted(
            len(c.samples) / c.sample_rate for c in published
        )
        assert durations == pytest.approx(published_durations)

    async def test_failing_subscriber_does_not_break_dubbing(self):
        cfg = _cfg()
        spans = [STTSpan(0.0, 1.0, "a", "en"), STTSpan(9.0, 10.0, "b", "en")]

        def broken(_clip):
            raise RuntimeError("подписчик упал")

        clips = await _synthesize_all(
            cfg, spans, ["раз", "два"], ["male", "male"], None, on_clip=broken
        )

        assert len(clips) == 2


class TestPickReferences:
    def test_prefers_loud_long_segment(self):
        rate = 16_000
        audio = np.zeros(rate * 20, dtype=np.float32)
        # Тихий обрывок и хороший шестисекундный фрагмент
        audio[rate : rate * 2] = 0.01
        audio[rate * 5 : rate * 11] = 0.3
        spans = [
            STTSpan(1.0, 2.0, "тихо", "en"),
            STTSpan(5.0, 11.0, "нормальная реплика", "en"),
        ]

        references = pick_references(audio, rate, spans, ["male", "male"])

        assert set(references) == {"male"}
        picked = references["male"]
        assert len(picked.samples) / picked.sample_rate == pytest.approx(6.0, abs=0.1)
        assert picked.text == "нормальная реплика"

    def test_silence_gives_no_reference(self):
        rate = 16_000
        audio = np.zeros(rate * 10, dtype=np.float32)
        spans = [STTSpan(0.0, 5.0, "тишина", "en")]

        assert pick_references(audio, rate, spans, ["male"]) == {}

    def test_length_mismatch_is_an_error(self):
        with pytest.raises(ValueError):
            pick_references(np.zeros(10, dtype=np.float32), 16_000, [], ["male"])
