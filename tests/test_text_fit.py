"""Укладка перевода в тайминг текстом, а не темпом речи.

Если перевод заведомо не влезает в свой слот, модель перевода просят
переписать его короче. Сжатие формулировкой не слышно, ускорение речи —
слышно сразу, поэтому этот проход идёт до озвучки.
"""
from __future__ import annotations

import pytest

from uvt.config import AppConfig
from uvt.dub import _text_budget, _translate_all
from uvt.interfaces import STTSpan, TranslationEngine
from uvt.registry import register

LONG_LINE = (
    "Возможно, я не знаю, в какую сферу вы пойдёте, но немного откровенных "
    "вещей может помочь вам попасть в разные индустрии"
)


class _ShorteningTranslator(TranslationEngine):
    """Переводит в верхнем регистре и записывает запросы на сжатие."""

    supports_shorten = True
    instances: list["_ShorteningTranslator"] = []

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.shorten_calls: list[tuple[str, int]] = []
        _ShorteningTranslator.instances.append(self)

    async def translate(self, text, source_lang, target_lang, context):
        return str(text)

    async def translate_batch(self, texts, source_lang, target_lang):
        return [str(text) for text in texts]

    async def shorten(self, text, target_lang, max_chars):
        self.shorten_calls.append((text, max_chars))
        return "Коротко."


register("translation", "test-shorten")(_ShorteningTranslator)


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.target_lang = "ru"
    cfg.translation.engine = "test-shorten"
    cfg.tts.duration_per_char = {"ru:default": 0.066}
    return cfg


@pytest.fixture(autouse=True)
def _reset():
    _ShorteningTranslator.instances.clear()
    yield
    _ShorteningTranslator.instances.clear()


class TestBudget:
    def test_budget_follows_slot_and_compression(self):
        cfg = _cfg()
        spans = [STTSpan(0.0, 2.0, "a"), STTSpan(3.0, 4.0, "b")]
        # слот 2.85 с × max_compression 1.25 / 0.066 с на символ
        assert _text_budget(cfg, spans, 0, "male") == int(2.85 * 1.25 / 0.066)

    def test_last_line_has_no_budget(self):
        cfg = _cfg()
        spans = [STTSpan(0.0, 2.0, "a")]
        assert _text_budget(cfg, spans, 0, "male") is None

    def test_without_rate_table_there_is_no_budget(self):
        cfg = _cfg()
        cfg.tts.duration_per_char = {}
        spans = [STTSpan(0.0, 2.0, "a"), STTSpan(3.0, 4.0, "b")]
        assert _text_budget(cfg, spans, 0, "male") is None


class TestFitStage:
    async def test_overlong_translation_is_shortened(self):
        cfg = _cfg()
        spans = [
            STTSpan(0.0, 2.0, LONG_LINE, "en"),
            STTSpan(3.0, 4.0, "Okay.", "en"),
        ]

        result = await _translate_all(cfg, spans, "en", ["male", "male"], None)

        translator = _ShorteningTranslator.instances[0]
        assert len(translator.shorten_calls) == 1, "сжимать нужно только длинную реплику"
        text, budget = translator.shorten_calls[0]
        assert text == LONG_LINE
        assert budget == int(2.85 * 1.25 / 0.066)
        assert result[0] == "Коротко."
        assert result[1] == "Okay."

    async def test_short_translation_is_left_alone(self):
        cfg = _cfg()
        spans = [STTSpan(0.0, 2.0, "Yes.", "en"), STTSpan(3.0, 4.0, "Okay.", "en")]

        result = await _translate_all(cfg, spans, "en", ["male", "male"], None)

        assert _ShorteningTranslator.instances[0].shorten_calls == []
        assert result == ["Yes.", "Okay."]

    async def test_engine_without_shorten_skips_the_stage(self):
        cfg = _cfg()
        cfg.translation.engine = "dummy"
        spans = [
            STTSpan(0.0, 2.0, LONG_LINE, "en"),
            STTSpan(3.0, 4.0, "Okay.", "en"),
        ]

        # Движок без поддержки сжатия просто не участвует в стадии укладки
        result = await _translate_all(cfg, spans, "en", ["male", "male"], None)
        assert len(result) == 2
        assert _ShorteningTranslator.instances == []
