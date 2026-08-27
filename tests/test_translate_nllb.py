"""NLLB CTranslate2 adapter: language framing, batching, and decode."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from uvt.engines.translate_nllb import NLLBCTranslate2Translator, _language_code


class FakeTokenizer:
    def __init__(self):
        self.encoded = []

    def encode_batch(self, texts, add_special_tokens):
        self.encoded.append((list(texts), add_special_tokens))
        return [SimpleNamespace(tokens=text.split()) for text in texts]

    @staticmethod
    def token_to_id(token):
        return {"Привіт": 10, "світе": 11, "</s>": 2}.get(token)

    @staticmethod
    def decode(ids, skip_special_tokens):
        assert skip_special_tokens is True
        return "  Привіт   світе  " if ids == [10, 11, 2] else ""


class FakeTranslator:
    def __init__(self):
        self.call = None

    def translate_batch(self, source_tokens, **kwargs):
        self.call = (source_tokens, kwargs)
        return [
            SimpleNamespace(hypotheses=[["ukr_Cyrl", "Привіт", "світе", "</s>"]])
            for _ in source_tokens
        ]


async def test_nllb_uses_source_suffix_target_prefix_and_batch_settings():
    cfg = SimpleNamespace(batch_size=32, beam_size=1, max_decoding_length=128)
    engine = NLLBCTranslate2Translator(cfg)
    engine.batch_hint = 32
    engine._beam_size = 1
    engine._max_decoding_length = 128
    engine._tokenizer = FakeTokenizer()
    engine._translator = FakeTranslator()

    result = await engine.translate_batch(["Hello world"], "en-US", "uk")

    assert result == ["Привіт світе"]
    assert engine._tokenizer.encoded == [(["Hello world"], False)]
    source, options = engine._translator.call
    assert source == [["Hello", "world", "</s>", "eng_Latn"]]
    assert options["target_prefix"] == [["ukr_Cyrl"]]
    assert options["beam_size"] == 1
    assert options["max_batch_size"] == 32


def test_nllb_language_mapping_is_explicit():
    assert _language_code("ru-RU") == "rus_Cyrl"
    assert _language_code("ukr_Cyrl") == "ukr_Cyrl"
    with pytest.raises(RuntimeError, match="язык 'auto'"):
        _language_code(None)
