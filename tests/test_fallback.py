"""Cloud-движки должны переключаться на локальный резерв без сети и моделей."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from uvt.config import AppConfig, STTFallbackConfig, TranslationConfig, TranslationFallbackConfig
from uvt.dub import _transcribe_all, _translate_all
from uvt.interfaces import STTEngine, STTResult, STTSpan, TranslationEngine


class _CloudSTTFails(STTEngine):
    async def transcribe(self, *_args):
        raise RuntimeError("HTTP 400 from cloud STT")

    async def transcribe_long(self, *_args, **_kwargs):
        raise RuntimeError("HTTP 400 from cloud STT")


class _LocalSTT(STTEngine):
    async def transcribe(self, *_args):
        return STTResult("local speech", "en")

    async def transcribe_long(self, *_args, **_kwargs):
        return [STTSpan(0.0, 1.0, "local speech", "en")]


class _CloudTranslatorFails(TranslationEngine):
    # Несколько remote-пачек стартуют параллельно; после отказа все они не
    # должны одновременно занять память локального Ollama.
    batch_hint = 1
    concurrency_hint = 3

    async def translate(self, *_args):
        raise RuntimeError("HTTP 429 from cloud translator")

    async def translate_batch_tagged(self, *_args):
        raise RuntimeError("HTTP 429 from cloud translator")


class _LargeBatchCloudTranslatorFails(_CloudTranslatorFails):
    batch_hint = 20
    concurrency_hint = 1


class _LocalTranslator(TranslationEngine):
    async def translate(self, text, _source_lang, target_lang, _context):
        return f"LOCAL {text} [{target_lang}]"

    async def translate_batch_tagged(self, texts, _source_lang, target_lang, _genders):
        return [f"LOCAL {text} [{target_lang}]" for text in texts]


class _BatchProbeLocalTranslator(_LocalTranslator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.batch_hint = cfg.batch_size
        self.calls: list[list[str]] = []

    async def translate_batch_tagged(self, texts, source_lang, target_lang, genders):
        self.calls.append(list(texts))
        return await super().translate_batch_tagged(texts, source_lang, target_lang, genders)


class _SerialProbeLocalTranslator(_LocalTranslator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.active = 0
        self.max_active = 0

    async def translate_batch_tagged(self, texts, source_lang, target_lang, genders):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            return await super().translate_batch_tagged(texts, source_lang, target_lang, genders)
        finally:
            self.active -= 1


class _CloudTranslatorWorks(TranslationEngine):
    async def translate(self, text, _source_lang, target_lang, _context):
        return f"CLOUD {text} [{target_lang}]"


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.stt.engine = "openai-compatible"
    cfg.stt.base_url = "https://api.example.test/v1"
    cfg.stt.fallback = STTFallbackConfig(engine="test-local-stt", model="small")
    cfg.translation.engine = "openai-compatible"
    cfg.translation.base_url = "https://api.example.test/v1"
    cfg.translation.fallback = TranslationFallbackConfig(
        engine="test-local-translation",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen-test",
    )
    return cfg


def _patch_engines(monkeypatch, *, primary_translation=None, local_translation=None, created=None):
    import uvt.fallback as fallback_module

    calls: list[tuple[str, str]] = []

    def create(kind, name, cfg):
        calls.append((kind, name))
        if kind == "stt" and name == "openai-compatible":
            return _CloudSTTFails(cfg)
        if kind == "stt" and name == "test-local-stt":
            return _LocalSTT(cfg)
        if kind == "translation" and name == "openai-compatible":
            return (primary_translation or _CloudTranslatorFails)(cfg)
        if kind == "translation" and name == "test-local-translation":
            engine = (local_translation or _LocalTranslator)(cfg)
            if created is not None:
                created.append(engine)
            return engine
        raise AssertionError((kind, name))

    monkeypatch.setattr(fallback_module.registry, "create", create)
    return calls


async def test_batch_stt_400_switches_to_local_whisper(monkeypatch):
    cfg = _cfg()
    calls = _patch_engines(monkeypatch)

    spans = await _transcribe_all(cfg, np.zeros(16_000, dtype=np.float32), None, None)

    assert [span.text for span in spans] == ["local speech"]
    assert calls == [("stt", "openai-compatible"), ("stt", "test-local-stt")]


async def test_batch_gpt_error_switches_to_local_translation(monkeypatch):
    cfg = _cfg()
    calls = _patch_engines(monkeypatch)
    spans = [STTSpan(0.0, 1.0, "hello", "en"), STTSpan(1.0, 2.0, "world", "en")]

    translated = await _translate_all(cfg, spans, "en", ["male", "female"], None)

    assert translated == ["LOCAL hello [ru]", "LOCAL world [ru]"]
    assert calls == [
        ("translation", "openai-compatible"),
        ("translation", "test-local-translation"),
    ]


async def test_cloud_batch_is_split_for_local_fallback_after_refusal(monkeypatch):
    """20 cloud-строк не должны одним запросом попасть в локальный Qwen."""
    cfg = _cfg()
    cfg.translation.fallback.batch_size = 2
    created = []
    _patch_engines(
        monkeypatch,
        primary_translation=_LargeBatchCloudTranslatorFails,
        local_translation=_BatchProbeLocalTranslator,
        created=created,
    )
    spans = [STTSpan(float(i), float(i + 1), f"line {i}", "en") for i in range(6)]

    translated = await _translate_all(cfg, spans, "en", None, None)

    assert translated == [f"LOCAL line {i} [ru]" for i in range(6)]
    assert len(created) == 1
    assert created[0].calls == [
        ["line 0", "line 1"],
        ["line 2", "line 3"],
        ["line 4", "line 5"],
    ]


async def test_failover_serializes_concurrent_batches_for_local_ollama(monkeypatch):
    """Три cloud-запроса могут упасть разом, local LLM вызывается по одному."""
    cfg = _cfg()
    created = []
    _patch_engines(
        monkeypatch,
        local_translation=_SerialProbeLocalTranslator,
        created=created,
    )
    spans = [STTSpan(float(i), float(i + 1), f"line {i}", "en") for i in range(6)]

    translated = await _translate_all(cfg, spans, "en", None, None)

    assert all(text and text.startswith("LOCAL line") for text in translated)
    assert len(created) == 1
    assert created[0].max_active == 1


async def test_working_cloud_translation_never_starts_local_reserve(monkeypatch):
    from uvt.fallback import create_translation_engine

    cfg = _cfg()
    calls = _patch_engines(monkeypatch, primary_translation=_CloudTranslatorWorks)
    engine = create_translation_engine(cfg)
    await engine.warmup()
    try:
        assert await engine.translate("hello", "en", "ru", []) == "CLOUD hello [ru]"
    finally:
        await engine.close()

    assert calls == [("translation", "openai-compatible")]


async def test_local_reserve_error_keeps_cloud_cause(monkeypatch):
    import uvt.fallback as fallback_module

    cfg = _cfg()

    class BrokenLocalSTT(STTEngine):
        async def warmup(self):
            raise RuntimeError("local model missing")

        async def transcribe(self, *_args):
            return None

    def create(kind, name, section):
        if kind == "stt" and name == "openai-compatible":
            return _CloudSTTFails(section)
        if kind == "stt" and name == "test-local-stt":
            return BrokenLocalSTT(section)
        raise AssertionError((kind, name))

    monkeypatch.setattr(fallback_module.registry, "create", create)
    with pytest.raises(RuntimeError, match="облачное распознавание.*HTTP 400.*локальный резерв"):
        await _transcribe_all(cfg, np.zeros(16_000, dtype=np.float32), None, None)


def test_openai_refusal_is_an_explicit_fallback_signal():
    from uvt.engines.translate_openai import _response_content

    with pytest.raises(RuntimeError, match="отказалась переводить"):
        _response_content(
            {
                "choices": [
                    {"message": {"content": None, "refusal": "policy refusal"}, "finish_reason": "stop"}
                ]
            }
        )


async def test_missing_numbered_cloud_line_is_a_failover_signal():
    """Неполный batch нельзя маскировать исходной строкой и считать успехом."""
    from uvt.engines.translate_openai import OpenAICompatibleTranslator

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "1. Первый перевод"}, "finish_reason": "stop"}
                ]
            }

    class Client:
        async def post(self, *_args, **_kwargs):
            return Response()

    engine = OpenAICompatibleTranslator(TranslationConfig(model="qwen-test"))
    engine._base = "https://api.example.test/v1"
    engine._template = "Translate faithfully."
    engine._client = Client()

    with pytest.raises(RuntimeError, match="не вернула 1 строк из 2"):
        await engine.translate_batch_tagged(["one", "two"], "en", "ru", None)
