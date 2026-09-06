"""Source context crosses file batches but never becomes extra translated output."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from uvt.config import AppConfig, TranslationConfig
from uvt.dub import _file_translation_context, _translate_all
from uvt.engines.translate_hymt_mlx import HyMTMLXTranslator
from uvt.engines.translate_mlx_chat import MlxChatTranslator
from uvt.engines.translate_openai import OpenAICompatibleTranslator
from uvt.interfaces import STTSpan, TranslationEngine


class ContextProbe(TranslationEngine):
    batch_hint = 2
    concurrency_hint = 1

    def __init__(self, *, mismatch=False, refuse=False):
        super().__init__(TranslationConfig())
        self.calls = []
        self.closed = False
        self.mismatch = mismatch
        self.refuse = refuse

    async def translate(self, *args):
        raise AssertionError("supported context must survive singleton retries")

    async def translate_batch_contextual(self, texts, source, target, genders, *, before=(), after=()):
        self.calls.append((list(texts), source, list(genders or []), list(before), list(after)))
        if self.mismatch and len(texts) > 1:
            return ["Неверная длина"]
        if self.refuse and len(texts) > 1:
            return ["Я не могу выполнить эту просьбу. Моя задача — предоставлять безопасную информацию." for _ in texts]
        return [f"Перевод реплики {text.split()[-1]}" for text in texts]

    async def close(self):
        self.closed = True


def spans(languages=None):
    return [STTSpan(float(i), i + 0.8, f"Line {i}", lang)
            for i, lang in enumerate(languages or ["en"] * 8)]


async def test_file_context_crosses_batch_boundaries_preserves_order_roles_and_language(monkeypatch):
    probe = ContextProbe()
    monkeypatch.setattr("uvt.dub.create_translation_engine", lambda *args, **kwargs: probe)
    cfg = AppConfig()
    cfg.translation.engine = "test-context"
    genders = ["female", "male"] * 4
    items = spans(["en", "en", "de", "de", "en", "en", "en", "en"])
    result = await _translate_all(cfg, items, None, genders, None)
    assert result == [f"Перевод реплики {i}" for i in range(8)]
    assert [call[1] for call in probe.calls] == ["en", "de", "en", "en"]
    second = probe.calls[1]
    assert second[0] == ["Line 2", "Line 3"]
    assert second[2] == ["female", "male"]
    assert second[3] == [("Line 0", "female"), ("Line 1", "male")]
    assert second[4] == [("Line 4", "female"), ("Line 5", "male"), ("Line 6", "female")]
    assert probe.closed
    assert [(item.start, item.end) for item in items] == [(float(i), i + 0.8) for i in range(8)]


@pytest.mark.parametrize("failure", ["mismatch", "refuse"])
async def test_retry_keeps_both_neighbours_and_exact_output_cardinality(monkeypatch, failure):
    probe = ContextProbe(**{failure: True})
    monkeypatch.setattr("uvt.dub.create_translation_engine", lambda *args, **kwargs: probe)
    cfg = AppConfig()
    cfg.translation.engine = "test-context"
    result = await _translate_all(cfg, spans(), "en", None, None)
    assert len(result) == 8
    assert result[3] == "Перевод реплики 3"
    retry = next(call for call in probe.calls if call[0] == ["Line 3"])
    assert retry[3][-1][0] == "Line 2"
    assert retry[4][0][0] == "Line 4"


def test_context_budget_and_disabled_window():
    items = [STTSpan(i, i + 1, str(i) + "x" * 5000, "en") for i in range(30)]
    before, after = _file_translation_context(items, [15, 16], None, 6)
    assert sum(len(text) for text, _ in before + after) <= 2400
    assert len(before) <= 6 and len(after) <= 6
    assert before[-1][0].startswith("14")
    assert after[0][0].startswith("17")
    assert _file_translation_context(items, [15], None, 0) == ([], [])


@pytest.mark.parametrize("model_class", [MlxChatTranslator, HyMTMLXTranslator])
async def test_local_context_is_prompt_only_no_extra_generations(monkeypatch, model_class):
    cfg = TranslationConfig(context_pairs=0, file_context_lines=2)
    model = model_class(cfg)
    prompts = []
    generated = []
    def encode(messages, **kwargs):
        prompts.append(deepcopy(messages))
        return [len(prompts)]
    model._tokenizer = SimpleNamespace(apply_chat_template=encode)
    monkeypatch.setattr(model, "_load", lambda: None)
    def generate(tokens):
        generated.extend(tokens)
        return [f"Перевод {i}" for i in range(len(tokens))]
    monkeypatch.setattr(model, "_generate", generate)
    result = await model.translate_batch_contextual(
        ["Target first.", "", "Target second."], "en", "ru", ["female", "", "male"],
        before=[("Earlier turn.", "male")], after=[("Following turn.", "female")],
    )
    assert result == ["Перевод 0", "", "Перевод 1"]
    assert len(generated) == 2
    first = "\n".join(message["content"] for message in prompts[0])
    second = "\n".join(message["content"] for message in prompts[1])
    assert "Earlier turn." in first
    assert "Following turn." in second
    assert "female" in first or "женщина" in first
    assert "male" in second or "мужчина" in second
    assert first.endswith("Target first.") or "Переведи только отмеченную реплику: Target first." in first
    # The existing live method remains past-only and context_pairs=0 really disables history.
    prompts.clear()
    generated.clear()
    await model.translate("Current live turn.", "en", "ru", [("Past turn.", "Прошлая.")])
    assert len(generated) == 1
    assert "Past turn." not in str(prompts)


class FakeResponse:
    def __init__(self, text):
        self.text = text
    def raise_for_status(self):
        pass
    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


async def test_openai_prompt_separates_numbered_targets_from_role_context(monkeypatch):
    cfg = TranslationConfig(model="gpt-4o-mini")
    model = OpenAICompatibleTranslator(cfg)
    calls = []
    async def post(url, **kwargs):
        calls.append(kwargs)
        return FakeResponse("1. Первый перевод\n2. Второй перевод")
    async def close():
        pass
    model._client = SimpleNamespace(post=post, aclose=close)
    model._base = "http://fixture.invalid"
    model._template = "Translate from {source_lang} into {target_lang}. {glossary}"
    try:
        result = await model.translate_batch_contextual(
            ["First target", "Second target"], "en", "ru", ["male", "female"],
            before=[("Previous context", "female")], after=[("Following context", "male")],
        )
        assert result == ["Первый перевод", "Второй перевод"]
        user = calls[0]["json"]["messages"][1]["content"]
        assert "PRECEDING dialogue" in user and "FOLLOWING dialogue" in user
        assert "[F] Previous context" in user and "[M] Following context" in user
        current = user.split("CURRENT lines to translate, and only these:\n")[1]
        assert current.splitlines() == ["1. [M] First target", "2. [F] Second target"]
        assert "Previous context" not in current
    finally:
        await model.close()


@pytest.mark.parametrize("answer", ["1. Текущая\n2. Лишний перевод контекста", "1. Текущая\n1. Повтор номера"])
async def test_openai_rejects_extra_context_output(monkeypatch, answer):
    model = OpenAICompatibleTranslator(TranslationConfig(model="gpt-4o-mini"))
    async def post(url, **kwargs):
        return FakeResponse(answer)
    async def close():
        pass
    model._client = SimpleNamespace(post=post, aclose=close)
    model._base = "http://fixture.invalid"
    model._template = "Translate from {source_lang} into {target_lang}. {glossary}"
    try:
        with pytest.raises(RuntimeError, match="нумерацию"):
            await model.translate_batch_contextual(["Current"], "en", "ru", None,
                                                   after=[("Context", "")])
    finally:
        await model.close()


async def test_cloud_to_local_fallback_keeps_context_when_rebatching(monkeypatch):
    from uvt.fallback import FailoverTranslator, _FallbackActivated
    cfg = AppConfig()
    cfg.translation.fallback = {"engine": "dummy", "model": "local"}
    wrapper = FailoverTranslator.__new__(FailoverTranslator)
    TranslationEngine.__init__(wrapper, cfg.translation)
    calls = []
    class Delegate:
        active = SimpleNamespace(batch_hint=2)
        async def call(self, method, texts, source, target, genders, **kwargs):
            if kwargs.pop("defer_fallback_retry", False):
                raise _FallbackActivated()
            calls.append((list(texts), kwargs["before"], kwargs["after"]))
            return ["Перевод " + text for text in texts]
    wrapper._delegate = Delegate()
    result = await wrapper.translate_batch_contextual(
        ["A", "B", "C", "D"], "en", "ru", ["male"] * 4,
        before=[("Previous", "female")], after=[("Next", "female")],
    )
    assert result == ["Перевод A", "Перевод B", "Перевод C", "Перевод D"]
    assert calls[0][1] == [("Previous", "female")]
    assert ("C", "male") in calls[0][2]
    assert ("B", "male") in calls[1][1]
    assert calls[1][2] == [("Next", "female")]
