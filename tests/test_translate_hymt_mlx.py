"""Hy-MT integration contracts; no model download or GPU required."""
from copy import deepcopy
from types import SimpleNamespace
import sys

import pytest

from uvt.config import TranslationConfig
from uvt.engines.translate_hymt_mlx import HyMTMLXTranslator, DEFAULT_MODEL, DEFAULT_REVISION


class Tokenizer:
    eos_token_ids = {120020}

    def __init__(self):
        self.messages = []
        self.options = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(deepcopy(messages))
        self.options.append(kwargs)
        return [len(self.messages)]


def engine(tmp_path, monkeypatch, answers=None, **overrides):
    tokenizer = Tokenizer()
    calls = []

    def generate(model, tok, prompts, **kwargs):
        calls.append((prompts, kwargs, tok.eos_token_ids))
        return SimpleNamespace(texts=answers if answers is not None else ["Перевод."] * len(prompts))

    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(
        load=lambda **kwargs: (object(), tokenizer), batch_generate=generate,
    ))
    model = HyMTMLXTranslator(TranslationConfig(engine="hymt-mlx", model=str(tmp_path), **overrides))
    return model, tokenizer, calls


@pytest.mark.parametrize("target,name", [("ru", "Russian"), ("uk", "Ukrainian"), ("uk_UA", "Ukrainian")])
async def test_native_prompt_target_and_eos(tmp_path, monkeypatch, target, name):
    model, tokenizer, calls = engine(tmp_path, monkeypatch, max_tokens=96)
    await model.translate_batch(["The train leaves in 20 minutes."], "en", target)
    assert [message["role"] for message in tokenizer.messages[0]] == ["user"]
    assert f"Translate into {name}." in tokenizer.messages[0][0]["content"]
    assert tokenizer.messages[0][0]["content"].endswith("The train leaves in 20 minutes.")
    assert tokenizer.options[0]["enable_thinking"] is False
    assert tokenizer.options[0]["add_generation_prompt"] is True
    assert calls[0][1]["max_tokens"] == 96
    assert calls[0][2] == {120020}


async def test_context_genders_glossary_and_count(tmp_path, monkeypatch):
    model, tokenizer, calls = engine(tmp_path, monkeypatch, glossary=["train -> поезд"], context_pairs=1)
    result = await model.translate_batch_tagged(["", "Have you bought the tickets?", "I am ready.", ""], "en", "ru", ["", "male", "female", ""])
    assert result == ["", "Перевод.", "Перевод.", ""]
    assert len(calls[0][0]) == 2
    second = tokenizer.messages[1][0]["content"]
    assert "Have you bought the tickets?" in second
    assert "speaker is female" in second
    assert "train -> поезд" in second
    assert second.endswith("I am ready.")


async def test_live_history_only_generates_current_phrase(tmp_path, monkeypatch):
    model, tokenizer, calls = engine(tmp_path, monkeypatch, context_pairs=2)
    history = [("Too old.", "Очень давно."), ("Where is the station?", "Где станция?"), ("Behind the park.", "За парком.")]
    await model.translate("Thank you.", "en", "uk", history)
    assert len(calls) == 1 and len(calls[0][0]) == 1
    prompt = tokenizer.messages[0][0]["content"]
    assert "Too old." not in prompt
    assert "Where is the station?" in prompt and "Behind the park." in prompt
    assert "Где станция?" not in prompt
    assert prompt.endswith("Thank you.")


async def test_disabled_context_and_unknown_gender(tmp_path, monkeypatch):
    model, tokenizer, _ = engine(tmp_path, monkeypatch, context_pairs=0)
    await model.translate_batch_tagged(["Before.", "I am ready."], "en", "uk", ["unknown", "unknown"])
    prompt = tokenizer.messages[1][0]["content"]
    assert "Before." not in prompt
    assert "speaker is male" not in prompt and "speaker is female" not in prompt


async def test_empty_input_does_not_load_model(tmp_path, monkeypatch):
    model, tokenizer, calls = engine(tmp_path, monkeypatch)
    assert await model.translate_batch(["", "  "], "en", "ru") == ["", ""]
    assert await model.translate(" ", "en", "ru", [("History.", "История.")]) == ""
    assert not calls and not tokenizer.messages and model._model is None


async def test_invalid_target_fails_before_loading(tmp_path, monkeypatch):
    model, _, calls = engine(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="язык перевода"):
        await model.translate_batch(["Hello."], "en", "unknown")
    assert not calls and model._model is None


async def test_batch_count_mismatch_is_not_silently_truncated(tmp_path, monkeypatch):
    model, _, _ = engine(tmp_path, monkeypatch, answers=["Один."])
    with pytest.raises(RuntimeError, match="размер пакета"):
        await model.translate_batch(["One.", "Two."], "en", "ru")


@pytest.mark.parametrize("raw,expected", [
    ("20 минут.\nНе опаздывайте.", "20 минут. Не опаздывайте."),
    ("2026 год.", "2026 год."),
    ("<think>reasoning</think>«Да.»", "Да."),
])
def test_cleanup_preserves_numbers_and_complete_translation(raw, expected):
    assert HyMTMLXTranslator._clean(raw) == expected


async def test_no_extra_shortening_generation(tmp_path, monkeypatch):
    model, _, calls = engine(tmp_path, monkeypatch)
    assert model.supports_shorten is False
    text = "Полная реплика с сохранением смысла."
    assert await model.shorten(text, "ru", 5) == text
    assert not calls


def test_default_revision_is_pinned_and_download_is_opt_in(monkeypatch, tmp_path):
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    model = HyMTMLXTranslator(TranslationConfig(engine="hymt-mlx", model=DEFAULT_MODEL))
    assert model._model_path() == tmp_path
    assert calls == [{"repo_id": DEFAULT_MODEL, "revision": DEFAULT_REVISION, "local_files_only": True}]
