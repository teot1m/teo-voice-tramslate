"""Unit tests for the dedicated MLX TranslateGemma engine."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import sys
import threading
from types import SimpleNamespace

import pytest

from uvt.engines.translate_mlx_translategemma import (
    _DEFAULT_MODEL,
    _DEFAULT_REVISION,
    _PARAKEET_LANGUAGE_CODES,
    MlxTranslateGemmaTranslator,
    _language_code,
)


def _model_dir(tmp_path):
    model_dir = tmp_path / "translategemma"
    model_dir.mkdir()
    for name in (
        "config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "model.safetensors",
    ):
        (model_dir / name).write_text("test", encoding="utf-8")
    return model_dir


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.added_eos_tokens: list[str] = []

    def add_eos_token(self, token: str):
        self.added_eos_tokens.append(token)

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), dict(kwargs)))
        return [len(self.calls)]


def _fake_mlx_lm(tokenizer, *, batch_generate=None):
    load_calls: list[dict[str, object]] = []
    generate_calls: list[tuple[object, object, object, dict[str, object]]] = []
    model = object()

    def load(**kwargs):
        load_calls.append(dict(kwargs))
        return model, tokenizer

    def default_batch_generate(loaded_model, loaded_tokenizer, prompts, **kwargs):
        generate_calls.append(
            (loaded_model, loaded_tokenizer, deepcopy(prompts), dict(kwargs))
        )
        return SimpleNamespace(texts=[f"перевод-{prompt[0]}" for prompt in prompts])

    module = SimpleNamespace(
        load=load,
        batch_generate=batch_generate or default_batch_generate,
    )
    return module, model, load_calls, generate_calls


def test_translategemma_language_mapping_is_strict():
    for code in _PARAKEET_LANGUAGE_CODES:
        assert _language_code(code) == code

    assert _language_code("en-US") == "en"
    assert _language_code("pt-BR") == "pt"
    assert _language_code("de_DE") == "de"
    assert _language_code("rus_Cyrl") == "ru"
    assert _language_code("ukr-Cyrl") == "uk"

    for unsupported in (None, "auto", "und", "zh"):
        with pytest.raises(RuntimeError, match="25 языков Parakeet v3"):
            _language_code(unsupported)


@pytest.mark.asyncio
async def test_warmup_resolves_pinned_cache_without_network(monkeypatch, tmp_path):
    model_dir = _model_dir(tmp_path)
    snapshot_calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs):
        snapshot_calls.append(dict(kwargs))
        return str(model_dir)

    tokenizer = _FakeTokenizer()
    mlx_lm, model, load_calls, _generate_calls = _fake_mlx_lm(tokenizer)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)

    engine = MlxTranslateGemmaTranslator(
        SimpleNamespace(model=_DEFAULT_MODEL, allow_download=False)
    )
    await engine.warmup()

    assert snapshot_calls == [
        {
            "repo_id": _DEFAULT_MODEL,
            "revision": _DEFAULT_REVISION,
            "local_files_only": True,
        }
    ]
    assert load_calls == [
        {"path_or_hf_repo": str(model_dir), "lazy": False}
    ]
    assert engine._model is model


@pytest.mark.asyncio
async def test_batch_uses_independent_structured_prompts_and_preserves_order(
    monkeypatch, tmp_path
):
    model_dir = _model_dir(tmp_path)
    tokenizer = _FakeTokenizer()
    mlx_lm, model, _load_calls, generate_calls = _fake_mlx_lm(tokenizer)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)

    progress: list[float] = []
    engine = MlxTranslateGemmaTranslator(
        SimpleNamespace(
            model=str(model_dir),
            batch_size=2,
            max_tokens=96,
            allow_download=False,
        )
    )
    result = await engine.translate_batch(
        ["First line", "Second line", "Third line"],
        "en-US",
        "uk-UA",
        progress.append,
    )

    assert result == ["перевод-1", "перевод-2", "перевод-3"]
    assert tokenizer.added_eos_tokens == ["<end_of_turn>"]
    assert progress == pytest.approx([2 / 3, 1.0])
    assert len(generate_calls) == 2
    assert generate_calls[0] == (
        model,
        tokenizer,
        [[1], [2]],
        {"max_tokens": 96, "verbose": False},
    )
    assert generate_calls[1][2] == [[3]]

    assert [
        call[0][0]["content"][0]["text"] for call in tokenizer.calls
    ] == ["First line", "Second line", "Third line"]
    for messages, kwargs in tokenizer.calls:
        assert messages == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "uk",
                        "text": messages[0]["content"][0]["text"],
                    }
                ],
            }
        ]
        assert kwargs == {"tokenize": True, "add_generation_prompt": True}


@pytest.mark.asyncio
async def test_batch_rejects_cardinality_mismatch(monkeypatch, tmp_path):
    model_dir = _model_dir(tmp_path)
    tokenizer = _FakeTokenizer()

    def short_batch(_model, _tokenizer, _prompts, **_kwargs):
        return SimpleNamespace(texts=["только одна строка"])

    mlx_lm, _model, _load_calls, _generate_calls = _fake_mlx_lm(
        tokenizer, batch_generate=short_batch
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    engine = MlxTranslateGemmaTranslator(
        SimpleNamespace(model=str(model_dir), batch_size=8)
    )

    with pytest.raises(RuntimeError, match="нарушила размер пакета"):
        await engine.translate_batch(["one", "two"], "en", "ru")


@pytest.mark.asyncio
async def test_batch_rejects_runaway_generation(monkeypatch, tmp_path):
    model_dir = _model_dir(tmp_path)
    tokenizer = _FakeTokenizer()

    def runaway(_model, _tokenizer, _prompts, **_kwargs):
        return SimpleNamespace(texts=["повтор " * 1000])

    mlx_lm, _model, _load_calls, _generate_calls = _fake_mlx_lm(
        tokenizer, batch_generate=runaway
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    engine = MlxTranslateGemmaTranslator(SimpleNamespace(model=str(model_dir)))

    with pytest.raises(RuntimeError, match="генерация не остановилась"):
        await engine.translate_batch(["short"], "en", "ru")


@pytest.mark.asyncio
async def test_cancellation_waits_for_native_worker(monkeypatch, tmp_path):
    model_dir = _model_dir(tmp_path)
    tokenizer = _FakeTokenizer()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_batch(_model, _tokenizer, prompts, **_kwargs):
        started.set()
        release.wait(timeout=5)
        finished.set()
        return SimpleNamespace(texts=["готово" for _prompt in prompts])

    mlx_lm, _model, _load_calls, _generate_calls = _fake_mlx_lm(
        tokenizer, batch_generate=blocking_batch
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    engine = MlxTranslateGemmaTranslator(SimpleNamespace(model=str(model_dir)))

    task = asyncio.create_task(engine.translate_batch(["hello"], "en", "ru"))
    assert await asyncio.to_thread(started.wait, 1.0)
    task.cancel()
    try:
        await asyncio.sleep(0.05)
        assert not task.done()
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()

    await engine.close()
    assert engine._model is None
    assert engine._tokenizer is None


@pytest.mark.asyncio
async def test_missing_local_cache_has_actionable_error(monkeypatch):
    def snapshot_download(**_kwargs):
        raise FileNotFoundError("not cached")

    tokenizer = _FakeTokenizer()
    mlx_lm, _model, _load_calls, _generate_calls = _fake_mlx_lm(tokenizer)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    engine = MlxTranslateGemmaTranslator(
        SimpleNamespace(model=_DEFAULT_MODEL, allow_download=False)
    )

    with pytest.raises(RuntimeError, match="uvt setup-mac-local"):
        await engine.warmup()
