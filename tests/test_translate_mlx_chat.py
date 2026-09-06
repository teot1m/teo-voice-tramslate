"""Контекстный перевод диалога локальной chat-моделью.

Проверяется то, чего не мог дать однострочный шаблон TranslateGemma: соседние
реплики в промпте, пол говорящего, глоссарий и форма обращения, а также
сжатие слишком длинной реплики под тайминг.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from uvt.config import TranslationConfig
from uvt.engines.translate_mlx_chat import MlxChatTranslator


class _FakeTokenizer:
    """Возвращает сам промпт, чтобы тест видел, что ушло в модель."""

    def __init__(self) -> None:
        self.prompts: list[list[dict[str, str]]] = []
        self.kwargs: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append(deepcopy(messages))
        self.kwargs.append(dict(kwargs))
        return [len(self.prompts)]


def _install_fake_mlx(monkeypatch, tokenizer, answers):
    """Подменяет mlx_lm: answers(prompts) → список строк ответа."""
    calls: dict[str, object] = {"generate": []}

    def load(**_kwargs):
        return object(), tokenizer

    def batch_generate(_model, _tokenizer, prompts, **kwargs):
        calls["generate"].append((deepcopy(prompts), dict(kwargs)))
        return SimpleNamespace(texts=answers(prompts))

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        SimpleNamespace(load=load, batch_generate=batch_generate),
    )
    return calls


def _engine(tmp_path, **overrides) -> MlxChatTranslator:
    model_dir = tmp_path / "chat-model"
    model_dir.mkdir(exist_ok=True)
    cfg = TranslationConfig(engine="mlx-chat", model=str(model_dir), **overrides)
    return MlxChatTranslator(cfg)


DIALOGUE = [
    "So, remember that we talked about some...",
    "Oh yeah, the casting?",
    "Right, right, right.",
]


class TestContextPrompt:
    async def test_neighbour_lines_are_in_the_prompt(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        _install_fake_mlx(monkeypatch, tokenizer, lambda prompts: ["перевод"] * len(prompts))
        engine = _engine(tmp_path)

        await engine.translate_batch_tagged(DIALOGUE, "en", "ru", None)

        # Промпт средней реплики видит и предыдущую, и следующую
        middle = tokenizer.prompts[1][-1]["content"]
        assert "the casting?" in middle
        assert "we talked about some" in middle
        assert "Right, right, right." in middle
        assert "→" in middle, "нужная реплика должна быть отмечена"

    async def test_speaker_gender_reaches_the_prompt(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        _install_fake_mlx(monkeypatch, tokenizer, lambda prompts: ["перевод"] * len(prompts))
        engine = _engine(tmp_path)

        await engine.translate_batch_tagged(
            ["I am not sure anymore."], "en", "ru", ["female"]
        )

        user = tokenizer.prompts[0][-1]["content"]
        assert "женщина" in user
        assert "[Ж]" in user

    async def test_system_rules_carry_glossary_and_address(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        _install_fake_mlx(monkeypatch, tokenizer, lambda prompts: ["перевод"] * len(prompts))
        engine = _engine(
            tmp_path,
            glossary=["modeling → модельный бизнес"],
            address="ты",
        )

        await engine.translate_batch_tagged(["You said modeling."], "en", "ru", None)

        system = tokenizer.prompts[0][0]["content"]
        assert "модельный бизнес" in system
        assert "«ты»" in system
        assert "русский" in system and "английского" in system
        # Явный запрет на то, что портило дорожку
        assert "скобках" in system
        assert "одной строкой" in system

    async def test_thinking_is_disabled_when_template_supports_it(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        _install_fake_mlx(monkeypatch, tokenizer, lambda prompts: ["перевод"] * len(prompts))
        engine = _engine(tmp_path)

        await engine.translate_batch_tagged(["Yes."], "en", "ru", None)

        assert tokenizer.kwargs[0].get("enable_thinking") is False


class TestAnswerCleanup:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("<think>долгие размышления</think>\nДа, конечно.", "Да, конечно."),
            ('"Да, конечно."', "Да, конечно."),
            ("1. Да, конечно.", "Да, конечно."),
            ("→ Да, конечно.", "Да, конечно."),
            ("«Да, конечно.»", "Да, конечно."),
            ("\n\n  Да, конечно.  \n", "Да, конечно."),
        ],
    )
    def test_single_speakable_line_is_kept(self, raw, expected):
        assert MlxChatTranslator._clean(raw) == expected

    async def test_empty_lines_stay_empty_and_size_matches(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        _install_fake_mlx(monkeypatch, tokenizer, lambda prompts: ["перевод"] * len(prompts))
        engine = _engine(tmp_path)

        result = await engine.translate_batch_tagged(["", "Yes.", "   "], "en", "ru", None)

        assert result == ["", "перевод", ""]

    async def test_batch_size_mismatch_is_an_error(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        _install_fake_mlx(monkeypatch, tokenizer, lambda _prompts: ["одна"])
        engine = _engine(tmp_path)

        with pytest.raises(RuntimeError, match="размер пакета"):
            await engine.translate_batch_tagged(["a", "b"], "en", "ru", None)


class TestShorten:
    def test_capability_is_declared(self):
        assert MlxChatTranslator.supports_shorten is True

    async def test_long_line_is_rewritten_shorter(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        _install_fake_mlx(monkeypatch, tokenizer, lambda _p: ["Коротко и по делу."])
        engine = _engine(tmp_path)
        long_line = "Очень длинная реплика, которая заведомо не влезает в свой тайминг."

        result = await engine.shorten(long_line, "ru", 30)

        assert result == "Коротко и по делу."
        system = tokenizer.prompts[0][0]["content"]
        assert "30 символов" in system

    async def test_already_short_line_is_untouched(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        _install_fake_mlx(monkeypatch, tokenizer, lambda _p: ["не должно вызываться"])
        engine = _engine(tmp_path)

        assert await engine.shorten("Да.", "ru", 30) == "Да."
        assert tokenizer.prompts == []

    async def test_model_answer_longer_than_original_is_rejected(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        _install_fake_mlx(
            monkeypatch,
            tokenizer,
            lambda _p: ["Ещё более длинная и подробная формулировка вместо короткой."],
        )
        engine = _engine(tmp_path)
        original = "Длинная реплика для укладки."

        assert await engine.shorten(original, "ru", 10) == original


class TestLiveTranslation:
    async def test_history_is_context_for_one_current_generation(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        calls = _install_fake_mlx(monkeypatch, tokenizer, lambda prompts: ["Спасибо."] * len(prompts))
        engine = _engine(tmp_path, context_pairs=3)
        history = [
            ("Outside the window.", "За окном."),
            ("Have you seen my keys?", "Ты видел мои ключи?"),
            ("They are on the table.", "Они на столе."),
            ("Near the blue cup.", "Возле синей чашки."),
        ]

        assert await engine.translate("Thank you.", "en", "ru", history) == "Спасибо."

        # One model invocation containing one prompt, regardless of history size.
        assert len(calls["generate"]) == 1
        assert len(calls["generate"][0][0]) == 1
        assert len(tokenizer.prompts) == 1
        prompt = tokenizer.prompts[0][-1]["content"]
        for source, _ in history[-3:]:
            assert source in prompt
        assert history[0][0] not in prompt
        assert "→ 4. Thank you." in prompt
        assert "Переведи только отмеченную реплику: Thank you." in prompt

    async def test_current_line_without_history_still_generates_once(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        calls = _install_fake_mlx(monkeypatch, tokenizer, lambda prompts: ["Привет."] * len(prompts))
        engine = _engine(tmp_path)

        assert await engine.translate("Hello.", "en", "ru", []) == "Привет."
        assert len(calls["generate"]) == 1
        assert len(calls["generate"][0][0]) == 1
        assert "→ 1. Hello." in tokenizer.prompts[0][-1]["content"]

    async def test_blank_current_line_does_not_translate_history(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        calls = _install_fake_mlx(monkeypatch, tokenizer, lambda prompts: ["Старый ответ."] * len(prompts))
        engine = _engine(tmp_path)

        assert await engine.translate("   ", "en", "ru", [("Old line.", "Старая реплика.")]) == ""
        assert calls["generate"] == []
        assert tokenizer.prompts == []


    async def test_disabled_history_ignores_supplied_past_lines(self, tmp_path, monkeypatch):
        tokenizer = _FakeTokenizer()
        calls = _install_fake_mlx(monkeypatch, tokenizer, lambda prompts: ["Привет."] * len(prompts))
        engine = _engine(tmp_path, context_pairs=0)

        assert await engine.translate("Hello.", "en", "ru", [("Old line.", "Старая реплика.")]) == "Привет."
        assert len(calls["generate"][0][0]) == 1
        assert "Old line." not in tokenizer.prompts[0][-1]["content"]
        assert "→ 1. Hello." in tokenizer.prompts[0][-1]["content"]
