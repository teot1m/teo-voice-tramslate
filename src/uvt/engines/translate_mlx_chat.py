"""Контекстный перевод диалогов локальной chat-моделью через mlx-lm.

Чем это отличается от ``translategemma-mlx``. У TranslateGemma официальный
шаблон принимает ровно одну строку: соседних реплик, пола говорящего и
глоссария в нём нет места. На диалоге это даёт ровно те ошибки, которые видно
в логах дубляжа: «Oh you are?» → «Кто вы?», «I am not» → «Я не являюсь»,
«modeling» → «моделирование» вместо «модельного бизнеса», скачки ты/вы между
соседними репликами и «хотел(а)» там, где пол известен.

Здесь каждая реплика переводится в контексте своей пачки: модель видит
предыдущие и следующие строки диалога, знает пол говорящего, глоссарий и
требуемую форму обращения. Плюс ``shorten`` — второй проход, который
переписывает слишком длинный перевод короче, чтобы реплика уложилась в свой
тайминг текстом, а не ускорением речи.

Модель задаётся профилем. Она должна переводить то, что сказано: у диалогов
с ругательствами и откровенными сценами модель с жёстким safety-фильтром
вместо перевода отдаёт отказ (dub его отбракует и озвучка потеряет реплику).
"""
from __future__ import annotations

import asyncio
import gc
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from uvt.interfaces import TranslationEngine
from uvt.registry import register

log = logging.getLogger("uvt.translate.mlx_chat")

_DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
_DEFAULT_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_LANGUAGE_NAMES = {
    "en": "английского",
    "ru": "русского",
    "uk": "украинского",
    "de": "немецкого",
    "fr": "французского",
    "es": "испанского",
    "it": "итальянского",
    "pt": "португальского",
    "pl": "польского",
    "cs": "чешского",
    "nl": "нидерландского",
    "sv": "шведского",
    "tr": "турецкого",
    "ja": "японского",
    "zh": "китайского",
}
_TARGET_NAMES = {
    "ru": "русский",
    "uk": "украинский",
    "en": "английский",
    "de": "немецкий",
    "fr": "французский",
    "es": "испанский",
}


def _language_name(code: str | None, table: dict[str, str], fallback: str) -> str:
    root = str(code or "").strip().lower().replace("_", "-").split("-", 1)[0]
    return table.get(root, fallback)


def _positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


@register("translation", "mlx-chat")
class MlxChatTranslator(TranslationEngine):
    """Перевод пачки реплик с контекстом диалога на одной локальной модели."""

    supports_shorten = True

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.batch_hint = _positive_int(getattr(cfg, "batch_size", None), default=8)
        # Одна MLX-модель и один GPU-воркер: параллельные пачки не ускоряют.
        self.concurrency_hint = 1
        self._max_tokens = _positive_int(getattr(cfg, "max_tokens", None), default=256)
        self._context_lines = _positive_int(
            getattr(cfg, "context_pairs", None), default=3
        )
        self._address = str(getattr(cfg, "address", "") or "ты").strip()
        self._glossary = [
            str(item).strip()
            for item in (getattr(cfg, "glossary", None) or [])
            if str(item).strip()
        ]
        self._style = str(getattr(cfg, "style", "") or "").strip()
        self._lock = asyncio.Lock()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._mlx_lm: Any | None = None

    # --- жизненный цикл модели ---

    def _model_path(self) -> Path:
        configured = str(getattr(self.cfg, "model", "") or _DEFAULT_MODEL)
        direct = Path(configured).expanduser()
        if direct.is_dir():
            return direct.resolve()
        revision = str(getattr(self.cfg, "revision", "") or "")
        if not revision and configured == _DEFAULT_MODEL:
            revision = _DEFAULT_REVISION
        allow_download = bool(getattr(self.cfg, "allow_download", False))
        try:
            from huggingface_hub import snapshot_download

            return Path(
                snapshot_download(
                    repo_id=configured,
                    revision=revision or None,
                    local_files_only=not allow_download,
                )
            )
        except Exception as exc:  # noqa: BLE001 — подсказываем действие
            action = (
                "проверьте доступ к Hugging Face"
                if allow_download
                else "скачайте модель один раз с allow_download: true"
            )
            raise RuntimeError(
                f"mlx-chat: модель {configured} не установлена; {action}"
            ) from exc

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import mlx_lm
        except ImportError as exc:
            raise RuntimeError(
                'mlx-chat требует mlx-lm: pip install -e ".[mac-local]"'
            ) from exc

        model_path = self._model_path()
        try:
            model, tokenizer = mlx_lm.load(path_or_hf_repo=str(model_path), lazy=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"mlx-chat: не удалось загрузить {model_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._mlx_lm = mlx_lm
        self._model = model
        self._tokenizer = tokenizer
        log.info("mlx-chat: модель загружена: %s", model_path.name)

    async def warmup(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._load)

    async def close(self) -> None:
        async with self._lock:
            self._model = None
            self._tokenizer = None
            self._mlx_lm = None
            gc.collect()

    # --- промпты ---

    def _system_prompt(self, source_lang: str | None, target_lang: str) -> str:
        source = _language_name(source_lang, _LANGUAGE_NAMES, "исходного языка")
        target = _language_name(target_lang, _TARGET_NAMES, target_lang)
        rules = [
            f"Ты переводчик закадрового дубляжа: переводишь реплики диалога с {source} на {target}.",
            "Отвечай ТОЛЬКО переводом указанной реплики, одной строкой.",
            "Никаких пояснений, комментариев, кавычек, нумерации и вариантов через слэш.",
            "Не пиши формы в скобках вида «готов(а)» — выбери форму по полу говорящего.",
            "Сохраняй разговорность, резкость и грубость оригинала: ругательства "
            "переводи ругательствами, не смягчай и не заменяй описанием.",
            "Реплику произносят вслух: без сокращений и цифр, числа словами.",
            "Не добавляй того, чего нет в реплике, и не объединяй её с соседними.",
            f"Обращение к собеседнику — «{self._address}», одинаково во всех репликах.",
            "Если реплика — обрывок или неразборчива, переведи её как обрывок, "
            "не достраивая смысл.",
        ]
        if self._glossary:
            rules.append("Термины переводи так: " + "; ".join(self._glossary) + ".")
        if self._style:
            rules.append(self._style)
        return "\n".join(rules)

    def _user_prompt(
        self,
        texts: Sequence[str],
        index: int,
        genders: Sequence[str] | None,
    ) -> str:
        start = max(0, index - self._context_lines)
        end = min(len(texts), index + self._context_lines + 1)
        lines = []
        for position in range(start, end):
            marker = "→" if position == index else " "
            speaker = ""
            if genders is not None and position < len(genders):
                speaker = "Ж" if str(genders[position]).lower().startswith(("f", "ж")) else "М"
                speaker = f" [{speaker}]"
            lines.append(f"{marker} {position - start + 1}.{speaker} {texts[position]}")
        gender_note = ""
        if genders is not None and index < len(genders):
            female = str(genders[index]).lower().startswith(("f", "ж"))
            gender_note = (
                " Говорит женщина — используй женские формы."
                if female
                else " Говорит мужчина — используй мужские формы."
            )
        return (
            "Фрагмент диалога (стрелкой отмечена нужная реплика):\n"
            + "\n".join(lines)
            + f"\n\nПереведи только отмеченную реплику: {texts[index]}"
            + gender_note
        )

    def _encode(self, system: str, user: str) -> list[int]:
        assert self._tokenizer is not None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            # Thinking-модели иначе тратят весь лимит токенов на рассуждения.
            return self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )

    @staticmethod
    def _clean(text: str) -> str:
        """Оставляет от ответа модели одну произносимую строку."""
        without_think = _THINK_BLOCK.sub(" ", str(text))
        for line in without_think.splitlines():
            candidate = line.strip().strip("«»\"'")
            # Модель иногда нумерует ответ или повторяет разметку промпта.
            candidate = re.sub(r"^[→\-\*\d\.\)\s]+", "", candidate).strip()
            if candidate:
                return candidate
        return ""

    # --- перевод ---

    def _generate(self, prompts: list[list[int]]) -> list[str]:
        self._load()
        assert self._mlx_lm is not None
        try:
            response = self._mlx_lm.batch_generate(
                self._model,
                self._tokenizer,
                prompts,
                max_tokens=self._max_tokens,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"mlx-chat: пакетная генерация не выполнена: {type(exc).__name__}: {exc}"
            ) from exc
        generated = getattr(response, "texts", None)
        if (
            not isinstance(generated, Sequence)
            or isinstance(generated, (str, bytes))
            or len(generated) != len(prompts)
        ):
            actual = len(generated) if isinstance(generated, Sequence) else "?"
            raise RuntimeError(
                f"mlx-chat: нарушен размер пакета: ожидалось {len(prompts)}, получено {actual}"
            )
        return [self._clean(item) for item in generated]

    async def translate_batch_tagged(
        self,
        texts: Sequence[str],
        source_lang: str | None,
        target_lang: str,
        genders: Sequence[str] | None,
    ) -> list[str]:
        items = list(texts)
        if not items:
            return []
        system = self._system_prompt(source_lang, target_lang)
        indexes = [i for i, text in enumerate(items) if str(text).strip()]
        if not indexes:
            return ["" for _ in items]

        async with self._lock:
            await asyncio.to_thread(self._load)
            prompts = [self._encode(system, self._user_prompt(items, i, genders)) for i in indexes]
            outputs = await asyncio.to_thread(self._generate, prompts)

        result = ["" for _ in items]
        for index, value in zip(indexes, outputs, strict=True):
            result[index] = value
        return result

    async def translate_batch(
        self, texts: Sequence[str], source_lang: str | None, target_lang: str
    ) -> list[str]:
        return await self.translate_batch_tagged(texts, source_lang, target_lang, None)

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Sequence[tuple[str, str]],
    ) -> str:
        # Контекст live-пути — пары (оригинал, перевод); для промпта нужны
        # только оригиналы соседних реплик.
        history = [str(pair[0]) for pair in list(context)[-self._context_lines :]]
        batch = [*history, str(text)]
        results = await self.translate_batch_tagged(batch, source_lang, target_lang, None)
        return results[-1]

    async def shorten(self, text: str, target_lang: str, max_chars: int) -> str:
        """Переписывает перевод короче, чтобы он уложился в тайминг реплики.

        Сжатие текстом лучше ускорения речи: скороговорка слышна сразу, а
        более короткая формулировка — нет.
        """
        phrase = " ".join(str(text).split())
        if not phrase or max_chars <= 0 or len(phrase) <= max_chars:
            return phrase
        target = _language_name(target_lang, _TARGET_NAMES, target_lang)
        system = (
            f"Ты редактор дубляжа. Перепиши реплику на {target} короче, "
            f"не длиннее {max_chars} символов, сохранив смысл, тон и грубость. "
            "Ответь только переписанной репликой, одной строкой, без пояснений."
        )
        user = f"Реплика ({len(phrase)} символов): {phrase}"
        async with self._lock:
            await asyncio.to_thread(self._load)
            prompt = self._encode(system, user)
            outputs = await asyncio.to_thread(self._generate, [prompt])
        candidate = outputs[0] if outputs else ""
        # Если модель не справилась — оставляем исходный перевод: пусть лучше
        # реплика выйдет за слот, чем потеряет смысл.
        return candidate if candidate and len(candidate) < len(phrase) else phrase
