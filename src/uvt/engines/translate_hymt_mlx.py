"""Hy-MT2: specialised local translation with its native user-only template.

Shares the serial MLX lifecycle and batch-count checks with mlx-chat, but
does not use its Qwen role prompt or a second paraphrasing pass. The source
text is translated once; neighbouring lines are context, never extra outputs.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from pathlib import Path

from uvt.engines.translate_mlx_chat import MlxChatTranslator, _positive_int
from uvt.registry import register

DEFAULT_MODEL = "mlx-community/Hy-MT2-1.8B-4bit"
DEFAULT_REVISION = "e5c6fe56c7b3bc77fae5ae92db31f2178f1e6912"
_LANGUAGES = {
    "en": "English", "zh": "Chinese", "ru": "Russian", "uk": "Ukrainian",
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "pl": "Polish", "cs": "Czech", "nl": "Dutch",
    "sv": "Swedish", "tr": "Turkish", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "hi": "Hindi", "id": "Indonesian", "vi": "Vietnamese",
    "th": "Thai", "he": "Hebrew", "fa": "Persian", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "fi": "Finnish", "da": "Danish",
    "no": "Norwegian", "ms": "Malay", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "tl": "Filipino",
}
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _target_name(language: str) -> str:
    code = str(language).strip().lower().replace("_", "-").split("-", 1)[0]
    if code in _LANGUAGES:
        return _LANGUAGES[code]
    # Full language names are already valid in the upstream instruction.
    for name in _LANGUAGES.values():
        if str(language).strip().casefold() == name.casefold():
            return name
    raise ValueError(f"hymt-mlx: укажите поддерживаемый язык перевода, получено {language!r}")


@register("translation", "hymt-mlx")
class HyMTMLXTranslator(MlxChatTranslator):
    """One Hy-MT2 generation per phrase, with bounded dialogue context."""

    supports_shorten = False

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.batch_hint = _positive_int(getattr(cfg, "batch_size", None), default=4)
        self._max_tokens = _positive_int(getattr(cfg, "max_tokens", None), default=160)

    def _model_path(self) -> Path:
        configured = str(getattr(self.cfg, "model", "") or DEFAULT_MODEL)
        direct = Path(configured).expanduser()
        if direct.is_dir():
            return direct.resolve()
        revision = str(getattr(self.cfg, "revision", "") or "")
        if not revision and configured == DEFAULT_MODEL:
            revision = DEFAULT_REVISION
        allow_download = bool(getattr(self.cfg, "allow_download", False))
        try:
            from huggingface_hub import snapshot_download

            return Path(snapshot_download(
                repo_id=configured,
                revision=revision or None,
                local_files_only=not allow_download,
            ))
        except Exception as exc:
            raise RuntimeError(
                f"hymt-mlx: модель {configured} не установлена; "
                "подготовьте Hy-MT2 через настройку локальных моделей"
            ) from exc

    def _prompt(
        self,
        texts: Sequence[str],
        index: int,
        target_lang: str,
        genders: Sequence[str] | None,
        *, context_lines: int | None = None,
    ) -> str:
        target = _target_name(target_lang)
        parts = []
        window = self._context_lines if context_lines is None else context_lines
        start = max(0, index - window)
        end = min(len(texts), index + window + 1)
        context = []
        for i in range(start, end):
            if i == index or not str(texts[i]).strip():
                continue
            role = str(genders[i]).lower() if genders is not None and i < len(genders) else ""
            speaker = "female" if role.startswith(("f", "ж")) else "male" if role.startswith(("m", "м")) else "unknown"
            context.append(f"[{'Before' if i < index else 'After'}; speaker {speaker}] {texts[i]}")
        if context:
            parts.append("Context (for understanding only; do not translate it):\n" + "\n".join(context))
        if self._glossary:
            parts.append("Reference translations:\n" + "\n".join(self._glossary))
        preferences = []
        if genders is not None and index < len(genders):
            gender = str(genders[index]).lower()
            if gender.startswith(("f", "ж")):
                preferences.append("The speaker is female; use feminine forms where needed.")
            elif gender.startswith(("m", "м")):
                preferences.append("The speaker is male; use masculine forms where needed.")
        # Keep the usual address choice consistent without adding a long Qwen
        # role prompt. Style/glossary are optional user-supplied preferences.
        if self._address:
            preferences.append(f"Form of address to the listener: {self._address}.")
        if self._style:
            preferences.append(self._style)
        if preferences:
            parts.append("Translation preferences:\n" + "\n".join(preferences))
        # Official Hy-MT instruction, full English language name, source last.
        parts.append(
            f"Translate into {target}. Note you should only output translated result "
            f"without any explanation:\n\n{texts[index]}"
        )
        return "\n\n".join(parts)

    def _encode_prompt(self, prompt: str) -> list[int]:
        assert self._tokenizer is not None
        # Hy-MT2 has its own no-thinking template and EOS (120020). The MLX
        # tokenizer reads EOS from its config; never substitute Qwen tokens.
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @staticmethod
    def _clean(text: str) -> str:
        # A translation can legitimately start with a number or contain more
        # than one sentence. Do not use the chat cleaner that strips digits
        # and discards all lines after the first.
        cleaned = " ".join(_THINK.sub(" ", str(text)).split()).strip()
        if len(cleaned) >= 2 and (cleaned[0], cleaned[-1]) in {('"', '"'), ('«', '»')}:
            return cleaned[1:-1].strip()
        return cleaned

    async def translate_batch_tagged(
        self,
        texts: Sequence[str],
        source_lang: str | None,
        target_lang: str,
        genders: Sequence[str] | None,
    ) -> list[str]:
        items = list(texts)
        indexes = [i for i, value in enumerate(items) if str(value).strip()]
        if not indexes:
            return ["" for _ in items]
        # Validate before loading weights or scheduling GPU work.
        _target_name(target_lang)
        async with self._lock:
            await asyncio.to_thread(self._load)
            prompts = [self._encode_prompt(self._prompt(items, i, target_lang, genders)) for i in indexes]
            outputs = await asyncio.to_thread(self._generate, prompts)
        result = ["" for _ in items]
        for index, value in zip(indexes, outputs, strict=True):
            result[index] = value
        return result

    async def translate_batch_contextual(
        self, texts: Sequence[str], source_lang: str | None, target_lang: str,
        genders: Sequence[str] | None, *,
        before: Sequence[tuple[str, str]] = (), after: Sequence[tuple[str, str]] = (),
    ) -> list[str]:
        items = [text for text, _role in before] + list(texts) + [text for text, _role in after]
        roles = [role for _text, role in before] + list(genders or [""] * len(texts)) + [role for _text, role in after]
        offset = len(before)
        indexes = [offset + i for i, text in enumerate(texts) if str(text).strip()]
        if not indexes:
            return ["" for _ in texts]
        _target_name(target_lang)
        window = max(0, min(6, int(getattr(self.cfg, "file_context_lines", 3))))
        async with self._lock:
            await asyncio.to_thread(self._load)
            prompts = [self._encode_prompt(self._prompt(
                items, i, target_lang, roles, context_lines=window,
            )) for i in indexes]
            outputs = await asyncio.to_thread(self._generate, prompts)
        result = ["" for _ in texts]
        for index, value in zip(indexes, outputs, strict=True):
            result[index - offset] = value
        return result

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Sequence[tuple[str, str]],
    ) -> str:
        if not str(text).strip():
            return ""
        _target_name(target_lang)
        history = [str(pair[0]) for pair in list(context)[-self._context_lines:]] if self._context_lines else []
        items = [*history, str(text)]
        async with self._lock:
            await asyncio.to_thread(self._load)
            prompt = self._encode_prompt(self._prompt(items, len(items) - 1, target_lang, None))
            outputs = await asyncio.to_thread(self._generate, [prompt])
        return outputs[0]

    async def shorten(self, text: str, target_lang: str, max_chars: int) -> str:
        # Specialised translation was not validated for paraphrasing; avoid
        # an extra generation that can damage meaning and increase latency.
        return text
