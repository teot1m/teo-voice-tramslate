"""Перевод через любой OpenAI-совместимый chat-endpoint (ТЗ §4).

Один движок покрывает: OpenAI, DeepSeek, Grok (x.ai), Groq, OpenRouter,
Ollama (http://localhost:11434/v1), LM Studio (http://localhost:1234/v1),
llama.cpp server и любые пользовательские endpoint.

Системный промпт — настраиваемый шаблон (ТЗ §5): встроенный по умолчанию,
путь к файлу или сам текст шаблона в конфиге.
"""
from __future__ import annotations

import importlib.resources
import logging
import os
import re
from pathlib import Path
from typing import Sequence

from uvt.interfaces import TranslationEngine
from uvt.registry import register

log = logging.getLogger("uvt.translate.openai")

_LANG_NAMES = {
    "ru": "Russian", "en": "English", "uk": "Ukrainian", "de": "German",
    "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "pl": "Polish",
    "tr": "Turkish", "ar": "Arabic", "hi": "Hindi", "nl": "Dutch",
    "cs": "Czech", "kk": "Kazakh", "be": "Belarusian", "und": "the source language",
}


def _lang_name(code: str | None) -> str:
    if not code:
        return "the source language"
    return _LANG_NAMES.get(code.split("-")[0].lower(), code)


class _SafeDict(dict):
    """format_map без KeyError: незнакомые плейсхолдеры остаются как есть."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _is_local_host(base_url: str) -> bool:
    from urllib.parse import urlparse

    return (urlparse(base_url).hostname or "") in ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def _require_key_for_remote(base_url: str, key: str, key_env: str) -> None:
    """Облачные endpoint без ключа не работают — падаем сразу с понятной ошибкой
    (локальные Ollama/LM Studio ключа не требуют)."""
    from urllib.parse import urlparse

    if not key and not _is_local_host(base_url):
        host = urlparse(base_url).hostname or base_url
        raise RuntimeError(
            f"для {host} нужен API-ключ: задайте переменную окружения {key_env} "
            f"(export {key_env}=...)"
        )


async def _ping_local(client, base_url: str) -> None:
    """Локальный endpoint должен отвечать сразу — иначе понятная ошибка вместо
    молчаливых сбоев на каждой пачке перевода."""
    if not _is_local_host(base_url):
        return
    try:
        await client.get(f"{base_url}/models", timeout=3.0)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"локальный LLM-сервер не отвечает на {base_url} — запустите Ollama "
            "(brew services start ollama) или LM Studio"
        ) from exc


def load_template(spec: str | None) -> str:
    if spec is None:
        return (
            importlib.resources.files("uvt")
            .joinpath("prompts/default.md")
            .read_text(encoding="utf-8")
        )
    path = Path(spec).expanduser()
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return spec  # шаблон задан прямо в конфиге


def _response_content(payload: object) -> str:
    """Извлекает текст или явно сигнализирует failover о refusal/пустом ответе."""
    if not isinstance(payload, dict):
        raise RuntimeError("модель вернула не-JSON ответ перевода")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("модель не вернула choices для перевода")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("модель не вернула сообщение перевода")
    refusal = message.get("refusal")
    if refusal:
        raise RuntimeError(f"модель отказалась переводить: {str(refusal).strip()[:240]}")
    if choice.get("finish_reason") == "content_filter":
        raise RuntimeError("модель остановила перевод content filter")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("модель вернула пустой текст перевода")
    return content.strip()


@register("translation", "openai-compatible")
class OpenAICompatibleTranslator(TranslationEngine):
    async def warmup(self) -> None:
        import httpx

        cfg = self.cfg
        self._base = str(cfg.base_url).rstrip("/")
        key = os.environ.get(cfg.api_key_env, "")
        _require_key_for_remote(self._base, key, cfg.api_key_env)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.AsyncClient(timeout=cfg.timeout_s, headers=headers)
        await _ping_local(self._client, self._base)
        self._template = load_template(cfg.prompt_template)
        # Маленьким локальным моделям большие/параллельные пачки чаще ломают
        # нумерацию и давят память. Облаку оставляем прежний быстрый маршрут.
        local = _is_local_host(self._base)
        self.batch_hint = int(cfg.batch_size or (8 if local else 20))
        self.concurrency_hint = int(cfg.concurrency or (1 if local else 3))

    async def close(self) -> None:
        if hasattr(self, "_client"):
            await self._client.aclose()

    def _system_prompt(self, source_lang: str, target_lang: str) -> str:
        glossary = "\n".join(f"- {line}" for line in self.cfg.glossary) or "(none)"
        return self._template.format_map(
            _SafeDict(
                source_lang=_lang_name(source_lang),
                target_lang=_lang_name(target_lang),
                glossary=glossary,
            )
        )

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Sequence[tuple[str, str]],
    ) -> str:
        messages = [{"role": "system", "content": self._system_prompt(source_lang, target_lang)}]
        for src, dst in context:
            messages.append({"role": "user", "content": src})
            messages.append({"role": "assistant", "content": dst})
        messages.append({"role": "user", "content": text})

        response = await self._client.post(
            f"{self._base}/chat/completions",
            json={
                "model": self.cfg.model,
                "messages": messages,
                "temperature": self.cfg.temperature,
            },
        )
        response.raise_for_status()
        return _response_content(response.json())

    async def translate_batch(
        self, texts: Sequence[str], source_lang: str | None, target_lang: str
    ) -> list[str]:
        return await self.translate_batch_tagged(texts, source_lang, target_lang, None)

    async def translate_batch_tagged(
        self,
        texts: Sequence[str],
        source_lang: str | None,
        target_lang: str,
        genders: Sequence[str] | None,
    ) -> list[str]:
        """Пачка пронумерованных реплик одним запросом: быстрее в ~20 раз и
        качественнее — модель видит соседние реплики как контекст. Теги [M]/[F]
        подсказывают род говорящего («я готова», а не «я готов»)."""
        return await self.translate_batch_contextual(texts, source_lang, target_lang, genders)

    async def translate_batch_contextual(
        self, texts: Sequence[str], source_lang: str | None, target_lang: str,
        genders: Sequence[str] | None, *,
        before: Sequence[tuple[str, str]] = (), after: Sequence[tuple[str, str]] = (),
    ) -> list[str]:
        system = self._system_prompt(source_lang or "und", target_lang) + (
            "\n\nYou will receive several numbered lines from one video, in order. "
            "A line may start with a speaker tag [M] (male speaker) or [F] (female "
            "speaker): use it to choose grammatical gender in the translation and "
            "NEVER include the tag in your output. "
            "Translate EACH line separately, using neighbouring lines only as context. "
            "Reply with exactly the same numbering, one line per item, in the format "
            "'<number>. <translation>'. No other text."
        )
        if genders is not None:
            user = "\n".join(
                f"{i}. [{'F' if g == 'female' else 'M'}] {' '.join(t.split())}"
                for i, (t, g) in enumerate(zip(texts, genders), 1)
            )
        else:
            user = "\n".join(f"{i}. {' '.join(t.split())}" for i, t in enumerate(texts, 1))
        if before or after:
            def context_block(label, lines):
                tagged = []
                for text, role in lines:
                    normalized = str(role).lower()
                    speaker = "F" if normalized.startswith(("f", "ж")) else "M" if normalized.startswith(("m", "м")) else "unknown"
                    tagged.append(f"[{speaker}] {' '.join(str(text).split())}")
                return label + " (context only, DO NOT translate or number):\n" + "\n".join(tagged)
            context = []
            if before:
                context.append(context_block("PRECEDING dialogue", before))
            if after:
                context.append(context_block("FOLLOWING dialogue", after))
            user = "\n\n".join(context) + "\n\nCURRENT lines to translate, and only these:\n" + user
            system += " Use preceding and following context to resolve references and keep names, terminology and form of address consistent. Never include context lines in your answer."

        # Большая пачка длинных реплик генерируется долго — таймаут растёт с размером
        timeout = max(float(self.cfg.timeout_s), 20.0 + 8.0 * len(texts))
        response = await self._client.post(
            f"{self._base}/chat/completions",
            json={
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.cfg.temperature,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = _response_content(response.json())
        parsed = _parse_numbered(content)
        if any(index < 1 or index > len(texts) for index in parsed) or _misaligned(texts, parsed):
            # слабая модель вернула исходные строки со сдвигом нумерации —
            # такой перевод озвучил бы каждую реплику чужим текстом
            raise RuntimeError("модель сбила нумерацию пачки")
        out: list[str] = []
        missing = 0
        for i, original in enumerate(texts, 1):
            value = (parsed.get(i) or "").strip()
            # слабые модели иногда возвращают тег обратно — вычищаем
            value = re.sub(r"^\[[MFmfМЖмж]\]\s*", "", value)
            if value:
                out.append(value)
            else:
                out.append(original)
                missing += 1
        if missing:
            # Для cloud это сигнал FailoverTranslator сразу перейти на
            # локальный LLM для всей пачки. Без failover _translate_all
            # штатно деградирует до построчного перевода тем же движком.
            # Нельзя молча оставить source: валидатор увидит непустой список
            # и отказ/обрыв ответа провайдера будет скрыт.
            raise RuntimeError(
                f"модель не вернула {missing} строк из {len(texts)} в нумерованной пачке"
            )
        return out


def _misaligned(texts: Sequence[str], parsed: dict[int, str]) -> bool:
    """Сдвиг нумерации: «переводы» совпадают с СОСЕДНИМИ исходными строками."""

    def norm(s: str) -> str:
        return " ".join(s.split()).lower()

    sources = {norm(t): i for i, t in enumerate(texts, 1)}
    echoes = 0
    for number, value in parsed.items():
        source_index = sources.get(norm(value))
        if source_index is not None and source_index != number:
            echoes += 1
    return echoes >= max(2, len(texts) // 3)


def _parse_numbered(text: str) -> dict[int, str]:
    """Разбирает ответ вида '1. …' построчно; продолжения без номера
    приклеиваются к предыдущей строке."""
    result: dict[int, str] = {}
    last: int | None = None
    for line in text.splitlines():
        match = re.match(r"\s*(\d+)\s*[.):\-–—]\s*(.*\S)\s*$", line)
        if match:
            last = int(match.group(1))
            if last in result:
                raise RuntimeError("модель повторила нумерацию пачки")
            result[last] = match.group(2)
        elif last is not None and line.strip():
            result[last] += " " + line.strip()
    return result
