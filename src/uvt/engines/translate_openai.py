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
        # Маленьким локальным моделям большие пачки не по зубам — сбивают нумерацию
        self.batch_hint = 8 if _is_local_host(self._base) else 20

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
        return response.json()["choices"][0]["message"]["content"].strip()

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
        content = response.json()["choices"][0]["message"]["content"]
        parsed = _parse_numbered(content)
        if _misaligned(texts, parsed):
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
            log.warning("пакетный перевод: %d строк из %d без ответа — оставлены как есть", missing, len(texts))
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
            result[last] = match.group(2)
        elif last is not None and line.strip():
            result[last] += " " + line.strip()
    return result
