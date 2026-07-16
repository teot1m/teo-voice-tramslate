"""Бесплатный перевод через публичный endpoint Google Translate — без ключа.

Тот же механизм используют браузерные переводчики и субтитры VOT. Оплаты и
регистрации нет; качество ниже LLM (буквальнее, без контекста тона), для
закадрового перевода обычно достаточно. Endpoint неофициальный: Google может
ограничивать частые запросы — при сбоях пачка автоматически переводится по
одной реплике, а совсем надёжные альтернативы — Ollama (локально, бесплатно)
или openai-compatible.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from uvt.interfaces import TranslationEngine
from uvt.registry import register

log = logging.getLogger("uvt.translate.google")

_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


@register("translation", "google-free")
class GoogleFreeTranslator(TranslationEngine):
    async def warmup(self) -> None:
        import httpx

        timeout = float(getattr(self.cfg, "timeout_s", 30.0) or 30.0)
        self._client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": _UA})

    async def close(self) -> None:
        if hasattr(self, "_client"):
            await self._client.aclose()

    @staticmethod
    def _code(lang: str | None) -> str:
        if not lang or lang == "und":
            return "auto"
        return lang.split("-")[0].lower()

    async def _request(self, text: str, source_lang: str | None, target_lang: str) -> str:
        response = await self._client.get(
            _ENDPOINT,
            params={
                "client": "gtx",
                "sl": self._code(source_lang),
                "tl": self._code(target_lang) or "en",
                "dt": "t",
                "q": text,
            },
        )
        response.raise_for_status()
        data = response.json()
        # data[0] — список кусков [переведённый, исходный, …]; переносы строк сохраняются
        return "".join(chunk[0] for chunk in data[0] if chunk and chunk[0])

    async def translate(
        self, text: str, source_lang: str, target_lang: str, context: Sequence
    ) -> str:
        return (await self._request(text, source_lang, target_lang)).strip()

    async def translate_batch(
        self, texts: Sequence[str], source_lang: str | None, target_lang: str
    ) -> list[str]:
        joined = "\n".join(" ".join(t.split()) for t in texts)
        result = await self._request(joined, source_lang, target_lang)
        lines = [line.strip() for line in result.split("\n")]
        if len(lines) == len(texts) and all(lines):
            return lines
        log.warning(
            "google-free: пачка распалась (%d → %d строк) — перевожу по одной",
            len(texts), len(lines),
        )
        semaphore = asyncio.Semaphore(4)

        async def one(text: str) -> str:
            async with semaphore:
                return await self.translate(text, source_lang or "und", target_lang, [])

        return list(await asyncio.gather(*[one(t) for t in texts]))
