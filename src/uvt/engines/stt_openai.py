"""Облачный STT через OpenAI-совместимый endpoint /audio/transcriptions.

Работает с OpenAI (whisper-1, gpt-4o-mini-transcribe) и Groq
(base_url=https://api.groq.com/openai/v1, model=whisper-large-v3-turbo —
бесплатный тариф: ~2 часа аудио в час, 228× реального времени).

Для дубляжа весь файл уходит ОДНИМ запросом (response_format=verbose_json,
сжатие FLAC) — сегменты с таймкодами возвращает сам сервис.
"""
from __future__ import annotations

import io
import logging
import os

import numpy as np

from uvt.audio import wav_bytes
from uvt.interfaces import STTEngine, STTResult, STTSpan
from uvt.registry import register

log = logging.getLogger("uvt.stt.openai")

_MAX_UPLOAD_BYTES = 38_000_000  # лимит Groq free — 40 МБ, оставляем запас


def _error_detail(response) -> str:
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("code") or "")
            else:
                detail = str(error or "")
    except Exception:  # noqa: BLE001 — в ошибке может быть HTML от proxy
        pass
    if not detail:
        detail = response.text
    return " ".join(str(detail).split())[:360]


def _require_success(response) -> None:
    """Не теряем текст HTTP 400: он нужен для понятного local failover."""
    if response.is_success:
        return
    detail = _error_detail(response)
    suffix = f" — {detail}" if detail else ""
    raise RuntimeError(f"облачное STT вернуло HTTP {response.status_code}{suffix}")


def _is_response_format_incompatible(response) -> bool:
    """Модель (например gpt-4o(-mini)-transcribe) не умеет verbose_json с сегментами.

    Это не сбой облака — просто у batch-режима нет пары для этой модели, поэтому
    вызывающий код должен уйти на VAD + пореплечное распознавание тем же
    облачным движком, а не на локальный резерв.
    """
    if response.status_code != 400:
        return False
    detail = _error_detail(response).lower()
    return "response_format" in detail and "verbose_json" in detail


@register("stt", "openai-compatible")
class OpenAICompatibleSTT(STTEngine):
    async def warmup(self) -> None:
        import httpx

        from uvt.engines.translate_openai import _require_key_for_remote

        self._base = str(self.cfg.base_url).rstrip("/")
        key = os.environ.get(self.cfg.api_key_env, "")
        _require_key_for_remote(self._base, key, self.cfg.api_key_env)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.AsyncClient(timeout=30.0, headers=headers)

    async def close(self) -> None:
        if hasattr(self, "_client"):
            await self._client.aclose()

    async def transcribe(
        self, samples: np.ndarray, sample_rate: int, language: str | None
    ) -> STTResult | None:
        data = {"model": self.cfg.model or "whisper-1"}
        if language:
            data["language"] = language
        response = await self._client.post(
            f"{self._base}/audio/transcriptions",
            data=data,
            files={"file": ("segment.wav", wav_bytes(samples, sample_rate), "audio/wav")},
        )
        _require_success(response)
        payload = response.json()
        text = (payload.get("text") or "").strip()
        if not text:
            return None
        return STTResult(text=text, language=payload.get("language") or language)

    async def transcribe_long(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str | None,
        progress=None,
    ) -> list[STTSpan] | None:
        """Весь файл одним запросом: сегменты с таймкодами из verbose_json."""
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="FLAC")
        payload = buffer.getvalue()
        if len(payload) > _MAX_UPLOAD_BYTES:
            log.info(
                "файл больше лимита загрузки (%.0f МБ) — переключаюсь на локальную нарезку",
                len(payload) / 1e6,
            )
            return None  # dub возьмёт путь VAD + пореплечных запросов

        data = {
            "model": self.cfg.model or "whisper-1",
            "response_format": "verbose_json",
        }
        if language:
            data["language"] = language
        response = await self._client.post(
            f"{self._base}/audio/transcriptions",
            data=data,
            files={"file": ("audio.flac", payload, "audio/flac")},
            timeout=300.0,
        )
        if not response.is_success and _is_response_format_incompatible(response):
            log.info(
                "модель %s не поддерживает verbose_json — переключаюсь на VAD + "
                "пореплечное распознавание тем же облачным движком",
                data["model"],
            )
            return None
        _require_success(response)
        body = response.json()
        detected = body.get("language") or language
        spans = [
            STTSpan(
                float(seg.get("start") or 0.0),
                float(seg.get("end") or 0.0),
                (seg.get("text") or "").strip(),
                detected,
            )
            for seg in body.get("segments") or []
            if (seg.get("text") or "").strip()
        ]
        if progress is not None:
            progress(1, 1)
        return spans
