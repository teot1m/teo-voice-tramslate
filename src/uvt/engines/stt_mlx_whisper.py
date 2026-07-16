"""Whisper на GPU Apple Silicon через MLX (mlx-whisper) — только macOS.

faster-whisper (CTranslate2) работает на CPU; на M-чипах GPU простаивает.
MLX гоняет Whisper на Metal: large-v3-turbo на M2 быстрее CPU-варианта
small при качестве large-модели. Установка: pip install mlx-whisper.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import numpy as np

from uvt.engines.stt_faster_whisper import words_to_spans
from uvt.interfaces import STTEngine, STTResult, STTSpan
from uvt.registry import register

log = logging.getLogger("uvt.stt.mlx")

# Короткие имена → репозитории mlx-community; полный путь тоже принимается
_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


@register("stt", "mlx-whisper")
class MlxWhisperSTT(STTEngine):
    async def warmup(self) -> None:
        import mlx_whisper  # noqa: F401 — ранняя проверка зависимости

        self._repo = _REPOS.get(str(self.cfg.model), str(self.cfg.model))
        log.info("mlx-whisper: модель %s (GPU Apple Silicon)", self._repo)

    def _transcribe(self, samples: np.ndarray, language: str | None, words: bool) -> dict:
        import mlx_whisper

        return mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self._repo,
            language=language,
            word_timestamps=words,
        )

    async def transcribe(
        self, samples: np.ndarray, sample_rate: int, language: str | None
    ) -> STTResult | None:
        if sample_rate != 16000:
            raise ValueError("mlx-whisper ожидает 16 кГц")
        result = await asyncio.to_thread(self._transcribe, samples, language, False)
        text = (result.get("text") or "").strip()
        if not text:
            return None
        return STTResult(text=text, language=result.get("language") or language)

    async def transcribe_long(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str | None,
        progress=None,
    ) -> list[STTSpan] | None:
        if sample_rate != 16000:
            raise ValueError("mlx-whisper ожидает 16 кГц")
        result = await asyncio.to_thread(self._transcribe, samples, language, True)
        detected = result.get("language") or language
        spans: list[STTSpan] = []
        for segment in result.get("segments") or []:
            raw_words = segment.get("words") or []
            if raw_words:
                words = [SimpleNamespace(**w) for w in raw_words]
                spans.extend(words_to_spans(words, detected))
            elif (segment.get("text") or "").strip():
                spans.append(
                    STTSpan(
                        float(segment.get("start") or 0.0),
                        float(segment.get("end") or 0.0),
                        segment["text"].strip(),
                        detected,
                    )
                )
        if progress is not None:
            progress(1, 1)
        return spans