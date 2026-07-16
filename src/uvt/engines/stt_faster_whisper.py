"""Faster Whisper (CTranslate2) — основной локальный STT (ТЗ §21, MVP).

device="auto": CUDA при наличии, иначе CPU (int8). На Apple Silicon работает
CPU-режим; Metal у CTranslate2 нет — для macOS это ожидаемо.
"""
from __future__ import annotations

import asyncio
import logging
import re

import numpy as np

from uvt.interfaces import STTEngine, STTResult, STTSpan
from uvt.registry import register

log = logging.getLogger("uvt.stt.faster-whisper")

_SENTENCE_END = re.compile(r"[.!?…]+[\"»')\]]*$")
_MAX_SPAN_S = 12.0  # предохранитель: фраза без знаков препинания режется по длине


def words_to_spans(words, language: str | None) -> list[STTSpan]:
    """Группирует пословные таймкоды Whisper в предложения.

    Пакетный Whisper отдаёт блоки до ~30 с — для дубляжа это слишком грубо:
    озвучка блока стартует с его начала и расходится с фразами внутри.
    Предложения дают пофразовую синхронизацию.
    """
    spans: list[STTSpan] = []
    current: list = []

    def flush() -> None:
        if not current:
            return
        text = "".join(w.word for w in current).strip()
        if text:
            spans.append(
                STTSpan(float(current[0].start), float(current[-1].end), text, language)
            )
        current.clear()

    for word in words:
        current.append(word)
        too_long = (word.end - current[0].start) >= _MAX_SPAN_S
        if _SENTENCE_END.search(word.word.strip()) or too_long:
            flush()
    flush()
    return spans


# Модель тяжёлая — держим по одному экземпляру на конфигурацию на весь процесс,
# иначе каждое нажатие кнопки в браузере перезагружало бы её заново.
_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


@register("stt", "faster-whisper")
class FasterWhisperSTT(STTEngine):
    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        from faster_whisper import WhisperModel

        cfg = self.cfg
        key = (str(cfg.model), str(cfg.device), str(cfg.compute_type))
        model = _MODEL_CACHE.get(key)
        if model is None:
            log.info(
                "загружаю Whisper '%s' (device=%s) в память — дальше будет мгновенно; "
                "скачивание с сети происходит только один раз за всё время",
                cfg.model, cfg.device,
            )
            model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
            _MODEL_CACHE[key] = model
        self._model = model

    async def transcribe(
        self, samples: np.ndarray, sample_rate: int, language: str | None
    ) -> STTResult | None:
        if sample_rate != 16000:
            raise ValueError("faster-whisper ожидает 16 кГц (это гарантирует CaptureService)")
        return await asyncio.to_thread(self._transcribe, samples, language)

    def _transcribe(self, samples: np.ndarray, language: str | None) -> STTResult | None:
        segments, info = self._model.transcribe(
            samples,
            language=language,
            beam_size=self.cfg.beam_size,
            vad_filter=False,  # нарезкой уже занялся наш VAD
            condition_on_previous_text=False,  # меньше галлюцинаций на потоке
            without_timestamps=True,
        )
        parts = [s.text.strip() for s in segments if s.text.strip()]
        text = " ".join(parts)
        if not text:
            return None
        return STTResult(
            text=text,
            language=info.language,
            confidence=float(info.language_probability),
        )

    async def transcribe_long(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str | None,
        progress=None,
    ) -> list[STTSpan] | None:
        """Пакетное распознавание целого файла: один проход, таймкоды, батчи."""
        if sample_rate != 16000:
            raise ValueError("faster-whisper ожидает 16 кГц")
        return await asyncio.to_thread(self._transcribe_long, samples, language, progress)

    def _transcribe_long(self, samples: np.ndarray, language, progress) -> list[STTSpan]:
        from faster_whisper import BatchedInferencePipeline

        if getattr(self, "_batched", None) is None:
            self._batched = BatchedInferencePipeline(model=self._model)
        batch_size = int(getattr(self.cfg, "batch_size", 8))
        total_s = len(samples) / 16000
        segments, info = self._batched.transcribe(
            samples,
            language=language,
            batch_size=batch_size,
            beam_size=self.cfg.beam_size,
            word_timestamps=True,  # для пофразовой нарезки (синхронность дубляжа)
        )
        spans: list[STTSpan] = []
        for segment in segments:
            words = getattr(segment, "words", None)
            if words:
                spans.extend(words_to_spans(words, info.language))
            elif segment.text.strip():
                spans.append(
                    STTSpan(float(segment.start), float(segment.end), segment.text.strip(), info.language)
                )
            if progress is not None:
                progress(min(float(segment.end), total_s), total_s)
        return spans
