"""Pinned Nemotron 3.5 ASR through MLX Audio, with native chunk timestamps.

The public UVT interface still receives completed utterances/files. Native
streaming is used internally without retranscribing accumulated audio.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import math
import threading
from pathlib import Path

import numpy as np

from uvt.interfaces import STTEngine, STTResult, STTSpan
from uvt.registry import register

log = logging.getLogger("uvt.stt.nemotron-mlx")
MODEL_REPO = "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
MODEL_REVISION = "7279359e4481b5e9e185a318bd618e429c6d86cd"
_NATIVE_GUARD = threading.Lock()


async def _native_call(operation, *, skip_if_cancelled=True):
    """Cancel between native chunks and drain the active one before closing."""
    cancel = threading.Event()

    def run():
        with _NATIVE_GUARD:
            if skip_if_cancelled and cancel.is_set():
                return None
            return operation(cancel)

    task = asyncio.create_task(asyncio.to_thread(run))
    cancelled = False
    while True:
        try:
            value = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
            cancel.set()
        except Exception:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
    if cancelled:
        raise asyncio.CancelledError
    return value


def _result_spans(result, duration: float, language: str | None) -> list[STTSpan]:
    if result is None:
        return []
    spans = []
    for sentence in (getattr(result, "sentences", None) or ()):
        text = " ".join(str(getattr(sentence, "text", "") or "").split())
        try:
            start = float(sentence.start)
            end = float(sentence.end)
        except (AttributeError, TypeError, ValueError):
            continue
        if not text or not math.isfinite(start) or not math.isfinite(end):
            continue
        start = max(0.0, start)
        end = min(duration, end)
        if start < end:
            spans.append(STTSpan(start, end, text, language))
    if not spans and str(getattr(result, "text", "") or "").strip() and duration > 0:
        spans = [STTSpan(0.0, duration, str(result.text).strip(), language)]
    return spans


@register("stt", "nemotron-mlx")
class NemotronMLX(STTEngine):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._lock = asyncio.Lock()
        self._model = None
        self._mx = None
        self._chunk_seconds = float(getattr(cfg, "chunk_seconds", 5.0) or 5.0)
        if not .08 <= self._chunk_seconds <= 30.0:
            raise ValueError("Nemotron: chunk_seconds должен быть от 0.08 до 30")

    def _load(self, cancel):
        if self._model is not None:
            return
        try:
            import mlx.core as mx
            from mlx_audio.stt.utils import load_model
        except ImportError as exc:
            raise RuntimeError('Nemotron: установите зависимости pip install -e ".[local-next]"') from exc
        path = Path(str(getattr(self.cfg, "model", ".models/nemotron"))).expanduser()
        required = ("config.json", "model.safetensors", "tokenizer.model")
        if not path.is_dir() or any(not (path / name).is_file() for name in required):
            raise RuntimeError("Nemotron: модель не подготовлена; выполните uvt setup-mac-local --preset nemotron")
        log.info("Nemotron MLX: загружаю локальную модель…")
        self._model = load_model(str(path.resolve()), lazy=False)
        self._mx = mx
        log.info("Nemotron MLX: модель готова")

    async def warmup(self):
        async with self._lock:
            await _native_call(self._load)

    def _language(self, value):
        code = str(value or "auto").strip().replace("_", "-")
        if code.lower() in {"", "auto", "und"}:
            return "auto"
        if code.lower() == "ua":
            code = "uk"
        prompts = self._model.prompt_dictionary
        if code in prompts:
            return code
        lower = {key.lower(): key for key in prompts}
        if code.lower() in lower:
            return lower[code.lower()]
        root = code.lower().split("-", 1)[0]
        if root in prompts:
            return root
        raise ValueError(f"Nemotron не поддерживает выбранный язык: {value}")

    @staticmethod
    def _audio(samples, sample_rate):
        if sample_rate != 16000:
            raise ValueError("Nemotron ожидает звук 16000 Гц")
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Nemotron ожидает одномерный монофонический звук")
        if not np.isfinite(audio).all():
            raise ValueError("Nemotron: звук содержит некорректные отсчёты")
        return audio

    def _recognize(self, audio, language, progress, cancel):
        prompt = self._language(language)
        duration = len(audio) / 16000
        if not duration:
            return []
        latest = None
        iterator = self._model.stream_generate(
            self._mx.array(audio), language=prompt, chunk_duration=self._chunk_seconds,
        )
        try:
            for index, result in enumerate(iterator, 1):
                if cancel.is_set():
                    return []
                # Results are cumulative. Retain the last one rather than
                # appending and duplicating every earlier sentence.
                latest = result
                if progress:
                    progress(min(duration, index * self._chunk_seconds), duration)
        finally:
            close = getattr(iterator, "close", None)
            if close:
                close()
        if cancel.is_set():
            return []
        detected = None if prompt == "auto" else prompt.lower().split("-", 1)[0]
        if detected is None and latest is not None and str(latest.text).strip():
            import langid
            detected = langid.classify(str(latest.text)[:20000])[0]
        return _result_spans(latest, duration, detected)

    async def transcribe_long(self, samples, sample_rate, language, progress=None):
        audio = self._audio(samples, sample_rate)
        if not len(audio):
            return []
        loop = asyncio.get_running_loop()
        callback = (lambda done, total: loop.call_soon_threadsafe(progress, done, total)) if progress else None
        async with self._lock:
            if self._model is None:
                raise RuntimeError("Nemotron: сначала подготовьте модель")
            return await _native_call(lambda cancel: self._recognize(audio, language, callback, cancel))

    async def transcribe(self, samples, sample_rate, language):
        spans = await self.transcribe_long(samples, sample_rate, language)
        if not spans:
            return None
        return STTResult(" ".join(span.text for span in spans), language=spans[0].language)

    async def close(self):
        async with self._lock:
            def release(cancel):
                self._model = None
                if self._mx is not None:
                    self._mx.clear_cache()
                gc.collect()
            await _native_call(release, skip_if_cancelled=False)
