"""Whisper на GPU Apple Silicon через MLX (mlx-whisper) — только macOS.

faster-whisper (CTranslate2) работает на CPU; на M-чипах GPU простаивает.
MLX гоняет Whisper на Metal: large-v3-turbo на M2 быстрее CPU-варианта
small при качестве large-модели. Установка: pip install mlx-whisper.
"""
from __future__ import annotations

import asyncio
from collections import Counter
import gc
import logging
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

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

# ``mlx-whisper`` owns a process-global ModelHolder.  All native calls,
# including release, must therefore be serialized across engine instances.
_MODEL_EXECUTION_LOCK = threading.Lock()


def _locked_call(func: Callable[..., Any], *args) -> Any:
    with _MODEL_EXECUTION_LOCK:
        return func(*args)


async def _wait_for_native(func: Callable[..., Any], *args) -> Any:
    """Run MLX in a thread without leaving it alive after task cancellation."""
    worker = asyncio.create_task(asyncio.to_thread(_locked_call, func, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # Cancelling ``to_thread`` only cancels its asyncio wrapper; the native
        # Metal work keeps running.  Wait for the current bounded chunk before
        # propagating cancellation so close/the next job cannot race it.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        try:
            worker.result()
        except Exception:  # noqa: BLE001 - original cancellation is authoritative
            log.exception("mlx-whisper worker failed while the job was cancelled")
        raise


@register("stt", "mlx-whisper")
class MlxWhisperSTT(STTEngine):
    async def warmup(self) -> None:
        import mlx_whisper  # noqa: F401 — ранняя проверка зависимости

        configured = _REPOS.get(str(self.cfg.model), str(self.cfg.model))
        direct = Path(configured).expanduser()
        if direct.is_dir():
            self._repo = str(direct)
        else:
            revision = str(getattr(self.cfg, "revision", "") or "")
            allow_download = bool(getattr(self.cfg, "allow_download", True))
            try:
                from huggingface_hub import snapshot_download

                self._repo = snapshot_download(
                    repo_id=configured,
                    revision=revision or None,
                    local_files_only=not allow_download,
                )
            except Exception as exc:  # noqa: BLE001 - actionable setup error
                raise RuntimeError(
                    f"MLX Whisper {configured} не установлен — "
                    "один раз выполните: uvt setup-mac-local"
                ) from exc
        log.info("mlx-whisper: загружаю модель %s (GPU Apple Silicon)…", self._repo)
        started = time.perf_counter()
        await _wait_for_native(self._load_model)
        log.info("mlx-whisper: модель готова за %.1f с", time.perf_counter() - started)

    def _load_model(self) -> None:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        # Load/evaluate the pinned local snapshot now, not on the first
        # multi-minute transcription call.  This makes startup readiness real.
        ModelHolder.get_model(self._repo, mx.float16)

    def _transcribe(self, samples: np.ndarray, language: str | None, words: bool) -> dict:
        import mlx_whisper

        return mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self._repo,
            language=language,
            word_timestamps=words,
            verbose=None,
            temperature=0.0,
            condition_on_previous_text=False,
        )

    async def transcribe(
        self, samples: np.ndarray, sample_rate: int, language: str | None
    ) -> STTResult | None:
        if sample_rate != 16000:
            raise ValueError("mlx-whisper ожидает 16 кГц")
        result = await _wait_for_native(self._transcribe, samples, language, False)
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
        total_samples = len(samples)
        total_s = total_samples / sample_rate
        if not total_samples:
            if progress is not None:
                progress(0.0, 0.0)
            return []

        chunk_s = float(getattr(self.cfg, "chunk_seconds", 60.0) or 60.0)
        overlap_s = float(getattr(self.cfg, "overlap_seconds", 2.0) or 0.0)
        if chunk_s < 15.0:
            raise ValueError("mlx-whisper: chunk_seconds должен быть не меньше 15")
        if overlap_s < 0.0 or overlap_s >= chunk_s / 2:
            raise ValueError("mlx-whisper: overlap_seconds должен быть >= 0 и меньше половины chunk_seconds")

        chunk_samples = max(1, int(round(chunk_s * sample_rate)))
        overlap_samples = int(round(overlap_s * sample_rate))
        step_samples = chunk_samples - overlap_samples
        all_words: list[SimpleNamespace] = []
        fallback_spans: list[STTSpan] = []
        languages: Counter[str] = Counter()
        detected_for_next = language

        for chunk_start in range(0, total_samples, step_samples):
            chunk_end = min(total_samples, chunk_start + chunk_samples)
            chunk = np.ascontiguousarray(samples[chunk_start:chunk_end], dtype=np.float32)
            result = await _wait_for_native(
                self._transcribe, chunk, detected_for_next, True
            )

            detected = str(result.get("language") or detected_for_next or "und")
            if detected_for_next is None and detected != "und":
                # Whisper's whole-file mode also chooses one dominant language;
                # keeping it after the first chunk avoids repeated detection.
                detected_for_next = detected
            languages[detected] += max(1, len(str(result.get("text") or "")))

            start_s = chunk_start / sample_rate
            end_s = chunk_end / sample_rate
            keep_from = start_s if chunk_start == 0 else start_s + overlap_s / 2
            keep_to = end_s if chunk_end == total_samples else end_s - overlap_s / 2

            for segment in result.get("segments") or []:
                raw_words = segment.get("words") or []
                if raw_words:
                    for raw in raw_words:
                        word = SimpleNamespace(**raw)
                        word.start = float(word.start) + start_s
                        word.end = float(word.end) + start_s
                        midpoint = (word.start + word.end) / 2
                        if keep_from <= midpoint < keep_to or (
                            chunk_end == total_samples and midpoint <= keep_to
                        ):
                            all_words.append(word)
                    continue

                text = str(segment.get("text") or "").strip()
                seg_start = float(segment.get("start") or 0.0) + start_s
                seg_end = float(segment.get("end") or 0.0) + start_s
                midpoint = (seg_start + seg_end) / 2
                if text and (
                    keep_from <= midpoint < keep_to
                    or (chunk_end == total_samples and midpoint <= keep_to)
                ):
                    fallback_spans.append(
                        STTSpan(seg_start, seg_end, text, detected)
                    )

            completed_s = total_s if chunk_end == total_samples else keep_to
            if progress is not None:
                progress(completed_s, total_s)
            if chunk_end == total_samples:
                break

        final_language = language or (
            languages.most_common(1)[0][0] if languages else None
        )
        spans = words_to_spans(all_words, final_language)
        spans.extend(
            STTSpan(item.start, item.end, item.text, final_language or item.language)
            for item in fallback_spans
        )
        spans.sort(key=lambda item: (item.start, item.end))
        return spans

    async def close(self) -> None:
        """Release Whisper before the translation stage loads."""
        await _wait_for_native(self._release_model)

    def _release_model(self) -> None:
        try:
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder

            if ModelHolder.model_path == getattr(self, "_repo", None):
                ModelHolder.model = None
                ModelHolder.model_path = None
            mx.clear_cache()
        finally:
            gc.collect()
