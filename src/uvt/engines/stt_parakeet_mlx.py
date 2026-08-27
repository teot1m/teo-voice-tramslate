"""Fast multilingual Parakeet ASR on Apple Silicon through MLX.

``parakeet-mlx`` currently exposes file-level chunking, but that call keeps a
worker busy until the whole file has finished.  UVT chunks the in-memory 16 kHz
audio itself so cancellation can stop after the active chunk and still wait
for that worker to leave MLX in a well-defined state.
"""

from __future__ import annotations

import asyncio
import gc
import importlib
import logging
import math
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uvt.interfaces import STTEngine, STTResult, STTSpan
from uvt.registry import register

log = logging.getLogger("uvt.stt.parakeet-mlx")

_DEFAULT_REPO = "mlx-community/parakeet-tdt-0.6b-v3"
_REPOS = {
    "parakeet-tdt-0.6b-v3": _DEFAULT_REPO,
    "parakeet-v3": _DEFAULT_REPO,
    "v3": _DEFAULT_REPO,
}

# Languages listed by the NVIDIA v3 model card.  In particular, EN/RU/UK are
# explicit members rather than being accepted accidentally by a generic code.
_PARAKEET_LANGUAGES = frozenset(
    {
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "uk",
    }
)
_LANGUAGE_ALIASES = {"ua": "uk"}

_MIN_CHUNK_SECONDS = 60.0
_MAX_CHUNK_SECONDS = 120.0
_DEFAULT_OVERLAP_SECONDS = 15.0
_MIN_SPAN_SECONDS = 0.001
_LANGID_TEXT_LIMIT = 20_000

# A loaded 0.6B model is shared by jobs and engine instances.  The model cache
# is protected during construction; inference is serialized separately below.
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_GUARD = threading.Lock()
_MODEL_EXECUTION_GUARD = threading.Lock()

_LANGID_IDENTIFIER: Any | None = None
_LANGID_GUARD = threading.Lock()


async def _thread_call_and_drain_on_cancel(
    call: Callable[..., Any], *args: Any
) -> Any:
    """Run blocking work and never abandon its thread after cancellation.

    ``asyncio.to_thread`` cannot kill a running Python/Metal call.  Shielding
    the worker and delaying ``CancelledError`` until it completes prevents an
    old inference from continuing invisibly after a job is marked cancelled.
    Repeated cancellation requests are handled by the loop as well.
    """

    worker = asyncio.create_task(asyncio.to_thread(call, *args))
    cancelled = False
    while True:
        try:
            value = await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
            continue
        except Exception as exc:
            if cancelled:
                log.debug("worker failed while cancellation was draining", exc_info=exc)
                raise asyncio.CancelledError from None
            raise
        if cancelled:
            raise asyncio.CancelledError
        return value


async def _serial_model_call_and_drain_on_cancel(
    call: Callable[..., Any], *args: Any
) -> Any:
    """Serialize MLX across loops and skip work cancelled while queued."""

    cancel_requested = threading.Event()

    def guarded_call() -> Any:
        with _MODEL_EXECUTION_GUARD:
            if cancel_requested.is_set():
                return None
            return call(*args)

    worker = asyncio.create_task(asyncio.to_thread(guarded_call))
    cancelled = False
    while True:
        try:
            value = await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
            cancel_requested.set()
            continue
        except Exception as exc:
            if cancelled:
                log.debug(
                    "model worker failed while cancellation was draining",
                    exc_info=exc,
                )
                raise asyncio.CancelledError from None
            raise
        if cancelled:
            raise asyncio.CancelledError
        return value


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bounded_config_float(
    value: Any, *, default: float, minimum: float, maximum: float
) -> float:
    number = _finite_float(value)
    if number is None:
        number = default
    return min(max(number, minimum), maximum)


def _normalise_language(language: str | None) -> str | None:
    if language is None:
        return None
    code = str(language).strip().lower().replace("_", "-")
    if code in {"", "auto", "und"}:
        return None
    code = _LANGUAGE_ALIASES.get(code, code)
    code = _LANGUAGE_ALIASES.get(code.split("-", 1)[0], code.split("-", 1)[0])
    if code not in _PARAKEET_LANGUAGES:
        supported = ", ".join(sorted(_PARAKEET_LANGUAGES))
        raise ValueError(
            f"Parakeet v3 does not support language {language!r}; "
            f"supported ISO codes: {supported}"
        )
    return code


def _detect_language_blocking(text: str) -> tuple[str | None, float | None]:
    """Classify one aggregate transcript, restricted to Parakeet languages."""

    compact = " ".join(text.split())[:_LANGID_TEXT_LIMIT]
    if not compact:
        return None, None

    global _LANGID_IDENTIFIER
    with _LANGID_GUARD:
        if _LANGID_IDENTIFIER is None:
            from langid.langid import LanguageIdentifier, model

            identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
            identifier.set_languages(sorted(_PARAKEET_LANGUAGES))
            _LANGID_IDENTIFIER = identifier
        code, confidence = _LANGID_IDENTIFIER.classify(compact)

    code = str(code).lower()
    if code not in _PARAKEET_LANGUAGES:
        return None, None
    probability = _finite_float(confidence)
    if probability is not None:
        probability = min(max(probability, 0.0), 1.0)
    return code, probability


def _result_text(result: Any) -> str:
    text = " ".join(str(getattr(result, "text", "") or "").split())
    if text:
        return text
    return " ".join(
        str(getattr(sentence, "text", "") or "").strip()
        for sentence in (getattr(result, "sentences", None) or [])
        if str(getattr(sentence, "text", "") or "").strip()
    ).strip()


def _result_to_spans(
    result: Any,
    *,
    offset_seconds: float,
    chunk_seconds: float,
    language: str | None,
) -> list[STTSpan]:
    """Convert third-party alignments to finite, bounded UVT timestamps."""

    raw_sentences = [
        sentence
        for sentence in (getattr(result, "sentences", None) or [])
        if str(getattr(sentence, "text", "") or "").strip()
    ]
    if not raw_sentences:
        text = _result_text(result)
        if not text or chunk_seconds <= 0:
            return []
        return [
            STTSpan(
                max(0.0, offset_seconds),
                max(0.0, offset_seconds + chunk_seconds),
                text,
                language,
            )
        ]

    spans: list[STTSpan] = []
    count = len(raw_sentences)
    for index, sentence in enumerate(raw_sentences):
        text = " ".join(str(sentence.text).split())
        fallback_start = chunk_seconds * index / count
        fallback_end = chunk_seconds * (index + 1) / count

        start = _finite_float(getattr(sentence, "start", None))
        end = _finite_float(getattr(sentence, "end", None))
        duration = _finite_float(getattr(sentence, "duration", None))
        if start is None:
            start = fallback_start
        if end is None and duration is not None and duration > 0:
            end = start + duration
        if end is None:
            end = fallback_end
        if end < start:
            start, end = end, start

        start = min(max(start, 0.0), chunk_seconds)
        end = min(max(end, 0.0), chunk_seconds)
        if end <= start and chunk_seconds > 0:
            minimum = min(_MIN_SPAN_SECONDS, chunk_seconds)
            start = min(start, max(0.0, chunk_seconds - minimum))
            end = min(chunk_seconds, max(start + minimum, fallback_end))

        spans.append(
            STTSpan(
                max(0.0, offset_seconds + start),
                max(0.0, offset_seconds + end),
                text,
                language,
            )
        )

    spans.sort(key=lambda span: (span.start, span.end))
    return spans


def _merge_chunk_spans(
    existing: list[STTSpan],
    incoming: list[STTSpan],
    *,
    chunk_start: float,
    overlap_seconds: float,
) -> list[STTSpan]:
    """Give each overlapping recognition result half of the overlap window."""

    if not existing:
        return incoming
    if not incoming:
        return existing
    if overlap_seconds <= 0:
        return sorted([*existing, *incoming], key=lambda span: (span.start, span.end))

    boundary = chunk_start + overlap_seconds / 2.0
    left = [span for span in existing if (span.start + span.end) / 2.0 < boundary]
    right = [span for span in incoming if (span.start + span.end) / 2.0 >= boundary]
    return sorted([*left, *right], key=lambda span: (span.start, span.end))


@register("stt", "parakeet-mlx")
class ParakeetMlxSTT(STTEngine):
    """Parakeet v3 with bounded chunks and a process-local model cache."""

    supported_languages = _PARAKEET_LANGUAGES
    concurrency_hint = 1

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        configured = str(getattr(cfg, "model", _DEFAULT_REPO) or _DEFAULT_REPO)
        self._configured_model = _REPOS.get(configured, configured)
        self._model: Any | None = None
        self._model_path: str | None = None
        self._runtime: tuple[Any, Any, Any] | None = None

        self.chunk_seconds = _bounded_config_float(
            getattr(cfg, "chunk_seconds", getattr(cfg, "chunk_duration", 120.0)),
            default=120.0,
            minimum=_MIN_CHUNK_SECONDS,
            maximum=_MAX_CHUNK_SECONDS,
        )
        self.overlap_seconds = _bounded_config_float(
            getattr(cfg, "overlap_seconds", getattr(cfg, "overlap_duration", 15.0)),
            default=_DEFAULT_OVERLAP_SECONDS,
            minimum=0.0,
            maximum=self.chunk_seconds / 2.0,
        )

    async def warmup(self) -> None:
        log.info("parakeet-mlx: loading model %s...", self._configured_model)
        await self._ensure_model()
        log.info("parakeet-mlx: model ready: %s (GPU Apple Silicon)", self._model_path)

    def _load_model_blocking(self) -> None:
        if self._model is not None:
            return

        parakeet = importlib.import_module("parakeet_mlx")
        parakeet_audio = importlib.import_module("parakeet_mlx.audio")
        mx = importlib.import_module("mlx.core")

        direct = Path(self._configured_model).expanduser()
        if direct.is_dir():
            model_path = direct.resolve()
        else:
            revision = str(getattr(self.cfg, "revision", "") or "")
            allow_download = bool(getattr(self.cfg, "allow_download", True))
            cache_dir = getattr(self.cfg, "cache_dir", None)
            try:
                from huggingface_hub import snapshot_download

                model_path = Path(
                    snapshot_download(
                        repo_id=self._configured_model,
                        revision=revision or None,
                        cache_dir=cache_dir,
                        local_files_only=not allow_download,
                        allow_patterns=["config.json", "model.safetensors"],
                    )
                ).resolve()
            except Exception as exc:  # noqa: BLE001 - actionable setup error
                raise RuntimeError(
                    f"Parakeet MLX {self._configured_model} is not installed; "
                    "run `uvt setup-mac-local` once"
                ) from exc

        missing = [
            name
            for name in ("config.json", "model.safetensors")
            if not (model_path / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Parakeet MLX snapshot {model_path} is incomplete "
                f"(missing {', '.join(missing)}); run `uvt setup-mac-local` again"
            )

        cache_key = str(model_path)
        with _MODEL_CACHE_GUARD:
            model = _MODEL_CACHE.get(cache_key)
            if model is None:
                model = parakeet.from_pretrained(cache_key)
                parameters = getattr(model, "parameters", None)
                evaluate = getattr(mx, "eval", None)
                if callable(parameters) and callable(evaluate):
                    evaluate(parameters())
                _MODEL_CACHE[cache_key] = model

        self._model = model
        self._model_path = cache_key
        self._runtime = (mx, parakeet_audio.get_logmel, model)

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        await _serial_model_call_and_drain_on_cancel(self._load_model_blocking)

    def _infer_chunk_blocking(self, samples: np.ndarray) -> Any:
        if self._runtime is None:
            raise RuntimeError("Parakeet MLX model was not loaded")
        mx, get_logmel, model = self._runtime
        clean = np.nan_to_num(
            np.asarray(samples, dtype=np.float32),
            copy=True,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        clean = np.clip(clean, -1.0, 1.0)
        try:
            audio = mx.array(clean)
            mel = get_logmel(audio, model.preprocessor_config)
            generated = model.generate(mel)
            if not generated:
                raise RuntimeError("Parakeet MLX returned no transcription result")
            return generated[0]
        finally:
            clear_cache = getattr(mx, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()

    async def _infer_chunk(self, samples: np.ndarray) -> Any:
        await self._ensure_model()
        return await _serial_model_call_and_drain_on_cancel(
            self._infer_chunk_blocking, samples
        )

    async def transcribe(
        self, samples: np.ndarray, sample_rate: int, language: str | None
    ) -> STTResult | None:
        if sample_rate != 16_000:
            raise ValueError("parakeet-mlx expects 16 kHz audio")
        explicit_language = _normalise_language(language)
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return None

        result = await self._infer_chunk(audio)
        text = _result_text(result)
        if not text:
            return None

        detected, confidence = (explicit_language, None)
        if detected is None:
            detected, confidence = await _thread_call_and_drain_on_cancel(
                _detect_language_blocking, text
            )
        return STTResult(text=text, language=detected, confidence=confidence)

    async def transcribe_long(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str | None,
        progress=None,
    ) -> list[STTSpan] | None:
        if sample_rate != 16_000:
            raise ValueError("parakeet-mlx expects 16 kHz audio")
        explicit_language = _normalise_language(language)
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        total_samples = int(audio.size)
        if total_samples == 0:
            if progress is not None:
                progress(0.0, 0.0)
            return []

        total_seconds = total_samples / sample_rate
        chunk_samples = max(1, int(round(self.chunk_seconds * sample_rate)))
        overlap_samples = min(
            int(round(self.overlap_seconds * sample_rate)), chunk_samples - 1
        )
        step_samples = max(1, chunk_samples - overlap_samples)
        if progress is not None:
            progress(0.0, total_seconds)

        spans: list[STTSpan] = []
        start_sample = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = np.ascontiguousarray(audio[start_sample:end_sample])
            result = await self._infer_chunk(chunk)

            chunk_start = start_sample / sample_rate
            chunk_duration = len(chunk) / sample_rate
            incoming = _result_to_spans(
                result,
                offset_seconds=chunk_start,
                chunk_seconds=chunk_duration,
                language=explicit_language,
            )
            actual_overlap = (
                min(self.overlap_seconds, chunk_duration) if start_sample else 0.0
            )
            spans = _merge_chunk_spans(
                spans,
                incoming,
                chunk_start=chunk_start,
                overlap_seconds=actual_overlap,
            )

            if progress is not None:
                progress(end_sample / sample_rate, total_seconds)
            if end_sample >= total_samples:
                break
            start_sample += step_samples

        detected_language = explicit_language
        if detected_language is None and spans:
            detected_language, _ = await _thread_call_and_drain_on_cancel(
                _detect_language_blocking, " ".join(span.text for span in spans)
            )
        if detected_language is not None:
            for span in spans:
                span.language = detected_language
        return spans

    async def close(self) -> None:
        """Release Parakeet before the translation model enters unified RAM."""

        await _serial_model_call_and_drain_on_cancel(self._release_model_blocking)

    def _release_model_blocking(self) -> None:
        model = self._model
        model_path = self._model_path
        self._model = None
        self._model_path = None
        runtime = self._runtime
        self._runtime = None

        if model_path is not None:
            with _MODEL_CACHE_GUARD:
                if _MODEL_CACHE.get(model_path) is model:
                    _MODEL_CACHE.pop(model_path, None)

        if runtime is not None:
            clear_cache = getattr(runtime[0], "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
        gc.collect()
