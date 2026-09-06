"""Отделение речи от фона перед распознаванием и сборкой дорожки.

Три проблемы дубляжа решаются одним шагом:

- STT на музыке и шумах зацикливается и выдумывает текст; на чистой речи это
  почти прекращается;
- образец голоса для клонирования, взятый вместе с музыкой, переносит в
  озвучку и музыку;
- в миксе оригинальную речь приходилось приглушать вместе с фоном (ducking),
  из-за чего пропадали и звуки сцены. Если речь отделена, фон играет целиком,
  а на месте исходной речи звучит только перевод.

Модель — Demucs (htdemucs). Работает через torch, на Apple Silicon — на MPS.
Это тяжёлый шаг: он окупается на музыкальных дорожках и мешает там, где нужна
скорость, поэтому включается профилем, а не по умолчанию.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
import inspect
import logging
from pathlib import Path
import sys
import threading
from dataclasses import dataclass

import numpy as np

from uvt.audio import resample

log = logging.getLogger("uvt.separate")

DEFAULT_MODEL = "htdemucs"
_VOCAL_SOURCE = "vocals"

ProgressCallback = Callable[[float, float], None]


class SeparationCancelled(asyncio.CancelledError):
    """Cooperative cancellation after the current native inference chunk."""


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise SeparationCancelled("отделение речи отменено")


def _load_cached_hf_model(model_name: str):
    """Use Demucs' HF format without allowing metadata requests or downloads."""
    import yaml
    from huggingface_hub import hf_hub_download
    from demucs.apply import BagOfModels
    from demucs.hf import DEFAULT_NAMESPACE, hf_repo_name, load_safetensors_model

    name = model_name.removeprefix("hf://")
    namespace = DEFAULT_NAMESPACE
    if "/" in name:
        namespace, name = name.split("/", 1)
    repo_id = f"{namespace}/{hf_repo_name(name)}"
    bag_path = hf_hub_download(repo_id, f"{name}.yaml", local_files_only=True)
    with open(bag_path, encoding="utf-8") as stream:
        bag = yaml.safe_load(stream)
    if not isinstance(bag, dict) or not isinstance(bag.get("models"), list) or not bag["models"]:
        raise RuntimeError("локальное описание модели Demucs повреждено")
    # Resolve every cached file before allocating any weights. An incomplete
    # HF cache can then fall back to legacy checkpoints without retaining a
    # partially loaded bag in unified memory.
    paths = [
        hf_hub_download(repo_id, f"{signature}.safetensors", local_files_only=True)
        for signature in bag["models"]
    ]
    models = [load_safetensors_model(path) for path in paths]
    return BagOfModels(models, bag.get("weights"), bag.get("segment"))


def _load_cached_legacy_model(model_name: str):
    """Retain support for the original torch.hub/checkpoints *.th cache."""
    import torch
    from demucs.pretrained import REMOTE_ROOT
    from demucs.repo import AnyModelRepo, BagOnlyRepo, LocalRepo

    checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    if not checkpoints.is_dir():
        raise FileNotFoundError("локальный кэш контрольных точек Demucs отсутствует")
    models = LocalRepo(checkpoints)
    bags = BagOnlyRepo(REMOTE_ROOT, models)
    return AnyModelRepo(models, bags).get_model(model_name)


def _load_separation_model(model_name: str, *, allow_download: bool):
    if allow_download:
        from demucs.pretrained import get_model
        return get_model(model_name)
    for loader in (_load_cached_hf_model, _load_cached_legacy_model):
        try:
            return loader(model_name)
        except Exception as exc:  # noqa: BLE001 - both cache formats are optional
            log.debug("локальная модель Demucs недоступна через %s: %s", loader.__name__, type(exc).__name__)
    raise RuntimeError(
        f"модель Demucs {model_name!r} не найдена или повреждена в локальном кэше; "
        "подготовьте её заранее или отключите отделение речи от фона. "
        "Автоматическое скачивание отключено"
    )


@dataclass(slots=True)
class SeparatedAudio:
    """Речь и фон в той же частоте и форме, что и вход."""

    speech: np.ndarray
    background: np.ndarray
    sample_rate: int


def _resolve_device(configured: str) -> str:
    value = str(configured or "auto").lower()
    if value not in ("auto", ""):
        return value
    try:
        import torch

        if sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 — без torch отдаём cpu, ошибка придёт позже
        pass
    return "cpu"


def _as_stereo(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        return np.stack([data, data], axis=1)
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2]


def separate_speech(
    samples: np.ndarray,
    rate: int,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    shifts: int = 0,
    overlap: float = 0.25,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    allow_download: bool = False,
) -> SeparatedAudio:
    """Separate speech, reporting completed source seconds from native chunks.

    ``progress`` runs on the same worker thread as this function. Async callers
    must publish it through their event loop. Set ``cancel_event`` when the
    awaiter is cancelled, then drain its worker: the next native chunk boundary
    stops inference. Cancellation never becomes an ordinary separation fallback.
    """
    _raise_if_cancelled(cancel_event)
    if rate <= 0 or shifts < 0 or not 0 <= overlap < 1:
        raise ValueError("Demucs: rate должен быть положительным, shifts >= 0, overlap от 0 до 1")
    stereo = _as_stereo(samples)
    duration = len(stereo) / rate
    if progress is not None:
        progress(0.0, duration)
    if not len(stereo):
        return SeparatedAudio(stereo[:, 0].copy(), stereo.copy(), rate)
    try:
        import torch
        from demucs.apply import apply_model
    except ImportError as exc:
        raise RuntimeError("отделение речи требует demucs: pip install demucs") from exc
    supports_callback = "callback" in inspect.signature(apply_model).parameters
    if not supports_callback and (progress is not None or cancel_event is not None):
        raise RuntimeError(
            "установленная версия Demucs не поддерживает прогресс и отмену по фрагментам; "
            "обновите Demucs или отключите отделение речи от фона"
        )
    _raise_if_cancelled(cancel_event)
    log.info("подготавливаю %s для отделения речи (%s)…", model_name, "сеть разрешена" if allow_download else "локальный кэш")
    model = _load_separation_model(model_name, allow_download=allow_download)
    resolved_device = _resolve_device(device)
    sources = normalized = wav = reference = None
    try:
        _raise_if_cancelled(cancel_event)
        model.eval()
        if _VOCAL_SOURCE not in model.sources:
            raise RuntimeError(f"у модели {model_name} нет источника vocals: {model.sources}")
        model_rate = int(getattr(model, "samplerate", 44100))
        work = stereo if rate == model_rate else resample(stereo, rate, model_rate)
        duration = len(work) / model_rate
        log.info("отделяю речь от фона (%s, %s, %.1f с звука)…", model_name, resolved_device, duration)
        wav = torch.from_numpy(np.ascontiguousarray(work.T, dtype=np.float32))
        reference = wav.mean(0)
        scale = float(reference.std()) or 1.0
        shift = float(reference.mean())
        normalized = (wav - shift) / scale
        segment_seconds = [float(getattr(part, "segment", 8.0)) for part in getattr(model, "models", [model])]
        shift_count = max(1, shifts)
        completed: dict[tuple[int, int], float] = {}
        previous = 0.0

        def on_chunk(info: dict) -> None:
            nonlocal previous
            _raise_if_cancelled(cancel_event)
            if info.get("state") != "end":
                return
            # MPS dispatch is asynchronous. Synchronize before declaring a chunk
            # complete; Demucs immediately copies it to CPU for overlap-add too.
            if resolved_device == "mps":
                torch.mps.synchronize()
            elif resolved_device.startswith("cuda"):
                torch.cuda.synchronize(resolved_device)
            _raise_if_cancelled(cancel_event)
            model_index = min(max(0, int(info.get("model_idx_in_bag", 0))), len(segment_seconds) - 1)
            shift_index = min(max(0, int(info.get("shift_idx", 0))), shift_count - 1)
            covered = min(duration, max(0, int(info.get("segment_offset", 0))) / model_rate + segment_seconds[model_index])
            key = (model_index, shift_index)
            completed[key] = max(completed.get(key, 0.0), covered)
            # Never show 100% before all sources are assembled and resampled.
            done = min(duration * .99, sum(completed.values()) / (len(segment_seconds) * shift_count))
            done = max(previous, done)
            if done > previous:
                previous = done
                log.info("отделение речи: %.1f / %.1f с (%.0f%%)", done, duration, 100 * done / duration)
                if progress is not None:
                    progress(done, duration)
            _raise_if_cancelled(cancel_event)

        kwargs = dict(device=resolved_device, shifts=shifts, overlap=overlap, progress=False, split=True, num_workers=0)
        if supports_callback:
            kwargs["callback"] = on_chunk
        with torch.no_grad():
            sources = apply_model(model, normalized[None], **kwargs)[0]
        _raise_if_cancelled(cancel_event)
        sources = sources * scale + shift
        index = model.sources.index(_VOCAL_SOURCE)
        speech = sources[index].cpu().numpy().T.astype(np.float32)
        background = np.zeros_like(speech)
        for position in range(sources.shape[0]):
            _raise_if_cancelled(cancel_event)
            if position != index:
                background += np.asarray(sources[position].cpu().numpy().T, dtype=np.float32)
        if rate != model_rate:
            speech = resample(speech, model_rate, rate)
            background = resample(background, model_rate, rate)
        _raise_if_cancelled(cancel_event)
        if progress is not None:
            progress(duration, duration)
        log.info("речь и фон разделены")
        return SeparatedAudio(speech=speech, background=background, sample_rate=rate)
    finally:
        # Demucs moves bag members back to CPU only on its success path. A
        # cancellation callback must release their MPS weights as well.
        sources = normalized = wav = reference = None
        try:
            model.to("cpu")
            if resolved_device == "mps":
                torch.mps.empty_cache()
            elif resolved_device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            log.debug("не удалось полностью освободить кэш Demucs", exc_info=True)
