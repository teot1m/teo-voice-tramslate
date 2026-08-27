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

import logging
import sys
from dataclasses import dataclass

import numpy as np

from uvt.audio import resample

log = logging.getLogger("uvt.separate")

DEFAULT_MODEL = "htdemucs"
_VOCAL_SOURCE = "vocals"


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
) -> SeparatedAudio:
    """Делит дорожку на речь и фон.

    ``samples`` — моно или стерео float32. Результат возвращается в той же
    частоте, что и вход: внутри звук приводится к частоте модели и обратно.
    """
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise RuntimeError(
            "отделение речи требует demucs: pip install demucs"
        ) from exc

    stereo = _as_stereo(samples)
    if not len(stereo):
        return SeparatedAudio(stereo[:, 0].copy(), stereo.copy(), rate)

    model = get_model(model_name)
    model.eval()
    if _VOCAL_SOURCE not in model.sources:
        raise RuntimeError(
            f"модель {model_name} не выделяет речь; источники: {model.sources}"
        )
    model_rate = int(getattr(model, "samplerate", 44100))

    work = stereo if rate == model_rate else resample(stereo, rate, model_rate)
    resolved_device = _resolve_device(device)
    log.info(
        "отделяю речь от фона (%s, %s, %.1f с звука)…",
        model_name,
        resolved_device,
        len(work) / model_rate,
    )

    wav = torch.from_numpy(np.ascontiguousarray(work.T, dtype=np.float32))
    # Нормализация как в demucs.separate: модель обучена на таком масштабе.
    reference = wav.mean(0)
    scale = float(reference.std()) or 1.0
    shift = float(reference.mean())
    normalized = (wav - shift) / scale

    with torch.no_grad():
        sources = apply_model(
            model,
            normalized[None],
            device=resolved_device,
            shifts=shifts,
            split=True,
            overlap=overlap,
            progress=False,
        )[0]
    sources = sources * scale + shift

    index = model.sources.index(_VOCAL_SOURCE)
    speech = sources[index].cpu().numpy().T.astype(np.float32)
    # Фон — сумма остальных источников: так в нём не остаётся остатков речи,
    # в отличие от вычитания вокала из микса.
    background = np.zeros_like(speech)
    for position in range(sources.shape[0]):
        if position == index:
            continue
        background += sources[position].cpu().numpy().T.astype(np.float32)

    if rate != model_rate:
        speech = resample(speech, model_rate, rate)
        background = resample(background, model_rate, rate)
    log.info("речь и фон разделены")
    return SeparatedAudio(speech=speech, background=background, sample_rate=rate)
