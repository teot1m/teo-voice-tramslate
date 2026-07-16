"""Аудио-утилиты: моно, ресемплинг, кодирование/декодирование байтов."""
from __future__ import annotations

import io
import subprocess

import numpy as np


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples.astype(np.float32, copy=False)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    import soxr

    return soxr.resample(samples, src_rate, dst_rate).astype(np.float32, copy=False)


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """PCM16 WAV в память — для отправки в облачные STT."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def decode_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Декодирует сжатое аудио (mp3/ogg/wav) в float32 моно.

    Сначала libsndfile (в колёсах soundfile ≥0.12 есть поддержка mp3),
    при неудаче — ffmpeg из PATH.
    """
    try:
        import soundfile as sf

        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return to_mono(np.asarray(samples)), int(rate)
    except Exception:  # noqa: BLE001 — пробуем ffmpeg
        pass
    return _ffmpeg_decode(data)


def _ffmpeg_decode(data: bytes, rate: int = 24000) -> tuple[np.ndarray, int]:
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", "pipe:0",
                "-f", "f32le", "-acodec", "pcm_f32le",
                "-ac", "1", "-ar", str(rate), "pipe:1",
            ],
            input=data,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "не удалось декодировать аудио: нет ни libsndfile с mp3, ни ffmpeg в PATH"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg не смог декодировать аудио: {exc.stderr.decode(errors='ignore')}") from exc
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), rate
