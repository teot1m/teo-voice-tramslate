"""Нарезка непрерывного звука на реплики — общая логика для живого конвейера
(VADService) и офлайн-дубляжа (uvt dub).

Счёт ведётся в кадрах по 512 сэмплов (32 мс при 16 кГц); отметки времени
сегментов — секунды от начала потока (для файла это позиция в файле).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from uvt.events import SpeechSegment, Trace
from uvt.interfaces import VADEngine

FRAME = 512
RATE = 16000
FRAME_MS = FRAME * 1000 / RATE  # 32 мс


@dataclass(frozen=True)
class SegmenterParams:
    on_threshold: float
    off_threshold: float
    min_silence_frames: int
    min_speech_frames: int
    max_frames: int
    pre_roll_frames: int
    keep_tail_frames: int

    @classmethod
    def from_config(cls, vcfg, preset) -> "SegmenterParams":
        min_silence_ms = (
            vcfg.min_silence_ms if vcfg.min_silence_ms is not None else preset.min_silence_ms
        )
        pre_roll_ms = vcfg.pre_roll_ms if vcfg.pre_roll_ms is not None else preset.pre_roll_ms
        max_segment_s = (
            vcfg.max_segment_s if vcfg.max_segment_s is not None else preset.max_segment_s
        )
        min_speech = max(1, round(vcfg.min_speech_ms / FRAME_MS))
        return cls(
            on_threshold=vcfg.threshold,
            off_threshold=max(0.15, vcfg.threshold - 0.15),
            min_silence_frames=max(1, round(min_silence_ms / FRAME_MS)),
            min_speech_frames=min_speech,
            max_frames=max(min_speech, round(max_segment_s * 1000 / FRAME_MS)),
            pre_roll_frames=max(1, round(pre_roll_ms / FRAME_MS)),
            keep_tail_frames=max(1, round(160 / FRAME_MS)),
        )


async def create_vad_engine(vcfg, log: logging.Logger) -> VADEngine:
    """Создаёт VAD из конфига; при недоступности деградирует до 'energy'."""
    from uvt import registry

    try:
        engine = registry.create("vad", vcfg.engine, vcfg)
        await engine.warmup()
        return engine
    except Exception as exc:  # noqa: BLE001
        if vcfg.engine == "energy":
            raise
        log.warning("VAD '%s' недоступен (%s) — переключаюсь на 'energy'", vcfg.engine, exc)
        engine = registry.create("vad", "energy", vcfg)
        await engine.warmup()
        return engine


class Segmenter:
    """Отсечка по паузе + пред-буфер + потолок длины сегмента."""

    def __init__(self, engine: VADEngine, params: SegmenterParams) -> None:
        self.engine = engine
        self.p = params
        self._pre: deque[tuple[int, np.ndarray]] = deque(maxlen=params.pre_roll_frames)
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0  # позиция в сэмплах от начала потока
        self._active = False
        self._frames: list[tuple[int, np.ndarray]] = []
        self._silence = 0
        self._speech = 0

    def feed(self, samples: np.ndarray) -> list[SpeechSegment]:
        self._buf = np.concatenate([self._buf, samples.astype(np.float32, copy=False)])
        out: list[SpeechSegment] = []
        while len(self._buf) >= FRAME:
            frame, self._buf = self._buf[:FRAME], self._buf[FRAME:]
            segment = self._step(frame)
            if segment is not None:
                out.append(segment)
        return out

    def flush(self) -> SpeechSegment | None:
        """Конец потока: закрыть активный сегмент (файл кончился на речи)."""
        if not self._active:
            return None
        self._active = False
        return self._finalize()

    def _step(self, frame: np.ndarray) -> SpeechSegment | None:
        start = self._pos
        self._pos += FRAME
        prob = self.engine.prob(frame)

        if not self._active:
            self._pre.append((start, frame))
            if prob >= self.p.on_threshold:
                self._active = True
                self._frames = list(self._pre)
                self._pre.clear()
                self._silence = 0
                self._speech = 1
            return None

        self._frames.append((start, frame))
        if prob >= self.p.off_threshold:
            self._speech += 1
            self._silence = 0
        else:
            self._silence += 1

        if self._silence >= self.p.min_silence_frames or len(self._frames) >= self.p.max_frames:
            self._active = False
            return self._finalize()
        return None

    def _finalize(self) -> SpeechSegment | None:
        frames = self._frames
        self._frames = []
        cut = self._silence - self.p.keep_tail_frames
        self._silence = 0
        speech = self._speech
        self._speech = 0

        if 0 < cut < len(frames):
            frames = frames[:-cut]
        if speech < self.p.min_speech_frames or not frames:
            return None

        samples = np.concatenate([f for _, f in frames])
        trace = Trace()
        trace.mark("speech_end")
        return SpeechSegment(
            samples=samples,
            sample_rate=RATE,
            start_ts=frames[0][0] / RATE,
            end_ts=(frames[-1][0] + FRAME) / RATE,
            trace=trace,
        )
