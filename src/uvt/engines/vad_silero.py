"""Silero VAD через onnxruntime — лёгкая модель (~2 МБ), без PyTorch.

Модель скачивается один раз в ~/.cache/uvt/models. Поддерживаются v5
(вход state[2,1,128]) и v4 (входы h/c[2,1,64]) — формат определяется по
именам входов ONNX-графа.
"""
from __future__ import annotations

import asyncio
import logging
import os
import urllib.request
from pathlib import Path

import numpy as np

from uvt.interfaces import VADEngine
from uvt.registry import register

log = logging.getLogger("uvt.vad.silero")

MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)


def _cache_dir() -> Path:
    root = os.environ.get("UVT_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")), "uvt"
    )
    path = Path(root).expanduser() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


@register("vad", "silero")
class SileroVAD(VADEngine):
    frame_samples = 512

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        import onnxruntime as ort

        model_path = Path(self.cfg.model_path).expanduser() if self.cfg.model_path else self._download()
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        input_names = {inp.name for inp in self._session.get_inputs()}
        self._v5 = "state" in input_names
        self._sr = np.array(16000, dtype=np.int64)
        self._reset_state()
        log.info("Silero VAD загружен (%s): %s", "v5" if self._v5 else "v4", model_path)

    def _reset_state(self) -> None:
        if self._v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            # v5 ждёт на входе 64 сэмпла контекста от предыдущего кадра: [1, 576]
            self._context = np.zeros((1, 64), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def _download(self) -> Path:
        path = _cache_dir() / "silero_vad.onnx"
        if path.is_file() and path.stat().st_size > 100_000:
            return path
        log.info("скачиваю модель Silero VAD (~2 МБ)…")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)  # noqa: S310 — https на GitHub
        tmp.rename(path)
        return path

    def prob(self, frame: np.ndarray) -> float:
        x = frame[None, :].astype(np.float32)
        if self._v5:
            x = np.concatenate([self._context, x], axis=1)
            output, self._state = self._session.run(
                None, {"input": x, "state": self._state, "sr": self._sr}
            )
            self._context = x[:, -64:]
        else:
            output, self._h, self._c = self._session.run(
                None, {"input": x, "sr": self._sr, "h": self._h, "c": self._c}
            )
        return float(np.asarray(output).reshape(-1)[0])
