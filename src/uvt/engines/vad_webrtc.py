"""WebRTC VAD (ТЗ §3) — требует webrtcvad и кадры ровно 10/20/30 мс,
поэтому движок внутри перенарезает поток на кадры по 30 мс."""
from __future__ import annotations

import numpy as np

from uvt.interfaces import VADEngine
from uvt.registry import register

_WEBRTC_FRAME = 480  # 30 мс при 16 кГц


@register("vad", "webrtc")
class WebRTCVAD(VADEngine):
    frame_samples = 512

    async def warmup(self) -> None:
        import webrtcvad

        aggressiveness = int(getattr(self.cfg, "webrtc_aggressiveness", 2))
        self._vad = webrtcvad.Vad(aggressiveness)
        self._buf = np.zeros(0, dtype=np.float32)
        self._last = 0.0

    def prob(self, frame: np.ndarray) -> float:
        self._buf = np.concatenate([self._buf, frame])
        while len(self._buf) >= _WEBRTC_FRAME:
            chunk, self._buf = self._buf[:_WEBRTC_FRAME], self._buf[_WEBRTC_FRAME:]
            pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            self._last = 1.0 if self._vad.is_speech(pcm, 16000) else 0.0
        return self._last
