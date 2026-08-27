"""IndexTTS-2: тембр из образца, интонация — из самой исходной реплики.

Единственная из доступных локальных моделей, которая разводит две вещи,
всегда слипавшиеся в дубляже:

- ``spk_audio_prompt`` задаёт, ЧЕЙ голос звучит (образец из оригинала);
- ``emo_audio_prompt`` задаёт, КАК он звучит — на вход подаётся звук самой
  переводимой реплики, поэтому шёпот остаётся шёпотом, а крик — криком.

Плюс ``duration_factor`` для укладки в тайминг: если естественная озвучка не
влезла в слот, движок повторяет синтез с рассчитанным коэффициентом, а не
ускоряет готовый звук.

Установка не через PyPI: репозиторий index-tts + скачанные веса, путь к ним
задаётся в профиле (``model_dir``, ``cfg_path``).
"""
from __future__ import annotations

import asyncio
import gc
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from uvt.interfaces import TTSEngine, VoiceReference
from uvt.registry import register

log = logging.getLogger("uvt.tts.indextts")

_MIN_REF_S = 0.5
# Ниже 0.75 речь звучит скороговоркой — лучше дать реплике выйти за слот,
# сборка сдвинет следующую.
_MIN_DURATION_FACTOR = 0.75
_SLOT_TOLERANCE_S = 0.25


@register("tts", "indextts")
class IndexTTS2Engine(TTSEngine):
    supports_duration = True
    supports_reference = True
    supports_emotion = True

    async def warmup(self) -> None:
        try:
            from indextts.infer_v2 import IndexTTS2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "движок tts=indextts требует пакет indextts (репозиторий "
                "index-tts) и скачанные веса; укажите tts.model_dir и tts.cfg_path"
            ) from exc

        model_dir = str(getattr(self.cfg, "model_dir", "") or "").strip()
        if not model_dir:
            raise RuntimeError("IndexTTS-2: в профиле не задан tts.model_dir с весами")
        self._model_dir = Path(model_dir).expanduser()
        cfg_path = str(getattr(self.cfg, "cfg_path", "") or "").strip()
        self._cfg_path = (
            Path(cfg_path).expanduser() if cfg_path else self._model_dir / "config.yaml"
        )
        if not self._cfg_path.is_file():
            raise RuntimeError(f"IndexTTS-2: не найден конфиг {self._cfg_path}")

        self._emo_alpha = float(getattr(self.cfg, "emo_alpha", 0.85) or 0.85)
        self._use_fp16 = bool(getattr(self.cfg, "use_fp16", False))
        self._min_factor = float(
            getattr(self.cfg, "min_duration_factor", _MIN_DURATION_FACTOR)
            or _MIN_DURATION_FACTOR
        )
        self._fallback_ref = str(getattr(self.cfg, "reference_wav", "") or "").strip()
        self._tmp = tempfile.TemporaryDirectory(prefix="uvt-indextts-")
        self._ref_cache: dict[str, Path] = {}
        self._lock = asyncio.Lock()
        self._counter = 0

        from indextts.infer_v2 import IndexTTS2 as _Index

        def build() -> Any:
            return _Index(
                cfg_path=str(self._cfg_path),
                model_dir=str(self._model_dir),
                use_fp16=self._use_fp16,
                use_cuda_kernel=False,
                use_deepspeed=False,
            )

        log.info("IndexTTS-2: загружаю веса из %s…", self._model_dir)
        self._model = await asyncio.to_thread(build)
        log.info("IndexTTS-2: модель готова")

    async def close(self) -> None:
        self._model = None
        self._ref_cache = {}
        tmp = getattr(self, "_tmp", None)
        if tmp is not None:
            tmp.cleanup()
            self._tmp = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _write_wav(self, reference: VoiceReference, name: str) -> Path:
        import soundfile as sf

        samples = np.ascontiguousarray(reference.samples, dtype=np.float32)
        if len(samples) / max(reference.sample_rate, 1) < _MIN_REF_S:
            raise RuntimeError(f"IndexTTS-2: образец '{name}' короче {_MIN_REF_S} с")
        path = Path(self._tmp.name) / f"{name}.wav"
        sf.write(path, samples, reference.sample_rate, subtype="PCM_16")
        return path

    def _speaker_wav(self, reference: VoiceReference | None) -> Path:
        if reference is not None:
            cached = self._ref_cache.get(reference.label)
            if cached is None:
                cached = self._write_wav(reference, f"spk-{reference.label.replace('/', '_')}")
                self._ref_cache[reference.label] = cached
            return cached
        if self._fallback_ref:
            path = Path(self._fallback_ref).expanduser()
            if not path.is_file():
                raise RuntimeError(f"IndexTTS-2: образец {path} не найден")
            return path
        raise RuntimeError(
            "IndexTTS-2: нет образца голоса — задайте tts.reference_wav или "
            "включите отбор образцов из оригинала"
        )

    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        return await self.synthesize_slot(text, language)

    async def synthesize_slot(
        self,
        text: str,
        language: str,
        *,
        target_duration: float | None = None,
        reference: VoiceReference | None = None,
        emotion: VoiceReference | None = None,
        speed: float | None = None,
    ) -> tuple[np.ndarray, int]:
        phrase = " ".join(str(text).split())
        if not phrase:
            return np.zeros(0, dtype=np.float32), 24000

        speaker_wav = self._speaker_wav(reference)
        emotion_wav: Path | None = None
        if emotion is not None:
            try:
                self._counter += 1
                emotion_wav = self._write_wav(emotion, f"emo-{self._counter}")
            except RuntimeError:
                # Слишком короткий фрагмент для переноса просодии — не беда,
                # синтезируем без него.
                emotion_wav = None

        base_factor = 1.0 / float(speed) if speed else 1.0

        def run(factor: float) -> tuple[np.ndarray, int]:
            import soundfile as sf

            self._counter += 1
            out = Path(self._tmp.name) / f"gen-{self._counter}.wav"
            kwargs: dict[str, Any] = {
                "spk_audio_prompt": str(speaker_wav),
                "text": phrase,
                "output_path": str(out),
                "verbose": False,
            }
            if emotion_wav is not None:
                kwargs["emo_audio_prompt"] = str(emotion_wav)
                kwargs["emo_alpha"] = self._emo_alpha
            if abs(factor - 1.0) > 0.01:
                kwargs["duration_factor"] = round(factor, 3)
            self._model.infer(**kwargs)
            samples, sample_rate = sf.read(out, dtype="float32", always_2d=False)
            out.unlink(missing_ok=True)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            return np.ascontiguousarray(samples, dtype=np.float32), int(sample_rate)

        async with self._lock:
            if self._model is None:
                raise RuntimeError("IndexTTS-2: движок уже закрыт")
            samples, rate = await asyncio.to_thread(run, base_factor)

            if target_duration is None or rate <= 0:
                return samples, rate
            duration = len(samples) / rate
            if duration <= target_duration + _SLOT_TOLERANCE_S:
                return samples, rate

            # Не влезли: пересинтезируем с коэффициентом длительности вместо
            # ускорения готового клипа.
            factor = max(self._min_factor, target_duration / duration) * base_factor
            if factor >= base_factor - 0.01:
                return samples, rate
            log.debug(
                "IndexTTS-2: реплика %.2f с не влезла в %.2f с — повтор с factor=%.2f",
                duration,
                target_duration,
                factor,
            )
            return await asyncio.to_thread(run, factor)
