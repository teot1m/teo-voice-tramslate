"""Piper TTS loaded once in the UVT process.

The previous adapter started a new ``piper`` process for every subtitle line.
On a fanless 8 GB Mac that repeated ONNX model loading dominated synthesis.
This adapter uses Piper's Python API, keeps one voice session per gender, and
releases it when the dubbing stage finishes. ONNX Runtime sessions support
concurrent inference, so the common TTS semaphore can process two phrases at
once without loading duplicate voice models.
"""
from __future__ import annotations

import asyncio
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from uvt.interfaces import TTSEngine
from uvt.registry import register


@register("tts", "piper")
class PiperTTS(TTSEngine):
    async def warmup(self) -> None:
        try:
            import piper  # noqa: F401 - fail before a job starts
        except ImportError as exc:
            raise RuntimeError(
                'локальная озвучка требует piper-tts: pip install -e ".[mac-local]"'
            ) from exc

        self._voice_dir = Path(
            str(
                getattr(self.cfg, "voice_dir", "")
                or "~/.local/share/uvt/piper"
            )
        ).expanduser()
        configured = getattr(self.cfg, "voice_models", {}) or {}
        self._voice_models = {
            str(key).lower(): self._expand_model_path(str(value))
            for key, value in dict(configured).items()
        }
        legacy = str(getattr(self.cfg, "model_path", "") or "").strip()
        self._legacy_model = self._expand_model_path(legacy) if legacy else None
        if not self._voice_models and self._legacy_model is None:
            raise RuntimeError(
                "Piper: не заданы voice_models/model_path — выполните "
                "uvt setup-mac-local и используйте профиль local-balanced"
            )

        for model_path in set(self._voice_models.values()) | (
            {self._legacy_model} if self._legacy_model else set()
        ):
            self._validate_model(model_path)
        self._voices: dict[Path, Any] = {}
        self._load_lock = asyncio.Lock()

    def _expand_model_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self._voice_dir / path

    @staticmethod
    def _model_sample_rate(model_path: Path) -> int:
        sidecar = Path(f"{model_path}.json")
        if not sidecar.is_file():
            raise RuntimeError(
                "Piper: рядом с моделью не найден metadata-файл "
                f"{sidecar.name} с audio.sample_rate"
            )
        try:
            data: Any = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Piper: не удалось прочитать {sidecar.name}: нужен корректный JSON"
            ) from exc
        audio = data.get("audio") if isinstance(data, dict) else None
        rate = audio.get("sample_rate") if isinstance(audio, dict) else None
        if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
            raise RuntimeError(
                f"Piper: в {sidecar.name} нет корректного audio.sample_rate"
            )
        return rate

    def _validate_model(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise RuntimeError(
                f"Piper: голос {model_path} не установлен — "
                "один раз выполните: uvt setup-mac-local"
            )
        rate = self._model_sample_rate(model_path)
        configured = getattr(self.cfg, "sample_rate", None)
        if configured is not None and configured != rate:
            raise RuntimeError(
                "Piper: tts.sample_rate не совпадает с audio.sample_rate "
                f"модели ({configured} != {rate})"
            )

    def _resolve_model(self, language: str) -> Path:
        root = str(language or "").replace("_", "-").split("-", 1)[0].lower()
        requested_voice = str(getattr(self.cfg, "voice_id", "") or "").strip()
        if requested_voice:
            for key, model_path in self._voice_models.items():
                key_language = key.split(":", 1)[0]
                if model_path.stem != requested_voice:
                    continue
                if key_language not in {root, "default", "male", "female"}:
                    raise RuntimeError(
                        f"Piper: голос '{requested_voice}' не подходит для языка '{language}'"
                    )
                return model_path
            raise RuntimeError(
                f"Piper: голос '{requested_voice}' отсутствует в tts.voice_models"
            )

        gender = str(getattr(self.cfg, "voice_gender", "auto") or "auto").lower()
        gender = "female" if gender.startswith(("f", "ж")) else "male"
        for key in (
            f"{root}:{gender}",
            f"{root}:default",
            gender,
            "default",
        ):
            if key in self._voice_models:
                return self._voice_models[key]
        if self._legacy_model is not None:
            return self._legacy_model
        raise RuntimeError(
            f"Piper: для языка '{language}' и голоса '{gender}' нет модели; "
            "выполните uvt setup-mac-local или добавьте tts.voice_models"
        )

    async def _voice(self, model_path: Path):
        voice = self._voices.get(model_path)
        if voice is not None:
            return voice
        async with self._load_lock:
            voice = self._voices.get(model_path)
            if voice is None:
                from piper import PiperVoice

                worker = asyncio.create_task(
                    asyncio.to_thread(PiperVoice.load, model_path)
                )
                try:
                    voice = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    # Loading also runs in a native worker.  Drain it before
                    # releasing _load_lock so timeout/cancel cannot race a
                    # second load or engine.close().
                    result = await asyncio.gather(worker, return_exceptions=True)
                    if result and not isinstance(result[0], BaseException):
                        self._voices[model_path] = result[0]
                    raise
                self._voices[model_path] = voice
            return voice

    @staticmethod
    def _synthesize(voice, text: str, speed: float) -> tuple[np.ndarray, int]:
        from piper import SynthesisConfig

        syn_config = SynthesisConfig(
            # Piper length_scale < 1 speaks faster.  Keep the range natural;
            # the common dub mixer handles any remaining timing pressure.
            length_scale=max(0.625, min(1.0, 1.0 / max(float(speed), 1.0))),
            normalize_audio=True,
        )
        chunks = list(voice.synthesize(text, syn_config=syn_config))
        if not chunks:
            return np.zeros(0, dtype=np.float32), int(voice.config.sample_rate)
        samples = np.concatenate(
            [np.asarray(chunk.audio_float_array, dtype=np.float32) for chunk in chunks]
        )
        return samples, int(chunks[0].sample_rate)

    async def _speak(
        self, text: str, language: str, speed: float
    ) -> tuple[np.ndarray, int]:
        voice = await self._voice(self._resolve_model(language))
        # Cancelling asyncio.to_thread only stops the await; the native ONNX
        # worker keeps running. Shield and drain the already-started bounded
        # phrase so close()/the next job never race a ghost Piper inference.
        worker = asyncio.create_task(
            asyncio.to_thread(self._synthesize, voice, text, speed)
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            await asyncio.gather(worker, return_exceptions=True)
            raise

    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        return await self._speak(text, language, 1.0)

    async def synthesize_rated(
        self, text: str, language: str, speed: float
    ) -> tuple[np.ndarray, int]:
        return await self._speak(text, language, speed)

    async def close(self) -> None:
        self._voices = {}
        await asyncio.to_thread(gc.collect)
