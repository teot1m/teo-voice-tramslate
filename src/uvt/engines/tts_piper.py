"""Piper CLI — полностью локальный TTS без ключа и сетевых запросов.

UVT намеренно не скачивает голосовые модели сам: пользователь явно указывает
``tts.model_path`` и устанавливает Piper удобным способом. Такой контракт
делает профиль ``private`` действительно офлайн, в отличие от Edge TTS.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from uvt.interfaces import TTSEngine
from uvt.registry import register


@register("tts", "piper")
class PiperTTS(TTSEngine):
    """Тонкий адаптер к ``piper --output_raw``.

    Поддерживаются любые локально установленные Piper voice models. Язык
    выбирается самой моделью, поэтому ``language`` здесь нужен лишь для общего
    контракта TTS.
    """

    async def warmup(self) -> None:
        binary = str(getattr(self.cfg, "binary", "piper") or "piper")
        self._binary = shutil.which(binary) or (binary if Path(binary).is_file() else None)
        if self._binary is None:
            raise RuntimeError(
                "piper не найден: установите piper-tts и укажите tts.binary, "
                "либо выберите Edge/OpenAI TTS"
            )
        model = str(getattr(self.cfg, "model_path", "") or "")
        model_path = Path(model).expanduser()
        if not model or not model_path.is_file():
            raise RuntimeError(
                "piper: укажите существующий tts.model_path (.onnx); "
                "модели не скачиваются автоматически ради приватности"
            )
        self._model = str(model_path)
        self._sample_rate = self._model_sample_rate(model_path)
        self._validate_configured_sample_rate(self._sample_rate)

    @staticmethod
    def _model_sample_rate(model_path: Path) -> int:
        """Read the rate Piper writes next to every ``.onnx`` model.

        Piper writes raw PCM to stdout.  The CLI does not put a WAV header on
        that stream, so guessing a rate here changes both pitch and duration.
        The adjacent model metadata is the only authoritative local source;
        deliberately do not fetch models or metadata from the network.
        """
        sidecar = Path(f"{model_path}.json")
        if not sidecar.is_file():
            raise RuntimeError(
                "piper: рядом с моделью не найден metadata-файл "
                f"{sidecar.name} с audio.sample_rate; без него UVT не будет "
                "угадывать частоту PCM (иначе исказятся темп и высота голоса)"
            )
        try:
            data: Any = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"piper: не удалось прочитать {sidecar.name}: "
                "нужен корректный JSON Piper с audio.sample_rate"
            ) from exc

        audio = data.get("audio") if isinstance(data, dict) else None
        rate = audio.get("sample_rate") if isinstance(audio, dict) else None
        # bool is an int subclass, but it is never a valid audio sample rate.
        if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
            raise RuntimeError(
                f"piper: в {sidecar.name} нет корректного целого "
                "audio.sample_rate; UVT отказывается угадывать частоту PCM"
            )
        return rate

    def _validate_configured_sample_rate(self, model_rate: int) -> None:
        """Make an old explicit ``tts.sample_rate`` fail loudly on mismatch.

        ``sample_rate`` used to override Piper's actual model rate.  Preserve
        compatibility for existing profiles only when it agrees with metadata;
        otherwise a 16 kHz/22.05 kHz mismatch would sound deceptively valid
        while being too slow/fast and low/high pitched.
        """
        configured = getattr(self.cfg, "sample_rate", None)
        if configured is None:
            return
        if (
            isinstance(configured, bool)
            or not isinstance(configured, int)
            or configured <= 0
        ):
            raise RuntimeError(
                "piper: tts.sample_rate должен быть положительным целым или "
                "не задан вовсе; частота берётся из .onnx.json"
            )
        if configured != model_rate:
            raise RuntimeError(
                "piper: tts.sample_rate не совпадает с audio.sample_rate "
                f"модели ({configured} != {model_rate}); удалите настройку "
                "или укажите частоту из .onnx.json"
            )

    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        del language  # язык зашит в выбранной Piper-модели
        return await asyncio.to_thread(self._synthesize, text)

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        try:
            proc = subprocess.run(
                [self._binary, "--model", self._model, "--output_raw"],
                input=text.encode("utf-8"),
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:  # бинарник мог исчезнуть после warmup
            raise RuntimeError("piper не найден во время синтеза") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"piper не смог синтезировать речь: {detail or exc}") from exc
        samples = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
        return samples, self._sample_rate
