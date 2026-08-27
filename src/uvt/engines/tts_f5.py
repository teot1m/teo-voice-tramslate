"""F5-TTS: клонирование голоса из оригинала и синтез точно в тайминг.

Зачем он нужен рядом с Piper. Piper — фиксированный голос с фиксированной
просодией: одна интонация на весь фильм, а укладка в тайминги делается
ускорением готового звука (atempo), что слышно как перемотка. F5-TTS —
flow-matching модель: она генерирует мел заданной длины, поэтому реплику
можно синтезировать сразу в доступный слот, без ускорения; а голос берётся
из короткого образца самой исходной дорожки, поэтому тембр и манера
совпадают с настоящим говорящим.

Важная деталь API: ``fix_duration`` в F5-TTS — это полная длительность
(образец + сгенерированная речь), поэтому движок сам прибавляет длину
образца к запрошенному слоту. Выход модели — 24 кГц моно.

Русский язык: базовый чекпойнт обучен на английском и китайском, поэтому для
ru/uk в профиле указывается путь к файнтюну (``model_path`` + ``vocab_path``).
Без файнтюна движок работает, но русская фонетика будет с акцентом.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from uvt.interfaces import TTSEngine, VoiceReference
from uvt.registry import register

log = logging.getLogger("uvt.tts.f5")

_DEFAULT_MODEL = "F5TTS_v1_Base"
_OUTPUT_RATE = 24000
_MIN_SLOT_S = 0.35     # короче слота фиксация длительности только вредит
_MIN_REF_S = 0.5


@register("tts", "f5")
class F5TTSEngine(TTSEngine):
    supports_duration = True
    supports_reference = True

    async def warmup(self) -> None:
        try:
            from f5_tts.api import F5TTS  # noqa: F401 — падаем до старта задачи
        except ImportError as exc:
            raise RuntimeError(
                "движок tts=f5 требует f5-tts: pip install f5-tts "
                "(на Apple Silicon работает через torch MPS)"
            ) from exc

        self._model_name = str(getattr(self.cfg, "model", "") or _DEFAULT_MODEL)
        # Явный путь в профиле побеждает; иначе веса берутся из закреплённого
        # снимка Hub, установленного `uvt setup-mac-local --preset natural`.
        hub_ckpt, hub_vocab = self._hub_files()
        self._ckpt = self._optional_path("model_path") or hub_ckpt
        self._vocab = self._optional_path("vocab_path") or hub_vocab
        for path in (self._ckpt, self._vocab):
            if path is not None and not path.is_file():
                raise RuntimeError(f"F5-TTS: файл модели {path} не найден")
        self._nfe_step = int(getattr(self.cfg, "nfe_step", 32) or 32)
        self._speed = float(getattr(self.cfg, "speed", 1.0) or 1.0)
        self._cfg_strength = float(getattr(self.cfg, "cfg_strength", 2.0) or 2.0)
        self._min_slot_s = float(getattr(self.cfg, "min_slot_s", _MIN_SLOT_S) or _MIN_SLOT_S)
        seed = getattr(self.cfg, "seed", None)
        self._seed = int(seed) if seed is not None else None
        self._device = self._resolve_device()

        # Статический образец на случай, когда в ролике не нашлось чистого
        # фрагмента речи (например одна короткая реплика на весь файл).
        self._fallback_ref = self._optional_path("reference_wav")
        self._fallback_ref_text = str(getattr(self.cfg, "reference_text", "") or "")
        if self._fallback_ref is not None and not self._fallback_ref.is_file():
            raise RuntimeError(f"F5-TTS: образец {self._fallback_ref} не найден")

        self._tmp = tempfile.TemporaryDirectory(prefix="uvt-f5-ref-")
        self._ref_cache: dict[str, tuple[Path, str, float]] = {}
        self._lock = asyncio.Lock()

        from f5_tts.api import F5TTS as _F5

        def build() -> Any:
            return _F5(
                model=self._model_name,
                ckpt_file=str(self._ckpt) if self._ckpt else "",
                vocab_file=str(self._vocab) if self._vocab else "",
                device=self._device,
            )

        log.info(
            "F5-TTS: загружаю %s%s на %s…",
            self._model_name,
            f" ({self._ckpt.name})" if self._ckpt else "",
            self._device,
        )
        self._model = await asyncio.to_thread(build)
        log.info("F5-TTS: модель готова")

    def _hub_files(self) -> tuple[Path | None, Path | None]:
        """Веса и словарь из pinned-ревизии Hub без сетевых сюрпризов.

        По умолчанию ``allow_download: false``, как и у остальных локальных
        движков: задача не должна сама тянуть модель во время дубляжа.
        """
        repo = str(getattr(self.cfg, "hf_repo", "") or "").strip()
        if not repo:
            return None, None
        revision = str(getattr(self.cfg, "hf_revision", "") or "").strip() or None
        allow_download = bool(getattr(self.cfg, "allow_download", False))

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                'F5-TTS: hf_repo требует huggingface-hub: pip install -e ".[mac-local]"'
            ) from exc

        def fetch(name: str) -> Path | None:
            if not name:
                return None
            try:
                return Path(
                    hf_hub_download(
                        repo_id=repo,
                        filename=name,
                        revision=revision,
                        local_files_only=not allow_download,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — подсказываем установку
                raise RuntimeError(
                    f"F5-TTS: {repo}/{name} нет в локальном кэше — выполните "
                    "uvt setup-mac-local --preset natural"
                ) from exc

        return (
            fetch(str(getattr(self.cfg, "hf_model_file", "") or "").strip()),
            fetch(str(getattr(self.cfg, "hf_vocab_file", "") or "").strip()),
        )

    def _optional_path(self, key: str) -> Path | None:
        raw = str(getattr(self.cfg, key, "") or "").strip()
        return Path(raw).expanduser() if raw else None

    def _resolve_device(self) -> str:
        configured = str(getattr(self.cfg, "device", "auto") or "auto").lower()
        if configured not in ("auto", ""):
            return configured
        try:
            import torch

            if sys.platform == "darwin" and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001 — без torch отдаём cpu и падём позже понятно
            pass
        return "cpu"

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

            if self._device == "mps":
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001 — освобождение памяти не должно ронять задачу
            pass

    def _store_reference(self, reference: VoiceReference) -> tuple[Path, str, float]:
        """Кладёт образец в WAV один раз на говорящего — F5 принимает путь."""
        cached = self._ref_cache.get(reference.label)
        if cached is not None:
            return cached

        import soundfile as sf

        samples = np.ascontiguousarray(reference.samples, dtype=np.float32)
        duration = len(samples) / max(reference.sample_rate, 1)
        if duration < _MIN_REF_S:
            raise RuntimeError(
                f"F5-TTS: образец голоса '{reference.label}' короче {_MIN_REF_S} с"
            )
        path = Path(self._tmp.name) / f"{reference.label.replace('/', '_')}.wav"
        sf.write(path, samples, reference.sample_rate, subtype="PCM_16")
        # Пустая расшифровка допустима: F5 распознает образец сам.
        item = (path, str(reference.text or ""), duration)
        self._ref_cache[reference.label] = item
        return item

    def _resolve_reference(
        self, reference: VoiceReference | None
    ) -> tuple[Path, str, float]:
        if reference is not None:
            return self._store_reference(reference)
        if self._fallback_ref is not None:
            import soundfile as sf

            info = sf.info(str(self._fallback_ref))
            return self._fallback_ref, self._fallback_ref_text, float(info.duration)
        raise RuntimeError(
            "F5-TTS: нет образца голоса — дубляж не передал фрагмент из оригинала, "
            "а tts.reference_wav в профиле не задан"
        )

    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        return await self.synthesize_slot(text, language)

    async def synthesize_rated(
        self, text: str, language: str, speed: float
    ) -> tuple[np.ndarray, int]:
        """Совместимость со старым путём: темп задаётся прямо модели.

        Это лучше, чем atempo поверх готового клипа: речь остаётся живой,
        меняется только скорость произнесения.
        """
        return await self.synthesize_slot(text, language, speed=speed)

    async def synthesize_slot(
        self,
        text: str,
        language: str,
        *,
        target_duration: float | None = None,
        reference: VoiceReference | None = None,
        emotion: VoiceReference | None = None,  # F5 не переносит просодию отдельно
        speed: float | None = None,
    ) -> tuple[np.ndarray, int]:
        phrase = " ".join(str(text).split())
        if not phrase:
            return np.zeros(0, dtype=np.float32), _OUTPUT_RATE

        ref_path, ref_text, ref_seconds = self._resolve_reference(reference)
        fix_duration: float | None = None
        if target_duration is not None and target_duration >= self._min_slot_s:
            # fix_duration в F5 — суммарная длина (образец + синтез).
            fix_duration = ref_seconds + float(target_duration)

        rate = float(speed if speed is not None else self._speed)

        def run() -> tuple[np.ndarray, int]:
            wav, sample_rate, _spec = self._model.infer(
                ref_file=str(ref_path),
                ref_text=ref_text,
                gen_text=phrase,
                show_info=lambda *args, **kwargs: None,
                nfe_step=self._nfe_step,
                cfg_strength=self._cfg_strength,
                speed=rate,
                fix_duration=fix_duration,
                remove_silence=False,
                seed=self._seed,
            )
            samples = np.asarray(wav, dtype=np.float32).reshape(-1)
            return samples, int(sample_rate)

        async with self._lock:
            if self._model is None:
                raise RuntimeError("F5-TTS: движок уже закрыт")
            return await asyncio.to_thread(run)
