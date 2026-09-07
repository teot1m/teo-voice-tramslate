"""MOSS-TTS-Nano 100M: local CPU ONNX synthesis without TorchCodec.

Models are prepared explicitly by setup-mac-local. Synthesis never downloads
weights, runs third-party repository Python, or changes the working Torch stack.
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import math
import re
import unicodedata
import threading
from pathlib import Path
from typing import Any

import numpy as np

from uvt.config import configured_role_voice
from uvt.interfaces import TTSEngine, VoiceReference
from uvt.registry import register

log = logging.getLogger("uvt.tts.moss")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_LANGUAGES = frozenset({
    "zh", "en", "de", "es", "fr", "ja", "it", "hu", "ko", "ru",
    "fa", "ar", "pl", "pt", "cs", "da", "sv", "el", "tr",
})
DEFAULT_VOICES = {"male": "Adam", "female": "Bella"}
BUILTIN_VOICES = {"Adam": "male", "Nathan": "male", "Ava": "female", "Bella": "female"}


class MossSynthesisLimitError(RuntimeError):
    """Exhausted bounded recovery; never publish a dub with this line missing."""

    fatal_tts = True
    user_message = (
        "MOSS не смогла завершить реплику после ограниченных повторов и разделения текста. "
        "Выберите встроенный голос MOSS вместо клонирования из оригинала "
        "или профиль Hy-MT / Быстро с Piper и повторите перевод."
    )

    def __init__(self):
        super().__init__(self.user_message)


class _FrameLimitReached(Exception):
    pass


def _split_failed_chunk(text: str) -> list[str]:
    """Split near the middle without dropping text or breaking words/numbers."""
    minimum, maximum = len(text) // 4, len(text) * 3 // 4
    boundaries = [m.end() for m in re.finditer(r"[.!?;:,。！？](?:\s+|$)", text)
                  if minimum <= m.end() <= maximum]
    if not boundaries:
        boundaries = [m.end() for m in re.finditer(r"\s+", text)
                      if minimum <= m.end() <= maximum]
    if not boundaries:
        # Unspaced CJK can split between characters; a single word cannot.
        boundaries = [i for i in range(max(1, minimum), min(len(text), maximum + 1))
                      if all(unicodedata.east_asian_width(c) in {"W", "F"}
                             for c in text[i - 1:i + 1])]
    if not boundaries:
        return [text]
    cut = min(boundaries, key=lambda i: abs(i - len(text) / 2))
    parts = [text[:cut].strip(), text[cut:].strip()]
    return parts if all(parts) else [text]


class _SynthesisCancelled(Exception):
    pass


def _local_path(value: str | None, default: str) -> Path:
    path = Path(value or default).expanduser()
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _build_runtime(model_dir: Path, codec_dir: Path, **options):
    import sentencepiece as spm
    from uvt.moss_runtime.ort_cpu_runtime import OrtCpuRuntime

    class LocalRuntime(OrtCpuRuntime):
        def resolve_manifest_relative_path(self, relative_path):
            # Official manifest expects a fixed sibling directory name. Keep
            # the downloaded manifest intact and map only that codec reference.
            relative = str(relative_path).replace("\\", "/")
            if relative in {
                "../MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json",
                "../MOSS-Audio-Tokenizer-Nano-ONNX-CPU/codec_browser_onnx_meta.json",
            }: return codec_dir / "codec_browser_onnx_meta.json"
            return super().resolve_manifest_relative_path(relative_path)

    runtime = LocalRuntime(model_dir, execution_provider="cpu", **options)
    tokenizer_path = runtime.resolve_manifest_relative_path(
        runtime.manifest["model_files"].get("tokenizer_model", "tokenizer.model")
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return runtime, tokenizer


def _text_chunks(text: str, tokenizer, budget: int) -> list[str]:
    """Preserve all text while keeping the autoregressive context bounded."""
    remaining = re.sub(r"\s+", " ", str(text)).strip()
    chunks = []
    while remaining:
        if len(tokenizer.encode(remaining, out_type=int)) <= budget:
            chunks.append(remaining)
            break
        low, high = 1, len(remaining)
        best = 1
        while low <= high:
            middle = (low + high) // 2
            if len(tokenizer.encode(remaining[:middle], out_type=int)) <= budget:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        # Prefer a sentence or word boundary near the end of the safe prefix.
        minimum = max(1, best // 2)
        punctuation = [m.end() for m in re.finditer(r"[.!?;:,。！？]", remaining[:best])]
        spaces = [m.end() for m in re.finditer(r"\s", remaining[:best])]
        boundaries = [p for p in punctuation if p >= minimum]
        if not boundaries:
            boundaries = [p for p in spaces if p >= minimum]
        cut = boundaries[-1] if boundaries else best
        piece, remaining = remaining[:cut].strip(), remaining[cut:].strip()
        if piece:
            chunks.append(piece)
    return chunks


@register("tts", "moss-onnx")
class MossOnnxTTS(TTSEngine):
    supports_reference = True

    async def warmup(self) -> None:
        model_dir = _local_path(getattr(self.cfg, "model_path", None), ".models/moss-tts")
        codec_dir = _local_path(getattr(self.cfg, "codec_path", None), ".models/moss-codec")
        for path in (model_dir / "browser_poc_manifest.json", model_dir / "tokenizer.model",
                     codec_dir / "codec_browser_onnx_meta.json"):
            if not path.is_file():
                raise RuntimeError("MOSS-TTS: модель не установлена — выполните "
                                   "uvt setup-mac-local --preset moss")
        sample_mode = str(getattr(self.cfg, "sample_mode", "fixed") or "fixed")
        if sample_mode not in {"fixed", "greedy", "full"}:
            raise RuntimeError("MOSS-TTS: sample_mode должен быть fixed, greedy или full")
        self._lock = asyncio.Lock()
        self._runtime = None
        self._reference_cache: dict[str, list[list[int]]] = {}
        self._budget = max(8, min(150, int(getattr(self.cfg, "max_text_tokens", 48))))
        self._max_frames = max(25, min(750, int(getattr(self.cfg, "max_new_frames", 375))))
        self._max_chunk_attempts = max(1, min(10, int(getattr(self.cfg, "max_chunk_attempts", 7))))
        self._reference_path = getattr(self.cfg, "reference_wav", None)
        if self._reference_path:
            self._reference_path = _local_path(self._reference_path, "")
            if not self._reference_path.is_file():
                raise RuntimeError(f"MOSS-TTS: образец голоса не найден: {self._reference_path}")
        options = dict(thread_count=max(1, min(8, int(getattr(self.cfg, "threads", 4)))),
                       max_new_frames=self._max_frames, sample_mode=sample_mode)
        try:
            self._runtime, self._tokenizer = await asyncio.to_thread(
                _build_runtime, model_dir, codec_dir, **options
            )
        except ImportError as exc:
            raise RuntimeError("MOSS-TTS требует onnxruntime и sentencepiece; "
                               "установите зависимости mac-local") from exc
        self._sample_rate = int(self._runtime.codec_meta["codec_config"]["sample_rate"])
        self._voices = {v["voice"]: v["prompt_audio_codes"]
                        for v in self._runtime.list_builtin_voices()}
        # Reject a stale/invalid voice before processing an entire video.
        self._resolve_voice()
        log.info("MOSS-TTS-Nano: CPU ONNX готов, %s Гц, %s потоков",
                 self._sample_rate, options["thread_count"])

    def _resolve_voice(self) -> str:
        requested = (str(getattr(self.cfg, "voice_id", "") or "").strip()
                     or configured_role_voice(self.cfg) or "")
        if not requested:
            gender = str(getattr(self.cfg, "voice_gender", "auto") or "auto").lower()
            requested = DEFAULT_VOICES["female" if gender.startswith("f") else "male"]
        if requested.startswith("ref_"):
            from uvt.voice_references import resolve_reference
            resolve_reference(requested)
            return requested
        if requested not in self._voices:
            raise RuntimeError(f"MOSS-TTS: неизвестный голос {requested}; выберите установленный голос из настроек")
        return requested

    @staticmethod
    def _validate_language(language: str) -> str:
        root = str(language or "").replace("_", "-").split("-", 1)[0].lower()
        if root not in SUPPORTED_LANGUAGES:
            raise RuntimeError(f"MOSS-TTS не поддерживает язык {language}; "
                               "для украинского выберите Piper")
        return root

    def _reference_codes(self, reference: VoiceReference | None) -> list[list[int]]:
        selected = str(getattr(self.cfg, "voice_id", "") or configured_role_voice(self.cfg) or "")
        if selected.startswith("ref_"):
            from uvt.voice_references import resolve_reference
            import soundfile as sf
            path, text = resolve_reference(selected)
            samples, rate = sf.read(path, dtype="float32", always_2d=True)
            reference = VoiceReference(samples.mean(axis=1), rate, text=text, label=selected)
        elif selected:
            return self._voices[self._resolve_voice()]
        if reference is None and self._reference_path:
            import soundfile as sf
            samples, rate = sf.read(self._reference_path, dtype="float32", always_2d=True)
            reference = VoiceReference(samples.mean(axis=1), rate, label="configured")
        if reference is None:
            return self._voices[self._resolve_voice()]
        rate = int(reference.sample_rate)
        if rate <= 0:
            raise ValueError("MOSS-TTS: неверная частота образца голоса")
        samples = np.asarray(reference.samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("MOSS-TTS: образец голоса должен быть моно")
        samples = samples[:rate * 10]
        if samples.size < rate // 2 or not np.isfinite(samples).all():
            raise ValueError("MOSS-TTS: нужен корректный образец голоса длиной от 0.5 секунды")
        digest = hashlib.blake2b(samples.tobytes(), digest_size=16)
        digest.update(str(rate).encode("ascii"))
        key = digest.hexdigest()
        if key in self._reference_cache:
            return self._reference_cache[key]
        if rate != self._sample_rate:
            from scipy.signal import resample_poly
            divisor = math.gcd(rate, self._sample_rate)
            samples = resample_poly(samples, self._sample_rate // divisor, rate // divisor)
        channels = int(self._runtime.codec_meta["codec_config"]["channels"])
        waveform = np.repeat(samples[None, None, :], channels, axis=1).astype(np.float32)
        session = self._runtime.sessions["codec_encode"]
        outputs = session.run(None, {"waveform": waveform,
            "input_lengths": np.asarray([waveform.shape[-1]], dtype=np.int32)})
        named = dict(zip((out.name for out in session.get_outputs()), outputs, strict=True))
        length = int(np.asarray(named["audio_code_lengths"]).reshape(-1)[0])
        if length <= 0:
            raise RuntimeError("MOSS-TTS: не удалось прочитать образец голоса")
        codes = np.asarray(named["audio_codes"], dtype=np.int32)[0, :length].tolist()
        if len(self._reference_cache) >= 8:
            self._reference_cache.pop(next(iter(self._reference_cache)))
        self._reference_cache[key] = codes
        return codes

    def _frame_limit(self, text: str) -> int:
        # A deliberately generous bound, including pauses. The extra probe frame
        # distinguishes EOS exactly at the limit from unfinished generation.
        weight = sum(3 if unicodedata.east_asian_width(c) in {"W", "F"} else 1
                     for c in text if not c.isspace())
        seconds = max(8.0, 3.0 + weight / 6.0)
        hop = int(self._runtime.codec_meta["codec_config"]["downsample_rate"])
        return min(self._max_frames - 1, math.ceil(seconds * self._sample_rate / hop))

    def _generate_chunk(self, chunk: str, prompt_codes, cancelled: threading.Event) -> np.ndarray:
        def check_cancel(*_):
            if cancelled.is_set():
                raise _SynthesisCancelled()

        check_cancel()
        request = self._runtime.build_voice_clone_request_rows(
            prompt_codes, self._tokenizer.encode(chunk, out_type=int))
        limit = self._frame_limit(chunk)
        defaults = self._runtime.manifest["generation_defaults"]
        previous_limit = defaults["max_new_frames"]
        # _speak serializes access and drains the worker before releasing its lock.
        try:
            defaults["max_new_frames"] = limit + 1
            frames = self._runtime.generate_audio_frames(request, on_frame=check_cancel)
        finally:
            defaults["max_new_frames"] = previous_limit
        check_cancel()
        if not frames:
            raise RuntimeError("MOSS-TTS вернула пустую озвучку; попробуйте другой голос")
        if len(frames) > limit:
            raise _FrameLimitReached()
        channels, length = self._runtime.decode_full_audio(frames)
        check_cancel()
        audio = np.asarray(channels, dtype=np.float32).mean(axis=0)[:length]
        if not audio.size or not np.isfinite(audio).all():
            raise RuntimeError("MOSS-TTS вернула некорректный звук")
        return audio

    def _synthesize(self, text: str, reference: VoiceReference | None, cancelled: threading.Event
                    ) -> tuple[np.ndarray, int]:
        def check_cancel():
            if cancelled.is_set():
                raise _SynthesisCancelled()

        check_cancel()
        prompt_codes = self._reference_codes(reference)
        chunks = _text_chunks(text, self._tokenizer, self._budget)
        audio_chunks = []
        for chunk in chunks:
            attempts = 0

            def recover(part: str, depth: int = 0) -> list[np.ndarray]:
                nonlocal attempts
                check_cancel()
                if attempts >= self._max_chunk_attempts:
                    raise MossSynthesisLimitError()
                attempts += 1
                try:
                    return [self._generate_chunk(part, prompt_codes, cancelled)]
                except _FrameLimitReached:
                    check_cancel()
                    parts = _split_failed_chunk(part) if depth < 2 else [part]
                    log.warning(
                        "MOSS: предел фрагмента (%d символов, попытка %d/%d); %s",
                        len(part), attempts, self._max_chunk_attempts,
                        "делю текст" if len(parts) > 1 else "повторяю с тем же голосом",
                    )
                    if len(parts) > 1:
                        return [audio for child in parts for audio in recover(child, depth + 1)]
                    # Fixed sampling parameters still draw fresh values. One retry
                    # may recover a stalled generation without changing the voice.
                    check_cancel()
                    if attempts >= self._max_chunk_attempts:
                        raise MossSynthesisLimitError() from None
                    attempts += 1
                    try:
                        return [self._generate_chunk(part, prompt_codes, cancelled)]
                    except _FrameLimitReached:
                        raise MossSynthesisLimitError() from None

            for audio in recover(chunk):
                if audio_chunks:
                    audio_chunks.append(np.zeros(int(self._sample_rate * 0.15), dtype=np.float32))
                audio_chunks.append(audio)
        return (np.concatenate(audio_chunks).astype(np.float32) if audio_chunks
                else np.empty(0, dtype=np.float32)), self._sample_rate

    async def _speak(self, text: str, language: str, reference=None):
        self._validate_language(language)
        if not str(text).strip():
            return np.empty(0, dtype=np.float32), self._sample_rate
        async with self._lock:
            if self._runtime is None:
                raise RuntimeError("MOSS-TTS уже остановлена")
            cancelled = threading.Event()
            worker = asyncio.create_task(asyncio.to_thread(self._synthesize, text, reference, cancelled))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled.set()
                # Do not release native sessions until the per-frame callback
                # has stopped the worker, including a second cancellation.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not worker.cancelled():
                    worker.exception()
                raise

    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        return await self._speak(text, language)

    async def synthesize_slot(self, text: str, language: str, *,
                              target_duration=None, reference=None, emotion=None):
        # This model clones timbre but does not implement duration conditioning.
        return await self._speak(text, language, reference)

    async def close(self) -> None:
        if not hasattr(self, "_lock"):
            return
        async with self._lock:
            self._reference_cache = {}
            self._runtime = None
            self._tokenizer = None
            gc.collect()
