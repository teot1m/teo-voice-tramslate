"""Local TranslateGemma translation on Apple Silicon through ``mlx-lm``.

TranslateGemma is not a generic chat model.  Its official template expects one
structured user item containing the source language, target language, and text.
Each subtitle is therefore rendered as an independent prompt and MLX batches
those prompts at the inference layer instead of asking the model to preserve
numbered lines in one free-form response.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import gc
import logging
from pathlib import Path
from typing import Any

from uvt.interfaces import TranslationEngine
from uvt.registry import register


log = logging.getLogger("uvt.translate.mlx_translategemma")

_DEFAULT_MODEL = "mlx-community/translategemma-4b-it-4bit"
_DEFAULT_REVISION = "5788ec08c047f3f2e17808101b8d9566ac930d58"

# TranslateGemma's chat template wants ISO 639-1 codes.  Keep the accepted set
# aligned with the 25 languages emitted by Parakeet v3, while also accepting the
# common three-letter values emitted by Whisper/FLORES for UVT's primary trio.
_PARAKEET_LANGUAGE_CODES = (
    "en",
    "es",
    "fr",
    "de",
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "et",
    "fi",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
    "ru",
    "uk",
)
_LANGUAGE_CODES = {code: code for code in _PARAKEET_LANGUAGE_CODES}
_LANGUAGE_CODES.update({"eng": "en", "rus": "ru", "ukr": "uk"})

ProgressCallback = Callable[[float], None]


def _language_code(language: str | None) -> str:
    raw = str(language or "").strip().lower().replace("_", "-")
    root = raw.split("-", 1)[0]
    code = _LANGUAGE_CODES.get(root)
    if code is None:
        shown = language if language else "auto/und"
        raise RuntimeError(
            "TranslateGemma MLX: поддерживаются только 25 языков Parakeet v3 "
            f"({', '.join(_PARAKEET_LANGUAGE_CODES)}); "
            f"получен '{shown}'"
        )
    return code


def _positive_int(value: object, *, name: str, default: int) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"TranslateGemma MLX: {name} должен быть положительным целым числом"
        ) from exc
    if parsed < 1:
        raise RuntimeError(
            f"TranslateGemma MLX: {name} должен быть положительным целым числом"
        )
    return parsed


async def _wait_for_worker(worker: asyncio.Task[Any]) -> Any:
    """Await a thread worker without abandoning it when its caller is cancelled.

    Cancelling ``asyncio.to_thread`` only cancels the asyncio wrapper; MLX keeps
    using the model in its native worker.  Waiting here prevents ``close()`` or
    another request from clearing/reusing that model while inference is active.
    """

    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # A second cancellation must not make us abandon the native worker.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        try:
            worker.result()
        except Exception:  # noqa: BLE001 - cancellation remains authoritative
            log.exception("TranslateGemma MLX worker failed while task was cancelled")
        raise


@register("translation", "translategemma-mlx")
class MlxTranslateGemmaTranslator(TranslationEngine):
    """Dedicated, lazily loaded TranslateGemma MLX translator."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.batch_hint = _positive_int(
            getattr(cfg, "batch_size", None), name="batch_size", default=8
        )
        # A single MLX model and unified-memory GPU worker must not be entered by
        # concurrent dub batches.  ``dub._translate_all`` respects this hint.
        self.concurrency_hint = 1
        self._max_tokens = _positive_int(
            getattr(cfg, "max_tokens", None), name="max_tokens", default=256
        )
        self._lock = asyncio.Lock()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._mlx_lm: Any | None = None
        self._resolved_model_path: Path | None = None

    def _model_path(self) -> Path:
        configured = str(getattr(self.cfg, "model", "") or _DEFAULT_MODEL)
        direct = Path(configured).expanduser()
        if direct.is_dir():
            model_path = direct.resolve()
        else:
            revision = str(getattr(self.cfg, "revision", "") or "")
            if not revision and configured == _DEFAULT_MODEL:
                revision = _DEFAULT_REVISION
            allow_download = bool(getattr(self.cfg, "allow_download", False))
            try:
                from huggingface_hub import snapshot_download

                model_path = Path(
                    snapshot_download(
                        repo_id=configured,
                        revision=revision or None,
                        local_files_only=not allow_download,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - turn cache errors actionable
                action = (
                    "проверьте доступ к Hugging Face и повторите подготовку"
                    if allow_download
                    else "один раз выполните uvt setup-mac-local"
                )
                raise RuntimeError(
                    f"TranslateGemma MLX: модель {configured} не установлена; {action}"
                ) from exc

        self._validate_model_path(model_path)
        return model_path

    @staticmethod
    def _validate_model_path(model_path: Path) -> None:
        has_weights = any(model_path.glob("*.safetensors"))
        has_tokenizer = (model_path / "tokenizer.json").is_file() or (
            model_path / "tokenizer.model"
        ).is_file()
        required = (
            (model_path / "config.json").is_file()
            and (model_path / "chat_template.jinja").is_file()
            and has_weights
            and has_tokenizer
        )
        if not required:
            raise RuntimeError(
                "TranslateGemma MLX: кэш модели неполный в "
                f"{model_path}; повторите uvt setup-mac-local"
            )

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import mlx_lm
        except ImportError as exc:
            raise RuntimeError(
                'TranslateGemma MLX требует mlx-lm; установите зависимости '
                'командой pip install -e ".[mac-local]"'
            ) from exc

        model_path = self._model_path()
        try:
            model, tokenizer = mlx_lm.load(
                path_or_hf_repo=str(model_path),
                lazy=False,
            )
        except Exception as exc:  # noqa: BLE001 - library errors need local context
            raise RuntimeError(
                f"TranslateGemma MLX: не удалось загрузить модель из {model_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # Gemma's chat turn terminator is distinct from tokenizer.eos_token_id.
        # mlx-lm otherwise keeps emitting ``<end_of_turn>`` until max_tokens,
        # which looks like a hang and can turn one subtitle into kilobytes.
        add_eos_token = getattr(tokenizer, "add_eos_token", None)
        if callable(add_eos_token):
            try:
                add_eos_token("<end_of_turn>")
            except Exception as exc:  # noqa: BLE001 - incompatible tokenizer
                raise RuntimeError(
                    "TranslateGemma MLX: tokenizer не распознал <end_of_turn>"
                ) from exc

        self._mlx_lm = mlx_lm
        self._model = model
        self._tokenizer = tokenizer
        self._resolved_model_path = model_path
        log.info("TranslateGemma MLX загружена: %s", model_path.name)

    def _translate_chunk(
        self, texts: Sequence[str], source_code: str, target_code: str
    ) -> list[str]:
        self._load()
        assert self._mlx_lm is not None
        assert self._model is not None
        assert self._tokenizer is not None

        outputs = [""] * len(texts)
        nonempty: list[tuple[int, str]] = [
            (index, text) for index, text in enumerate(texts) if text.strip()
        ]
        if not nonempty:
            return outputs

        prompts: list[list[int]] = []
        for _index, text in nonempty:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_code,
                            "target_lang_code": target_code,
                            "text": text,
                        }
                    ],
                }
            ]
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            except Exception as exc:  # noqa: BLE001 - explain template mismatch
                raise RuntimeError(
                    "TranslateGemma MLX: официальный structured chat template "
                    f"не применился: {type(exc).__name__}: {exc}"
                ) from exc
            prompts.append(prompt)

        try:
            response = self._mlx_lm.batch_generate(
                self._model,
                self._tokenizer,
                prompts,
                max_tokens=self._max_tokens,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - actionable batch failure
            raise RuntimeError(
                "TranslateGemma MLX: пакетный перевод не выполнен: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        generated = getattr(response, "texts", None)
        if (
            not isinstance(generated, Sequence)
            or isinstance(generated, (str, bytes))
            or len(generated) != len(nonempty)
        ):
            actual = len(generated) if isinstance(generated, Sequence) else "?"
            raise RuntimeError(
                "TranslateGemma MLX: модель нарушила размер пакета: "
                f"ожидалось {len(nonempty)}, получено {actual}"
            )

        for (index, source), translated in zip(nonempty, generated, strict=True):
            if not isinstance(translated, str) or not translated.strip():
                raise RuntimeError(
                    f"TranslateGemma MLX: пустой перевод для реплики {index + 1}"
                )
            cleaned = translated.split("<end_of_turn>", 1)[0].strip()
            if not cleaned:
                raise RuntimeError(
                    f"TranslateGemma MLX: пустой перевод для реплики {index + 1}"
                )
            max_reasonable_chars = max(512, len(source) * 8 + 256)
            if len(cleaned) > max_reasonable_chars:
                raise RuntimeError(
                    "TranslateGemma MLX: генерация не остановилась после перевода "
                    f"реплики {index + 1} ({len(cleaned)} символов)"
                )
            outputs[index] = cleaned
        return outputs

    async def _run_thread(self, func: Callable[..., Any], *args: Any) -> Any:
        worker = asyncio.create_task(asyncio.to_thread(func, *args))
        return await _wait_for_worker(worker)

    async def warmup(self) -> None:
        async with self._lock:
            await self._run_thread(self._load)

    async def close(self) -> None:
        # The same lock covers loading and every inference call.  If close races
        # a translation, it cannot release unified-memory buffers prematurely.
        async with self._lock:
            await self._run_thread(self._release)

    def _release(self) -> None:
        self._model = None
        self._tokenizer = None
        self._mlx_lm = None
        self._resolved_model_path = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass
        gc.collect()

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Sequence[tuple[str, str]],
    ) -> str:
        # TranslateGemma's supported template has no history/system-message slot.
        del context
        return (await self.translate_batch([text], source_lang, target_lang))[0]

    async def translate_batch(
        self,
        texts: Sequence[str],
        source_lang: str | None,
        target_lang: str,
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        source_code = _language_code(source_lang)
        target_code = _language_code(target_lang)
        materialized = list(texts)
        for index, text in enumerate(materialized):
            if not isinstance(text, str):
                raise TypeError(
                    "TranslateGemma MLX: текст реплики "
                    f"{index + 1} должен быть строкой"
                )

        total = len(materialized)
        if not total:
            if progress is not None:
                progress(1.0)
            return []
        if source_code == target_code:
            if progress is not None:
                progress(1.0)
            return materialized

        translated: list[str] = []
        async with self._lock:
            for start in range(0, total, self.batch_hint):
                chunk = materialized[start : start + self.batch_hint]
                result = await self._run_thread(
                    self._translate_chunk, chunk, source_code, target_code
                )
                translated.extend(result)
                if progress is not None:
                    progress(min(len(translated) / total, 1.0))

        if len(translated) != total:
            raise RuntimeError(
                "TranslateGemma MLX: внутренний размер результата не совпал "
                f"с запросом ({len(translated)} != {total})"
            )
        return translated

    async def translate_batch_tagged(
        self,
        texts: Sequence[str],
        source_lang: str | None,
        target_lang: str,
        genders: Sequence[str] | None,
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        # Adding [M]/[F] to source text is outside TranslateGemma's official
        # template and may leak those tags into dubbed speech.
        del genders
        return await self.translate_batch(texts, source_lang, target_lang, progress)
