"""Fast in-process local translation with NLLB and CTranslate2.

This engine is intended for memory-constrained Apple Silicon machines.  It
loads a quantized translation model only for the translation stage and does
not require an Ollama/LM Studio daemon or a general-purpose chat model.
"""
from __future__ import annotations

import asyncio
import gc
import logging
from pathlib import Path
from typing import Sequence

from uvt.interfaces import TranslationEngine
from uvt.registry import register

log = logging.getLogger("uvt.translate.nllb")

_DEFAULT_MODEL = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"
_DEFAULT_REVISION = "70f572adafa4794890ce7826156a4209717855af"

# ISO/Whisper language codes -> FLORES-200 codes expected by NLLB.
_LANG_CODES = {
    "ar": "arb_Arab",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "cs": "ces_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "kk": "kaz_Cyrl",
    "ko": "kor_Hang",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "nl": "nld_Latn",
    "no": "nob_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}


def _language_code(language: str | None) -> str:
    raw = str(language or "").strip()
    if raw in _LANG_CODES.values():
        return raw
    root = raw.replace("_", "-").split("-", 1)[0].lower()
    try:
        return _LANG_CODES[root]
    except KeyError:
        raise RuntimeError(
            f"NLLB: язык '{language or 'auto'}' не распознан; "
            "укажите source_lang/target_lang ISO-кодом, например ru, uk или en"
        ) from None


@register("translation", "nllb-ct2")
class NLLBCTranslate2Translator(TranslationEngine):
    """NLLB INT8 on CPU, batched in one UVT process."""

    async def warmup(self) -> None:
        self.batch_hint = int(getattr(self.cfg, "batch_size", None) or 32)
        # One CTranslate2 worker already uses several CPU threads.  Starting
        # several outer batches only duplicates queues and raises peak memory.
        self.concurrency_hint = 1
        self._beam_size = int(getattr(self.cfg, "beam_size", 1) or 1)
        self._max_decoding_length = int(
            getattr(self.cfg, "max_decoding_length", 256) or 256
        )
        await asyncio.to_thread(self._load)

    def _model_path(self) -> Path:
        model = str(getattr(self.cfg, "model", "") or _DEFAULT_MODEL)
        direct = Path(model).expanduser()
        if direct.is_dir():
            return direct

        revision = str(
            getattr(self.cfg, "revision", "")
            or (_DEFAULT_REVISION if model == _DEFAULT_MODEL else "")
        )
        try:
            from huggingface_hub import snapshot_download

            return Path(
                snapshot_download(
                    repo_id=model,
                    revision=revision or None,
                    local_files_only=not bool(
                        getattr(self.cfg, "allow_download", False)
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - actionable setup error
            raise RuntimeError(
                f"локальная модель перевода {model} не установлена — "
                "один раз выполните: uvt setup-mac-local"
            ) from exc

    def _load(self) -> None:
        import ctranslate2
        from tokenizers import Tokenizer

        model_path = self._model_path()
        tokenizer_path = model_path / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise RuntimeError(
                f"NLLB: в {model_path} нет tokenizer.json; повторите uvt setup-mac-local"
            )

        threads = max(1, int(getattr(self.cfg, "threads", 4) or 4))
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._translator = ctranslate2.Translator(
            str(model_path),
            device="cpu",
            compute_type=str(getattr(self.cfg, "compute_type", "int8") or "int8"),
            inter_threads=1,
            intra_threads=threads,
        )
        log.info(
            "NLLB загружен в процессе UVT: %s, INT8, %d CPU-потока",
            model_path.name,
            threads,
        )

    async def close(self) -> None:
        await asyncio.to_thread(self._release)

    def _release(self) -> None:
        self._translator = None
        self._tokenizer = None
        gc.collect()

    def _translate_many(
        self,
        texts: Sequence[str],
        source_lang: str | None,
        target_lang: str,
    ) -> list[str]:
        if not texts:
            return []
        source_code = _language_code(source_lang)
        target_code = _language_code(target_lang)
        if source_code == target_code:
            return list(texts)

        # NLLB language is selected by the forced first target token.  The raw
        # tokenizer is enough here and avoids importing the much larger
        # Transformers runtime into an 8 GB process.
        # NLLB's legacy source format is ``tokens </s> src_lang``.  The
        # standalone tokenizer.json cannot know src_lang and otherwise appends
        # ``<unk>`` as a placeholder, which noticeably damages translation.
        encoded = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
        source_tokens = [item.tokens + ["</s>", source_code] for item in encoded]
        results = self._translator.translate_batch(
            source_tokens,
            target_prefix=[[target_code] for _ in texts],
            beam_size=self._beam_size,
            max_decoding_length=self._max_decoding_length,
            max_batch_size=self.batch_hint,
            batch_type="examples",
        )

        translated: list[str] = []
        for result in results:
            tokens = list(result.hypotheses[0])
            if tokens and tokens[0] == target_code:
                tokens = tokens[1:]
            token_ids = [
                token_id
                for token in tokens
                if (token_id := self._tokenizer.token_to_id(token)) is not None
            ]
            text = " ".join(
                self._tokenizer.decode(token_ids, skip_special_tokens=True).split()
            )
            translated.append(text)
        return translated

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context,
    ) -> str:
        del context
        return (
            await asyncio.to_thread(
                self._translate_many, [text], source_lang, target_lang
            )
        )[0]

    async def translate_batch(
        self, texts: Sequence[str], source_lang: str | None, target_lang: str
    ) -> list[str]:
        return await asyncio.to_thread(
            self._translate_many, texts, source_lang, target_lang
        )

    async def translate_batch_tagged(
        self,
        texts: Sequence[str],
        source_lang: str | None,
        target_lang: str,
        genders,
    ) -> list[str]:
        # NLLB is a dedicated MT model, not a chat model.  Gender already
        # present in Slavic source text is preserved; synthetic prompt tags
        # would reduce translation quality and may leak into speech.
        del genders
        return await self.translate_batch(texts, source_lang, target_lang)
