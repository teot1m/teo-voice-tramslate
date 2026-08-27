"""Ленивый локальный резерв для облачных STT и переводчиков.

Cloud-профиль не обязан падать целиком из-за временной ошибки API, квоты или
отказа модели. Этот модуль переключает только remote openai-compatible
движки на явно настроенный localhost-резерв, не отправляя OpenAI-ключ в
локальный Ollama/LM Studio и не перехватывая отмену задачи.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from urllib.parse import urlparse

from uvt import registry
from uvt.config import AppConfig
from uvt.interfaces import STTEngine, STTResult, STTSpan, TranslationEngine

log = logging.getLogger("uvt.fallback")


class ApprovalGate:
    """Ждёт явного решения пользователя перед переходом на локальный резерв.

    ``on_request`` вызывается синхронно в момент отказа облака — например,
    чтобы перевести job в состояние "awaiting_approval" на HTTP-сервере.
    Само ожидание — без таймаута, но остаётся отменяемым: CancelledError из
    ``request`` пробрасывается как обычно и корректно останавливает задачу.
    """

    def __init__(self, on_request: Callable[[str, str], None] | None = None) -> None:
        self._on_request = on_request
        self._pending: asyncio.Event | None = None
        self._approved = False

    async def request(self, kind: str, cause: str) -> bool:
        self._pending = asyncio.Event()
        self._approved = False
        if self._on_request is not None:
            self._on_request(kind, cause)
        await self._pending.wait()
        return self._approved

    def resolve(self, approved: bool) -> None:
        self._approved = approved
        if self._pending is not None:
            self._pending.set()


def _is_loopback_endpoint(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        host = (urlparse(value).hostname or "").lower()
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "::1"}


def _is_remote_openai_compatible(section: object) -> bool:
    return (
        str(getattr(section, "engine", "") or "") == "openai-compatible"
        and not _is_loopback_endpoint(getattr(section, "base_url", None))
    )


def _brief(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return text[:360] + ("…" if len(text) > 360 else "")


def _engine_label(section: object) -> str:
    engine = str(getattr(section, "engine", "unknown") or "unknown")
    model = str(getattr(section, "model", "") or "")
    return f"{engine}/{model}" if model else engine


class _FallbackActivated(Exception):
    """Локальный переводчик включён; batch нужно повторить короткими пачками.

    Это внутренний сигнал между ``_EngineFailover`` и ``FailoverTranslator``.
    Его нельзя показывать пользователю и нельзя применять к одиночным вызовам:
    им безопасно повторить запрос сразу на резервном движке.
    """


class _EngineFailover:
    """Переключает один Engine на localhost-резерв не более одного раза."""

    def __init__(
        self,
        kind: str,
        primary_cfg: object,
        fallback_cfg: object,
        approval: ApprovalGate | None = None,
    ) -> None:
        self.kind = kind
        self.primary_cfg = primary_cfg
        self.fallback_cfg = fallback_cfg
        self.approval = approval
        self.primary = None
        self.fallback = None
        self.active = None
        self.cause: BaseException | None = None
        self._switch_lock = asyncio.Lock()
        # В момент отказа cloud несколько уже запущенных batch-задач могут
        # одновременно перейти на один локальный Ollama. Не раздуваем его KV
        # cache тремя запросами на малом Mac: локальный резерв обслуживаем
        # последовательно.
        self._fallback_call_lock = asyncio.Lock()

    @property
    def using_fallback(self) -> bool:
        return self.fallback is not None and self.active is self.fallback

    async def warmup(self) -> None:
        if self.active is not None:
            return
        self.primary = registry.create(self.kind, self.primary_cfg.engine, self.primary_cfg)
        self.active = self.primary
        try:
            await self.primary.warmup()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — резерв нужен именно при ошибке провайдера
            await self._activate_fallback(exc)

    def _validate_local_fallback(self) -> None:
        # Название fallback не гарантирует локальность: не маскируем ошибку
        # одного облака вызовом другого и не передаём туда секреты.
        if (
            str(getattr(self.fallback_cfg, "engine", "") or "") == "openai-compatible"
            and not _is_loopback_endpoint(getattr(self.fallback_cfg, "base_url", None))
        ):
            raise RuntimeError(
                "локальный резерв должен указывать на localhost (Ollama, LM Studio или llama.cpp), "
                "а не на внешний API"
            )

    async def _activate_fallback(self, cause: BaseException) -> None:
        async with self._switch_lock:
            if self.fallback is not None:
                return
            self.cause = cause
            label = "распознавание" if self.kind == "stt" else "перевод"
            if self.approval is not None:
                approved = await self.approval.request(self.kind, _brief(cause))
                if not approved:
                    raise RuntimeError(
                        f"облачное {label} не удалось ({_brief(cause)}); переход на "
                        f"локальный резерв {_engine_label(self.fallback_cfg)} отклонён"
                    )
            try:
                self._validate_local_fallback()
                candidate = registry.create(
                    self.kind, self.fallback_cfg.engine, self.fallback_cfg
                )
                await candidate.warmup()
            except asyncio.CancelledError:
                raise
            except Exception as fallback_error:  # noqa: BLE001 — добавляем исходную причину
                raise RuntimeError(
                    f"облачное {label} недоступно ({_brief(cause)}); "
                    f"локальный резерв {_engine_label(self.fallback_cfg)} не запустился "
                    f"({_brief(fallback_error)})"
                ) from fallback_error

            self.fallback = candidate
            self.active = candidate
            log.warning(
                "облачное %s не удалось (%s) — переключаю задачу на локальный резерв %s",
                label,
                _brief(cause),
                _engine_label(self.fallback_cfg),
            )

    async def call(
        self,
        method: str,
        *args,
        validate: Callable[[object], None] | None = None,
        defer_fallback_retry: bool = False,
        **kwargs,
    ):
        await self.warmup()
        active = self.active
        assert active is not None
        # ``_translate_all`` мог разрезать реплики по cloud batch_hint ещё до
        # отказа. Не отдаём такую большую пачку маленькому локальному LLM:
        # FailoverTranslator поймает сигнал и разделит её по batch_hint резерва.
        if defer_fallback_retry and active is self.fallback:
            raise _FallbackActivated()

        async def invoke(engine):
            if engine is self.fallback:
                async with self._fallback_call_lock:
                    return await getattr(engine, method)(*args, **kwargs)
            return await getattr(engine, method)(*args, **kwargs)

        try:
            result = await invoke(active)
            if validate is not None:
                validate(result)
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — remote HTTP/timeout/refusal и malformed response
            if active is self.fallback:
                label = "распознавание" if self.kind == "stt" else "перевод"
                raise RuntimeError(
                    f"локальный резерв {label} {_engine_label(self.fallback_cfg)} не справился "
                    f"после ошибки облака ({_brief(self.cause or exc)}): {_brief(exc)}"
                ) from exc
            await self._activate_fallback(exc)
            if defer_fallback_retry:
                raise _FallbackActivated() from exc
            fallback = self.active
            assert fallback is not None
            try:
                result = await invoke(fallback)
                if validate is not None:
                    validate(result)
                return result
            except asyncio.CancelledError:
                raise
            except Exception as fallback_error:  # noqa: BLE001
                label = "распознавание" if self.kind == "stt" else "перевод"
                raise RuntimeError(
                    f"локальный резерв {label} {_engine_label(self.fallback_cfg)} не справился "
                    f"после ошибки облака ({_brief(self.cause or exc)}): {_brief(fallback_error)}"
                ) from fallback_error

    async def close(self) -> None:
        closed: set[int] = set()
        for engine in (self.primary, self.fallback):
            if engine is None or id(engine) in closed:
                continue
            closed.add(id(engine))
            await engine.close()


class FailoverSTT(STTEngine):
    """STT, который при ошибке remote OpenAI повторяет запрос локальному Whisper."""

    def __init__(self, cfg: AppConfig, approval: ApprovalGate | None = None) -> None:
        super().__init__(cfg.stt)
        assert cfg.stt.fallback is not None
        self._delegate = _EngineFailover("stt", cfg.stt, cfg.stt.fallback, approval=approval)

    @property
    def using_fallback(self) -> bool:
        return self._delegate.using_fallback

    @property
    def concurrency_hint(self) -> int:
        active = self._delegate.active or self._delegate.primary
        return int(getattr(active, "concurrency_hint", self.cfg.concurrency or 1))

    async def warmup(self) -> None:
        await self._delegate.warmup()

    async def close(self) -> None:
        await self._delegate.close()

    async def transcribe(
        self, samples, sample_rate: int, language: str | None
    ) -> STTResult | None:
        return await self._delegate.call("transcribe", samples, sample_rate, language)

    async def transcribe_long(
        self,
        samples,
        sample_rate: int,
        language: str | None,
        progress=None,
    ) -> list[STTSpan] | None:
        return await self._delegate.call(
            "transcribe_long", samples, sample_rate, language, progress=progress
        )


def _require_text(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("модель не вернула текст перевода")


def _require_batch(value: object, expected_count: int) -> None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != expected_count
        or not value
        or any(
            not isinstance(item, str) or not item.strip() for item in value
        )
    ):
        raise RuntimeError("модель не вернула полный пакет переводов")


class FailoverTranslator(TranslationEngine):
    """Переводчик, липко переключающийся с облака на локальный LLM."""

    def __init__(self, cfg: AppConfig, approval: ApprovalGate | None = None) -> None:
        super().__init__(cfg.translation)
        assert cfg.translation.fallback is not None
        self._delegate = _EngineFailover(
            "translation", cfg.translation, cfg.translation.fallback, approval=approval
        )

    @property
    def batch_hint(self) -> int:
        active = self._delegate.active or self._delegate.primary
        return int(getattr(active, "batch_hint", 20))

    @property
    def concurrency_hint(self) -> int:
        active = self._delegate.active or self._delegate.primary
        return int(getattr(active, "concurrency_hint", 3))

    @property
    def using_fallback(self) -> bool:
        return self._delegate.using_fallback

    async def warmup(self) -> None:
        await self._delegate.warmup()

    async def close(self) -> None:
        await self._delegate.close()

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Sequence[tuple[str, str]],
    ) -> str:
        return await self._delegate.call(
            "translate", text, source_lang, target_lang, context, validate=_require_text
        )

    async def _translate_active_fallback_in_batches(
        self,
        texts: Sequence[str],
        source_lang: str | None,
        target_lang: str,
        genders: Sequence[str] | None,
        *,
        tagged: bool,
    ) -> list[str]:
        """Повторяет cloud-пачку небольшими запросами к уже включённой LLM.

        В начале batch-дубляжа размер и параллелизм выбираются для облака. Если
        тот отказался уже посреди работы, нельзя повторять 20 реплик одним
        запросом в Ollama: на малом Mac он дольше отвечает и чаще сбивает
        нумерацию. Используем настроенный размер резервной модели и сохраняем
        соответствие тегов пола строкам.
        """
        items = list(texts)
        if not items:
            return []
        active = self._delegate.active
        try:
            chunk_size = max(1, int(getattr(active, "batch_hint", 4)))
        except (TypeError, ValueError):
            chunk_size = 4
        gender_items = list(genders) if genders is not None else None
        translated: list[str] = []

        for start in range(0, len(items), chunk_size):
            end = start + chunk_size
            chunk = items[start:end]
            if tagged:
                chunk_genders = gender_items[start:end] if gender_items is not None else None
                result = await self._delegate.call(
                    "translate_batch_tagged",
                    chunk,
                    source_lang,
                    target_lang,
                    chunk_genders,
                    validate=lambda value, count=len(chunk): _require_batch(value, count),
                )
            else:
                result = await self._delegate.call(
                    "translate_batch",
                    chunk,
                    source_lang,
                    target_lang,
                    validate=lambda value, count=len(chunk): _require_batch(value, count),
                )
            translated.extend(result)
        return list(translated)

    async def translate_batch(
        self, texts: Sequence[str], source_lang: str | None, target_lang: str
    ) -> list[str]:
        try:
            result = await self._delegate.call(
                "translate_batch",
                texts,
                source_lang,
                target_lang,
                validate=lambda value: _require_batch(value, len(texts)),
                defer_fallback_retry=True,
            )
        except _FallbackActivated:
            return await self._translate_active_fallback_in_batches(
                texts, source_lang, target_lang, None, tagged=False
            )
        return list(result)

    async def translate_batch_tagged(
        self,
        texts: Sequence[str],
        source_lang: str | None,
        target_lang: str,
        genders: Sequence[str] | None,
    ) -> list[str]:
        try:
            result = await self._delegate.call(
                "translate_batch_tagged",
                texts,
                source_lang,
                target_lang,
                genders,
                validate=lambda value: _require_batch(value, len(texts)),
                defer_fallback_retry=True,
            )
        except _FallbackActivated:
            return await self._translate_active_fallback_in_batches(
                texts, source_lang, target_lang, genders, tagged=True
            )
        return list(result)


def create_stt_engine(cfg: AppConfig, approval: ApprovalGate | None = None):
    """Создаёт primary STT или failover для явно заданного cloud-профиля."""
    if cfg.stt.fallback is not None and _is_remote_openai_compatible(cfg.stt):
        return FailoverSTT(cfg, approval=approval)
    return registry.create("stt", cfg.stt.engine, cfg.stt)


def create_translation_engine(cfg: AppConfig, approval: ApprovalGate | None = None):
    """Создаёт primary translator или failover для явно заданного cloud-профиля."""
    if cfg.translation.fallback is not None and _is_remote_openai_compatible(cfg.translation):
        return FailoverTranslator(cfg, approval=approval)
    return registry.create("translation", cfg.translation.engine, cfg.translation)
