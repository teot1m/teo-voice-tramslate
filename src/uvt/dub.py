"""Офлайн-дубляж: перевод видео/аудиофайла с заменой голоса (ТЗ §13).

Схема «звук → текст с таймкодами → перевод → озвучка по таймкодам», каждая
стадия пакетная:

1. STT: faster-whisper распознаёт весь файл одним пакетным проходом
   (BatchedInferencePipeline) и сразу отдаёт реплики с таймкодами. Движок без
   пакетного режима → резервный путь: наш VAD + пореплечное распознавание.
2. Перевод: реплики уходят в LLM пачками (по объёму текста); упавшая пачка
   переводится по одной реплике; совсем не переведённые реплики не озвучиваются.
3. TTS: озвучка параллельно, клипы кладутся точно на свои таймкоды.

Сборка — два варианта:
- полный микс (uvt dub → файл): оригинал приглушается под репликами (ducking),
  поверх кладётся переведённая речь;
- только голос перевода (mix_original=False, сервер браузерной кнопки):
  тишина между репликами, оригинал играет сам плеер на странице — так нет
  дублирования звука.
"""
from __future__ import annotations

import asyncio
import threading
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Callable

import numpy as np

from uvt import registry
from uvt.audio import resample, to_mono
from uvt.config import PRESETS, AppConfig
from uvt.fallback import ApprovalGate, create_stt_engine, create_translation_engine
from uvt.diarize import assign_speakers
from uvt.gender import estimate_gender_f0
from uvt.history import EXPORTERS, HistoryEntry
from uvt.interfaces import STTEngine, STTSpan, VoiceReference
from uvt.segmenter import Segmenter, SegmenterParams, create_vad_engine
from uvt.separate import separate_speech
from uvt.services.stt import _is_junk
from uvt.services.translate import _same_lang
from uvt.text_quality import collapse_repeats, is_vocalization, sanitize_translation
from uvt.voices import pick_references

log = logging.getLogger("uvt.dub")

PIPE_RATE = 16000     # частота конвейера распознавания
MIX_RATE = 48000      # частота итоговой дорожки
SEPARATION_RATE = 44100  # нативная частота Demucs
RAMP_S = 0.05         # плавность приглушения, 50 мс
BATCH_MAX_ITEMS = 20  # реплик на один запрос к LLM…
BATCH_MAX_CHARS = 2500  # …но не больше этого объёма текста
TTS_CONCURRENCY = 4
_DUB_SENTENCE_END = re.compile(r"[.!?…]+[\"»')\]]*$")
_MAX_COMPACT_SPAN_S = 12.0
_MAX_COMPACT_GAP_S = 0.9

ProgressFn = Callable[[int, int], None]
StageFn = Callable[[str, float, str], None]
_StageCb = Callable[[float], None]


@dataclass(slots=True)
class ClipReady:
    """Готовая озвученная реплика — публикуется сразу после укладки в слот.

    Нужна для прогрессивного дубляжа: подписчик (сервер браузерной кнопки)
    получает реплику, как только она готова, и может проиграть её в видео, не
    дожидаясь окончания обработки всего файла.
    """

    index: int
    source_start: float
    source_end: float
    original: str
    translated: str
    voice_style: str
    speaker: str
    samples: np.ndarray
    sample_rate: int


ClipCallback = Callable[[ClipReady], None]


@dataclass(slots=True)
class _SynthClip:
    """Озвученная реплика до размещения на итоговой временной шкале."""

    index: int
    source_start: float
    samples: np.ndarray
    sample_rate: int


def _compact_stt_spans(spans: list[STTSpan]) -> list[STTSpan]:
    """Join Whisper fragments that are really parts of one sentence.

    MLX Whisper can split a long narration into several timestamp segments
    even when there is no sentence boundary.  Synthesizing every fragment
    separately makes Piper pay ONNX overhead hundreds of times and produces
    choppy speech.  Keep real punctuation, long pauses, language changes, and
    the 12-second sync guard as hard boundaries.
    """
    if not spans:
        return []

    compacted: list[STTSpan] = []
    current = STTSpan(
        spans[0].start,
        spans[0].end,
        " ".join(spans[0].text.split()),
        spans[0].language,
    )
    for item in spans[1:]:
        text = " ".join(item.text.split())
        gap = max(0.0, item.start - current.end)
        language_changed = bool(
            current.language and item.language and current.language != item.language
        )
        can_join = (
            not _DUB_SENTENCE_END.search(current.text)
            and gap <= _MAX_COMPACT_GAP_S
            and (item.end - current.start) <= _MAX_COMPACT_SPAN_S
            and not language_changed
        )
        if can_join:
            current = STTSpan(
                current.start,
                item.end,
                f"{current.text} {text}".strip(),
                current.language or item.language,
            )
            continue
        compacted.append(current)
        current = STTSpan(item.start, item.end, text, item.language)
    compacted.append(current)
    return compacted


async def _run_blocking(func: Callable, /, *args, _on_cancel: Callable[[], None] | None = None, **kwargs):
    """Keep the event loop responsive and drain I/O before temporary files close."""
    worker = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        if _on_cancel is not None:
            _on_cancel()
        # Cancelling to_thread does not stop its thread or subprocess. Wait for
        # completion before a caller removes its temporary input/output folder.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            worker.result()
        except Exception:
            log.debug("background I/O failed while cancellation was draining", exc_info=True)
        raise


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def _decode_file(path: Path, rate: int, channels: int) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "f32le", "-ac", str(channels), "-ar", str(rate), "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    data = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return data.reshape(-1, channels) if channels > 1 else data


def _has_video(path: Path) -> bool:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() != ""


def is_url(source: str) -> bool:
    return bool(re.match(r"^https?://", source, re.IGNORECASE))


def _find_ytdlp() -> str | None:
    """yt-dlp из текущего venv (рядом с интерпретатором) или из PATH."""
    local = Path(sys.executable).with_name("yt-dlp")
    if local.exists():
        return str(local)
    return shutil.which("yt-dlp")


def _ytdlp_js_args() -> list[str]:
    """Явно включает доступный JS runtime для современных YouTube challenge.

    yt-dlp включает Deno автоматически, но Node/Bun требует флага. UVT не
    скачивает runtime сам: используем только уже установленный executable.
    """
    for runtime in ("deno", "node", "bun"):
        executable = shutil.which(runtime)
        if executable:
            return ["--js-runtimes", f"{runtime}:{executable}"]
    return []


def download_url(url: str, dest_dir: Path) -> Path:
    """Скачивает ролик через yt-dlp (YouTube, Twitch VOD, Vimeo и т.д.)."""
    ytdlp = _find_ytdlp()
    if ytdlp is None:
        raise RuntimeError("для ссылок нужен yt-dlp: pip install yt-dlp")
    log.info("скачиваю ролик через yt-dlp…")
    try:
        subprocess.run(
            [
                ytdlp,
                "--ignore-config",
                *_ytdlp_js_args(),
                # у YouTube бывают дорожки авто-дубляжа — берём оригинальную
                "-f", "bv*+ba[format_note*=original]/bv*+ba/b",
                "--merge-output-format", "mkv",
                "-o", str(dest_dir / "%(title).80s.%(ext)s"), url,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"yt-dlp не смог скачать {url} — проверьте, что это настоящая ссылка "
            "на видео (скопируйте её из адресной строки браузера)"
        ) from exc
    files = sorted(dest_dir.iterdir(), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("yt-dlp ничего не скачал")
    return files[0]


async def dub(
    cfg: AppConfig,
    source: str,
    output: str | Path | None = None,
    duck_db: float = -12.0,
    keep_original: bool = True,
) -> Path:
    """Дублирует локальный файл или ссылку. Возвращает путь к результату."""
    if is_url(source):
        with tempfile.TemporaryDirectory(prefix="uvt-dub-") as td:
            downloaded = await _run_blocking(download_url, source, Path(td))
            if output is None:
                output = Path.cwd() / f"{downloaded.stem}.dub.mkv"
            return await dub_file(cfg, downloaded, output, duck_db, keep_original)
    return await dub_file(cfg, Path(source), output, duck_db, keep_original)


# --- стадия 1: звук → реплики с таймкодами ---

async def _transcribe_all(
    cfg: AppConfig,
    mono16: np.ndarray,
    source_lang: str | None,
    progress: _StageCb | None,
    approval: ApprovalGate | None = None,
    stt_engine: STTEngine | None = None,
) -> list[STTSpan]:
    stt = stt_engine or create_stt_engine(cfg, approval=approval)
    try:
        # Own the engine before loading native weights. If the job is cancelled
        # during warm-up, ``close`` must still run before another Metal task.
        await stt.warmup()

        def stt_progress(done_s: float, total_s: float) -> None:
            if progress is not None:
                progress(done_s / max(total_s, 1e-6))

        spans = await stt.transcribe_long(mono16, PIPE_RATE, source_lang, progress=stt_progress)
        if spans is not None:
            raw_count = len(spans)
            spans = _compact_stt_spans(spans)
            log.info(
                "пакетное распознавание: %d фрагментов → %d реплик",
                raw_count,
                len(spans),
            )
            return spans

        # Резервный путь: движок без пакетного режима → наш VAD + по одной
        vad_engine = await create_vad_engine(cfg.vad, log)
        try:
            segmenter = Segmenter(
                vad_engine, SegmenterParams.from_config(cfg.vad, PRESETS[cfg.latency.preset])
            )
            segments = []
            for i in range(0, len(mono16), PIPE_RATE):
                segments.extend(segmenter.feed(mono16[i : i + PIPE_RATE]))
            tail = segmenter.flush()
            if tail is not None:
                segments.append(tail)
            log.info("VAD-путь: %d реплик", len(segments))

            parallelism = max(1, int(getattr(stt, "concurrency_hint", 1)))
            log.info(
                "распознавание VAD-фрагментов: параллельность %d", parallelism
            )
            limit = asyncio.Semaphore(parallelism)
            completed = 0
            results: list[STTSpan | None] = [None] * len(segments)

            async def transcribe_one(index: int, seg) -> None:
                nonlocal completed
                async with limit:
                    result = await stt.transcribe(
                        seg.samples, seg.sample_rate, source_lang
                    )
                if result is not None and result.text.strip():
                    results[index] = STTSpan(
                        seg.start_ts, seg.end_ts, result.text, result.language
                    )
                completed += 1
                if progress is not None:
                    progress(completed / max(len(segments), 1))

            tasks = [
                asyncio.create_task(transcribe_one(index, seg))
                for index, seg in enumerate(segments)
            ]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            return [span for span in results if span is not None]
        finally:
            await vad_engine.close()
    finally:
        await stt.close()


# --- стадия 2: перевод пачками с деградацией до пореплечного ---

def _make_batches(
    spans: list[STTSpan],
    max_items: int = BATCH_MAX_ITEMS,
    langs: Sequence[str | None] | None = None,
) -> list[list[int]]:
    """Группирует индексы реплик: не больше max_items и BATCH_MAX_CHARS.

    ``langs`` — исходный язык каждой реплики. В одном ролике встречаются
    несколько языков (английский диалог со вставками на чешском); пачка с
    единым языком заставляла переводчик читать чешскую строку как английскую.
    Поэтому смена языка — жёсткая граница пачки.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    chars = 0
    current_lang: str | None = None
    for i, span in enumerate(spans):
        lang = langs[i] if langs is not None else None
        lang_changed = bool(current) and langs is not None and lang != current_lang
        if current and (
            lang_changed
            or len(current) >= max_items
            or chars + len(span.text) > BATCH_MAX_CHARS
        ):
            batches.append(current)
            current, chars = [], 0
        if not current:
            current_lang = lang
        current.append(i)
        chars += len(span.text)
    if current:
        batches.append(current)
    return batches


def _file_translation_context(
    spans: list[STTSpan], batch: list[int], genders: list[str] | None,
    lines: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Nearest source neighbours, bounded to 2400 characters across both sides."""
    count = max(0, min(6, lines))
    candidates = []
    for distance in range(1, count + 1):
        for index in (batch[0] - distance, batch[-1] + distance):
            if 0 <= index < len(spans):
                candidates.append(index)
    remaining = 2400
    selected: dict[int, tuple[str, str]] = {}
    for index in candidates:
        text = " ".join(spans[index].text.split())
        if not text or remaining <= 0:
            continue
        text = text[:min(400, remaining)]
        role = genders[index] if genders and index < len(genders) else ""
        selected[index] = (text, role)
        remaining -= len(text)
    before = [selected[i] for i in sorted(selected) if i < batch[0]]
    after = [selected[i] for i in sorted(selected) if i > batch[-1]]
    return before, after


async def _translate_all(
    cfg: AppConfig,
    spans: list[STTSpan],
    source_lang: str | None,
    genders: list[str] | None,
    progress: _StageCb | None,
    approval: ApprovalGate | None = None,
) -> list[str | None]:
    """Возвращает переводы по репликам; None — перевод не удался совсем."""
    if cfg.translation.engine in ("none", "passthrough"):
        return [span.text for span in spans]

    translator = create_translation_engine(cfg, approval=approval)
    try:
        # Keep model loading inside the same cancellation-safe lifecycle as
        # inference and release.
        await translator.warmup()

        detected = Counter(span.language or "und" for span in spans).most_common(1)[0][0]
        lang = source_lang or (None if detected == "und" else detected)
        # Язык каждой реплики отдельно: явный source_lang побеждает, иначе
        # берётся язык самой реплики, а «und» подменяется доминирующим по
        # файлу. Один язык на весь ролик ломал многоязычные диалоги.
        span_langs: list[str | None] = [
            lang if source_lang else ((span.language or None) or lang)
            for span in spans
        ]
        distinct = sorted({value for value in span_langs if value})
        if len(distinct) > 1:
            log.info("в ролике несколько исходных языков: %s", ", ".join(distinct))
        translated: list[str | None] = [None] * len(spans)
        # Реплики, где модель ответила отказом/пояснением вместо перевода:
        # такие уходят на повторную попытку по одной.
        rejected: set[int] = set()
        completed = 0
        # Cloud выдерживает несколько пачек, но локальная LLM сообщает hint=1:
        # на малой unified-memory машине параллельные контексты ухудшают
        # стабильность и не ускоряют Ollama, который обычно всё равно очередит.
        parallelism = max(1, int(getattr(translator, "concurrency_hint", 3)))
        limit = asyncio.Semaphore(parallelism)

        def accept(index: int, value: str | None) -> None:
            """Кладёт перевод после проверки: отказ модели не попадёт в TTS."""
            gender = genders[index] if genders else "male"
            clean = sanitize_translation(value, spans[index].text, gender)
            if clean is None:
                if value:
                    rejected.add(index)
                    log.warning(
                        "реплика @%.1f с: модель ответила не переводом, повторю отдельно",
                        spans[index].start,
                    )
                translated[index] = None
                return
            rejected.discard(index)
            translated[index] = clean

        context_lines = int(getattr(cfg.translation, "file_context_lines", 3))

        async def translate_contextual(batch: list[int]) -> list[str]:
            texts = [spans[i].text for i in batch]
            roles = [genders[i] for i in batch] if genders else None
            before, after = _file_translation_context(spans, batch, genders, context_lines)
            method = getattr(translator, "translate_batch_contextual", None)
            if callable(method):
                result = await method(texts, span_langs[batch[0]], cfg.target_lang, roles,
                                      before=before, after=after)
            else:
                result = await translator.translate_batch_tagged(texts, span_langs[batch[0]], cfg.target_lang, roles)
            if not isinstance(result, (list, tuple)) or len(result) != len(batch):
                raise RuntimeError("модель вернула неверное число переводов для пачки")
            return list(result)

        async def translate_one(index: int) -> str:
            # Retry keeps both sides of the same source window where supported.
            try:
                return (await translate_contextual([index]))[0]
            except Exception:
                history = [(spans[i].text, translated[i])
                           for i in range(max(0, index - context_lines), index)
                           if translated[i]]
                return await translator.translate(spans[index].text, span_langs[index] or "und",
                                                  cfg.target_lang, history)

        async def run_batch(batch: list[int]) -> None:
            nonlocal completed
            try:
                async with limit:
                    result = await translate_contextual(batch)
                for idx, value in zip(batch, result):
                    accept(idx, value)
            except Exception as exc:  # noqa: BLE001 — деградируем до пореплечного
                log.warning(
                    "пакетный перевод %d реплик не прошёл (%s: %s) — перевожу по одной",
                    len(batch), type(exc).__name__, exc,
                )
                for idx in batch:
                    try:
                        async with limit:
                            accept(
                                idx,
                                await translate_one(idx),
                            )
                    except Exception as exc2:  # noqa: BLE001
                        log.error(
                            "реплика @%.1f с осталась без перевода (%s: %s)",
                            spans[idx].start, type(exc2).__name__, exc2,
                        )
            completed += len(batch)
            if progress is not None:
                progress(completed / len(spans))
            log.info("перевод: %d/%d реплик", completed, len(spans))

        max_items = int(getattr(translator, "batch_hint", BATCH_MAX_ITEMS))
        tasks = [
            asyncio.create_task(run_batch(batch))
            for batch in _make_batches(spans, max_items, span_langs)
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            # Piper drains an already-started native phrase on cancellation.
            # Wait for those bounded workers before closing shared voices.
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # Допереводчик: слабые модели часть строк возвращают без изменений или
        # с чужим алфавитом — такие реплики переводим повторно по одной.
        target_root = cfg.target_lang.split("-")[0].lower()
        cjk = re.compile(r"[㐀-鿿぀-ヿ]")

        def needs_retry(index: int) -> bool:
            out = translated[index]
            if not out:
                # Отказ модели стоит попробовать ещё раз по одной реплике:
                # без соседей по пачке safety-фильтр часто не срабатывает.
                return index in rejected
            span = spans[index]
            if _same_lang(span.language, cfg.target_lang):
                return False
            unchanged = " ".join(out.split()).lower() == " ".join(span.text.split()).lower()
            alien = target_root not in ("zh", "ja") and bool(cjk.search(out))
            return unchanged or alien

        retry = [i for i in range(len(spans)) if needs_retry(i)]
        if retry:
            log.info("доперевожу %d реплик, оставшихся без перевода", len(retry))

            async def retry_one(index: int) -> None:
                try:
                    async with limit:
                        accept(
                            index,
                            await translate_one(index),
                        )
                except Exception as exc:  # noqa: BLE001 — остаётся как было
                    log.debug("доперевод @%.1f с не удался: %s", spans[index].start, exc)

            await asyncio.gather(*(retry_one(i) for i in retry))

        if getattr(translator, "supports_shorten", False):
            await _fit_texts_to_slots(cfg, spans, translated, genders, translator)
        return translated
    finally:
        await translator.close()


def _text_budget(cfg: AppConfig, spans: list[STTSpan], index: int, gender: str) -> int | None:
    """Сколько символов перевода уложится в слот реплики.

    Оценка по той же таблице ``duration_per_char``, что используется озвучкой:
    если движок перевода умеет сжимать текст, лучше сделать это словами до
    синтеза, чем ускорять готовую речь после.
    """
    slot = _slot_seconds(spans, index)
    if slot is None:
        return None
    rates = dict(getattr(cfg.tts, "duration_per_char", {}) or {})
    root = str(cfg.target_lang or "").replace("_", "-").split("-", 1)[0].lower()
    rate = float(
        rates.get(f"{root}:{gender}")
        or rates.get(f"{root}:default")
        or rates.get(gender)
        or rates.get("default")
        or 0.0
    )
    if rate <= 0:
        return None
    max_compression = float(
        getattr(cfg.tts, "max_compression", MAX_COMPRESSION) or MAX_COMPRESSION
    )
    return int(slot * max_compression / rate)


async def _fit_texts_to_slots(
    cfg: AppConfig,
    spans: list[STTSpan],
    translated: list[str | None],
    genders: list[str] | None,
    translator,
) -> None:
    """Просит модель переписать короче те реплики, что не влезают в тайминг."""
    tasks: list[tuple[int, int]] = []
    for index, text in enumerate(translated):
        if not text:
            continue
        gender = genders[index] if genders else "male"
        budget = _text_budget(cfg, spans, index, gender)
        # 15% запаса: из-за оценки по символам нет смысла трогать почти
        # подходящие реплики.
        if budget and len(text) > budget * 1.15:
            tasks.append((index, budget))
    if not tasks:
        return
    log.info("укладываю %d реплик в тайминг текстом, а не темпом речи", len(tasks))

    async def shorten_one(index: int, budget: int) -> None:
        original = translated[index]
        assert original is not None
        try:
            shortened = await translator.shorten(original, cfg.target_lang, budget)
        except Exception as exc:  # noqa: BLE001 — реплика останется длинной
            log.debug("сжатие реплики @%.1f с не удалось: %s", spans[index].start, exc)
            return
        gender = genders[index] if genders else "male"
        clean = sanitize_translation(shortened, spans[index].text, gender)
        if clean and len(clean) < len(original):
            translated[index] = clean

    await asyncio.gather(*(shorten_one(index, budget) for index, budget in tasks))


# --- стадия 3: озвучка по таймкодам ---

MAX_SPEEDUP = 1.6     # потолок ускорения озвучки для движков без контроля длительности
MAX_COMPRESSION = 1.25  # насколько быстрее естественного темпа можно просить модель
SLOT_MARGIN_S = 0.15  # зазор до следующей реплики
SLOT_TOLERANCE_S = 0.3  # превышение слота, которое не стоит исправлять
MAX_EMOTION_S = 8.0   # длиннее фрагмент интонации движку не нужен
CLIP_LEAD_S = 0.15    # озвучка стартует чуть раньше оригинала — синхроннее на слух
_FATAL_TTS_HTTP_STATUSES = {401, 402, 403}
_RETRYABLE_TTS_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
_TTS_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "elevenlabs": "ElevenLabs",
    "edge": "Microsoft Edge TTS",
}


def _is_fatal_tts_error(error: BaseException) -> bool:
    """Stop on provider failures or a local engine's exhausted recovery."""
    return (
        getattr(error, "fatal_tts", False) is True
        or _tts_http_status(error) in _FATAL_TTS_HTTP_STATUSES
    )


def _tts_http_status(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return int(status) if isinstance(status, int) else None


def _is_retryable_tts_error(error: BaseException) -> bool:
    return _tts_http_status(error) in _RETRYABLE_TTS_HTTP_STATUSES


def _tts_retry_delay(error: BaseException, attempt: int) -> float:
    """Use provider Retry-After when present, otherwise bounded exponential backoff."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    try:
        if raw is not None:
            return min(30.0, max(0.5, float(raw)))
    except (TypeError, ValueError):
        pass
    return min(30.0, max(1.0, 2.0 ** max(0, attempt - 1)))


def _tts_failure_reason(provider: str, error: BaseException) -> str:
    # ElevenLabs distinguishes unavailable library voices, API permissions,
    # credits and rate limits in its JSON body. Do not replace that verified
    # explanation with the old generic assumption that every 402 is quota.
    provider_reason = getattr(error, "user_message", None)
    if (
        (provider == "ElevenLabs" or getattr(error, "fatal_tts", False) is True)
        and isinstance(provider_reason, str)
        and provider_reason
    ):
        return provider_reason
    status = _tts_http_status(error)
    if status == 429:
        return (
            f"{provider} временно ограничил частоту запросов (HTTP 429); "
            "уменьшена параллельность, повторите задачу через минуту"
        )
    if status == 402:
        return (
            f"{provider} отклонил запрос по квоте/биллингу (HTTP 402): "
            "проверьте остаток символов бесплатного тарифа и лимит аккаунта"
        )
    if status in (401, 403):
        return f"{provider} отклонил ключ или доступ (HTTP {status}): проверьте API-ключ"
    return f"{provider} отклонил запрос (HTTP {status or '?'})"


def _fit_audio_tempo(samples: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    """Compress a synthesized clip with ffmpeg atempo without changing pitch."""
    if factor <= 1.01 or len(samples) == 0:
        return samples
    source = np.ascontiguousarray(samples, dtype=np.float32)
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "f32le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-af",
                f"atempo={factor:.6f}",
                "-f",
                "f32le",
                "pipe:1",
            ],
            input=source.tobytes(),
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"ffmpeg не смог скорректировать темп TTS ×{factor:.2f}") from exc
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _slot_seconds(spans: list[STTSpan], index: int) -> float | None:
    """Сколько секунд доступно реплике до старта следующей.

    None — ограничения нет (последняя реплика или слот слишком короткий,
    чтобы говорить о укладке).
    """
    if index + 1 >= len(spans):
        return None
    slot = spans[index + 1].start - spans[index].start - SLOT_MARGIN_S
    return slot if slot > 0.5 else None


async def _create_tts_engines(
    cfg: AppConfig, genders: list[str]
) -> tuple[dict[str, object], object]:
    """Движки озвучки: общий для клонирующих, по одному на пол для остальных.

    Piper и облачные движки задают голос моделью, поэтому мужской и женский
    нужны отдельными экземплярами. У клонирующего движка голос приходит
    образцом, и второй экземпляр только занял бы вторую копию весов в памяти —
    на 16 ГБ это разница между работой и свопом.
    """
    order = list(dict.fromkeys(genders)) or ["male"]
    engines: dict[str, object] = {}
    try:
        first_cfg = cfg.tts.model_copy(deep=True)
        first_cfg.voice_gender = order[0]
        first = registry.create("tts", cfg.tts.engine, first_cfg)
        # Register ownership before loading native weights: отменённая
        # инициализация должна освободиться вместе с готовыми голосами.
        engines[order[0]] = first
        await first.warmup()
        if getattr(first, "supports_reference", False):
            return {gender: first for gender in order}, first
        for gender in order[1:]:
            tts_cfg = cfg.tts.model_copy(deep=True)
            tts_cfg.voice_gender = gender
            engine = registry.create("tts", cfg.tts.engine, tts_cfg)
            engines[gender] = engine
            await engine.warmup()
        return engines, first
    except BaseException:
        await asyncio.gather(
            *(engine.close() for engine in dict.fromkeys(engines.values())),
            return_exceptions=True,
        )
        raise


async def _synthesize_all(
    cfg: AppConfig,
    spans: list[STTSpan],
    texts: list[str],
    genders: list[str],
    progress: _StageCb | None,
    references: dict[str, VoiceReference] | None = None,
    source_audio: np.ndarray | None = None,
    source_rate: int = PIPE_RATE,
    labels: list[str] | None = None,
    on_clip: ClipCallback | None = None,
) -> list[_SynthClip]:
    """Озвучивает реплики и укладывает их в тайминги оригинала.

    Два разных пути укладки:

    - движок умеет ``synthesize_slot`` с длительностью (F5-TTS, IndexTTS-2) —
      реплика синтезируется сразу в доступный слот, речь остаётся живой;
    - обычный движок (Piper, облако) — прежний путь: предсказание темпа по
      ``duration_per_char`` и остаточный atempo.

    ``references`` — образцы голоса из оригинала по метке говорящего,
    ``source_audio`` — исходная моно-дорожка, из неё берётся звук самой
    реплики как источник интонации для движков с переносом просодии.
    """
    voice_labels = list(labels) if labels is not None else list(genders)
    engines, probe = await _create_tts_engines(cfg, genders)
    # Reference-based engines share expensive weights between roles. Their
    # mutable config must retain one replica's role until all retries and
    # duration fitting finish, including cancellation of native inference.
    shared_role_locks = {
        id(engine): asyncio.Lock()
        for engine in engines.values()
        if sum(other is engine for other in engines.values()) > 1
    }

    duration_aware = bool(getattr(probe, "supports_duration", False))
    use_reference = bool(getattr(probe, "supports_reference", False)) and bool(references)
    use_emotion = (
        bool(getattr(probe, "supports_emotion", False)) and source_audio is not None
    )
    if duration_aware:
        log.info(
            "озвучка укладывается в тайминги самим движком — atempo не применяется"
        )
    if use_reference:
        log.info("голоса клонируются из оригинала: %d образцов", len(references or {}))
    if use_emotion:
        log.info("интонация переносится из исходных реплик")

    limit = asyncio.Semaphore(int(getattr(cfg.tts, "concurrency", TTS_CONCURRENCY)))
    request_interval = max(0.0, float(getattr(cfg.tts, "request_interval_s", 0.0) or 0.0))
    request_gate = asyncio.Lock()
    next_request_at = 0.0
    finished = 0
    resynthesized = 0   # уложены повторным синтезом в нужную длительность
    retimed = 0         # уложены ускорением готового звука (движки без duration)
    fatal_error: BaseException | None = None
    provider = _TTS_PROVIDER_LABELS.get(cfg.tts.engine, cfg.tts.engine)

    duration_rates = dict(getattr(cfg.tts, "duration_per_char", {}) or {})
    target_root = str(cfg.target_lang or "").replace("_", "-").split("-", 1)[0].lower()
    max_speedup = float(getattr(cfg.tts, "max_speedup", MAX_SPEEDUP) or MAX_SPEEDUP)
    max_compression = float(
        getattr(cfg.tts, "max_compression", MAX_COMPRESSION) or MAX_COMPRESSION
    )

    def natural_duration(text: str, gender: str) -> float | None:
        """Оценка естественной длительности реплики по настройкам голоса."""
        rate = float(
            duration_rates.get(f"{target_root}:{gender}")
            or duration_rates.get(f"{target_root}:default")
            or duration_rates.get(gender)
            or duration_rates.get("default")
            or 0.0
        )
        return len(text) * rate if rate > 0 else None

    def predicted_speed(index: int, text: str, gender: str) -> float:
        slot = _slot_seconds(spans, index)
        if slot is None:
            return 1.0
        estimated = natural_duration(text, gender)
        if estimated is None or estimated <= slot + 0.3:
            return 1.0
        return min(max_speedup, max(1.0, estimated / slot))

    def predicted_target(index: int, text: str, gender: str) -> float | None:
        """Длительность, которую просим у движка, если реплика не влезает.

        Пока реплика укладывается в слот — не ограничиваем: навязанная длина
        растянула бы короткую фразу и звучала бы медленнее живой речи.
        """
        slot = _slot_seconds(spans, index)
        if slot is None:
            return None
        estimated = natural_duration(text, gender)
        if estimated is None or estimated <= slot + SLOT_TOLERANCE_S:
            return None
        # Сильнее max_compression не жмём: остаток выйдет за слот, сборка
        # сдвинет следующую реплику — это слышно лучше скороговорки.
        return max(slot, estimated / max_compression)

    def emotion_reference(index: int) -> VoiceReference | None:
        if source_audio is None:
            return None
        span = spans[index]
        start = max(0, int(span.start * source_rate))
        end = min(len(source_audio), int(span.end * source_rate))
        if end - start < int(0.5 * source_rate):
            return None
        chunk = source_audio[start : min(end, start + int(MAX_EMOTION_S * source_rate))]
        return VoiceReference(
            samples=np.ascontiguousarray(chunk, dtype=np.float32),
            sample_rate=source_rate,
            label=f"emotion-{index}",
        )

    initial_speeds = [1.0] * len(texts)
    initial_targets: list[float | None] = [None] * len(texts)
    if duration_aware:
        initial_targets = [
            predicted_target(index, text, gender)
            for index, (text, gender) in enumerate(zip(texts, genders))
        ]
        fitted_count = sum(target is not None for target in initial_targets)
        if fitted_count:
            log.info("синтезирую %d реплик сразу в их тайминг", fitted_count)
    else:
        initial_speeds = [
            predicted_speed(index, text, gender)
            for index, (text, gender) in enumerate(zip(texts, genders))
        ]
        predicted_count = sum(speed > 1.0 for speed in initial_speeds)
        if predicted_count:
            log.info(
                "сразу ускоряю %d реплик по длительности выбранного голоса",
                predicted_count,
            )

    async def call_engine(
        index: int,
        text: str,
        gender: str,
        speed: float,
        target: float | None,
    ) -> tuple[np.ndarray, int]:
        engine = engines[gender]
        if duration_aware or use_reference or use_emotion:
            kwargs: dict[str, object] = {}
            if duration_aware and target is not None:
                kwargs["target_duration"] = target
            if use_reference:
                kwargs["reference"] = (references or {}).get(voice_labels[index])
            if use_emotion:
                kwargs["emotion"] = emotion_reference(index)
            return await engine.synthesize_slot(text, cfg.target_lang, **kwargs)
        if speed > 1.0:
            return await engine.synthesize_rated(text, cfg.target_lang, speed)
        return await engine.synthesize(text, cfg.target_lang)

    async def synth_once(
        index: int,
        span: STTSpan,
        text: str,
        gender: str,
        speed: float,
        target: float | None,
    ) -> _SynthClip | None:
        """Один синтез с повторами на временных ошибках провайдера."""
        nonlocal fatal_error, next_request_at
        try:
            max_attempts = 4 if cfg.tts.engine == "elevenlabs" else 2
            for attempt in range(1, max_attempts + 1):
                try:
                    async with limit:
                        if fatal_error is not None:
                            return None
                        if request_interval:
                            async with request_gate:
                                now = asyncio.get_running_loop().time()
                                wait_s = max(0.0, next_request_at - now)
                                next_request_at = max(now, next_request_at) + request_interval
                            if wait_s:
                                await asyncio.sleep(wait_s)
                        speech, rate = await call_engine(index, text, gender, speed, target)
                    break
                except Exception as exc:  # noqa: BLE001
                    if _is_fatal_tts_error(exc):
                        raise
                    if attempt == max_attempts or not _is_retryable_tts_error(exc):
                        raise
                    delay = _tts_retry_delay(exc, attempt)
                    log.warning(
                        "озвучка @%.1f с: %s — повторяю через %.1f с",
                        span.start,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
        except Exception as exc:  # noqa: BLE001 — одна реплика не валит дубляж
            if _is_fatal_tts_error(exc) or _tts_http_status(exc) == 429:
                if fatal_error is None:
                    fatal_error = exc
                    log.error(
                        "озвучка остановлена: %s",
                        _tts_failure_reason(provider, exc),
                    )
            else:
                log.error("озвучка реплики @%.1f с не удалась: %s", span.start, exc)
            return None
        return _SynthClip(index, span.start, speech, rate) if len(speech) else None

    async def fit_into_slot(index: int, clip: _SynthClip) -> _SynthClip:
        """Укладывает готовый клип в его слот.

        Укладка идёт сразу после синтеза реплики, а не отдельной фазой в конце:
        так реплику можно отдать наружу готовой (см. ``on_clip``) и включить в
        видео, пока остальные ещё считаются.
        """
        nonlocal resynthesized, retimed
        slot = _slot_seconds(spans, index)
        if slot is None:
            return clip
        duration = len(clip.samples) / clip.sample_rate
        if duration <= slot + SLOT_TOLERANCE_S:
            return clip

        if duration_aware:
            # Пересинтез в нужную длительность: речь остаётся живой, в отличие
            # от ускорения готового звука.
            target = max(slot, duration / max_compression)
            fitted = await synth_once(
                index, spans[index], texts[index], genders[index], 1.0, target
            )
            if fitted is not None:
                resynthesized += 1
                return fitted
            return clip

        current_speed = initial_speeds[index]
        max_extra_factor = max_speedup / max(current_speed, 1.0)
        tempo_factor = min(max_extra_factor, duration / slot)
        if tempo_factor <= 1.03:
            return clip
        async with limit:
            samples = await asyncio.to_thread(
                _fit_audio_tempo, clip.samples, clip.sample_rate, tempo_factor
            )
        retimed += 1
        return _SynthClip(clip.index, clip.source_start, samples, clip.sample_rate)

    async def synth(
        index: int,
        span: STTSpan,
        text: str,
        gender: str,
        speed: float = 1.0,
        target: float | None = None,
    ) -> _SynthClip | None:
        nonlocal finished
        try:
            clip = await synth_once(index, span, text, gender, speed, target)
            if clip is not None:
                clip = await fit_into_slot(index, clip)
                if on_clip is not None:
                    # Реплика готова окончательно — её уже можно проигрывать.
                    try:
                        on_clip(
                            ClipReady(
                                index=index,
                                source_start=span.start,
                                source_end=span.end,
                                original=span.text,
                                translated=text,
                                voice_style=gender,
                                speaker=voice_labels[index],
                                samples=clip.samples,
                                sample_rate=clip.sample_rate,
                            )
                        )
                    except Exception:  # noqa: BLE001 — подписчик не валит дубляж
                        log.exception("подписчик on_clip не обработал реплику %d", index)
            return clip
        finally:
            finished += 1
            if progress is not None:
                progress(finished / max(len(spans), 1))

    async def synth_with_role(
        index: int, span: STTSpan, text: str, gender: str,
        speed: float, target: float | None,
    ) -> _SynthClip | None:
        engine = engines[gender]
        lock = shared_role_locks.get(id(engine))
        if lock is None:
            return await synth(index, span, text, gender, speed, target)
        async with lock:
            previous_role = engine.cfg.voice_gender
            engine.cfg.voice_gender = gender
            try:
                return await synth(index, span, text, gender, speed, target)
            finally:
                engine.cfg.voice_gender = previous_role

    try:
        tasks = [
            asyncio.create_task(
                synth_with_role(i, s, t, g, initial_speeds[i], initial_targets[i])
            )
            for i, (s, t, g) in enumerate(zip(spans, texts, genders))
        ]
        try:
            clips = list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            # A cancelled gather can raise before every sibling has drained.
            # Keep role scopes and model ownership alive until they all finish.
            drained = asyncio.gather(*tasks, return_exceptions=True)
            while not drained.done():
                try:
                    await asyncio.shield(drained)
                except asyncio.CancelledError:
                    continue
            raise
        if fatal_error is not None:
            raise RuntimeError(
                f"озвучка остановлена: {_tts_failure_reason(provider, fatal_error)}"
            ) from fatal_error

        if resynthesized:
            log.info(
                "%d реплик пересинтезированы в свой тайминг вместо ускорения",
                resynthesized,
            )
        if retimed:
            log.info("%d реплик подогнаны atempo под тайминги", retimed)

        if progress is not None:
            progress(1.0)
        return [clip for clip in clips if clip is not None]
    finally:
        for engine in dict.fromkeys(engines.values()):
            await engine.close()


async def render_dub_track(
    cfg: AppConfig,
    input_path: Path,
    duck_db: float = -12.0,
    progress: ProgressFn | None = None,
    mix_original: bool = True,
    approval: ApprovalGate | None = None,
    stt_engine: STTEngine | None = None,
    on_clip: ClipCallback | None = None,
    on_stage: StageFn | None = None,
) -> tuple[np.ndarray, list[HistoryEntry]]:
    """Готовит дублированную дорожку.

    mix_original=True → полный микс (приглушенный оригинал + перевод, стерео);
    mix_original=False → только голос перевода на тишине (моно) — для
    браузера, где оригинал играет сам плеер.
    """
    input_path = Path(input_path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if cfg.tts.engine == "none":
        raise RuntimeError("для дубляжа нужен синтез речи: tts.engine != none")

    if cfg.tts.engine == "f5":
        from uvt.engines.tts_f5 import check_f5_audio_runtime

        await _run_blocking(check_f5_audio_runtime)

    registry.load_builtins()
    registry.load_plugin_dirs(cfg.plugin_dirs)

    # Прогресс стадий: STT 0–70 %, перевод 70–85 %, озвучка 85–97 %, сборка 100 %
    def stage(base: float, span_pct: float) -> _StageCb | None:
        if progress is None:
            return None
        return lambda fraction: progress(int(base + span_pct * min(max(fraction, 0.0), 1.0)), 100)

    total_started = time.perf_counter()
    stage_started = total_started
    log.info("декодирую %s…", input_path.name)
    if on_stage is not None:
        on_stage("decode", 0.0, "декодирую исходный звук…")
    # Отделение речи от фона: распознавание идёт по чистой речи, а в микс
    # попадает полный фон — приглушать его под репликами больше не нужно.
    background_mix: np.ndarray | None = None
    separation = getattr(cfg, "separation", None)
    if bool(getattr(separation, "enabled", False)):
        source = await _run_blocking(_decode_file, input_path, SEPARATION_RATE, 2)
        loop = asyncio.get_running_loop()
        cancel_separation = threading.Event()

        def separation_progress(done_seconds: float, total_seconds: float) -> None:
            if on_stage is not None and not cancel_separation.is_set():
                fraction = min(1.0, max(0.0, done_seconds / max(total_seconds, 0.001)))
                detail = f"отделяю речь от фона: {done_seconds:.0f} из {total_seconds:.0f} с звука"
                loop.call_soon_threadsafe(on_stage, "separate", fraction, detail)

        if on_stage is not None:
            on_stage("separate", 0.0, "готовлю модель отделения речи; этот режим требует времени…")
        try:
            parts = await _run_blocking(
                separate_speech,
                source,
                SEPARATION_RATE,
                progress=separation_progress,
                cancel_event=cancel_separation,
                _on_cancel=cancel_separation.set,
                model_name=str(getattr(separation, "model", "htdemucs") or "htdemucs"),
                device=str(getattr(separation, "device", "auto") or "auto"),
                shifts=int(getattr(separation, "shifts", 0) or 0),
                overlap=float(getattr(separation, "overlap", 0.25) or 0.25),
            )
        except Exception as exc:  # noqa: BLE001 — работаем по исходному звуку
            log.warning(
                "отделение речи не выполнено (%s: %s) — продолжаю по исходному звуку",
                type(exc).__name__,
                exc,
            )
            mono16 = resample(to_mono(source), SEPARATION_RATE, PIPE_RATE)
        else:
            mono16 = resample(to_mono(parts.speech), SEPARATION_RATE, PIPE_RATE)
            background_mix = resample(parts.background, SEPARATION_RATE, MIX_RATE)
        del source
    else:
        mono16 = await _run_blocking(_decode_file, input_path, PIPE_RATE, 1)
    log.info("декодирование завершено за %.1f с", time.perf_counter() - stage_started)
    duration = len(mono16) / PIPE_RATE
    if duration > 3600:
        log.warning("файл длиннее часа — обработка идёт в памяти, следите за RAM")

    source_lang = None if cfg.source_lang in (None, "", "auto") else cfg.source_lang
    translating = cfg.translation.engine not in ("none", "passthrough")

    # 1. Реплики с таймкодами
    stage_started = time.perf_counter()
    spans = await _transcribe_all(
        cfg,
        mono16,
        source_lang,
        stage(0, 70),
        approval=approval,
        stt_engine=stt_engine,
    )
    log.info("распознавание завершено за %.1f с", time.perf_counter() - stage_started)
    # Зацикливание STT («фраза фраза фраза…») сворачивается до одной копии:
    # иначе оно уходит в перевод и растягивает реплику далеко за её слот.
    spans = [
        STTSpan(s.start, s.end, collapse_repeats(" ".join(s.text.split())), s.language)
        for s in spans
        if not _is_junk(s.text)
    ]
    if not spans:
        raise RuntimeError("в файле не найдено речи — нечего дублировать")

    # Чистые вокализации («oh», «um ah um», «mm-hmm») не переводятся и не
    # озвучиваются: на их месте в миксе остаётся оригинальный звук, и это
    # звучит естественнее любого синтеза.
    voiced_spans = [s for s in spans if not is_vocalization(s.text)]
    skipped_vocal = len(spans) - len(voiced_spans)
    if skipped_vocal:
        log.info(
            "%d реплик — неречевые вокализации: оставляю оригинальный звук",
            skipped_vocal,
        )
    if not voiced_spans:
        raise RuntimeError(
            "в файле только неречевые вокализации — переводить нечего"
        )
    spans = voiced_spans

    # Реплики уже на целевом языке переводить и переозвучивать не нужно
    if translating:
        kept = [s for s in spans if not _same_lang(s.language, cfg.target_lang)]
        if not kept:
            raise RuntimeError(
                f"речь в ролике уже на языке перевода ({cfg.target_lang}) — "
                f"выберите другую пару, например {cfg.target_lang} → en"
            )
        if len(kept) < len(spans):
            log.info("%d реплик уже на '%s' — пропущены", len(spans) - len(kept), cfg.target_lang)
        spans = kept

    # Пол голоса — ДО перевода: теги [M]/[F] дают переводчику правильный род
    # («я готова», а не «я готов»), а озвучке — мужской/женский голос.
    configured_gender = str(getattr(cfg.tts, "voice_gender", "auto") or "auto").lower()
    speaker_config = getattr(cfg, "speaker", None)
    if configured_gender == "auto" and bool(getattr(speaker_config, "enabled", True)):
        # Реплики раскладываются по говорящим целиком по файлу, а роль голоса
        # определяется голосованием внутри кластера. Пореплечный F0 менял голос
        # посреди сцены и превращал диалог двух человек в десятки «спикеров».
        layout = assign_speakers(
            mono16,
            PIPE_RATE,
            spans,
            max_speakers=max(1, int(getattr(speaker_config, "max_speakers", 8) or 8)),
        )
        overrides = dict(getattr(speaker_config, "voice_map", {}) or {})
        for label, role in overrides.items():
            if str(role).lower() in {"male", "female"} and label in layout.speakers:
                layout.speakers[label] = str(role).lower()
        genders_all = [layout.speakers[label] for label in layout.labels]
        speaker_labels = list(layout.labels)
    elif configured_gender == "auto":
        # Разделение отключено настройкой: один голос на всех, роль по F0 файла.
        gender, _f0 = estimate_gender_f0(mono16, PIPE_RATE)
        genders_all = [gender or "male"] * len(spans)
        speaker_labels = ["speaker-1"] * len(spans)
    else:
        genders_all = [configured_gender] * len(spans)
        speaker_labels = [configured_gender] * len(spans)

    # 2. Перевод пачками (с полом говорящего)
    stage_started = time.perf_counter()
    translated = await _translate_all(
        cfg, spans, source_lang, genders_all, stage(70, 15), approval=approval
    )
    log.info("перевод завершён за %.1f с", time.perf_counter() - stage_started)
    kept_spans: list[STTSpan] = []
    kept_texts: list[str] = []
    kept_genders: list[str] = []
    kept_labels: list[str] = []
    dropped = 0
    for span, text, gender, label in zip(spans, translated, genders_all, speaker_labels):
        if translating and text is None:
            dropped += 1
            continue
        kept_texts.append(text if text is not None else span.text)
        kept_spans.append(span)
        kept_genders.append(gender)
        kept_labels.append(label)
    if dropped:
        log.warning("%d реплик без перевода — озвучены не будут, там останется оригинал", dropped)
    if not kept_spans:
        raise RuntimeError(
            "перевод не удался ни для одной реплики — проверьте LLM (ключ, модель, лимиты)"
        )
    for span, text in zip(kept_spans, kept_texts):
        log.info("[@%7.1f с] %s ⇒ %s", span.start, span.text, text)

    # 3. Озвучка параллельно
    stage_started = time.perf_counter()
    # Клонирующему движку нужны образцы голоса из самого ролика: тембр и манера
    # берутся у настоящего говорящего, а не у фиксированного голоса модели.
    references: dict[str, VoiceReference] | None = None
    try:
        tts_class = registry.engine_class("tts", cfg.tts.engine)
    except KeyError:
        tts_class = None
    if tts_class is not None and getattr(tts_class, "supports_reference", False):
        references = pick_references(mono16, PIPE_RATE, kept_spans, kept_labels)
    clips = await _synthesize_all(
        cfg,
        kept_spans,
        kept_texts,
        kept_genders,
        stage(85, 12),
        references=references,
        source_audio=mono16,
        source_rate=PIPE_RATE,
        labels=kept_labels,
        on_clip=on_clip,
    )
    log.info("озвучка завершена за %.1f с", time.perf_counter() - stage_started)
    if not clips:
        raise RuntimeError("озвучка не удалась ни для одной реплики")

    # 4. Сборка дорожки
    log.info("собираю дорожку (%d реплик, режим %s)…", len(clips), "микс" if mix_original else "только голос")
    native: np.ndarray | None = None
    if mix_original:
        native = (
            background_mix
            if background_mix is not None
            else await _run_blocking(_decode_file, input_path, MIX_RATE, 2)
        )
    # Речь уже удалена из фона — приглушать его нечего ради чего.
    duck_gain = 1.0 if background_mix is not None else float(10 ** (duck_db / 20))
    ramp = int(RAMP_S * MIX_RATE)
    total = len(native) if native is not None else int(duration * MIX_RATE)
    tts_track = np.zeros(total, dtype=np.float32)
    # Browser playback mixes the original itself. A full-length ducking envelope
    # would waste 691 MB per hour of audio at 48 kHz in this mode.
    envelope = np.ones(total, dtype=np.float32) if native is not None else None
    cursor = 0
    # Индекс реплики → её реальные границы в готовой дорожке. Раньше браузер
    # угадывал длительность из числа символов, из-за чего ducking расходился
    # с произнесённой фразой.
    tts_bounds: dict[int, tuple[float, float]] = {}

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0

    for item in clips:
        start_ts = item.source_start
        clip = resample(item.samples, item.sample_rate, MIX_RATE)
        # Выравнивание громкости: перевод не должен звучать тише оригинала.
        # Целевой уровень — RMS оригинальной речи в этом месте, +20 % сверху.
        orig_from = int(start_ts * PIPE_RATE)
        orig_to = orig_from + int(len(clip) * PIPE_RATE / MIX_RATE)
        target_rms = _rms(mono16[orig_from:orig_to]) * 1.2
        clip_rms = _rms(clip)
        if target_rms > 1e-4 and clip_rms > 1e-4:
            clip = clip * float(np.clip(target_rms / clip_rms, 0.6, 4.0))
        # старт чуть раньше оригинала (CLIP_LEAD_S) — так закадровый звучит синхроннее
        start = max(int((start_ts - CLIP_LEAD_S) * MIX_RATE), cursor, 0)
        end = start + len(clip)
        if end > total:  # TTS длиннее хвоста файла — дорожка удлиняется
            pad = end - total
            tts_track = np.concatenate([tts_track, np.zeros(pad, dtype=np.float32)])
            if envelope is not None:
                envelope = np.concatenate([envelope, np.ones(pad, dtype=np.float32)])
            if native is not None:
                native = np.concatenate([native, np.zeros((pad, 2), dtype=np.float32)])
            total = end
        tts_track[start:end] += clip
        if envelope is not None:
            ramp_from = max(0, start - ramp)
            if start > ramp_from:
                fade = np.linspace(1.0, duck_gain, start - ramp_from, dtype=np.float32)
                np.minimum(envelope[ramp_from:start], fade, out=envelope[ramp_from:start])
            np.minimum(envelope[start:end], duck_gain, out=envelope[start:end])
            ramp_to = min(total, end + ramp)
            if ramp_to > end:
                fade = np.linspace(duck_gain, 1.0, ramp_to - end, dtype=np.float32)
                np.minimum(envelope[end:ramp_to], fade, out=envelope[end:ramp_to])
        cursor = end
        tts_bounds[item.index] = (start / MIX_RATE, end / MIX_RATE)

    if native is not None:
        # Native audio is owned by this render. Reuse it instead of allocating
        # a second full stereo array (1.38 GB per hour at 48 kHz).
        np.multiply(native, envelope[:, None], out=native)
        mixed = native
        mixed[:, 0] += tts_track
        mixed[:, 1] += tts_track
        np.clip(mixed, -1.0, 1.0, out=mixed)
        track = mixed
    else:
        np.clip(tts_track, -1.0, 1.0, out=tts_track)
        track = tts_track

    entries = [
        HistoryEntry(
            start=span.start,
            end=span.end,
            original=span.text,
            translated=text,
            language=span.language or "und",
            target_lang=cfg.target_lang,
            tts_start=tts_bounds.get(i, (None, None))[0],
            tts_end=tts_bounds.get(i, (None, None))[1],
            # F0 — это выбор тембра TTS, не идентификация человека.
            voice_style=gender,
        )
        for i, (span, text, gender) in enumerate(zip(kept_spans, kept_texts, kept_genders))
    ]

    if progress is not None:
        progress(100, 100)
    log.info("дорожка подготовлена за %.1f с", time.perf_counter() - total_started)
    return track, entries


async def dub_file(
    cfg: AppConfig,
    input_path: Path,
    output: str | Path | None = None,
    duck_db: float = -12.0,
    keep_original: bool = True,
) -> Path:
    input_path = Path(input_path).expanduser()
    mixed, entries = await render_dub_track(cfg, input_path, duck_db=duck_db, mix_original=True)

    has_video = await _run_blocking(_has_video, input_path)
    if output is None:
        suffix = ".dub.mkv" if has_video else ".dub.wav"
        output = input_path.with_name(input_path.stem + suffix)
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    if has_video:
        with tempfile.TemporaryDirectory(prefix="uvt-mix-") as td:
            wav = Path(td) / "mix.wav"
            await _run_blocking(sf.write, wav, mixed, MIX_RATE, subtype="PCM_16")
            args = ["-i", str(input_path), "-i", str(wav), "-map", "0:v", "-map", "1:a"]
            if keep_original:
                args += ["-map", "0:a?"]
            args += [
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-metadata:s:a:0", f"language={cfg.target_lang}",
                "-metadata:s:a:0", "title=UVT dub",
                str(output_path),
            ]
            await _run_blocking(_ffmpeg, args)
    else:
        await _run_blocking(sf.write, output_path, mixed, MIX_RATE, subtype="PCM_16")

    for fmt in ("srt", "json"):
        output_path.with_suffix(f".{fmt}").write_text(EXPORTERS[fmt](entries), encoding="utf-8")

    log.info("готово: %s (+ .srt, .json)", output_path)
    return output_path
