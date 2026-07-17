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
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from uvt import registry
from uvt.audio import resample
from uvt.config import PRESETS, AppConfig
from uvt.fallback import ApprovalGate, create_stt_engine, create_translation_engine
from uvt.gender import estimate_gender_f0
from uvt.history import EXPORTERS, HistoryEntry
from uvt.interfaces import STTSpan
from uvt.segmenter import Segmenter, SegmenterParams, create_vad_engine
from uvt.services.stt import _is_junk
from uvt.services.translate import _same_lang

log = logging.getLogger("uvt.dub")

PIPE_RATE = 16000     # частота конвейера распознавания
MIX_RATE = 48000      # частота итоговой дорожки
RAMP_S = 0.05         # плавность приглушения, 50 мс
BATCH_MAX_ITEMS = 20  # реплик на один запрос к LLM…
BATCH_MAX_CHARS = 2500  # …но не больше этого объёма текста
TTS_CONCURRENCY = 4

ProgressFn = Callable[[int, int], None]
_StageCb = Callable[[float], None]


@dataclass(slots=True)
class _SynthClip:
    """Озвученная реплика до размещения на итоговой временной шкале."""

    index: int
    source_start: float
    samples: np.ndarray
    sample_rate: int


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
            downloaded = download_url(source, Path(td))
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
) -> list[STTSpan]:
    stt = create_stt_engine(cfg, approval=approval)
    await stt.warmup()
    try:
        def stt_progress(done_s: float, total_s: float) -> None:
            if progress is not None:
                progress(done_s / max(total_s, 1e-6))

        spans = await stt.transcribe_long(mono16, PIPE_RATE, source_lang, progress=stt_progress)
        if spans is not None:
            log.info("пакетное распознавание: %d реплик", len(spans))
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

            spans = []
            for n, seg in enumerate(segments, 1):
                result = await stt.transcribe(seg.samples, seg.sample_rate, source_lang)
                if progress is not None:
                    progress(n / max(len(segments), 1))
                if result is not None and result.text.strip():
                    spans.append(
                        STTSpan(seg.start_ts, seg.end_ts, result.text, result.language)
                    )
            return spans
        finally:
            await vad_engine.close()
    finally:
        await stt.close()


# --- стадия 2: перевод пачками с деградацией до пореплечного ---

def _make_batches(spans: list[STTSpan], max_items: int = BATCH_MAX_ITEMS) -> list[list[int]]:
    """Группирует индексы реплик: не больше max_items и BATCH_MAX_CHARS."""
    batches: list[list[int]] = []
    current: list[int] = []
    chars = 0
    for i, span in enumerate(spans):
        if current and (len(current) >= max_items or chars + len(span.text) > BATCH_MAX_CHARS):
            batches.append(current)
            current, chars = [], 0
        current.append(i)
        chars += len(span.text)
    if current:
        batches.append(current)
    return batches


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
    await translator.warmup()
    try:
        detected = Counter(span.language or "und" for span in spans).most_common(1)[0][0]
        lang = source_lang or (None if detected == "und" else detected)
        translated: list[str | None] = [None] * len(spans)
        completed = 0
        # Cloud выдерживает несколько пачек, но локальная LLM сообщает hint=1:
        # на малой unified-memory машине параллельные контексты ухудшают
        # стабильность и не ускоряют Ollama, который обычно всё равно очередит.
        parallelism = max(1, int(getattr(translator, "concurrency_hint", 3)))
        limit = asyncio.Semaphore(parallelism)

        async def run_batch(batch: list[int]) -> None:
            nonlocal completed
            texts = [spans[i].text for i in batch]
            batch_genders = [genders[i] for i in batch] if genders else None
            try:
                async with limit:
                    result = await translator.translate_batch_tagged(
                        texts, lang, cfg.target_lang, batch_genders
                    )
                for idx, value in zip(batch, result):
                    translated[idx] = value
            except Exception as exc:  # noqa: BLE001 — деградируем до пореплечного
                log.warning(
                    "пакетный перевод %d реплик не прошёл (%s: %s) — перевожу по одной",
                    len(batch), type(exc).__name__, exc,
                )
                for idx in batch:
                    try:
                        async with limit:
                            translated[idx] = await translator.translate(
                                spans[idx].text, lang or "und", cfg.target_lang, []
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
        tasks = [asyncio.create_task(run_batch(batch)) for batch in _make_batches(spans, max_items)]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise

        # Допереводчик: слабые модели часть строк возвращают без изменений или
        # с чужим алфавитом — такие реплики переводим повторно по одной.
        target_root = cfg.target_lang.split("-")[0].lower()
        cjk = re.compile(r"[㐀-鿿぀-ヿ]")

        def needs_retry(index: int) -> bool:
            out = translated[index]
            if not out:
                return False
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
                        translated[index] = await translator.translate(
                            spans[index].text, lang or "und", cfg.target_lang, []
                        )
                except Exception as exc:  # noqa: BLE001 — остаётся как было
                    log.debug("доперевод @%.1f с не удался: %s", spans[index].start, exc)

            await asyncio.gather(*(retry_one(i) for i in retry))
        return translated
    finally:
        await translator.close()


# --- стадия 3: озвучка по таймкодам ---

MAX_SPEEDUP = 1.6     # потолок ускорения озвучки
SLOT_MARGIN_S = 0.15  # зазор до следующей реплики
CLIP_LEAD_S = 0.15    # озвучка стартует чуть раньше оригинала — синхроннее на слух


async def _synthesize_all(
    cfg: AppConfig,
    spans: list[STTSpan],
    texts: list[str],
    genders: list[str],
    progress: _StageCb | None,
) -> list[_SynthClip]:
    # Отдельный движок на каждый нужный пол голоса (мужской/женский диалог)
    engines: dict[str, object] = {}
    for gender in dict.fromkeys(genders):
        tts_cfg = cfg.tts.model_copy(deep=True)
        tts_cfg.voice_gender = gender
        engine = registry.create("tts", cfg.tts.engine, tts_cfg)
        await engine.warmup()
        engines[gender] = engine

    limit = asyncio.Semaphore(int(getattr(cfg.tts, "concurrency", TTS_CONCURRENCY)))
    finished = 0

    async def synth(index: int, span: STTSpan, text: str, gender: str, speed: float = 1.0):
        nonlocal finished
        engine = engines[gender]
        try:
            for attempt in (1, 2):  # сетевой TTS иногда мигает — один повтор
                try:
                    async with limit:
                        if speed > 1.0:
                            speech, rate = await engine.synthesize_rated(text, cfg.target_lang, speed)
                        else:
                            speech, rate = await engine.synthesize(text, cfg.target_lang)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        raise
                    log.warning("озвучка @%.1f с: %s — повторяю", span.start, exc)
                    await asyncio.sleep(1.0)
        except Exception as exc:  # noqa: BLE001 — одна реплика не валит дубляж
            log.error("озвучка реплики @%.1f с не удалась: %s", span.start, exc)
            return None
        finally:
            finished += 1
            if progress is not None:
                progress(finished / max(len(spans) * 1.2, 1))
        return _SynthClip(index, span.start, speech, rate) if len(speech) else None

    try:
        tasks = [
            asyncio.create_task(synth(i, s, t, g))
            for i, (s, t, g) in enumerate(zip(spans, texts, genders))
        ]
        try:
            clips = list(await asyncio.gather(*tasks))
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise

        # Укладка в тайминги: озвучка длиннее промежутка до следующей реплики
        # сдвигала бы все последующие — пересинтезируем её быстрее.
        refit: list[tuple[int, float]] = []
        for i, clip in enumerate(clips):
            if clip is None or i + 1 >= len(spans):
                continue
            duration = len(clip.samples) / clip.sample_rate
            slot = spans[i + 1].start - spans[i].start - SLOT_MARGIN_S
            if slot > 0.5 and duration > slot + 0.3:
                refit.append((i, min(MAX_SPEEDUP, duration / slot)))
        if refit:
            log.info("ускоряю %d реплик, чтобы уложиться в тайминги оригинала", len(refit))
            refit_tasks = [
                asyncio.create_task(synth(i, spans[i], texts[i], genders[i], speed))
                for i, speed in refit
            ]
            refit_clips = await asyncio.gather(*refit_tasks)
            for (i, _speed), clip in zip(refit, refit_clips):
                if clip is not None:
                    clips[i] = clip

        return [clip for clip in clips if clip is not None]
    finally:
        for engine in engines.values():
            await engine.close()


async def render_dub_track(
    cfg: AppConfig,
    input_path: Path,
    duck_db: float = -12.0,
    progress: ProgressFn | None = None,
    mix_original: bool = True,
    approval: ApprovalGate | None = None,
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

    registry.load_builtins()
    registry.load_plugin_dirs(cfg.plugin_dirs)

    # Прогресс стадий: STT 0–70 %, перевод 70–85 %, озвучка 85–97 %, сборка 100 %
    def stage(base: float, span_pct: float) -> _StageCb | None:
        if progress is None:
            return None
        return lambda fraction: progress(int(base + span_pct * min(max(fraction, 0.0), 1.0)), 100)

    log.info("декодирую %s…", input_path.name)
    mono16 = _decode_file(input_path, PIPE_RATE, 1)
    duration = len(mono16) / PIPE_RATE
    if duration > 3600:
        log.warning("файл длиннее часа — обработка идёт в памяти, следите за RAM")

    source_lang = None if cfg.source_lang in (None, "", "auto") else cfg.source_lang
    translating = cfg.translation.engine not in ("none", "passthrough")

    # 1. Реплики с таймкодами
    spans = await _transcribe_all(cfg, mono16, source_lang, stage(0, 70), approval=approval)
    spans = [
        STTSpan(s.start, s.end, " ".join(s.text.split()), s.language)
        for s in spans
        if not _is_junk(s.text)
    ]
    if not spans:
        raise RuntimeError("в файле не найдено речи — нечего дублировать")

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
    if configured_gender == "auto":
        genders_all: list[str] = []
        last_gender = "male"
        for span in spans:
            segment = mono16[int(span.start * PIPE_RATE) : int(span.end * PIPE_RATE)]
            gender, f0 = estimate_gender_f0(segment, PIPE_RATE)
            log.debug(
                "голос @%.1f с: F0≈%s → %s",
                span.start,
                f"{f0:.0f} Гц" if f0 else "—",
                gender or f"не определился, наследую {last_gender}",
            )
            gender = gender or last_gender
            genders_all.append(gender)
            last_gender = gender
        female_count = genders_all.count("female")
        log.info(
            "голоса определены: %d муж., %d жен.",
            len(genders_all) - female_count, female_count,
        )
    else:
        genders_all = [configured_gender] * len(spans)

    # 2. Перевод пачками (с полом говорящего)
    translated = await _translate_all(
        cfg, spans, source_lang, genders_all, stage(70, 15), approval=approval
    )
    kept_spans: list[STTSpan] = []
    kept_texts: list[str] = []
    kept_genders: list[str] = []
    dropped = 0
    for span, text, gender in zip(spans, translated, genders_all):
        if translating and text is None:
            dropped += 1
            continue
        kept_texts.append(text if text is not None else span.text)
        kept_spans.append(span)
        kept_genders.append(gender)
    if dropped:
        log.warning("%d реплик без перевода — озвучены не будут, там останется оригинал", dropped)
    if not kept_spans:
        raise RuntimeError(
            "перевод не удался ни для одной реплики — проверьте LLM (ключ, модель, лимиты)"
        )
    for span, text in zip(kept_spans, kept_texts):
        log.info("[@%7.1f с] %s ⇒ %s", span.start, span.text, text)

    # 3. Озвучка параллельно
    clips = await _synthesize_all(cfg, kept_spans, kept_texts, kept_genders, stage(85, 12))
    if not clips:
        raise RuntimeError("озвучка не удалась ни для одной реплики")

    # 4. Сборка дорожки
    log.info("собираю дорожку (%d реплик, режим %s)…", len(clips), "микс" if mix_original else "только голос")
    native = _decode_file(input_path, MIX_RATE, 2) if mix_original else None
    duck_gain = float(10 ** (duck_db / 20))
    ramp = int(RAMP_S * MIX_RATE)
    total = len(native) if native is not None else int(duration * MIX_RATE)
    tts_track = np.zeros(total, dtype=np.float32)
    envelope = np.ones(total, dtype=np.float32)
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
            envelope = np.concatenate([envelope, np.ones(pad, dtype=np.float32)])
            if native is not None:
                native = np.concatenate([native, np.zeros((pad, 2), dtype=np.float32)])
            total = end
        tts_track[start:end] += clip
        ramp_from = max(0, start - ramp)
        if start > ramp_from:
            fade = np.linspace(1.0, duck_gain, start - ramp_from, dtype=np.float32)
            envelope[ramp_from:start] = np.minimum(envelope[ramp_from:start], fade)
        envelope[start:end] = np.minimum(envelope[start:end], duck_gain)
        ramp_to = min(total, end + ramp)
        if ramp_to > end:
            fade = np.linspace(duck_gain, 1.0, ramp_to - end, dtype=np.float32)
            envelope[end:ramp_to] = np.minimum(envelope[end:ramp_to], fade)
        cursor = end
        tts_bounds[item.index] = (start / MIX_RATE, end / MIX_RATE)

    if native is not None:
        mixed = native * envelope[:, None]
        mixed[:, 0] += tts_track
        mixed[:, 1] += tts_track
        np.clip(mixed, -1.0, 1.0, out=mixed)
        track = mixed
    else:
        track = np.clip(tts_track, -1.0, 1.0)

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

    has_video = _has_video(input_path)
    if output is None:
        suffix = ".dub.mkv" if has_video else ".dub.wav"
        output = input_path.with_name(input_path.stem + suffix)
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    if has_video:
        with tempfile.TemporaryDirectory(prefix="uvt-mix-") as td:
            wav = Path(td) / "mix.wav"
            sf.write(wav, mixed, MIX_RATE, subtype="PCM_16")
            args = ["-i", str(input_path), "-i", str(wav), "-map", "0:v", "-map", "1:a"]
            if keep_original:
                args += ["-map", "0:a?"]
            args += [
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-metadata:s:a:0", f"language={cfg.target_lang}",
                "-metadata:s:a:0", "title=UVT dub",
                str(output_path),
            ]
            _ffmpeg(args)
    else:
        sf.write(output_path, mixed, MIX_RATE, subtype="PCM_16")

    for fmt in ("srt", "json"):
        output_path.with_suffix(f".{fmt}").write_text(EXPORTERS[fmt](entries), encoding="utf-8")

    log.info("готово: %s (+ .srt, .json)", output_path)
    return output_path
