"""История сессии и экспорт субтитров: txt / srt / vtt / json (ТЗ §12)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from uvt.events import TOPIC_TRANSLATION, Translation
from uvt.services.base import Service


@dataclass(slots=True)
class HistoryEntry:
    start: float  # секунды от начала сессии
    end: float
    original: str
    translated: str
    language: str
    target_lang: str
    # Фактические границы дорожки TTS. Для live-сессии они могут быть неизвестны,
    # зато batch/browser использует их для корректного ducking и индикации sync.
    tts_start: float | None = None
    tts_end: float | None = None
    # Это идентификатор спикера/тембра, а не утверждение о поле человека.
    speaker_id: str | None = None
    speaker_confidence: float | None = None
    voice_style: str | None = None


def _fmt_ts(seconds: float, sep: str = ",") -> str:
    ms = max(0, round(seconds * 1000))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def _pick_text(entry: HistoryEntry, which: str) -> str:
    if which == "original":
        return entry.original
    if which == "both":
        return f"{entry.original}\n{entry.translated}"
    return entry.translated


def to_srt(entries: list[HistoryEntry], which: str = "translated") -> str:
    blocks = [
        f"{i}\n{_fmt_ts(e.start)} --> {_fmt_ts(e.end)}\n{_pick_text(e, which)}\n"
        for i, e in enumerate(entries, 1)
    ]
    return "\n".join(blocks)


def to_vtt(entries: list[HistoryEntry], which: str = "translated") -> str:
    blocks = [
        f"{_fmt_ts(e.start, sep='.')} --> {_fmt_ts(e.end, sep='.')}\n{_pick_text(e, which)}\n"
        for e in entries
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_txt(entries: list[HistoryEntry], which: str = "both") -> str:
    lines = [
        f"[{_fmt_ts(e.start)}] {e.original}\n→ {e.translated}\n"
        for e in entries
    ]
    return "\n".join(lines)


def to_json(entries: list[HistoryEntry], which: str = "both") -> str:
    return json.dumps([asdict(e) for e in entries], ensure_ascii=False, indent=2)


EXPORTERS = {"srt": to_srt, "vtt": to_vtt, "txt": to_txt, "json": to_json}


class HistoryService(Service):
    """Копит реплики сессии и сохраняет их при остановке конвейера."""

    name = "history"
    consumes = TOPIC_TRANSLATION
    produces = None

    def __init__(self, bus, cfg, metrics) -> None:
        super().__init__(bus, cfg, metrics)
        self.entries: list[HistoryEntry] = []
        self._anchor: float | None = None

    async def handle(self, translation: Translation):
        segment = translation.transcript.segment
        if self._anchor is None:
            self._anchor = segment.start_ts
        self.entries.append(
            HistoryEntry(
                start=max(0.0, segment.start_ts - self._anchor),
                end=max(0.0, segment.end_ts - self._anchor),
                original=translation.transcript.text,
                translated=translation.text,
                language=translation.transcript.language,
                target_lang=translation.target_lang,
            )
        )
        return None

    async def teardown(self) -> None:
        if not (self.cfg.history.enabled and self.entries):
            return
        session_dir = Path(self.cfg.history.dir).expanduser() / time.strftime("%Y%m%d-%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        for fmt in self.cfg.history.formats:
            exporter = EXPORTERS.get(fmt)
            if exporter is None:
                self.log.warning("неизвестный формат экспорта: %s", fmt)
                continue
            (session_dir / f"session.{fmt}").write_text(exporter(self.entries), encoding="utf-8")
        self.log.info("история сессии сохранена: %s (%d реплик)", session_dir, len(self.entries))
